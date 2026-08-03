"""Behavior classification and diverse ranking for HetroD selection.

The official v0.8 selector configures this ranking engine with stricter
semantic gates. The implementation remains in a separate module so its
candidate evidence and ranking constraints stay independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from .agent_selection import future_valid_frames, is_evaluated_type
from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .interaction_candidates import (
    AgentSelectionResult,
    FutureConflictSelectionConfig,
    _analyze_pair,
    _valid_motion_extent,
    _valid_path_length,
)


COMPLEX_BEHAVIORS = frozenset(
    {"crossing_merge", "head_on", "overtake"}
)
SIMPLE_BEHAVIORS = frozenset(
    {"following", "parallel", "moving_static", "low_motion"}
)


@dataclass(frozen=True)
class DiverseSelectionConfig:
    """Small, auditable set of diversity and interaction-strength parameters."""

    max_pairs: int = 4
    max_pair_endpoints: int = 8
    max_pairs_per_agent: int = 2
    max_following_pairs: int = 1
    max_parallel_pairs: int = 1
    max_moving_static_pairs: int = 1
    max_low_motion_pairs: int = 1
    max_same_complex_behavior_pairs: int = 2
    min_anchor_path_length_m: float = 2.0
    min_anchor_motion_extent_m: float = 1.0
    min_history_valid_frames: int = 5
    num_fallback_agents: int = 4
    min_fallback_agents: int = 2
    direction_window_s: float = 1.0
    same_direction_angle_deg: float = 30.0
    head_on_angle_deg: float = 150.0
    same_corridor_lateral_m: float = 3.0
    close_corridor_lateral_m: float = 1.5
    minimum_order_separation_m: float = 1.0
    counterfactual_horizon_s: float = 4.0
    counterfactual_distance_m: float = 5.0
    strong_arrival_gap_s: float = 2.5
    strong_response_score: float = 0.20
    static_risk_distance_m: float = 3.0
    use_local_static_semantics: bool = False
    static_speed_threshold_mps: float = 0.30
    require_corridor_persistence: bool = False
    overtake_uses_dedicated_tier: bool = False
    reject_low_motion_pairs: bool = False
    prefer_pair_type_diversity: bool = False
    allow_static_cap_for_new_anchor_type: bool = False
    max_moving_static_pairs_with_type_coverage: int = 2


def _local_direction(
    xy: np.ndarray,
    validity: np.ndarray,
    center_index: int,
    half_window: int,
) -> tuple[np.ndarray, float]:
    indices = np.flatnonzero(
        validity
        & (np.arange(validity.size) >= max(0, center_index - half_window))
        & (
            np.arange(validity.size)
            <= min(validity.size - 1, center_index + half_window)
        )
    )
    if indices.size < 2:
        return np.zeros(2, dtype=np.float32), 0.0
    displacement = xy[indices[-1]] - xy[indices[0]]
    norm = float(np.linalg.norm(displacement))
    if norm < 0.2:
        return np.zeros(2, dtype=np.float32), 0.0
    return displacement / norm, norm


def _local_speed(
    xy: np.ndarray,
    validity: np.ndarray,
    center_index: int,
    half_window: int,
    seconds_per_step: float,
) -> float:
    """Estimate path speed near a conflict without using whole-track extent."""
    indices = np.flatnonzero(
        validity
        & (np.arange(validity.size) >= max(0, center_index - half_window))
        & (
            np.arange(validity.size)
            <= min(validity.size - 1, center_index + half_window)
        )
    )
    if indices.size < 2:
        return 0.0
    frame_gaps = np.diff(indices)
    usable = frame_gaps > 0
    if not np.any(usable):
        return 0.0
    distances = np.linalg.norm(
        xy[indices[1:]] - xy[indices[:-1]],
        axis=-1,
    )
    durations = frame_gaps * seconds_per_step
    return float(distances[usable].sum() / durations[usable].sum())


def _position_at_or_near(
    xy: np.ndarray,
    validity: np.ndarray,
    index: int,
) -> np.ndarray | None:
    if validity[index]:
        return xy[index]
    usable = np.flatnonzero(validity)
    if usable.size == 0:
        return None
    nearest = int(usable[np.argmin(np.abs(usable - index))])
    return xy[nearest]


def _velocity_near_start(
    xy: np.ndarray,
    validity: np.ndarray,
    seconds_per_step: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    indices = np.flatnonzero(validity)
    if indices.size < 2:
        return None
    first = int(indices[0])
    later = indices[
        (indices > first)
        & (indices <= first + max(2, round(1.0 / seconds_per_step)))
    ]
    if later.size == 0:
        return None
    last = int(later[-1])
    velocity = (xy[last] - xy[first]) / (
        (last - first) * seconds_per_step
    )
    return xy[first], velocity


def _counterfactual_risk(
    first_xy: np.ndarray,
    second_xy: np.ndarray,
    first_valid: np.ndarray,
    second_valid: np.ndarray,
    horizon_s: float,
    seconds_per_step: float,
) -> dict[str, float | None]:
    first = _velocity_near_start(
        first_xy,
        first_valid,
        seconds_per_step,
    )
    second = _velocity_near_start(
        second_xy,
        second_valid,
        seconds_per_step,
    )
    if first is None or second is None:
        return {
            "counterfactual_ttc_s": None,
            "counterfactual_min_distance_m": float("inf"),
            "counterfactual_risk": 0.0,
        }
    first_position, first_velocity = first
    second_position, second_velocity = second
    relative_position = second_position - first_position
    relative_velocity = second_velocity - first_velocity
    velocity_squared = float(np.dot(relative_velocity, relative_velocity))
    closing = float(np.dot(relative_position, relative_velocity)) < 0.0
    if not closing or velocity_squared < 1e-6:
        closest_time = 0.0
    else:
        closest_time = float(
            np.clip(
                -np.dot(relative_position, relative_velocity)
                / velocity_squared,
                0.0,
                horizon_s,
            )
        )
    minimum_distance = float(
        np.linalg.norm(
            relative_position + closest_time * relative_velocity
        )
    )
    if not closing:
        risk = 0.0
        ttc = None
    else:
        distance_quality = math.exp(-minimum_distance / 2.0)
        timing_quality = max(1.0 - closest_time / horizon_s, 0.0)
        risk = distance_quality * timing_quality
        ttc = closest_time
    return {
        "counterfactual_ttc_s": ttc,
        "counterfactual_min_distance_m": minimum_distance,
        "counterfactual_risk": risk,
    }


def _relative_geometry(
    first_xy: np.ndarray,
    second_xy: np.ndarray,
    first_valid: np.ndarray,
    second_valid: np.ndarray,
    first_arrival: int,
    second_arrival: int,
    first_direction: np.ndarray,
    second_direction: np.ndarray,
    half_window: int,
) -> dict[str, float | bool]:
    middle = int(round(0.5 * (first_arrival + second_arrival)))
    synchronous = first_valid & second_valid
    local_indices = np.flatnonzero(
        synchronous
        & (np.arange(synchronous.size) >= max(0, middle - half_window))
        & (
            np.arange(synchronous.size)
            <= min(synchronous.size - 1, middle + half_window)
        )
    )
    if local_indices.size:
        distances = np.linalg.norm(
            second_xy[local_indices] - first_xy[local_indices],
            axis=-1,
        )
        closest_index = int(local_indices[np.argmin(distances)])
    else:
        closest_index = middle
    average_direction = first_direction + second_direction
    average_norm = float(np.linalg.norm(average_direction))
    if average_norm < 0.2:
        average_direction = first_direction
    else:
        average_direction = average_direction / average_norm

    def coordinates(index: int) -> tuple[float, float]:
        first_position = _position_at_or_near(
            first_xy,
            first_valid,
            index,
        )
        second_position = _position_at_or_near(
            second_xy,
            second_valid,
            index,
        )
        if first_position is None or second_position is None:
            return 0.0, float("inf")
        relative = second_position - first_position
        longitudinal = float(np.dot(relative, average_direction))
        lateral = abs(float(np.cross(average_direction, relative)))
        return longitudinal, lateral

    before_index = max(0, middle - half_window)
    after_index = min(synchronous.size - 1, middle + half_window)
    before_longitudinal, before_lateral = coordinates(before_index)
    current_longitudinal, current_lateral = coordinates(closest_index)
    after_longitudinal, after_lateral = coordinates(after_index)
    order_swapped = (
        before_longitudinal * after_longitudinal < 0.0
        and abs(before_longitudinal - after_longitudinal) >= 2.0
    )
    merged_corridor = (
        before_lateral >= 3.0
        and after_lateral <= 1.5
    )
    corridor_persistent = max(
        before_lateral,
        current_lateral,
        after_lateral,
    ) <= 3.0
    return {
        "closest_index": closest_index,
        "longitudinal_separation_m": abs(current_longitudinal),
        "lateral_separation_m": current_lateral,
        "before_lateral_separation_m": before_lateral,
        "after_lateral_separation_m": after_lateral,
        "order_swapped": order_swapped,
        "merged_corridor": merged_corridor,
        "corridor_persistent": corridor_persistent,
    }


def _classify_and_score(
    candidate: dict[str, Any],
    tracks: np.ndarray,
    validity: np.ndarray,
    active_indices: set[int],
    selection_config: FutureConflictSelectionConfig,
    diverse_config: DiverseSelectionConfig,
    metric_config: HetrodMetricConfig,
) -> dict[str, Any]:
    first_index, second_index = candidate["agent_indices"]
    first_arrival, second_arrival = candidate["arrival_indices"]
    first_xy = tracks[first_index, :, :2]
    second_xy = tracks[second_index, :, :2]
    first_valid = validity[first_index]
    second_valid = validity[second_index]
    half_window = max(
        2,
        int(
            round(
                diverse_config.direction_window_s
                / metric_config.seconds_per_step
            )
        ),
    )
    first_direction, first_motion = _local_direction(
        first_xy,
        first_valid,
        first_arrival,
        half_window,
    )
    second_direction, second_motion = _local_direction(
        second_xy,
        second_valid,
        second_arrival,
        half_window,
    )
    first_speed = _local_speed(
        first_xy,
        first_valid,
        first_arrival,
        half_window,
        metric_config.seconds_per_step,
    )
    second_speed = _local_speed(
        second_xy,
        second_valid,
        second_arrival,
        half_window,
        metric_config.seconds_per_step,
    )
    counterfactual = _counterfactual_risk(
        first_xy,
        second_xy,
        first_valid,
        second_valid,
        diverse_config.counterfactual_horizon_s,
        metric_config.seconds_per_step,
    )
    if diverse_config.use_local_static_semantics:
        first_static = (
            first_speed <= diverse_config.static_speed_threshold_mps
        )
        second_static = (
            second_speed <= diverse_config.static_speed_threshold_mps
        )
    else:
        first_static = first_index not in active_indices
        second_static = second_index not in active_indices
    if first_static != second_static:
        behavior = "moving_static"
        angle_deg = None
        geometry = {}
    elif first_static and second_static:
        behavior = "low_motion"
        angle_deg = None
        geometry = {}
    elif first_motion < 0.2 or second_motion < 0.2:
        behavior = "low_motion"
        angle_deg = None
        geometry = {}
    else:
        angle_deg = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(first_direction, second_direction),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        geometry = _relative_geometry(
            first_xy,
            second_xy,
            first_valid,
            second_valid,
            first_arrival,
            second_arrival,
            first_direction,
            second_direction,
            half_window,
        )
        if angle_deg >= diverse_config.head_on_angle_deg:
            behavior = "head_on"
        elif angle_deg > diverse_config.same_direction_angle_deg:
            behavior = "crossing_merge"
        elif geometry["merged_corridor"]:
            behavior = "crossing_merge"
        elif (
            geometry["order_swapped"]
            and (
                not diverse_config.require_corridor_persistence
                or geometry["corridor_persistent"]
            )
        ):
            behavior = "overtake"
        elif geometry["order_swapped"]:
            behavior = "crossing_merge"
        elif (
            geometry["lateral_separation_m"]
            <= diverse_config.same_corridor_lateral_m
            and geometry["longitudinal_separation_m"]
            >= diverse_config.minimum_order_separation_m
            and (
                not diverse_config.require_corridor_persistence
                or geometry["corridor_persistent"]
            )
        ):
            behavior = "following"
        else:
            behavior = "parallel"

    timing_quality = (
        max(
            1.0
            - candidate["arrival_gap_s"]
            / selection_config.max_arrival_gap_s,
            0.0,
        )
        if candidate["timed_path_conflict"]
        else 0.0
    )
    proximity_quality = (
        max(
            1.0
            - candidate["min_synchronous_distance_m"]
            / diverse_config.counterfactual_distance_m,
            0.0,
        )
        if math.isfinite(candidate["min_synchronous_distance_m"])
        else 0.0
    )
    closing_quality = min(
        candidate["closing_distance_m"]
        / max(selection_config.min_direct_closing_distance_m, 1e-6),
        1.0,
    )
    response_quality = min(candidate["motion_response_score"], 1.0)
    strength_score = (
        0.35 * timing_quality
        + 0.30 * float(counterfactual["counterfactual_risk"])
        + 0.20 * response_quality
        + 0.10 * closing_quality
        + 0.05 * proximity_quality
    )

    counterfactual_ttc = counterfactual["counterfactual_ttc_s"]
    counterfactual_distance = float(
        counterfactual["counterfactual_min_distance_m"]
    )
    direct_risk = (
        counterfactual_ttc is not None
        and counterfactual_ttc <= diverse_config.counterfactual_horizon_s
        and counterfactual_distance
        <= diverse_config.counterfactual_distance_m
    )
    has_response = (
        candidate["motion_response_score"]
        >= diverse_config.strong_response_score
    )
    moving_endpoint_has_response = any(
        speed > diverse_config.static_speed_threshold_mps
        and response["score"] >= diverse_config.strong_response_score
        for speed, response in zip(
            (first_speed, second_speed),
            candidate["responses"],
        )
    )
    tight_path_conflict = (
        candidate["timed_path_conflict"]
        and candidate["arrival_gap_s"]
        < diverse_config.strong_arrival_gap_s
    )

    if (
        behavior == "overtake"
        and diverse_config.overtake_uses_dedicated_tier
    ):
        tier = (
            "B"
            if candidate["direct_footprint_proximity"]
            and (direct_risk or moving_endpoint_has_response)
            else "C"
        )
    elif behavior in COMPLEX_BEHAVIORS:
        if tight_path_conflict and (
            direct_risk
            or has_response
            or candidate["direct_footprint_proximity"]
        ):
            tier = "A"
        elif tight_path_conflict or (
            candidate["direct_footprint_proximity"]
            and (
                has_response
                or candidate["closing_distance_m"]
                >= selection_config.min_direct_closing_distance_m
            )
        ):
            tier = "B"
        else:
            tier = "C"
    elif behavior == "following":
        tier = (
            "B"
            if direct_risk
            and (
                has_response
                or candidate["closing_distance_m"]
                >= selection_config.min_direct_closing_distance_m
            )
            else "C"
        )
    elif behavior == "moving_static":
        tier = (
            "B"
            if counterfactual_distance
            <= diverse_config.static_risk_distance_m
            and candidate["direct_footprint_proximity"]
            and (
                moving_endpoint_has_response
                if diverse_config.use_local_static_semantics
                else has_response
            )
            else "C"
        )
    elif behavior == "parallel":
        tier = "B" if direct_risk and has_response else "C"
    elif diverse_config.reject_low_motion_pairs:
        tier = "C"
    else:
        tier = "B" if tight_path_conflict and has_response else "C"

    candidate.update(counterfactual)
    candidate.update(geometry)
    candidate["behavior_category"] = behavior
    candidate["path_angle_deg"] = angle_deg
    candidate["local_speeds_mps"] = [first_speed, second_speed]
    candidate["interaction_tier"] = tier
    candidate["strength_score"] = strength_score
    candidate["is_cross_type"] = (
        candidate["agent_types"][0] != candidate["agent_types"][1]
    )
    return candidate


def select_diverse_interaction_agents(
    gt_scenario: dict[str, Any],
    diverse_config: DiverseSelectionConfig = DiverseSelectionConfig(),
    metric_config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> AgentSelectionResult:
    """Select high-confidence pairs while limiting simple behavior families."""
    selection_config = FutureConflictSelectionConfig(
        path_margin_m=metric_config.selection_path_margin_m,
        max_arrival_gap_s=metric_config.selection_max_arrival_gap_s,
        max_synchronous_distance_m=(
            metric_config.selection_max_synchronous_distance_m
        ),
        min_motion_response_score=(
            metric_config.selection_min_motion_response_score
        ),
        min_direct_closing_distance_m=(
            metric_config.selection_min_direct_closing_distance_m
        ),
        min_direct_motion_response_score=(
            metric_config.selection_min_direct_motion_response_score
        ),
    )
    track_masks = gt_scenario["track_masks"]
    history_valid_frames = track_masks[
        :, : metric_config.current_time_index + 1
    ].sum(dim=1)
    base_mask = (
        is_evaluated_type(gt_scenario["object_types"], metric_config)
        & track_masks[:, metric_config.current_time_index]
        & (
            history_valid_frames
            >= diverse_config.min_history_valid_frames
        )
        & (
            future_valid_frames(track_masks, metric_config)
            >= metric_config.min_future_valid_frames
        )
    )
    base_indices = torch.where(base_mask)[0].detach().cpu().tolist()
    tracks = gt_scenario["tracks"][
        :, metric_config.future_start_index :
    ].detach().cpu().numpy()
    validity = gt_scenario["track_masks"][
        :, metric_config.future_start_index :
    ].detach().cpu().numpy().astype(bool)
    object_ids = gt_scenario["object_ids"].detach().cpu().numpy()
    object_types = gt_scenario["object_types"].detach().cpu().numpy()
    path_lengths = np.asarray(
        [
            _valid_path_length(tracks[index, :, :2], validity[index])
            for index in range(tracks.shape[0])
        ],
        dtype=np.float32,
    )
    motion_extents = np.asarray(
        [
            _valid_motion_extent(tracks[index, :, :2], validity[index])
            for index in range(tracks.shape[0])
        ],
        dtype=np.float32,
    )
    active_indices = {
        index
        for index in base_indices
        if (
            path_lengths[index]
            >= diverse_config.min_anchor_path_length_m
            and motion_extents[index]
            >= diverse_config.min_anchor_motion_extent_m
        )
    }

    candidates = []
    for first_position, first_index in enumerate(base_indices):
        for second_index in base_indices[first_position + 1 :]:
            candidate = _analyze_pair(
                first_index=first_index,
                second_index=second_index,
                tracks=tracks,
                validity=validity,
                object_ids=object_ids,
                object_types=object_types,
                selection_config=selection_config,
                metric_config=metric_config,
            )
            if candidate is None:
                continue
            pair_active = set(candidate["agent_indices"]) & active_indices
            if not pair_active:
                continue
            candidates.append(
                _classify_and_score(
                    candidate,
                    tracks,
                    validity,
                    active_indices,
                    selection_config,
                    diverse_config,
                    metric_config,
                )
            )

    tier_order = {"A": 0, "B": 1, "C": 2}
    candidates.sort(
        key=lambda item: (
            tier_order[item["interaction_tier"]],
            -item["strength_score"],
            -item["score"],
            item["agent_ids"],
        )
    )
    strong_candidates = [
        candidate
        for candidate in candidates
        if candidate["interaction_tier"] in {"A", "B"}
    ]
    chosen: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    endpoint_indices: set[int] = set()
    degrees: dict[int, int] = {}
    behavior_counts: dict[str, int] = {}

    behavior_caps = {
        "following": diverse_config.max_following_pairs,
        "parallel": diverse_config.max_parallel_pairs,
        "moving_static": diverse_config.max_moving_static_pairs,
        "low_motion": diverse_config.max_low_motion_pairs,
        "crossing_merge": diverse_config.max_same_complex_behavior_pairs,
        "head_on": diverse_config.max_same_complex_behavior_pairs,
        "overtake": diverse_config.max_same_complex_behavior_pairs,
    }

    def try_add(candidate: dict[str, Any]) -> bool:
        if candidate in chosen:
            return False
        behavior = candidate["behavior_category"]
        endpoints = set(candidate["agent_indices"])
        anchors = endpoints & active_indices
        if not anchors:
            return False
        behavior_count = behavior_counts.get(behavior, 0)
        if behavior_count >= behavior_caps[behavior]:
            represented_anchor_types = {
                int(object_types[index]) for index in selected_indices
            }
            candidate_anchor_types = {
                int(object_types[index]) for index in anchors
            }
            allow_type_coverage = (
                diverse_config.allow_static_cap_for_new_anchor_type
                and behavior == "moving_static"
                and behavior_count
                < diverse_config.max_moving_static_pairs_with_type_coverage
                and bool(candidate_anchor_types - represented_anchor_types)
            )
            if not allow_type_coverage:
                return False
        if (
            len(endpoint_indices | endpoints)
            > diverse_config.max_pair_endpoints
        ):
            return False
        if any(
            degrees.get(index, 0) >= diverse_config.max_pairs_per_agent
            for index in endpoints
        ):
            return False
        chosen.append(candidate)
        selected_indices.update(anchors)
        endpoint_indices.update(endpoints)
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        for index in endpoints:
            degrees[index] = degrees.get(index, 0) + 1
        candidate["anchor_agent_ids"] = [
            int(object_ids[index]) for index in sorted(anchors)
        ]
        candidate["context_only_agent_ids"] = [
            int(object_ids[index])
            for index in candidate["agent_indices"]
            if index not in anchors
        ]
        candidate["pair_rank"] = len(chosen)
        return True

    strongest_complex = next(
        (
            candidate
            for candidate in strong_candidates
            if candidate["behavior_category"] in COMPLEX_BEHAVIORS
        ),
        None,
    )
    if strongest_complex is not None:
        try_add(strongest_complex)
    strongest_cross_type = next(
        (
            candidate
            for candidate in strong_candidates
            if candidate["is_cross_type"]
        ),
        None,
    )
    if strongest_cross_type is not None:
        try_add(strongest_cross_type)

    if diverse_config.prefer_pair_type_diversity:
        while len(chosen) < diverse_config.max_pairs:
            represented_pair_types = {
                candidate["pair_type"] for candidate in chosen
            }
            represented_anchor_types = {
                int(object_types[index]) for index in selected_indices
            }
            ranked = sorted(
                (
                    candidate
                    for candidate in strong_candidates
                    if candidate not in chosen
                ),
                key=lambda item: (
                    tier_order[item["interaction_tier"]],
                    item["behavior_category"] in behavior_counts,
                    item["pair_type"] in represented_pair_types,
                    not bool(
                        {
                            int(object_types[index])
                            for index in item["agent_indices"]
                            if index in active_indices
                        }
                        - represented_anchor_types
                    ),
                    -item["strength_score"],
                    -item["score"],
                ),
            )
            if not any(try_add(candidate) for candidate in ranked):
                break
    else:
        best_by_unseen_behavior: dict[str, dict[str, Any]] = {}
        for candidate in strong_candidates:
            best_by_unseen_behavior.setdefault(
                candidate["behavior_category"],
                candidate,
            )
        for candidate in sorted(
            best_by_unseen_behavior.values(),
            key=lambda item: (
                tier_order[item["interaction_tier"]],
                -item["strength_score"],
                -item["score"],
            ),
        ):
            if len(chosen) >= diverse_config.max_pairs:
                break
            try_add(candidate)
        for candidate in strong_candidates:
            if len(chosen) >= diverse_config.max_pairs:
                break
            try_add(candidate)

    selected = torch.zeros_like(base_mask)
    if not chosen and diverse_config.num_fallback_agents:
        def fallback_key(index: int) -> tuple[float, float, int, int]:
            return (
                -float(motion_extents[index]),
                -float(path_lengths[index]),
                -int(validity[index].sum()),
                int(object_ids[index]),
            )

        active_fallback = sorted(active_indices, key=fallback_key)
        fallback_indices: list[int] = []
        represented_types: set[int] = set()
        for index in active_fallback:
            object_type = int(object_types[index])
            if object_type not in represented_types:
                fallback_indices.append(index)
                represented_types.add(object_type)
                if (
                    len(fallback_indices)
                    >= diverse_config.num_fallback_agents
                ):
                    break
        for index in active_fallback:
            if (
                len(fallback_indices)
                >= diverse_config.num_fallback_agents
            ):
                break
            if index not in fallback_indices:
                fallback_indices.append(index)
        minimum = min(
            diverse_config.min_fallback_agents,
            diverse_config.num_fallback_agents,
            len(base_indices),
        )
        if len(fallback_indices) < minimum:
            remaining = sorted(
                (
                    index
                    for index in base_indices
                    if index not in fallback_indices
                ),
                key=fallback_key,
            )
            fallback_indices.extend(
                remaining[: minimum - len(fallback_indices)]
            )
        selected_indices.update(fallback_indices)
    if selected_indices:
        selected[list(sorted(selected_indices))] = True
    return AgentSelectionResult(
        anchor_mask=selected,
        pairs=chosen,
        mode="interactive_pairs" if chosen else "fallback_noninteractive",
    )
