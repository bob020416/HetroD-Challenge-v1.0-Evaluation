"""GT future-path interaction candidates for the HetroD evaluator.

This module implements the geometry and motion evidence shared by the official
v0.8 selector. Pair discovery may use same-type context, while the scored
cross-type component remains explicitly cross-type.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from .agent_selection import (
    future_valid_frames,
    is_evaluated_type,
)
from .config import DEFAULT_CONFIG, HetrodMetricConfig


@dataclass(frozen=True)
class FutureConflictSelectionConfig:
    """Small set of interpretable future-conflict selection parameters."""

    path_margin_m: float = 0.50
    max_arrival_gap_s: float = 3.0
    max_synchronous_distance_m: float = 5.0
    min_motion_response_score: float = 0.40
    # The score is diagnostic/ranking-only by default. A pair is eligible when
    # it passes the interpretable swept-path and arrival-gap hard conditions.
    min_pair_score: float = 0.0
    max_pairs: int = 0
    max_agents: int = 0
    max_pair_endpoints: int = 0
    max_pairs_per_agent: int = 0
    reserve_cross_type_pair: bool = False
    num_fallback_agents: int = 2
    min_fallback_agents: int = 2
    response_window_s: float = 1.0
    min_history_valid_frames: int = 5
    min_anchor_path_length_m: float = 2.0
    min_anchor_motion_extent_m: float = 1.0
    min_direct_closing_distance_m: float = 2.0
    min_direct_motion_response_score: float = 0.20


@dataclass(frozen=True)
class AgentSelectionResult:
    """Competition selection output with explicit anchor/context semantics."""

    anchor_mask: torch.Tensor
    pairs: list[dict[str, Any]]
    mode: str

    @property
    def pair_object_ids(self) -> list[list[int]]:
        return [pair["agent_ids"] for pair in self.pairs]

    @property
    def context_only_object_ids(self) -> list[int]:
        anchor_ids = {
            agent_id
            for pair in self.pairs
            for agent_id in pair.get("anchor_agent_ids", [])
        }
        return sorted(
            {
                agent_id
                for pair in self.pairs
                for agent_id in pair["agent_ids"]
                if agent_id not in anchor_ids
            }
        )


def _type_name(object_type: int, config: HetrodMetricConfig) -> str:
    names = {
        config.vehicle_type: "vehicle",
        config.two_wheeler_type: "two_wheeler",
        config.pedestrian_type: "pedestrian",
    }
    return names[object_type]


def _default_width(object_type: int, config: HetrodMetricConfig) -> float:
    if object_type == config.two_wheeler_type:
        return config.coverage_default_two_wheeler_width_m
    if object_type == config.pedestrian_type:
        return config.coverage_default_pedestrian_width_m
    return config.coverage_default_vehicle_width_m


def _median_width(
    tracks: np.ndarray,
    validity: np.ndarray,
    agent_index: int,
    object_type: int,
    config: HetrodMetricConfig,
) -> float:
    widths = tracks[agent_index, :, 4]
    usable = widths[validity[agent_index] & np.isfinite(widths) & (widths > 0.0)]
    if usable.size == 0:
        return _default_width(object_type, config)
    return float(np.median(usable))


def _wrapped_angle_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (second - first + np.pi) % (2.0 * np.pi) - np.pi


def _valid_path_length(
    xy: np.ndarray,
    validity: np.ndarray,
) -> float:
    consecutive = validity[1:] & validity[:-1]
    return float(
        np.linalg.norm(xy[1:] - xy[:-1], axis=-1)[consecutive].sum()
    )


def _valid_motion_extent(
    xy: np.ndarray,
    validity: np.ndarray,
) -> float:
    """Return maximum displacement from the first valid future position."""
    valid_xy = xy[validity]
    if valid_xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(valid_xy - valid_xy[0], axis=-1).max())


def _expanded_box_overlap(
    first_tracks: np.ndarray,
    second_tracks: np.ndarray,
    validity: np.ndarray,
    margin_m: float,
) -> np.ndarray:
    """Return synchronous overlap after expanding both oriented boxes.

    The expansion is an interaction gate, not a collision tolerance. A
    0.5-m margin catches a moving agent approaching a stopped vehicle without
    replacing the strict zero-tolerance collision metric.
    """
    result = np.zeros(validity.shape, dtype=bool)
    if not validity.any():
        return result
    first = first_tracks[validity]
    second = second_tracks[validity]
    delta = second[:, :2] - first[:, :2]
    first_yaw = first[:, 6]
    second_yaw = second[:, 6]
    first_cosine = np.cos(first_yaw)
    first_sine = np.sin(first_yaw)
    second_cosine = np.cos(second_yaw)
    second_sine = np.sin(second_yaw)
    first_half_length = 0.5 * first[:, 3] + margin_m
    first_half_width = 0.5 * first[:, 4] + margin_m
    second_half_length = 0.5 * second[:, 3] + margin_m
    second_half_width = 0.5 * second[:, 4] + margin_m
    relative_cosine = np.abs(np.cos(second_yaw - first_yaw))
    relative_sine = np.abs(np.sin(second_yaw - first_yaw))
    delta_x = delta[:, 0]
    delta_y = delta[:, 1]
    first_forward = np.abs(delta_x * first_cosine + delta_y * first_sine)
    first_lateral = np.abs(-delta_x * first_sine + delta_y * first_cosine)
    second_forward = np.abs(delta_x * second_cosine + delta_y * second_sine)
    second_lateral = np.abs(-delta_x * second_sine + delta_y * second_cosine)
    overlap = (
        (
            first_forward
            < first_half_length
            + second_half_length * relative_cosine
            + second_half_width * relative_sine
        )
        & (
            first_lateral
            < first_half_width
            + second_half_length * relative_sine
            + second_half_width * relative_cosine
        )
        & (
            second_forward
            < second_half_length
            + first_half_length * relative_cosine
            + first_half_width * relative_sine
        )
        & (
            second_lateral
            < second_half_width
            + first_half_length * relative_sine
            + first_half_width * relative_cosine
        )
    )
    result[validity] = overlap
    return result


def _motion_response(
    xy: np.ndarray,
    heading: np.ndarray,
    validity: np.ndarray,
    arrival_index: int,
    seconds_per_step: float,
    window_s: float,
) -> dict[str, float]:
    """Measure local braking or turning around arrival at a conflict region."""
    num_steps = xy.shape[0]
    window = max(2, int(round(window_s / seconds_per_step)))
    speed = np.full(num_steps, np.nan, dtype=np.float32)
    consecutive = validity[1:] & validity[:-1]
    speed[1:][consecutive] = (
        np.linalg.norm(xy[1:][consecutive] - xy[:-1][consecutive], axis=-1)
        / seconds_per_step
    )
    before = speed[
        max(0, arrival_index - window) : max(0, arrival_index - 1)
    ]
    around = speed[
        max(0, arrival_index - 1) : min(num_steps, arrival_index + window + 1)
    ]
    before = before[np.isfinite(before)]
    around = around[np.isfinite(around)]
    speed_before = float(np.median(before)) if before.size else 0.0
    minimum_around = float(np.min(around)) if around.size else speed_before
    speed_drop = max(speed_before - minimum_around, 0.0) / max(
        speed_before,
        0.5,
    )

    start = max(0, arrival_index - window)
    end = min(num_steps - 1, arrival_index + window)
    valid_heading = validity[start : end + 1]
    heading_window = heading[start : end + 1][valid_heading]
    if heading_window.size >= 2:
        heading_steps = np.abs(
            _wrapped_angle_delta(heading_window[:-1], heading_window[1:])
        )
        heading_change = float(np.sum(heading_steps))
    else:
        heading_change = 0.0
    turn_response = min(heading_change / math.radians(45.0), 1.0)
    return {
        "speed_drop": min(speed_drop, 1.0),
        "heading_change_rad": heading_change,
        "turn_response": turn_response,
        "score": max(min(speed_drop, 1.0), turn_response),
    }


def _analyze_pair(
    *,
    first_index: int,
    second_index: int,
    tracks: np.ndarray,
    validity: np.ndarray,
    object_ids: np.ndarray,
    object_types: np.ndarray,
    selection_config: FutureConflictSelectionConfig,
    metric_config: HetrodMetricConfig,
) -> dict[str, Any] | None:
    first_valid = validity[first_index]
    second_valid = validity[second_index]
    first_xy = tracks[first_index, :, :2]
    second_xy = tracks[second_index, :, :2]
    first_points = first_xy[first_valid]
    second_points = second_xy[second_valid]
    if first_points.size == 0 or second_points.size == 0:
        return None
    synchronous_valid = first_valid & second_valid
    expanded_overlap = _expanded_box_overlap(
        tracks[first_index],
        tracks[second_index],
        synchronous_valid,
        selection_config.path_margin_m,
    )
    has_direct_footprint_proximity = bool(expanded_overlap.any())

    first_width = _median_width(
        tracks,
        validity,
        first_index,
        int(object_types[first_index]),
        metric_config,
    )
    second_width = _median_width(
        tracks,
        validity,
        second_index,
        int(object_types[second_index]),
        metric_config,
    )
    clearance = (
        0.5 * first_width
        + 0.5 * second_width
        + selection_config.path_margin_m
    )

    # Cheap swept-path AABB rejection before the pairwise time query.
    if not has_direct_footprint_proximity and (
        first_points[:, 0].max() + clearance < second_points[:, 0].min()
        or second_points[:, 0].max() + clearance < first_points[:, 0].min()
        or first_points[:, 1].max() + clearance < second_points[:, 1].min()
        or second_points[:, 1].max() + clearance < first_points[:, 1].min()
    ):
        return None

    first_times = np.flatnonzero(first_valid)
    second_times = np.flatnonzero(second_valid)
    delta = first_points[:, None, :] - second_points[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    path_distance = float(distances.min())
    conflict = distances <= clearance
    if not conflict.any() and not has_direct_footprint_proximity:
        return None

    time_gaps = np.abs(
        first_times[:, None] - second_times[None, :]
    ) * metric_config.seconds_per_step
    conflict_gaps = np.where(conflict, time_gaps, np.inf)
    arrival_gap = float(conflict_gaps.min())
    has_timed_path_conflict = (
        arrival_gap < selection_config.max_arrival_gap_s
    )
    if (
        arrival_gap >= selection_config.max_arrival_gap_s
        and not has_direct_footprint_proximity
    ):
        return None
    if arrival_gap < selection_config.max_arrival_gap_s:
        closest_in_time = conflict & np.isclose(conflict_gaps, arrival_gap)
        conflict_distances = np.where(closest_in_time, distances, np.inf)
        flat_index = int(np.argmin(conflict_distances))
        first_local_index, second_local_index = np.unravel_index(
            flat_index,
            conflict_distances.shape,
        )
        first_arrival_index = int(first_times[first_local_index])
        second_arrival_index = int(second_times[second_local_index])
        conflict_distance = float(
            distances[first_local_index, second_local_index]
        )
    else:
        # Direct synchronous proximity remains meaningful at a censored
        # rollout boundary even when the agents' path-center tubes do not
        # intersect within the clip.
        direct_times = np.flatnonzero(expanded_overlap)
        direct_distances = np.linalg.norm(
            first_xy[direct_times] - second_xy[direct_times],
            axis=-1,
        )
        direct_index = int(np.argmin(direct_distances))
        first_arrival_index = int(direct_times[direct_index])
        second_arrival_index = first_arrival_index
        conflict_distance = float(direct_distances[direct_index])
        arrival_gap = 0.0

    if synchronous_valid.any():
        synchronous_distances = np.linalg.norm(
            first_xy[synchronous_valid] - second_xy[synchronous_valid],
            axis=-1,
        )
        synchronous_distance = float(synchronous_distances.min())
        closing_distance = max(
            float(synchronous_distances[0] - synchronous_distance),
            0.0,
        )
    else:
        synchronous_distance = float("inf")
        closing_distance = 0.0

    first_response = _motion_response(
        first_xy,
        tracks[first_index, :, 6],
        first_valid,
        first_arrival_index,
        metric_config.seconds_per_step,
        selection_config.response_window_s,
    )
    second_response = _motion_response(
        second_xy,
        tracks[second_index, :, 6],
        second_valid,
        second_arrival_index,
        metric_config.seconds_per_step,
        selection_config.response_window_s,
    )
    response_score = max(first_response["score"], second_response["score"])
    # A geometric conflict is only an interaction candidate when GT also
    # contains direct proximity, or a visible braking/turning response near
    # the conflict. The second branch retains agents that yielded early and
    # therefore never became close at the same timestep.
    same_type = object_types[first_index] == object_types[second_index]
    direct_has_evidence = (
        closing_distance >= selection_config.min_direct_closing_distance_m
        or response_score
        >= selection_config.min_direct_motion_response_score
    )
    if (
        has_direct_footprint_proximity
        and not has_timed_path_conflict
        and not direct_has_evidence
    ):
        return None
    if same_type:
        # Dense same-type traffic is not automatically interactive. Retain a
        # direct near-contact only when the pair is closing or one member
        # visibly responds; retain a yield-before-contact only with the
        # stronger response gate.
        if has_direct_footprint_proximity:
            if (
                closing_distance
                < selection_config.min_direct_closing_distance_m
                and response_score
                < selection_config.min_direct_motion_response_score
            ):
                return None
        elif response_score < selection_config.min_motion_response_score:
            return None
    elif (
        not has_direct_footprint_proximity
        and synchronous_distance >= selection_config.max_synchronous_distance_m
        and response_score < selection_config.min_motion_response_score
    ):
        return None
    spatial_quality = max(1.0 - conflict_distance / max(clearance, 1e-6), 0.0)
    timing_quality = max(
        1.0 - arrival_gap / selection_config.max_arrival_gap_s,
        0.0,
    )
    synchronous_quality = (
        max(1.0 - synchronous_distance / 5.0, 0.0)
        if math.isfinite(synchronous_distance)
        else 0.0
    )
    score = (
        0.45 * spatial_quality
        + 0.35 * timing_quality
        + 0.10 * synchronous_quality
        + 0.10 * response_score
    )
    first_type = int(object_types[first_index])
    second_type = int(object_types[second_index])
    type_names = sorted(
        (
            _type_name(first_type, metric_config),
            _type_name(second_type, metric_config),
        )
    )
    return {
        "agent_indices": [first_index, second_index],
        "agent_ids": [int(object_ids[first_index]), int(object_ids[second_index])],
        "agent_types": [
            _type_name(first_type, metric_config),
            _type_name(second_type, metric_config),
        ],
        "pair_type": "_".join(type_names),
        "score": score,
        "path_distance_m": path_distance,
        "conflict_distance_m": conflict_distance,
        "clearance_m": clearance,
        "arrival_gap_s": arrival_gap,
        "arrival_indices": [first_arrival_index, second_arrival_index],
        "min_synchronous_distance_m": synchronous_distance,
        "closing_distance_m": closing_distance,
        "direct_footprint_proximity": has_direct_footprint_proximity,
        "timed_path_conflict": has_timed_path_conflict,
        "future_path_only": (
            synchronous_distance >= selection_config.max_synchronous_distance_m
        ),
        "motion_response_score": response_score,
        "responses": [first_response, second_response],
    }


def select_future_conflict_agents(
    gt_scenario: dict[str, Any],
    selection_config: FutureConflictSelectionConfig = FutureConflictSelectionConfig(),
    metric_config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Select moving anchors from every high-confidence future-conflict pair.

    Zero-valued caps mean unlimited. Slow/static pair members remain recorded
    as context but are not selected as scoring anchors.
    """
    if (
        selection_config.max_pairs < 0
        or selection_config.max_agents < 0
        or selection_config.max_pair_endpoints < 0
        or selection_config.max_pairs_per_agent < 0
        or selection_config.num_fallback_agents < 0
        or selection_config.min_fallback_agents < 0
    ):
        raise ValueError("Future-conflict pair/agent caps cannot be negative.")
    track_masks = gt_scenario["track_masks"]
    history_valid_frames = track_masks[
        :, : metric_config.current_time_index + 1
    ].sum(dim=1)
    base_mask = (
        is_evaluated_type(gt_scenario["object_types"], metric_config)
        & track_masks[:, metric_config.current_time_index]
        & (
            history_valid_frames
            >= selection_config.min_history_valid_frames
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

    candidates = []
    for first_position, first_index in enumerate(base_indices):
        for second_index in base_indices[first_position + 1 :]:
            report = _analyze_pair(
                first_index=first_index,
                second_index=second_index,
                tracks=tracks,
                validity=validity,
                object_ids=object_ids,
                object_types=object_types,
                selection_config=selection_config,
                metric_config=metric_config,
            )
            if report is not None:
                candidates.append(report)
    candidates.sort(key=lambda item: item["score"], reverse=True)

    chosen_pairs = []
    selected_indices: set[int] = set()
    chosen_endpoint_indices: set[int] = set()
    pair_degrees: dict[int, int] = {}
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

    def active_indices(candidate: dict[str, Any]) -> set[int]:
        return {
            index
            for index in candidate["agent_indices"]
            if (
                path_lengths[index]
                >= selection_config.min_anchor_path_length_m
                and motion_extents[index]
                >= selection_config.min_anchor_motion_extent_m
            )
        }

    def try_add(candidate: dict[str, Any]) -> bool:
        if candidate["score"] < selection_config.min_pair_score:
            return False
        candidate_indices = active_indices(candidate)
        if not candidate_indices:
            return False
        endpoints = set(candidate["agent_indices"])
        if (
            selection_config.max_agents
            and len(selected_indices | candidate_indices)
            > selection_config.max_agents
        ):
            return False
        if (
            selection_config.max_pair_endpoints
            and len(chosen_endpoint_indices | endpoints)
            > selection_config.max_pair_endpoints
        ):
            return False
        if (
            selection_config.max_pairs_per_agent
            and any(
                pair_degrees.get(index, 0)
                >= selection_config.max_pairs_per_agent
                for index in endpoints
            )
        ):
            return False
        chosen_pairs.append(candidate)
        selected_indices.update(candidate_indices)
        chosen_endpoint_indices.update(endpoints)
        for index in endpoints:
            pair_degrees[index] = pair_degrees.get(index, 0) + 1
        candidate["anchor_agent_ids"] = [
            int(object_ids[index]) for index in sorted(candidate_indices)
        ]
        candidate["context_only_agent_ids"] = [
            int(object_ids[index])
            for index in candidate["agent_indices"]
            if index not in candidate_indices
        ]
        candidate["pair_rank"] = len(chosen_pairs)
        return True

    eligible_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate["score"] >= selection_config.min_pair_score
            and active_indices(candidate)
        )
    ]
    reserved_candidate = None
    if selection_config.reserve_cross_type_pair:
        reserved_candidate = next(
            (
                candidate
                for candidate in eligible_candidates
                if candidate["agent_types"][0]
                != candidate["agent_types"][1]
            ),
            None,
        )
        if reserved_candidate is not None:
            try_add(reserved_candidate)

    for candidate in eligible_candidates:
        if candidate is reserved_candidate:
            continue
        if (
            selection_config.max_pairs
            and len(chosen_pairs) >= selection_config.max_pairs
        ):
            break
        try_add(candidate)

    selected = torch.zeros_like(base_mask)
    if not selected_indices and selection_config.num_fallback_agents:
        # Preserve non-interactive scenarios for single-agent metrics without
        # inventing interaction pairs. Pick one active agent per type first,
        # then fill by activity. Only use low-motion agents when needed to
        # guarantee the small minimum fallback sample.
        def fallback_key(agent_index: int) -> tuple[float, float, int, int]:
            return (
                -float(motion_extents[agent_index]),
                -float(path_lengths[agent_index]),
                -int(validity[agent_index].sum()),
                int(object_ids[agent_index]),
            )

        active_fallback = sorted(
            (
                index
                for index in base_indices
                if (
                    path_lengths[index]
                    >= selection_config.min_anchor_path_length_m
                    and motion_extents[index]
                    >= selection_config.min_anchor_motion_extent_m
                )
            ),
            key=fallback_key,
        )
        fallback_count = selection_config.num_fallback_agents
        if selection_config.max_agents:
            fallback_count = min(fallback_count, selection_config.max_agents)
        fallback_indices: list[int] = []
        represented_types: set[int] = set()
        for agent_index in active_fallback:
            object_type = int(object_types[agent_index])
            if object_type not in represented_types:
                fallback_indices.append(agent_index)
                represented_types.add(object_type)
                if len(fallback_indices) >= fallback_count:
                    break
        for agent_index in active_fallback:
            if (
                len(fallback_indices) >= fallback_count
                or agent_index in fallback_indices
            ):
                continue
            fallback_indices.append(agent_index)
        minimum = min(
            selection_config.min_fallback_agents,
            fallback_count,
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
    return selected, chosen_pairs


def select_competition_agents(
    gt_scenario: dict[str, Any],
    metric_config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> AgentSelectionResult:
    """Run the fixed, competition-facing simple interaction selector."""
    selection_config = FutureConflictSelectionConfig(
        path_margin_m=metric_config.selection_path_margin_m,
        max_arrival_gap_s=metric_config.selection_max_arrival_gap_s,
        max_synchronous_distance_m=(
            metric_config.selection_max_synchronous_distance_m
        ),
        min_motion_response_score=(
            metric_config.selection_min_motion_response_score
        ),
        max_pairs=metric_config.selection_max_pairs,
        max_agents=metric_config.selection_max_pair_endpoints,
        max_pair_endpoints=metric_config.selection_max_pair_endpoints,
        max_pairs_per_agent=metric_config.selection_max_pairs_per_agent,
        reserve_cross_type_pair=True,
        num_fallback_agents=metric_config.selection_num_fallback_agents,
        min_fallback_agents=metric_config.selection_min_fallback_agents,
        min_history_valid_frames=(
            metric_config.selection_min_history_valid_frames
        ),
        min_anchor_path_length_m=(
            metric_config.selection_min_anchor_path_length_m
        ),
        min_anchor_motion_extent_m=(
            metric_config.selection_min_anchor_motion_extent_m
        ),
        min_direct_closing_distance_m=(
            metric_config.selection_min_direct_closing_distance_m
        ),
        min_direct_motion_response_score=(
            metric_config.selection_min_direct_motion_response_score
        ),
    )
    anchor_mask, pairs = select_future_conflict_agents(
        gt_scenario,
        selection_config,
        metric_config,
    )
    return AgentSelectionResult(
        anchor_mask=anchor_mask,
        pairs=pairs,
        mode="interactive_pairs" if pairs else "fallback_noninteractive",
    )


def competition_selection_audit(
    gt_scenario: dict[str, Any],
    result: AgentSelectionResult,
    metric_config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return transparent selection counts and exact reference-GT pair IDs."""
    object_ids = gt_scenario["object_ids"]
    object_types = gt_scenario["object_types"]
    selected_ids = [
        int(value)
        for value in object_ids[result.anchor_mask].detach().cpu().tolist()
    ]
    cross_type_pairs = [
        pair
        for pair in result.pairs
        if pair["agent_types"][0] != pair["agent_types"][1]
    ]
    pair_endpoint_ids = sorted(
        {
            agent_id
            for pair in result.pairs
            for agent_id in pair["agent_ids"]
        }
    )
    return {
        "mode": result.mode,
        "num_selected_anchors": int(result.anchor_mask.sum().item()),
        "selected_anchor_ids": selected_ids,
        "selected_anchor_type_counts": {
            _type_name(type_id, metric_config): int(
                (
                    result.anchor_mask
                    & (object_types == type_id)
                ).sum().item()
            )
            for type_id in metric_config.evaluated_object_types
        },
        "num_interaction_pairs": len(result.pairs),
        "num_pair_endpoints": len(pair_endpoint_ids),
        "pair_endpoint_ids": pair_endpoint_ids,
        "num_cross_type_interaction_pairs": len(cross_type_pairs),
        "interaction_pair_ids": result.pair_object_ids,
        "cross_type_interaction_pair_ids": [
            pair["agent_ids"] for pair in cross_type_pairs
        ],
        "context_only_agent_ids": result.context_only_object_ids,
    }
