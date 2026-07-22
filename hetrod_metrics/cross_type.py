from __future__ import annotations

from typing import Any

import torch

from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .features import HetrodFeatureBundle


def _cross_type_anchor_context_pair_mask(
    anchor_object_ids: torch.Tensor,
    anchor_types: torch.Tensor,
    context_object_ids: torch.Tensor,
    context_types: torch.Tensor,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    evaluated_types = torch.tensor(
        config.evaluated_object_types,
        device=anchor_types.device,
        dtype=anchor_types.dtype,
    )
    valid_anchor_type = torch.isin(anchor_types, evaluated_types)
    valid_context_type = torch.isin(context_types, evaluated_types)
    different_type = anchor_types[:, None] != context_types[None, :]
    not_self = anchor_object_ids[:, None] != context_object_ids[None, :]
    return (
        valid_anchor_type[:, None]
        & valid_context_type[None, :]
        & different_type
        & not_self
    )


def _deduplicate_bidirectional_pairs(
    pair_mask: torch.Tensor,
    anchor_object_ids: torch.Tensor,
    context_object_ids: torch.Tensor,
    context_anchor_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep one canonical direction when both agents are selected anchors."""
    context_is_anchor = context_anchor_mask.bool().unsqueeze(0)
    canonical_direction = anchor_object_ids[:, None] < context_object_ids[None, :]
    return pair_mask & (~context_is_anchor | canonical_direction)


def _pair_type_name(
    anchor_type: int,
    context_type: int,
    config: HetrodMetricConfig,
) -> str:
    pair = frozenset((anchor_type, context_type))
    names = {
        frozenset((config.vehicle_type, config.pedestrian_type)): "vehicle_pedestrian",
        frozenset((config.vehicle_type, config.two_wheeler_type)): "vehicle_two_wheeler",
        frozenset((config.pedestrian_type, config.two_wheeler_type)): "pedestrian_two_wheeler",
    }
    return names[pair]


def _empty_report(num_cross_type_pairs: int = 0) -> dict[str, Any]:
    return {
        "score": 1.0,
        "distance_proximity_to_gt": 1.0,
        "time_to_proximity_to_gt": 1.0,
        "ttc_proximity_to_gt": 1.0,
        "num_cross_type_pairs": num_cross_type_pairs,
        "num_included_pairs": 0,
        "pair_type_scores": {},
        "scoring_method": "closest_approach_error",
    }


def _minimum_and_time(
    distances: torch.Tensor,
    validity: torch.Tensor,
    seconds_per_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = torch.where(validity, distances, torch.full_like(distances, float("inf")))
    minimum, index = masked.min(dim=-1)
    return minimum, index.to(distances.dtype) * seconds_per_step


def compute_cross_type_interaction(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Score cross-type interaction using closest distance and its timing.

    A physical pair is included when either GT or at least one rollout comes
    within the distance gate. This union gate penalizes invented interactions
    while keeping the metric independent of a fragile constant-velocity TTP.
    """
    base_pair_mask = _cross_type_anchor_context_pair_mask(
        features.object_ids,
        features.object_types,
        features.context_object_ids,
        features.context_object_types,
        config,
    )
    base_pair_mask = _deduplicate_bidirectional_pairs(
        base_pair_mask,
        features.object_ids,
        features.context_object_ids,
        features.context_anchor_mask,
    )
    if not base_pair_mask.any():
        return _empty_report()

    future_steps = features.simulated_future.shape[2]
    future_end = config.future_start_index + future_steps
    gt_anchor_xy = features.gt_tracks[:, config.future_start_index : future_end, :2]
    gt_context_xy = features.context_gt_tracks[:, config.future_start_index : future_end, :2]
    sim_anchor_xy = features.simulated_future[..., :2]
    sim_context_xy = features.context_simulated_future[..., :2]

    pair_type_metrics: dict[str, dict[str, list[float]]] = {}
    pair_indices = torch.nonzero(base_pair_mask, as_tuple=False)
    num_base_pairs = int(pair_indices.shape[0])
    for start in range(0, num_base_pairs, config.cross_type_pair_chunk_size):
        chunk = pair_indices[start : start + config.cross_type_pair_chunk_size]
        anchor_indices, context_indices = chunk[:, 0], chunk[:, 1]
        pair_validity = (
            features.future_validity[anchor_indices]
            & features.context_future_validity[context_indices]
        )
        has_valid_frame = pair_validity.any(dim=-1)
        if not has_valid_frame.any():
            continue
        anchor_indices = anchor_indices[has_valid_frame]
        context_indices = context_indices[has_valid_frame]
        pair_validity = pair_validity[has_valid_frame]

        gt_distances = torch.linalg.norm(
            gt_anchor_xy[anchor_indices] - gt_context_xy[context_indices], dim=-1
        )
        sim_distances = torch.linalg.norm(
            sim_anchor_xy[:, anchor_indices]
            - sim_context_xy[:, context_indices],
            dim=-1,
        )
        gt_min, gt_time = _minimum_and_time(
            gt_distances, pair_validity, config.seconds_per_step
        )
        sim_validity = pair_validity.unsqueeze(0).expand(sim_distances.shape[0], -1, -1)
        sim_min, sim_time = _minimum_and_time(
            sim_distances, sim_validity, config.seconds_per_step
        )
        included = (gt_min < config.cross_type_pair_distance_gate_m) | (
            sim_min < config.cross_type_pair_distance_gate_m
        ).any(dim=0)
        if not included.any():
            continue

        distance_scores = (
            1.0 - torch.abs(sim_min - gt_min.unsqueeze(0)) / 5.0
        ).clamp(0.0, 1.0).mean(dim=0)
        time_scores = (
            1.0 - torch.abs(sim_time - gt_time.unsqueeze(0)) / 4.0
        ).clamp(0.0, 1.0).mean(dim=0)
        anchor_types = features.object_types[anchor_indices[included]].detach().cpu().tolist()
        context_types = (
            features.context_object_types[context_indices[included]].detach().cpu().tolist()
        )
        distance_values = distance_scores[included].detach().cpu().tolist()
        time_values = time_scores[included].detach().cpu().tolist()
        for anchor_type, context_type, distance, time in zip(
            anchor_types, context_types, distance_values, time_values
        ):
            type_name = _pair_type_name(anchor_type, context_type, config)
            metrics = pair_type_metrics.setdefault(
                type_name, {"distance": [], "time": []}
            )
            metrics["distance"].append(distance)
            metrics["time"].append(time)

    if not pair_type_metrics:
        return _empty_report(num_base_pairs)

    pair_type_scores = {}
    for type_name, metrics in pair_type_metrics.items():
        distance = sum(metrics["distance"]) / len(metrics["distance"])
        time = sum(metrics["time"]) / len(metrics["time"])
        pair_type_scores[type_name] = {
            "score": 0.5 * (distance + time),
            "distance_proximity_to_gt": distance,
            "time_to_proximity_to_gt": time,
            "ttc_proximity_to_gt": time,
            "num_pairs": len(metrics["distance"]),
        }

    distance_score = sum(
        item["distance_proximity_to_gt"] for item in pair_type_scores.values()
    ) / len(pair_type_scores)
    time_score = sum(
        item["time_to_proximity_to_gt"] for item in pair_type_scores.values()
    ) / len(pair_type_scores)
    return {
        "score": 0.5 * (distance_score + time_score),
        "distance_proximity_to_gt": distance_score,
        "time_to_proximity_to_gt": time_score,
        "ttc_proximity_to_gt": time_score,
        "num_cross_type_pairs": num_base_pairs,
        "num_included_pairs": sum(item["num_pairs"] for item in pair_type_scores.values()),
        "pair_distance_gate_m": config.cross_type_pair_distance_gate_m,
        "distance_error_scale_m": 5.0,
        "time_error_scale_s": 4.0,
        "scoring_method": "closest_approach_error",
        "interaction_definition": "minimum_distance_and_time_of_closest_approach",
        "pair_type_scores": pair_type_scores,
    }
