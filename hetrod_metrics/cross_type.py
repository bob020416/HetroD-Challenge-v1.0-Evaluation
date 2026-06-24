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
    valid_anchor_type = torch.isin(
        anchor_types,
        torch.tensor(config.evaluated_object_types, device=anchor_types.device, dtype=anchor_types.dtype),
    )
    valid_context_type = torch.isin(
        context_types,
        torch.tensor(config.evaluated_object_types, device=context_types.device, dtype=context_types.dtype),
    )
    different_type = anchor_types[:, None] != context_types[None, :]
    not_self = anchor_object_ids[:, None] != context_object_ids[None, :]
    return valid_anchor_type[:, None] & valid_context_type[None, :] & different_type & not_self


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


def _histogram_similarity(
    gt_values: torch.Tensor,
    sim_values: torch.Tensor,
    *,
    max_value: float,
    bins: int,
) -> float:
    gt_values = gt_values[torch.isfinite(gt_values)].clamp(0.0, max_value)
    sim_values = sim_values[torch.isfinite(sim_values)].clamp(0.0, max_value)
    if gt_values.numel() == 0 or sim_values.numel() == 0:
        return 1.0
    gt_hist = torch.histc(gt_values.float(), bins=bins, min=0.0, max=max_value)
    sim_hist = torch.histc(sim_values.float(), bins=bins, min=0.0, max=max_value)
    gt_hist = gt_hist / gt_hist.sum().clamp_min(1e-9)
    sim_hist = sim_hist / sim_hist.sum().clamp_min(1e-9)
    total_variation = 0.5 * torch.abs(gt_hist - sim_hist).sum()
    return float((1.0 - total_variation).clamp(0.0, 1.0).item())


def _anchor_context_distances(anchor_xy: torch.Tensor, context_xy: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(anchor_xy[..., :, None, :, :] - context_xy[..., None, :, :, :], dim=-1)


def _estimate_velocity(xy: torch.Tensor, seconds_per_step: float) -> torch.Tensor:
    velocity = torch.zeros_like(xy)
    velocity[..., 1:-1, :] = (xy[..., 2:, :] - xy[..., :-2, :]) / (2.0 * seconds_per_step)
    velocity[..., 0, :] = (xy[..., 1, :] - xy[..., 0, :]) / seconds_per_step
    velocity[..., -1, :] = (xy[..., -1, :] - xy[..., -2, :]) / seconds_per_step
    return velocity


def _first_time_to_proximity(
    relative_xy: torch.Tensor,
    relative_velocity_xy: torch.Tensor,
    radius: float,
    max_ttc_s: float,
) -> torch.Tensor:
    a = (relative_velocity_xy * relative_velocity_xy).sum(dim=-1)
    b = 2.0 * (relative_xy * relative_velocity_xy).sum(dim=-1)
    c = (relative_xy * relative_xy).sum(dim=-1) - radius * radius
    already_close = c <= 0.0
    discriminant = b * b - 4.0 * a * c
    can_solve = (a > 1e-8) & (discriminant >= 0.0)
    sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
    denom = 2.0 * torch.clamp(a, min=1e-8)
    t_enter = (-b - sqrt_disc) / denom
    t_exit = (-b + sqrt_disc) / denom
    future_hit = can_solve & (t_exit >= 0.0)
    t_hit = torch.where(t_enter >= 0.0, t_enter, torch.zeros_like(t_enter))
    return torch.where(
        already_close,
        torch.zeros_like(c),
        torch.where(future_hit, t_hit, torch.full_like(c, max_ttc_s)),
    ).clamp(0.0, max_ttc_s)


def _anchor_context_ttc(
    anchor_xy: torch.Tensor,
    context_xy: torch.Tensor,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    anchor_velocity = _estimate_velocity(anchor_xy, config.seconds_per_step)
    context_velocity = _estimate_velocity(context_xy, config.seconds_per_step)
    relative_xy = context_xy.unsqueeze(1) - anchor_xy.unsqueeze(2)
    relative_velocity = context_velocity.unsqueeze(1) - anchor_velocity.unsqueeze(2)
    return _first_time_to_proximity(
        relative_xy,
        relative_velocity,
        radius=config.cross_type_ttc_proximity_radius_m,
        max_ttc_s=config.cross_type_ttc_hist_max_s,
    )


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


def _event_occurrence_similarity(
    gt_values: torch.Tensor,
    sim_values: torch.Tensor,
    threshold: float,
) -> float:
    gt_occurs = float((gt_values < threshold).any().item())
    sim_occurs = (sim_values < threshold).any(dim=1).float().mean()
    return float((1.0 - torch.abs(sim_occurs - gt_occurs)).clamp(0.0, 1.0).item())


def _pair_metric_score(
    gt_values: torch.Tensor,
    sim_values: torch.Tensor,
    *,
    event_threshold: float,
    histogram_max: float,
    histogram_bins: int,
) -> tuple[float, float]:
    distribution_score = _histogram_similarity(
        gt_values,
        sim_values.flatten(),
        max_value=histogram_max,
        bins=histogram_bins,
    )
    event_score = _event_occurrence_similarity(
        gt_values,
        sim_values,
        threshold=event_threshold,
    )
    return distribution_score, event_score


def _empty_report(num_cross_type_pairs: int = 0) -> dict[str, Any]:
    return {
        "score": 1.0,
        "distance_proximity_to_gt": 1.0,
        "time_to_proximity_to_gt": 1.0,
        "ttc_proximity_to_gt": 1.0,
        "num_cross_type_pairs": num_cross_type_pairs,
        "num_included_pairs": 0,
        "pair_type_scores": {},
    }


def compute_cross_type_interaction(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
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

    future_end_index = config.future_start_index + features.simulated_future.shape[2]
    gt_anchor_xy = features.gt_tracks[
        :, config.future_start_index : future_end_index, :2
    ].unsqueeze(0)
    gt_context_xy = features.context_gt_tracks[
        :, config.future_start_index : future_end_index, :2
    ].unsqueeze(0)
    sim_anchor_xy = features.simulated_future[..., :2]
    sim_context_xy = features.context_simulated_future[..., :2]
    gt_anchor_validity = features.future_validity.unsqueeze(0)
    gt_context_validity = features.context_future_validity.unsqueeze(0)
    sim_anchor_validity = features.future_validity.unsqueeze(0).expand(sim_anchor_xy.shape[0], -1, -1)
    sim_context_validity = features.context_future_validity.unsqueeze(0).expand(sim_context_xy.shape[0], -1, -1)

    gt_distance_tensor = _anchor_context_distances(gt_anchor_xy, gt_context_xy)
    gt_ttc_tensor = _anchor_context_ttc(gt_anchor_xy, gt_context_xy, config)
    gt_pair_validity = gt_anchor_validity[..., :, None, :] & gt_context_validity[..., None, :, :]
    gt_pair_validity = gt_pair_validity & base_pair_mask.view(1, *base_pair_mask.shape, 1)
    gated_distances = torch.where(
        gt_pair_validity,
        gt_distance_tensor,
        torch.full_like(gt_distance_tensor, float("inf")),
    )
    gated_ttc = torch.where(
        gt_pair_validity,
        gt_ttc_tensor,
        torch.full_like(gt_ttc_tensor, float("inf")),
    )
    near_distance_pair = gated_distances.amin(dim=(0, 3)) < config.cross_type_pair_distance_gate_m
    near_ttc_pair = gated_ttc.amin(dim=(0, 3)) < config.cross_type_pair_ttc_gate_s
    pair_mask = base_pair_mask & (near_distance_pair | near_ttc_pair)
    if not pair_mask.any():
        return _empty_report(int(base_pair_mask.sum().item()))

    sim_distance_tensor = _anchor_context_distances(sim_anchor_xy, sim_context_xy)
    sim_ttc_tensor = _anchor_context_ttc(sim_anchor_xy, sim_context_xy, config)
    sim_pair_validity = sim_anchor_validity[..., :, None, :] & sim_context_validity[..., None, :, :]
    pair_type_metrics: dict[str, dict[str, list[float]]] = {}

    for anchor_index, context_index in torch.nonzero(pair_mask, as_tuple=False).tolist():
        gt_valid = gt_pair_validity[0, anchor_index, context_index]
        gt_pair_distances = gt_distance_tensor[0, anchor_index, context_index]
        gt_pair_ttc = gt_ttc_tensor[0, anchor_index, context_index]
        event_times = gt_valid & (
            (gt_pair_distances < config.cross_type_pair_distance_gate_m)
            | (gt_pair_ttc < config.cross_type_pair_ttc_gate_s)
        )
        if not event_times.any():
            continue

        sim_valid = sim_pair_validity[:, anchor_index, context_index, event_times]
        valid_rollouts = sim_valid.all(dim=1)
        if not valid_rollouts.any():
            continue

        gt_distances = gt_pair_distances[event_times]
        gt_ttc = gt_pair_ttc[event_times]
        sim_distances = sim_distance_tensor[
            valid_rollouts, anchor_index, context_index
        ][:, event_times]
        sim_ttc = sim_ttc_tensor[
            valid_rollouts, anchor_index, context_index
        ][:, event_times]
        distance_score, distance_event = _pair_metric_score(
            gt_distances,
            sim_distances,
            event_threshold=config.cross_type_distance_threshold_m,
            histogram_max=config.cross_type_distance_hist_max_m,
            histogram_bins=config.cross_type_distance_hist_bins,
        )
        ttc_score, ttc_event = _pair_metric_score(
            gt_ttc,
            sim_ttc,
            event_threshold=config.cross_type_ttc_threshold_s,
            histogram_max=config.cross_type_ttc_hist_max_s,
            histogram_bins=config.cross_type_ttc_hist_bins,
        )
        type_name = _pair_type_name(
            int(features.object_types[anchor_index].item()),
            int(features.context_object_types[context_index].item()),
            config,
        )
        metrics = pair_type_metrics.setdefault(
            type_name,
            {
                "distance": [],
                "ttc": [],
                "distance_distribution": [],
                "ttc_distribution": [],
                "distance_event": [],
                "ttc_event": [],
            },
        )
        metrics["distance"].append(distance_score)
        metrics["ttc"].append(ttc_score)
        metrics["distance_distribution"].append(distance_score)
        metrics["ttc_distribution"].append(ttc_score)
        metrics["distance_event"].append(distance_event)
        metrics["ttc_event"].append(ttc_event)

    if not pair_type_metrics:
        return _empty_report(int(base_pair_mask.sum().item()))

    pair_type_scores = {}
    for type_name, metrics in pair_type_metrics.items():
        distance = sum(metrics["distance"]) / len(metrics["distance"])
        ttc = sum(metrics["ttc"]) / len(metrics["ttc"])
        pair_type_scores[type_name] = {
            "score": 0.5 * distance + 0.5 * ttc,
            "distance_proximity_to_gt": distance,
            "time_to_proximity_to_gt": ttc,
            "ttc_proximity_to_gt": ttc,
            "distance_distribution_similarity": (
                sum(metrics["distance_distribution"]) / len(metrics["distance_distribution"])
            ),
            "ttc_distribution_similarity": (
                sum(metrics["ttc_distribution"]) / len(metrics["ttc_distribution"])
            ),
            "distance_event_similarity": (
                sum(metrics["distance_event"]) / len(metrics["distance_event"])
            ),
            "ttc_event_similarity": sum(metrics["ttc_event"]) / len(metrics["ttc_event"]),
            "num_pairs": len(metrics["distance"]),
        }

    distance_score = sum(
        metrics["distance_proximity_to_gt"] for metrics in pair_type_scores.values()
    ) / len(pair_type_scores)
    ttc_score = sum(
        metrics["ttc_proximity_to_gt"] for metrics in pair_type_scores.values()
    ) / len(pair_type_scores)
    num_scored_pairs = sum(metrics["num_pairs"] for metrics in pair_type_scores.values())

    return {
        "score": 0.5 * distance_score + 0.5 * ttc_score,
        "distance_proximity_to_gt": distance_score,
        "time_to_proximity_to_gt": ttc_score,
        "ttc_proximity_to_gt": ttc_score,
        "num_cross_type_pairs": int(base_pair_mask.sum().item()),
        "num_included_pairs": num_scored_pairs,
        "pair_distance_gate_m": config.cross_type_pair_distance_gate_m,
        "pair_ttc_gate_s": config.cross_type_pair_ttc_gate_s,
        "scoring_method": "pairwise_type_balanced_histogram_similarity",
        "interaction_definition": "constant_velocity_time_to_5m_proximity",
        "pair_type_scores": pair_type_scores,
    }
