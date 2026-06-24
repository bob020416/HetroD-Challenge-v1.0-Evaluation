from __future__ import annotations

from typing import Any

import torch

from wosac_fast_eval_tool.fast_sim_agents_metrics import interaction_features, map_metric_features

from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .features import HetrodFeatureBundle


def _mean_safe_score(unsafe: torch.Tensor, validity: torch.Tensor) -> float:
    if unsafe.shape != validity.shape:
        raise ValueError(f"Shape mismatch: {unsafe.shape} vs {validity.shape}")
    if not validity.any():
        return 1.0
    unsafe_rate = unsafe[validity].float().mean()
    return float((1.0 - unsafe_rate).clamp(0.0, 1.0).item())


def _type_balanced_safety_report(
    unsafe: torch.Tensor,
    validity: torch.Tensor,
    object_types: torch.Tensor,
    config: HetrodMetricConfig,
) -> dict[str, Any]:
    type_names = {
        config.vehicle_type: "vehicle",
        config.two_wheeler_type: "two_wheeler",
        config.pedestrian_type: "pedestrian",
    }
    by_type = {}
    present_scores = []
    total_unsafe = 0
    total_samples = 0
    for object_type in config.evaluated_object_types:
        type_name = type_names[object_type]
        type_mask = object_types == object_type
        type_validity = validity & type_mask.view(1, -1, 1)
        num_samples = int(type_validity.sum().item())
        if num_samples == 0:
            by_type[type_name] = None
            continue
        num_unsafe = int(unsafe[type_validity].sum().item())
        unsafe_rate = num_unsafe / num_samples
        score = 1.0 - unsafe_rate
        by_type[type_name] = {
            "score": score,
            "unsafe_rate": unsafe_rate,
            "num_unsafe": num_unsafe,
            "num_samples": num_samples,
            "num_agents": int(type_mask.sum().item()),
        }
        present_scores.append(score)
        total_unsafe += num_unsafe
        total_samples += num_samples
    return {
        "score": sum(present_scores) / len(present_scores) if present_scores else 1.0,
        "unsafe_rate": total_unsafe / total_samples if total_samples else 0.0,
        "num_unsafe": total_unsafe,
        "num_samples": total_samples,
        "by_type": by_type,
        "aggregation": "agent_type_macro_average",
    }


def _sim_boxes(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    future_steps = features.simulated_future.shape[2]
    future_end_index = config.future_start_index + future_steps
    dims = features.gt_tracks[
        :, config.future_start_index : future_end_index, 3:6
    ].to(features.simulated_future.device)
    dims = dims.unsqueeze(0).expand(features.simulated_future.shape[0], -1, -1, -1)
    return torch.cat(
        [
            features.simulated_future[..., 0:3],
            dims,
            features.simulated_future[..., 3:4],
        ],
        dim=-1,
    )


def _context_sim_boxes(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    future_steps = features.context_simulated_future.shape[2]
    future_end_index = config.future_start_index + future_steps
    dims = features.context_gt_tracks[
        :, config.future_start_index : future_end_index, 3:6
    ].to(features.context_simulated_future.device)
    dims = dims.unsqueeze(0).expand(features.context_simulated_future.shape[0], -1, -1, -1)
    return torch.cat(
        [
            features.context_simulated_future[..., 0:3],
            dims,
            features.context_simulated_future[..., 3:4],
        ],
        dim=-1,
    )


def collision_with_annotation_tolerance(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    boxes = _context_sim_boxes(features, config)
    distances = interaction_features.compute_distance_to_nearest_object(
        boxes=boxes,
        valid=features.context_future_validity,
        evaluated_object_mask=features.context_anchor_mask,
    )
    # Treat the margin as annotation/model tolerance: tiny box intersections are
    # ignored, and only overlaps deeper than the tolerance are unsafe.
    collisions = distances < -config.collision_tolerance_m
    validity = features.future_validity.unsqueeze(0).expand_as(collisions)
    report = _type_balanced_safety_report(
        collisions,
        validity,
        features.object_types,
        config,
    )
    return {
        **report,
        "collision_tolerance_m": config.collision_tolerance_m,
        "num_context_agents": int(features.context_object_ids.numel()),
        "definition": "box_overlap_depth_exceeds_annotation_tolerance",
    }


def collision_with_safe_margin(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Compatibility alias for the annotation-tolerant collision metric."""
    return collision_with_annotation_tolerance(features, config)


def _valid_region_margin_by_type(
    object_types: torch.Tensor,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    margin = torch.full(
        object_types.shape,
        config.vehicle_valid_region_margin_m,
        dtype=torch.float32,
        device=object_types.device,
    )
    margin = torch.where(
        object_types == config.two_wheeler_type,
        torch.tensor(config.two_wheeler_valid_region_margin_m, device=object_types.device),
        margin,
    )
    margin = torch.where(
        object_types == config.pedestrian_type,
        torch.tensor(config.pedestrian_valid_region_margin_m, device=object_types.device),
        margin,
    )
    return margin


def valid_region_margin(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if not features.road_edges:
        return {
            "score": 0.0,
            "unsafe_rate": 1.0,
            "by_type": {},
            "missing_road_edges": True,
            "valid_region_source": "missing",
        }
    boxes = _sim_boxes(features, config)
    evaluated_object_mask = torch.ones(
        features.object_ids.shape[0],
        dtype=torch.bool,
        device=features.object_ids.device,
    )
    distances = map_metric_features.compute_distance_to_road_edge(
        boxes=boxes,
        valid=features.future_validity,
        evaluated_object_mask=evaluated_object_mask,
        road_edge_polylines=features.road_edges,
    )
    margins = _valid_region_margin_by_type(features.object_types, config)
    outside = distances > margins.view(1, -1, 1)
    validity = features.future_validity.unsqueeze(0).expand_as(outside)
    return {
        **_type_balanced_safety_report(
            outside,
            validity,
            features.object_types,
            config,
        ),
        "missing_road_edges": False,
        "valid_region_source": "road_edge_margin_fallback",
        "type_margins_m": {
            "vehicle": config.vehicle_valid_region_margin_m,
            "two_wheeler": config.two_wheeler_valid_region_margin_m,
            "pedestrian": config.pedestrian_valid_region_margin_m,
        },
    }


def compute_safety(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    collision = collision_with_annotation_tolerance(features, config)
    valid_region = valid_region_margin(features, config)
    score = 0.5 * collision["score"] + 0.5 * valid_region["score"]
    return {
        "score": score,
        "collision_with_annotation_tolerance": collision,
        "collision_with_safe_margin": collision,
        "valid_region_margin": valid_region,
        "aggregation": "0.5_collision_plus_0.5_valid_region",
    }
