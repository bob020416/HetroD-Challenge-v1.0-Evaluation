from __future__ import annotations

from typing import Any

import torch

from wosac_fast_eval_tool.fast_sim_agents_metrics import interaction_features, map_metric_features

from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .features import HetrodFeatureBundle


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


def _gt_boxes(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    future_steps = features.simulated_future.shape[2]
    future_end_index = config.future_start_index + future_steps
    tracks = features.gt_tracks[:, config.future_start_index : future_end_index]
    return torch.cat([tracks[..., 0:6], tracks[..., 6:7]], dim=-1).unsqueeze(0)


def _type_balanced_rollout_collision_report(
    collided: torch.Tensor,
    valid_rollouts: torch.Tensor,
    object_types: torch.Tensor,
    config: HetrodMetricConfig,
) -> dict[str, Any]:
    """Aggregate one binary collision outcome per anchor and rollout."""
    type_names = {
        config.vehicle_type: "vehicle",
        config.two_wheeler_type: "two_wheeler",
        config.pedestrian_type: "pedestrian",
    }
    by_type = {}
    present_scores = []
    total_collided = 0
    total_rollouts = 0
    for object_type in config.evaluated_object_types:
        type_name = type_names[object_type]
        type_mask = object_types == object_type
        type_valid = valid_rollouts & type_mask.unsqueeze(0)
        num_rollouts = int(type_valid.sum().item())
        if num_rollouts == 0:
            by_type[type_name] = None
            continue
        num_collided = int(collided[type_valid].sum().item())
        score = 1.0 - num_collided / num_rollouts
        by_type[type_name] = {
            "score": score,
            "collision_rate": num_collided / num_rollouts,
            "unsafe_rate": num_collided / num_rollouts,
            "num_collided_agent_rollouts": num_collided,
            "num_valid_agent_rollouts": num_rollouts,
            # Compatibility with the dataset sufficient-statistic aggregator.
            "num_unsafe": num_collided,
            "num_samples": num_rollouts,
            "num_agents": int(type_mask.sum().item()),
        }
        present_scores.append(score)
        total_collided += num_collided
        total_rollouts += num_rollouts
    return {
        "score": sum(present_scores) / len(present_scores) if present_scores else 1.0,
        "collision_rate": total_collided / total_rollouts if total_rollouts else 0.0,
        "unsafe_rate": total_collided / total_rollouts if total_rollouts else 0.0,
        "num_collided_agent_rollouts": total_collided,
        "num_valid_agent_rollouts": total_rollouts,
        "num_unsafe": total_collided,
        "num_samples": total_rollouts,
        "by_type": by_type,
        "aggregation": "agent_type_macro_average_of_agent_rollout_collision_rate",
    }


def collision_rollout_rate(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    boxes = _context_sim_boxes(features, config)
    distances = interaction_features.compute_distance_to_nearest_object(
        boxes=boxes,
        valid=features.context_future_validity,
        evaluated_object_mask=features.context_anchor_mask,
    )
    frame_collisions = distances < 0.0
    frame_validity = features.future_validity.unsqueeze(0).expand_as(frame_collisions)
    valid_rollouts = frame_validity.any(dim=-1)
    collided = (frame_collisions & frame_validity).any(dim=-1)
    report = _type_balanced_rollout_collision_report(
        collided,
        valid_rollouts,
        features.object_types,
        config,
    )
    return {
        **report,
        "collision_tolerance_m": 0.0,
        "num_context_agents": int(features.context_object_ids.numel()),
        "definition": "any_strict_box_overlap_per_agent_rollout",
    }


def collision_with_safe_margin(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Compatibility alias for the strict rollout-level collision metric."""
    return collision_rollout_rate(features, config)


def collision_with_annotation_tolerance(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Deprecated v0.1 name retained as an API compatibility alias."""
    return collision_rollout_rate(features, config)


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


def _distance_to_road_edge_in_agent_chunks(
    boxes: torch.Tensor,
    validity: torch.Tensor,
    road_edges: list[torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    """Run the road-edge kernel with bounded anchor-agent memory."""
    parts = []
    num_agents = boxes.shape[1]
    for start in range(0, num_agents, chunk_size):
        end = min(start + chunk_size, num_agents)
        evaluated_object_mask = torch.ones(
            end - start,
            dtype=torch.bool,
            device=boxes.device,
        )
        parts.append(
            map_metric_features.compute_distance_to_road_edge(
                boxes=boxes[:, start:end],
                valid=validity[start:end],
                evaluated_object_mask=evaluated_object_mask,
                road_edge_polylines=road_edges,
            )
        )
    return torch.cat(parts, dim=1)


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
    gt_boxes = _gt_boxes(features, config)
    distances = _distance_to_road_edge_in_agent_chunks(
        boxes=boxes,
        validity=features.future_validity,
        road_edges=features.road_edges,
        chunk_size=config.valid_region_agent_chunk_size,
    )
    margins = _valid_region_margin_by_type(features.object_types, config)
    outside = distances > margins.view(1, -1, 1)
    validity = features.future_validity.unsqueeze(0).expand_as(outside)
    gt_distances = _distance_to_road_edge_in_agent_chunks(
        boxes=gt_boxes,
        validity=features.future_validity,
        road_edges=features.road_edges,
        chunk_size=config.valid_region_agent_chunk_size,
    )
    gt_outside = gt_distances > margins.view(1, -1, 1)
    gt_validity = features.future_validity.unsqueeze(0)

    type_names = {
        config.vehicle_type: "vehicle",
        config.two_wheeler_type: "two_wheeler",
        config.pedestrian_type: "pedestrian",
    }
    by_type = {}
    present_scores = []
    total_sim_outside = total_sim_samples = 0
    total_gt_outside = total_gt_samples = 0
    for object_type in config.evaluated_object_types:
        type_name = type_names[object_type]
        type_mask = features.object_types == object_type
        sim_valid = validity & type_mask.view(1, -1, 1)
        gt_valid = gt_validity & type_mask.view(1, -1, 1)
        num_sim = int(sim_valid.sum().item())
        num_gt = int(gt_valid.sum().item())
        if num_sim == 0 or num_gt == 0:
            by_type[type_name] = None
            continue
        num_sim_outside = int(outside[sim_valid].sum().item())
        num_gt_outside = int(gt_outside[gt_valid].sum().item())
        sim_rate = num_sim_outside / num_sim
        gt_rate = num_gt_outside / num_gt
        excess_rate = max(sim_rate - gt_rate, 0.0)
        score = 1.0 - excess_rate
        by_type[type_name] = {
            "score": score,
            "excess_outside_rate": excess_rate,
            "sim_outside_rate": sim_rate,
            "gt_outside_rate": gt_rate,
            "num_unsafe": num_sim_outside,
            "num_samples": num_sim,
            "gt_num_unsafe": num_gt_outside,
            "gt_num_samples": num_gt,
            "num_agents": int(type_mask.sum().item()),
        }
        present_scores.append(score)
        total_sim_outside += num_sim_outside
        total_sim_samples += num_sim
        total_gt_outside += num_gt_outside
        total_gt_samples += num_gt
    sim_rate = total_sim_outside / total_sim_samples if total_sim_samples else 0.0
    gt_rate = total_gt_outside / total_gt_samples if total_gt_samples else 0.0
    excess_rate = max(sim_rate - gt_rate, 0.0)
    return {
        "score": sum(present_scores) / len(present_scores) if present_scores else 1.0,
        "unsafe_rate": excess_rate,
        "excess_outside_rate": excess_rate,
        "sim_outside_rate": sim_rate,
        "gt_outside_rate": gt_rate,
        "num_unsafe": total_sim_outside,
        "num_samples": total_sim_samples,
        "gt_num_unsafe": total_gt_outside,
        "gt_num_samples": total_gt_samples,
        "by_type": by_type,
        "aggregation": "agent_type_macro_average_of_gt_relative_excess_rate",
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
    collision = collision_rollout_rate(features, config)
    valid_region = valid_region_margin(features, config)
    score = 0.5 * collision["score"] + 0.5 * valid_region["score"]
    return {
        "score": score,
        "collision_rollout_rate": collision,
        "collision_with_annotation_tolerance": collision,
        "collision_with_safe_margin": collision,
        "valid_region_margin": valid_region,
        "aggregation": "0.5_collision_plus_0.5_valid_region",
    }
