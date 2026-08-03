from __future__ import annotations

from typing import Any

import torch

from wosac_fast_eval_tool.fast_sim_agents_metrics import map_metric_features

from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .features import HetrodFeatureBundle
from .valid_region import (
    box_corners_xy,
    footprints_inside_region_records,
    footprints_inside_valid_regions,
    has_pedestrian_transition_regions,
    has_type_specific_valid_regions,
    points_in_valid_region,
)


def _effective_dimensions(
    dimensions: torch.Tensor,
    object_types: torch.Tensor,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    """Fill zero-sized annotations with stable type-specific footprints."""
    defaults = torch.tensor(
        [
            config.coverage_default_vehicle_length_m,
            config.coverage_default_vehicle_width_m,
            1.0,
        ],
        dtype=dimensions.dtype,
        device=dimensions.device,
    ).expand(object_types.shape[0], 3).clone()
    defaults[object_types == config.two_wheeler_type, :2] = torch.tensor(
        [
            config.coverage_default_two_wheeler_length_m,
            config.coverage_default_two_wheeler_width_m,
        ],
        dtype=dimensions.dtype,
        device=dimensions.device,
    )
    defaults[object_types == config.pedestrian_type, :2] = torch.tensor(
        [
            config.coverage_default_pedestrian_length_m,
            config.coverage_default_pedestrian_width_m,
        ],
        dtype=dimensions.dtype,
        device=dimensions.device,
    )
    defaults = defaults[:, None, :]
    return torch.where(dimensions > 0.0, dimensions, defaults)


def _sim_boxes(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    future_steps = features.simulated_future.shape[2]
    future_end_index = config.future_start_index + future_steps
    dims = _effective_dimensions(
        features.gt_tracks[
        :, config.future_start_index : future_end_index, 3:6
        ].to(features.simulated_future.device),
        features.object_types,
        config,
    )
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
    dims = _effective_dimensions(
        features.context_gt_tracks[
        :, config.future_start_index : future_end_index, 3:6
        ].to(features.context_simulated_future.device),
        features.context_object_types,
        config,
    )
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
    dimensions = _effective_dimensions(
        tracks[..., 3:6],
        features.object_types,
        config,
    )
    return torch.cat(
        [tracks[..., 0:3], dimensions, tracks[..., 6:7]],
        dim=-1,
    ).unsqueeze(0)


def _context_gt_boxes(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig,
) -> torch.Tensor:
    future_steps = features.context_simulated_future.shape[2]
    future_end_index = config.future_start_index + future_steps
    tracks = features.context_gt_tracks[
        :, config.future_start_index : future_end_index
    ]
    dimensions = _effective_dimensions(
        tracks[..., 3:6],
        features.context_object_types,
        config,
    )
    return torch.cat(
        [tracks[..., 0:3], dimensions, tracks[..., 6:7]],
        dim=-1,
    ).unsqueeze(0)


def _strict_oriented_box_overlap(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Return strict BEV overlap for paired oriented rectangles.

    Touching edges are not collisions. Inputs have a shared shape ending in the
    seven box fields ``[x, y, z, length, width, height, yaw]``.
    """
    delta = second[..., :2] - first[..., :2]
    first_yaw = first[..., 6]
    second_yaw = second[..., 6]
    first_cosine = torch.cos(first_yaw)
    first_sine = torch.sin(first_yaw)
    second_cosine = torch.cos(second_yaw)
    second_sine = torch.sin(second_yaw)
    first_half_length = 0.5 * first[..., 3]
    first_half_width = 0.5 * first[..., 4]
    second_half_length = 0.5 * second[..., 3]
    second_half_width = 0.5 * second[..., 4]
    relative_cosine = torch.abs(torch.cos(second_yaw - first_yaw))
    relative_sine = torch.abs(torch.sin(second_yaw - first_yaw))
    delta_x = delta[..., 0]
    delta_y = delta[..., 1]
    first_forward_distance = torch.abs(
        delta_x * first_cosine + delta_y * first_sine
    )
    first_lateral_distance = torch.abs(
        -delta_x * first_sine + delta_y * first_cosine
    )
    second_forward_distance = torch.abs(
        delta_x * second_cosine + delta_y * second_sine
    )
    second_lateral_distance = torch.abs(
        -delta_x * second_sine + delta_y * second_cosine
    )
    overlap_on_first_forward = first_forward_distance < (
            first_half_length
            + second_half_length * relative_cosine
            + second_half_width * relative_sine
    )
    overlap_on_first_lateral = first_lateral_distance < (
        first_half_width
        + second_half_length * relative_sine
        + second_half_width * relative_cosine
    )
    overlap_on_second_forward = second_forward_distance < (
        second_half_length
        + first_half_length * relative_cosine
        + first_half_width * relative_sine
    )
    overlap_on_second_lateral = second_lateral_distance < (
        second_half_width
        + first_half_length * relative_sine
        + first_half_width * relative_cosine
    )
    return (
        overlap_on_first_forward
        & overlap_on_first_lateral
        & overlap_on_second_forward
        & overlap_on_second_lateral
    )


def _collision_outcomes(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return new-sim, raw-sim, and GT collision outcomes.

    A simulated pair/frame overlap is new only when the same physical pair does
    not overlap in GT at that frame. This keeps zero geometric tolerance while
    preventing annotation overlap in GT from lowering the oracle score.
    """
    sim_anchor_boxes = _sim_boxes(features, config)
    sim_context_boxes = _context_sim_boxes(features, config)
    gt_anchor_boxes = _gt_boxes(features, config)
    gt_context_boxes = _context_gt_boxes(features, config)
    num_rollouts, num_anchors = sim_anchor_boxes.shape[:2]
    device = sim_anchor_boxes.device
    new_collided = torch.zeros(
        (num_rollouts, num_anchors),
        dtype=torch.bool,
        device=device,
    )
    raw_collided = torch.zeros_like(new_collided)
    gt_collided = torch.zeros(num_anchors, dtype=torch.bool, device=device)

    non_self = (
        features.object_ids[:, None]
        != features.context_object_ids[None, :]
    )
    pair_indices = torch.nonzero(non_self, as_tuple=False)
    for start in range(0, pair_indices.shape[0], config.collision_pair_chunk_size):
        chunk = pair_indices[start : start + config.collision_pair_chunk_size]
        anchor_indices = chunk[:, 0]
        context_indices = chunk[:, 1]
        pair_validity = (
            features.future_validity[anchor_indices]
            & features.context_future_validity[context_indices]
        )
        sim_overlap = _strict_oriented_box_overlap(
            sim_anchor_boxes[:, anchor_indices],
            sim_context_boxes[:, context_indices],
        ) & pair_validity.unsqueeze(0)
        gt_overlap = _strict_oriented_box_overlap(
            gt_anchor_boxes[:, anchor_indices],
            gt_context_boxes[:, context_indices],
        ) & pair_validity.unsqueeze(0)
        new_overlap = sim_overlap & ~gt_overlap

        for anchor_index in torch.unique(anchor_indices).tolist():
            anchor_pairs = anchor_indices == anchor_index
            raw_collided[:, anchor_index] |= sim_overlap[
                :, anchor_pairs
            ].any(dim=(1, 2))
            new_collided[:, anchor_index] |= new_overlap[
                :, anchor_pairs
            ].any(dim=(1, 2))
            gt_collided[anchor_index] |= gt_overlap[
                :, anchor_pairs
            ].any()
    return new_collided, raw_collided, gt_collided


def _type_balanced_rollout_collision_report(
    collided: torch.Tensor,
    valid_rollouts: torch.Tensor,
    object_types: torch.Tensor,
    config: HetrodMetricConfig,
    *,
    raw_collided: torch.Tensor | None = None,
    gt_collided: torch.Tensor | None = None,
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
    total_raw_collided = 0
    total_gt_collided = 0
    total_rollouts = 0
    gt_collided_rollouts = (
        gt_collided.unsqueeze(0).expand_as(valid_rollouts)
        if gt_collided is not None
        else torch.zeros_like(valid_rollouts)
    )
    raw_collided = raw_collided if raw_collided is not None else collided
    for object_type in config.evaluated_object_types:
        type_name = type_names[object_type]
        type_mask = object_types == object_type
        type_valid = valid_rollouts & type_mask.unsqueeze(0)
        num_rollouts = int(type_valid.sum().item())
        if num_rollouts == 0:
            by_type[type_name] = None
            continue
        num_collided = int(collided[type_valid].sum().item())
        num_raw_collided = int(raw_collided[type_valid].sum().item())
        num_gt_collided = int(gt_collided_rollouts[type_valid].sum().item())
        score = 1.0 - num_collided / num_rollouts
        by_type[type_name] = {
            "score": score,
            "collision_rate": num_collided / num_rollouts,
            "new_collision_rate": num_collided / num_rollouts,
            "raw_sim_collision_rate": num_raw_collided / num_rollouts,
            "gt_collision_rate": num_gt_collided / num_rollouts,
            "unsafe_rate": num_collided / num_rollouts,
            "num_collided_agent_rollouts": num_collided,
            "num_raw_collided_agent_rollouts": num_raw_collided,
            "num_gt_collided_agent_rollouts": num_gt_collided,
            "num_valid_agent_rollouts": num_rollouts,
            # Compatibility with the dataset sufficient-statistic aggregator.
            "num_unsafe": num_collided,
            "num_samples": num_rollouts,
            "num_agents": int(type_mask.sum().item()),
        }
        present_scores.append(score)
        total_collided += num_collided
        total_raw_collided += num_raw_collided
        total_gt_collided += num_gt_collided
        total_rollouts += num_rollouts
    return {
        "score": sum(present_scores) / len(present_scores) if present_scores else 1.0,
        "collision_rate": total_collided / total_rollouts if total_rollouts else 0.0,
        "new_collision_rate": total_collided / total_rollouts if total_rollouts else 0.0,
        "raw_sim_collision_rate": (
            total_raw_collided / total_rollouts if total_rollouts else 0.0
        ),
        "gt_collision_rate": (
            total_gt_collided / total_rollouts if total_rollouts else 0.0
        ),
        "unsafe_rate": total_collided / total_rollouts if total_rollouts else 0.0,
        "num_collided_agent_rollouts": total_collided,
        "num_raw_collided_agent_rollouts": total_raw_collided,
        "num_gt_collided_agent_rollouts": total_gt_collided,
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
    collided, raw_collided, gt_collided = _collision_outcomes(features, config)
    valid_rollouts = features.future_validity.any(dim=-1).unsqueeze(0).expand_as(
        collided
    )
    report = _type_balanced_rollout_collision_report(
        collided,
        valid_rollouts,
        features.object_types,
        config,
        raw_collided=raw_collided,
        gt_collided=gt_collided,
    )
    return {
        **report,
        "collision_tolerance_m": 0.0,
        "num_context_agents": int(features.context_object_ids.numel()),
        "definition": (
            "any_new_strict_pair_frame_box_overlap_per_agent_rollout_relative_to_gt"
        ),
    }


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


def _pedestrian_transition_masks(
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    future_validity: torch.Tensor,
    valid_regions: dict,
    valid_region_definition: dict,
    config: HetrodMetricConfig,
) -> dict[str, torch.Tensor | int | float]:
    """Classify pedestrian rollout frames using map semantics and a GT corridor."""
    gt_centers = gt_boxes[0, ..., :2]
    core_records = valid_regions["pedestrian_core"]
    crosswalk_records = valid_regions["pedestrian_crosswalk"]
    road_records = valid_regions["pedestrian_road"]

    gt_core = points_in_valid_region(
        gt_centers,
        core_records,
        chunk_size=config.valid_region_query_chunk_size,
    )
    gt_crosswalk = ~gt_core & points_in_valid_region(
        gt_centers,
        crosswalk_records,
        chunk_size=config.valid_region_query_chunk_size,
    )
    gt_road = ~gt_core & ~gt_crosswalk & points_in_valid_region(
        gt_centers,
        road_records,
        chunk_size=config.valid_region_query_chunk_size,
    )
    gt_supported = gt_core | gt_crosswalk | gt_road

    corridor_margin = float(
        valid_region_definition.get(
            "pedestrian_gt_road_corridor_margin_m",
            config.pedestrian_gt_road_corridor_margin_m,
        )
    )
    corners = box_corners_xy(boxes)

    def corridor_containment(anchor_mask: torch.Tensor) -> torch.Tensor:
        containment = torch.zeros(
            boxes.shape[:-1],
            dtype=torch.bool,
            device=boxes.device,
        )
        for agent_index in range(boxes.shape[1]):
            anchors = gt_centers[
                agent_index,
                anchor_mask[agent_index] & future_validity[agent_index],
            ]
            if anchors.numel() == 0:
                continue
            flat_corners = corners[:, agent_index].reshape(-1, 2)
            corner_inside = (
                torch.cdist(flat_corners, anchors).amin(dim=-1)
                <= corridor_margin
            )
            containment[:, agent_index] = corner_inside.reshape(
                boxes.shape[0],
                boxes.shape[2],
                4,
            ).all(dim=-1)
        return containment

    sim_core = footprints_inside_region_records(boxes, core_records, config)
    gt_footprint_core = footprints_inside_region_records(
        gt_boxes,
        core_records,
        config,
    )[0]
    # Preserve a perfect GT replay at semantic boundaries without making the
    # fallback globally available: only that pedestrian's supported GT centers
    # may supply the local boundary corridor.
    sim_core |= corridor_containment(gt_core & ~gt_footprint_core)
    sim_crosswalk = (
        ~sim_core
        & footprints_inside_region_records(boxes, crosswalk_records, config)
    )

    sim_corridor = corridor_containment(gt_road)

    sim_transition = ~sim_core & (sim_crosswalk | sim_corridor)
    spatial_offroad = ~sim_core & ~sim_transition
    evaluable = future_validity & gt_supported

    seconds_per_step = float(config.seconds_per_step)
    slack_seconds = float(
        valid_region_definition.get(
            "pedestrian_transition_time_slack_s",
            config.pedestrian_transition_time_slack_s,
        )
    )
    slack_frames = int(round(slack_seconds / seconds_per_step))
    gt_transition_frames = (
        ((gt_crosswalk | gt_road) & future_validity).sum(dim=-1)
    )
    sim_transition_frames = (
        sim_transition & evaluable.unsqueeze(0)
    ).sum(dim=-1)
    transition_budget = gt_transition_frames.unsqueeze(0) + slack_frames
    transition_overstay = torch.clamp(
        sim_transition_frames - transition_budget,
        min=0,
    )
    return {
        "gt_supported": gt_supported,
        "gt_core": gt_core,
        "gt_crosswalk": gt_crosswalk,
        "gt_road": gt_road,
        "spatial_offroad": spatial_offroad,
        "sim_transition": sim_transition,
        "transition_overstay": transition_overstay,
        "slack_frames": slack_frames,
        "slack_seconds": slack_seconds,
        "corridor_margin_m": corridor_margin,
    }


def valid_region_margin(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    has_polygon_regions = has_type_specific_valid_regions(
        features.valid_regions,
        config,
    )
    if not has_polygon_regions and not features.road_edges:
        return {
            "score": 0.0,
            "unsafe_rate": 1.0,
            "by_type": {},
            "missing_road_edges": True,
            "valid_region_source": "missing",
        }
    boxes = _sim_boxes(features, config)
    gt_boxes = _gt_boxes(features, config)
    transition_regions = has_pedestrian_transition_regions(
        features.valid_regions
    )
    transition_details = None
    if has_polygon_regions:
        outside = ~footprints_inside_valid_regions(
            boxes,
            features.object_types,
            features.valid_regions,
            config,
        )
        gt_outside = ~footprints_inside_valid_regions(
            gt_boxes,
            features.object_types,
            features.valid_regions,
            config,
        )
        valid_region_source = "type_specific_polygon"
        type_margins = features.valid_region_definition.get(
            "boundary_margin_m_by_region",
            features.valid_region_definition.get(
                "boundary_margin_m_by_agent_type",
                {},
            ),
        )
        pedestrian_mask = features.object_types == config.pedestrian_type
        if transition_regions and pedestrian_mask.any():
            transition_details = _pedestrian_transition_masks(
                boxes[:, pedestrian_mask],
                gt_boxes[:, pedestrian_mask],
                features.future_validity[pedestrian_mask],
                features.valid_regions,
                features.valid_region_definition,
                config,
            )
            outside[:, pedestrian_mask] = transition_details[
                "spatial_offroad"
            ]
            gt_outside[:, pedestrian_mask] = ~transition_details[
                "gt_supported"
            ].unsqueeze(0)
            valid_region_source = "type_specific_polygon_transition_aware"
    else:
        distances = _distance_to_road_edge_in_agent_chunks(
            boxes=boxes,
            validity=features.future_validity,
            road_edges=features.road_edges,
            chunk_size=config.valid_region_agent_chunk_size,
        )
        margins = _valid_region_margin_by_type(features.object_types, config)
        outside = distances > margins.view(1, -1, 1)
        gt_distances = _distance_to_road_edge_in_agent_chunks(
            boxes=gt_boxes,
            validity=features.future_validity,
            road_edges=features.road_edges,
            chunk_size=config.valid_region_agent_chunk_size,
        )
        gt_outside = gt_distances > margins.view(1, -1, 1)
        valid_region_source = "road_edge_margin_fallback"
        type_margins = {
            "vehicle": config.vehicle_valid_region_margin_m,
            "cyclist": config.two_wheeler_valid_region_margin_m,
            "pedestrian": config.pedestrian_valid_region_margin_m,
        }
    gt_validity = features.future_validity.unsqueeze(0)
    evaluable_gt_inside = gt_validity & ~gt_outside
    validity = evaluable_gt_inside.expand_as(outside)

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
        num_gt_outside = int(gt_outside[gt_valid].sum().item())
        total_gt_outside += num_gt_outside
        total_gt_samples += num_gt
        if num_sim == 0 or num_gt == 0:
            by_type[type_name] = None
            continue
        num_spatial_outside = int(outside[sim_valid].sum().item())
        gt_rate = num_gt_outside / num_gt
        num_transition_overstay = 0
        transition_audit = {}
        if (
            object_type == config.pedestrian_type
            and transition_details is not None
        ):
            num_transition_overstay = int(
                transition_details["transition_overstay"].sum().item()
            )
            transition_audit = {
                "num_spatial_offroad_frames": num_spatial_outside,
                "num_transition_overstay_frames": num_transition_overstay,
                "spatial_offroad_rate": num_spatial_outside / num_sim,
                "transition_overstay_rate": num_transition_overstay / num_sim,
                "excluded_map_unsupported_rate": gt_rate,
                "transition_time_slack_frames": transition_details[
                    "slack_frames"
                ],
                "transition_time_slack_s": transition_details[
                    "slack_seconds"
                ],
                "gt_road_corridor_margin_m": transition_details[
                    "corridor_margin_m"
                ],
            }
        num_sim_outside = num_spatial_outside + num_transition_overstay
        sim_rate = min(1.0, num_sim_outside / num_sim)
        score = 1.0 - sim_rate
        by_type[type_name] = {
            "score": score,
            "outside_rate_on_gt_inside_frames": sim_rate,
            "excess_outside_rate": sim_rate,
            "sim_outside_rate": sim_rate,
            "gt_outside_rate": gt_rate,
            "num_unsafe": num_sim_outside,
            "num_samples": num_sim,
            "gt_num_unsafe": num_gt_outside,
            "gt_num_samples": num_gt,
            "num_excluded_gt_outside_frames": num_gt_outside,
            "num_agents": int(type_mask.sum().item()),
            **transition_audit,
        }
        present_scores.append(score)
        total_sim_outside += num_sim_outside
        total_sim_samples += num_sim
    sim_rate = total_sim_outside / total_sim_samples if total_sim_samples else 0.0
    gt_rate = total_gt_outside / total_gt_samples if total_gt_samples else 0.0
    return {
        "score": sum(present_scores) / len(present_scores) if present_scores else 1.0,
        "unsafe_rate": sim_rate,
        "outside_rate_on_gt_inside_frames": sim_rate,
        "excess_outside_rate": sim_rate,
        "sim_outside_rate": sim_rate,
        "gt_outside_rate": gt_rate,
        "num_unsafe": total_sim_outside,
        "num_samples": total_sim_samples,
        "gt_num_unsafe": total_gt_outside,
        "gt_num_samples": total_gt_samples,
        "by_type": by_type,
        "aggregation": "agent_type_macro_average_on_gt_inside_frames",
        "missing_road_edges": not bool(features.road_edges),
        "valid_region_source": valid_region_source,
        "type_margins_m": type_margins,
        "polygon_schema_version": features.valid_region_definition.get(
            "schema_version"
        ),
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
        "valid_region_margin": valid_region,
        "aggregation": "0.5_collision_plus_0.5_valid_region",
    }
