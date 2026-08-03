"""Tensor-only queries for HetroD type-specific polygon valid regions."""

from __future__ import annotations

from typing import Any

import torch

from .config import DEFAULT_CONFIG, HetrodMetricConfig


def region_key_for_object_type(
    object_type: int,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> str:
    if object_type == config.vehicle_type:
        return "vehicle"
    if object_type == config.two_wheeler_type:
        return "cyclist"
    if object_type == config.pedestrian_type:
        return "pedestrian"
    raise ValueError(f"Unsupported HetroD object type: {object_type}")


def has_type_specific_valid_regions(
    valid_regions: Any,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> bool:
    if not isinstance(valid_regions, dict):
        return False
    return all(
        isinstance(valid_regions.get(region_key_for_object_type(object_type, config)), list)
        and bool(valid_regions[region_key_for_object_type(object_type, config)])
        for object_type in config.evaluated_object_types
    )


def has_pedestrian_transition_regions(valid_regions: Any) -> bool:
    if not isinstance(valid_regions, dict):
        return False
    return (
        all(
            isinstance(valid_regions.get(key), list)
            for key in (
                "pedestrian_core",
                "pedestrian_crosswalk",
                "pedestrian_road",
            )
        )
        and bool(valid_regions["pedestrian_core"])
        and bool(valid_regions["pedestrian_road"])
    )


def _closed_ring(ring: torch.Tensor, device: torch.device) -> torch.Tensor:
    ring = torch.as_tensor(ring, dtype=torch.float32, device=device)[..., :2]
    if ring.ndim != 2 or ring.shape[0] < 3:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if not torch.allclose(ring[0], ring[-1]):
        ring = torch.cat([ring, ring[:1]], dim=0)
    return ring


def _points_in_ring(points: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """Return boundary-inclusive point-in-polygon results for one ring."""
    ring = _closed_ring(ring, points.device)
    if ring.shape[0] < 4:
        return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)

    start = ring[:-1]
    end = ring[1:]
    px = points[:, 0:1]
    py = points[:, 1:2]
    x1 = start[:, 0].unsqueeze(0)
    y1 = start[:, 1].unsqueeze(0)
    x2 = end[:, 0].unsqueeze(0)
    y2 = end[:, 1].unsqueeze(0)

    denominator = y2 - y1
    safe_denominator = torch.where(
        denominator.abs() < 1e-12,
        torch.ones_like(denominator),
        denominator,
    )
    intersections = (
        ((y1 > py) != (y2 > py))
        & (px < (x2 - x1) * (py - y1) / safe_denominator + x1)
    )
    inside = (intersections.sum(dim=1) % 2) == 1

    segment = end - start
    point_delta = points[:, None, :] - start[None, :, :]
    denominator_2d = (segment * segment).sum(dim=-1).clamp_min(1e-12)
    projection = (
        (point_delta * segment[None, :, :]).sum(dim=-1)
        / denominator_2d.unsqueeze(0)
    ).clamp(0.0, 1.0)
    closest = start[None, :, :] + projection.unsqueeze(-1) * segment[None, :, :]
    on_boundary = (
        torch.linalg.norm(points[:, None, :] - closest, dim=-1).amin(dim=1)
        <= 1e-5
    )
    return inside | on_boundary


def points_in_valid_region(
    points: torch.Tensor,
    region_records: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Query a union of polygons with holes without requiring Shapely."""
    original_shape = points.shape[:-1]
    flat_points = points.reshape(-1, 2)
    result_parts = []
    for start_index in range(0, flat_points.shape[0], chunk_size):
        chunk = flat_points[start_index : start_index + chunk_size]
        chunk_inside = torch.zeros(
            chunk.shape[0],
            dtype=torch.bool,
            device=chunk.device,
        )
        for record in region_records:
            polygon_inside = _points_in_ring(chunk, record["exterior"])
            for hole in record.get("holes", []):
                polygon_inside &= ~_points_in_ring(chunk, hole)
            chunk_inside |= polygon_inside
        result_parts.append(chunk_inside)
    if not result_parts:
        return torch.zeros(original_shape, dtype=torch.bool, device=points.device)
    return torch.cat(result_parts).reshape(original_shape)


def box_corners_xy(boxes: torch.Tensor) -> torch.Tensor:
    """Return four oriented footprint corners for boxes [..., 7]."""
    center = boxes[..., :2]
    half_length = 0.5 * boxes[..., 3]
    half_width = 0.5 * boxes[..., 4]
    heading = boxes[..., 6]
    forward = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    lateral = torch.stack([-torch.sin(heading), torch.cos(heading)], dim=-1)
    forward = forward * half_length.unsqueeze(-1)
    lateral = lateral * half_width.unsqueeze(-1)
    return torch.stack(
        [
            center + forward + lateral,
            center + forward - lateral,
            center - forward + lateral,
            center - forward - lateral,
        ],
        dim=-2,
    )


def footprints_inside_valid_regions(
    boxes: torch.Tensor,
    object_types: torch.Tensor,
    valid_regions: dict[str, list[dict[str, Any]]],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Return [rollout, agent, step] full-footprint containment."""
    corners = box_corners_xy(boxes)
    inside = torch.zeros(
        boxes.shape[:-1],
        dtype=torch.bool,
        device=boxes.device,
    )
    for object_type in config.evaluated_object_types:
        type_mask = object_types == object_type
        if not type_mask.any():
            continue
        region_key = region_key_for_object_type(object_type, config)
        type_corners = corners[:, type_mask]
        corner_inside = points_in_valid_region(
            type_corners,
            valid_regions[region_key],
            chunk_size=config.valid_region_query_chunk_size,
        )
        inside[:, type_mask] = corner_inside.all(dim=-1)
    return inside


def footprints_inside_region_records(
    boxes: torch.Tensor,
    region_records: list[dict[str, Any]],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Return full-footprint containment in one polygon union."""
    corners = box_corners_xy(boxes)
    corner_inside = points_in_valid_region(
        corners,
        region_records,
        chunk_size=config.valid_region_query_chunk_size,
    )
    return corner_inside.all(dim=-1)
