"""Type-wise predicted occupancy coverage metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from wosac_fast_eval_tool.fast_sim_agents_metrics.map_metric_features import (
    _compute_signed_distance_to_polylines,
    _tensorize_polylines,
)

from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .features import HetrodFeatureBundle


@dataclass(frozen=True)
class _OccupancyRecord:
    agent_index: int
    object_type: int
    occupancy: torch.Tensor


def _type_name(object_type: int, config: HetrodMetricConfig) -> str:
    names = {
        config.vehicle_type: "vehicle",
        config.two_wheeler_type: "two_wheeler",
        config.pedestrian_type: "pedestrian",
    }
    return names[object_type]


def _valid_region_margin(object_type: int, config: HetrodMetricConfig) -> float:
    if object_type == config.two_wheeler_type:
        return config.two_wheeler_valid_region_margin_m
    if object_type == config.pedestrian_type:
        return config.pedestrian_valid_region_margin_m
    return config.vehicle_valid_region_margin_m


def _default_footprint(
    object_type: int,
    config: HetrodMetricConfig,
) -> tuple[float, float]:
    if object_type == config.two_wheeler_type:
        return (
            config.coverage_default_two_wheeler_length_m,
            config.coverage_default_two_wheeler_width_m,
        )
    if object_type == config.pedestrian_type:
        return (
            config.coverage_default_pedestrian_length_m,
            config.coverage_default_pedestrian_width_m,
        )
    return (
        config.coverage_default_vehicle_length_m,
        config.coverage_default_vehicle_width_m,
    )


def _rasterize_oriented_boxes(
    xy: torch.Tensor,
    yaw: torch.Tensor,
    length: float,
    width: float,
    resolution: float,
) -> torch.Tensor:
    """Return sparse [rollout, grid_x, grid_y] occupancy rows."""
    rows = []
    half_diagonal = 0.5 * math.hypot(length, width)
    cell_padding = 0.5 * resolution
    for rollout_index in range(xy.shape[0]):
        center = xy[rollout_index]
        min_index = torch.floor((center - half_diagonal) / resolution).to(torch.int64)
        max_index = torch.floor((center + half_diagonal) / resolution).to(torch.int64)
        grid_x = torch.arange(
            int(min_index[0].item()),
            int(max_index[0].item()) + 1,
            device=xy.device,
        )
        grid_y = torch.arange(
            int(min_index[1].item()),
            int(max_index[1].item()) + 1,
            device=xy.device,
        )
        index_x, index_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        cell_xy = torch.stack(
            [
                (index_x.flatten() + 0.5) * resolution,
                (index_y.flatten() + 0.5) * resolution,
            ],
            dim=-1,
        )
        delta = cell_xy - center
        cosine = torch.cos(yaw[rollout_index])
        sine = torch.sin(yaw[rollout_index])
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        occupied = (
            (local_x.abs() <= 0.5 * length + cell_padding)
            & (local_y.abs() <= 0.5 * width + cell_padding)
        )
        occupied_indices = torch.stack(
            [index_x.flatten()[occupied], index_y.flatten()[occupied]],
            dim=-1,
        )
        rollout_column = torch.full(
            (occupied_indices.shape[0], 1),
            rollout_index,
            dtype=torch.int64,
            device=xy.device,
        )
        rows.append(torch.cat([rollout_column, occupied_indices], dim=-1))
    return torch.cat(rows, dim=0)


def _build_occupancy_records(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig,
) -> list[_OccupancyRecord]:
    records = []
    future_steps = features.simulated_future.shape[2]
    dimension_slice = slice(
        config.future_start_index,
        config.future_start_index + future_steps,
    )
    dimensions = features.gt_tracks[:, dimension_slice, 3:5]
    for agent_index in range(features.object_ids.shape[0]):
        object_type = int(features.object_types[agent_index].item())
        for step_index in range(future_steps):
            if not features.future_validity[agent_index, step_index]:
                continue
            length = float(dimensions[agent_index, step_index, 0].item())
            width = float(dimensions[agent_index, step_index, 1].item())
            if length <= 0.0 or width <= 0.0:
                length, width = _default_footprint(object_type, config)
            occupancy = _rasterize_oriented_boxes(
                features.simulated_future[:, agent_index, step_index, :2],
                features.simulated_future[:, agent_index, step_index, 3],
                length,
                width,
                config.coverage_grid_resolution_m,
            )
            records.append(
                _OccupancyRecord(
                    agent_index=agent_index,
                    object_type=object_type,
                    occupancy=occupancy,
                )
            )
    return records


def _valid_grid_keys_by_type(
    records: list[_OccupancyRecord],
    road_edges: list[torch.Tensor],
    config: HetrodMetricConfig,
) -> dict[int, set[tuple[int, int]]]:
    valid_keys = {}
    (
        polylines,
        use_left_neighbor,
        use_right_neighbor,
        left_neighbors,
        right_neighbors,
    ) = _tensorize_polylines(road_edges, seg_length=50)
    for object_type in config.evaluated_object_types:
        type_cells = [
            record.occupancy[:, 1:]
            for record in records
            if record.object_type == object_type
        ]
        if not type_cells:
            continue
        cells = torch.unique(torch.cat(type_cells, dim=0), dim=0)
        valid_parts = []
        for start in range(0, cells.shape[0], config.coverage_map_query_chunk_size):
            chunk = cells[start : start + config.coverage_map_query_chunk_size]
            centers = (chunk.to(torch.float32) + 0.5) * config.coverage_grid_resolution_m
            xyz = torch.zeros(
                chunk.shape[0],
                3,
                dtype=torch.float32,
                device=chunk.device,
            )
            xyz[:, :2] = centers
            distances = _compute_signed_distance_to_polylines(
                xyzs=xyz,
                polylines=polylines,
                use_left_neighbor=use_left_neighbor,
                use_right_neighbor=use_right_neighbor,
                left_neighbors=left_neighbors,
                right_neighbors=right_neighbors,
                z_stretch=3.0,
            )
            valid_parts.append(
                chunk[
                    distances <= _valid_region_margin(object_type, config)
                ]
            )
        valid_cells = torch.cat(valid_parts, dim=0).detach().cpu().tolist()
        valid_keys[object_type] = {tuple(cell) for cell in valid_cells}
    return valid_keys


def _record_coverage(
    record: _OccupancyRecord,
    valid_keys: set[tuple[int, int]],
    resolution: float,
) -> tuple[float, float, float]:
    occupancy = record.occupancy.detach().cpu().tolist()
    per_rollout = {}
    total_cells = 0
    valid_cells = 0
    for rollout_index, grid_x, grid_y in occupancy:
        total_cells += 1
        key = (grid_x, grid_y)
        if key not in valid_keys:
            continue
        valid_cells += 1
        per_rollout.setdefault(rollout_index, set()).add(key)

    cell_sets = list(per_rollout.values())
    if not cell_sets:
        return 0.0, 0.0, 0.0
    union_cells = set().union(*cell_sets)
    sum_cells = sum(len(cells) for cells in cell_sets)
    baseline_cells = max(len(cells) for cells in cell_sets)
    max_incremental_cells = sum_cells - baseline_cells
    incremental_cells = len(union_cells) - baseline_cells
    score = (
        incremental_cells / max_incremental_cells
        if max_incremental_cells > 0
        else 0.0
    )
    union_area = len(union_cells) * resolution * resolution
    valid_fraction = valid_cells / total_cells if total_cells else 0.0
    return float(score), float(union_area), float(valid_fraction)


def compute_coverage(
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Compute type-balanced rollout occupancy diversity in the valid region."""
    if not features.road_edges:
        return {
            "score": 0.0,
            "by_type": {},
            "grid_resolution_m": config.coverage_grid_resolution_m,
            "missing_road_edges": True,
        }
    records = _build_occupancy_records(features, config)
    valid_keys = _valid_grid_keys_by_type(records, features.road_edges, config)
    per_agent = {}
    for record in records:
        score, union_area, valid_fraction = _record_coverage(
            record,
            valid_keys.get(record.object_type, set()),
            config.coverage_grid_resolution_m,
        )
        values = per_agent.setdefault(
            record.agent_index,
            {"scores": [], "areas": [], "valid_fractions": []},
        )
        values["scores"].append(score)
        values["areas"].append(union_area)
        values["valid_fractions"].append(valid_fraction)

    by_type = {}
    for object_type in config.evaluated_object_types:
        type_name = _type_name(object_type, config)
        agent_reports = [
            values
            for agent_index, values in per_agent.items()
            if int(features.object_types[agent_index].item()) == object_type
        ]
        if not agent_reports:
            by_type[type_name] = None
            continue
        agent_scores = [
            sum(report["scores"]) / len(report["scores"])
            for report in agent_reports
        ]
        by_type[type_name] = {
            "score": sum(agent_scores) / len(agent_scores),
            "num_agents": len(agent_reports),
            "num_agent_timesteps": sum(len(report["scores"]) for report in agent_reports),
            "mean_union_area_m2": sum(
                sum(report["areas"]) / len(report["areas"])
                for report in agent_reports
            )
            / len(agent_reports),
            "mean_valid_occupancy_fraction": sum(
                sum(report["valid_fractions"]) / len(report["valid_fractions"])
                for report in agent_reports
            )
            / len(agent_reports),
        }

    present_scores = [
        report["score"] for report in by_type.values() if report is not None
    ]
    return {
        "score": sum(present_scores) / len(present_scores) if present_scores else 0.0,
        "by_type": by_type,
        "grid_resolution_m": config.coverage_grid_resolution_m,
        "num_rollouts": int(features.simulated_future.shape[0]),
        "scoring_method": "normalized_incremental_box_union",
        "missing_road_edges": False,
        "valid_region_source": "road_edge_margin_fallback",
    }
