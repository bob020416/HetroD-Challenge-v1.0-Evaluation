from __future__ import annotations

import torch

from wosac_fast_eval_tool.fast_sim_agents_metrics import map_metric_features

from .config import DEFAULT_CONFIG, HetrodMetricConfig


def has_full_history(track_masks: torch.Tensor, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    return track_masks[:, : config.current_time_index + 1].all(dim=1)


def has_full_future(track_masks: torch.Tensor, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    return track_masks[:, config.future_start_index :].all(dim=1)


def future_path_length(
    tracks: torch.Tensor,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    xy = tracks[:, config.current_time_index :, :2]
    return torch.linalg.norm(xy[:, 1:] - xy[:, :-1], dim=-1).sum(dim=-1)


def future_displacement(
    tracks: torch.Tensor,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Compatibility alias for the path-length based motion filter."""
    return future_path_length(tracks, config)


def is_evaluated_type(object_types: torch.Tensor, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    valid_types = torch.tensor(config.evaluated_object_types, device=object_types.device, dtype=object_types.dtype)
    return torch.isin(object_types, valid_types)


def select_base_agents(gt_scenario: dict, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    """Return the base HetroD selection mask before map/cross-type interaction filters."""
    tracks = gt_scenario["tracks"]
    track_masks = gt_scenario["track_masks"]
    object_types = gt_scenario["object_types"]
    return (
        has_full_history(track_masks, config)
        & has_full_future(track_masks, config)
        & is_evaluated_type(object_types, config)
        & (future_path_length(tracks, config) > config.future_displacement_threshold_m)
    )


def distance_to_valid_region(gt_scenario: dict, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    """Return current-step signed distance to the scenario valid region.

    The current GT schema exposes WOSAC road edges but not a separate pedestrian
    valid-region layer. Until that schema exists, this uses the WOSAC road-edge
    signed distance for all evaluated types: negative means inside the drivable
    region, positive means outside. The selection threshold allows a small
    positive tolerance.
    """
    tracks = gt_scenario["tracks"]
    track_masks = gt_scenario["track_masks"]
    road_edges = gt_scenario.get("road_edges", [])
    num_agents = tracks.shape[0]
    if not road_edges:
        return torch.full((num_agents,), float("inf"), device=tracks.device)

    current = config.current_time_index
    boxes = torch.cat(
        [
            tracks[:, current : current + 1, 0:6],
            tracks[:, current : current + 1, 6:7],
        ],
        dim=-1,
    ).unsqueeze(0)
    valid = track_masks[:, current : current + 1]
    evaluated_object_mask = torch.ones(num_agents, dtype=torch.bool, device=tracks.device)
    distances = map_metric_features.compute_distance_to_road_edge(
        boxes=boxes,
        valid=valid,
        evaluated_object_mask=evaluated_object_mask,
        road_edge_polylines=road_edges,
    )
    distances = distances[0, :, 0]
    return torch.where(valid[:, 0], distances, torch.full_like(distances, float("inf")))


def cross_type_pair_mask(object_types: torch.Tensor, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    type_ok = is_evaluated_type(object_types, config)
    different_type = object_types[:, None] != object_types[None, :]
    not_self = ~torch.eye(object_types.shape[0], dtype=torch.bool, device=object_types.device)
    return type_ok[:, None] & type_ok[None, :] & different_type & not_self


def min_cross_type_distance(gt_scenario: dict, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    tracks = gt_scenario["tracks"]
    track_masks = gt_scenario["track_masks"]
    object_types = gt_scenario["object_types"]
    xy = tracks[:, config.current_time_index :, :2]
    valid = track_masks[:, config.current_time_index :]

    pair_mask = cross_type_pair_mask(object_types, config)
    pair_valid = valid[:, None, :] & valid[None, :, :] & pair_mask[:, :, None]
    distances = torch.linalg.norm(xy[:, None, :, :] - xy[None, :, :, :], dim=-1)
    distances = torch.where(pair_valid, distances, torch.full_like(distances, float("inf")))
    return distances.amin(dim=(1, 2))


def _first_time_to_proximity(
    relative_xy: torch.Tensor,
    relative_velocity_xy: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """First non-negative time when a constant-velocity pair enters radius."""
    a = (relative_velocity_xy * relative_velocity_xy).sum(dim=-1)
    b = 2.0 * (relative_xy * relative_velocity_xy).sum(dim=-1)
    c = (relative_xy * relative_xy).sum(dim=-1) - radius * radius
    already_close = c <= 0.0

    discriminant = b * b - 4.0 * a * c
    can_solve = (a > 1e-8) & (discriminant >= 0.0)
    sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
    t_enter = (-b - sqrt_disc) / (2.0 * torch.clamp(a, min=1e-8))
    t_exit = (-b + sqrt_disc) / (2.0 * torch.clamp(a, min=1e-8))
    future_hit = can_solve & (t_exit >= 0.0)
    t_hit = torch.where(t_enter >= 0.0, t_enter, torch.zeros_like(t_enter))
    return torch.where(
        already_close,
        torch.zeros_like(c),
        torch.where(future_hit, t_hit, torch.full_like(c, float("inf"))),
    )


def min_cross_type_ttc(gt_scenario: dict, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    """Return min cross-type time-to-proximity per agent from GT future states."""
    tracks = gt_scenario["tracks"]
    track_masks = gt_scenario["track_masks"]
    object_types = gt_scenario["object_types"]
    xy = tracks[:, config.current_time_index :, :2]
    velocity = tracks[:, config.current_time_index :, 7:9]
    valid = track_masks[:, config.current_time_index :]

    relative_xy = xy[None, :, :, :] - xy[:, None, :, :]
    relative_velocity = velocity[None, :, :, :] - velocity[:, None, :, :]
    ttc = _first_time_to_proximity(
        relative_xy,
        relative_velocity,
        radius=config.cross_type_ttc_proximity_radius_m,
    )
    pair_mask = cross_type_pair_mask(object_types, config)
    pair_valid = valid[:, None, :] & valid[None, :, :] & pair_mask[:, :, None]
    ttc = ttc + torch.arange(ttc.shape[-1], device=tracks.device, dtype=ttc.dtype) * config.seconds_per_step
    ttc = torch.where(pair_valid, ttc, torch.full_like(ttc, float("inf")))
    return ttc.amin(dim=(1, 2))


def select_agents(gt_scenario: dict, config: HetrodMetricConfig = DEFAULT_CONFIG) -> torch.Tensor:
    return (
        select_base_agents(gt_scenario, config)
        & (distance_to_valid_region(gt_scenario, config) < config.valid_region_distance_threshold_m)
        & (
            (min_cross_type_distance(gt_scenario, config) < config.cross_type_distance_threshold_m)
            | (min_cross_type_ttc(gt_scenario, config) < config.cross_type_ttc_threshold_s)
        )
    )


def selection_audit(
    gt_scenario: dict,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, int]:
    track_masks = gt_scenario["track_masks"]
    tracks = gt_scenario["tracks"]
    object_types = gt_scenario["object_types"]
    history = has_full_history(track_masks, config)
    future = has_full_future(track_masks, config)
    evaluated_type = is_evaluated_type(object_types, config)
    motion = future_path_length(tracks, config) > config.future_displacement_threshold_m
    valid_region = (
        distance_to_valid_region(gt_scenario, config)
        < config.valid_region_distance_threshold_m
    )
    interaction = (
        min_cross_type_distance(gt_scenario, config)
        < config.cross_type_distance_threshold_m
    ) | (
        min_cross_type_ttc(gt_scenario, config)
        < config.cross_type_ttc_threshold_s
    )
    cumulative = torch.ones_like(history)
    counts = {"total_agents": int(history.numel())}
    for name, condition in (
        ("full_history", history),
        ("full_future", future),
        ("evaluated_type", evaluated_type),
        ("future_path_length", motion),
        ("valid_region", valid_region),
        ("cross_type_interaction", interaction),
    ):
        cumulative &= condition
        counts[f"after_{name}"] = int(cumulative.sum().item())
    return counts
