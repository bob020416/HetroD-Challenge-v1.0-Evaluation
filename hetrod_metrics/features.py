from __future__ import annotations

from dataclasses import dataclass

import torch

from wosac_fast_eval_tool.fast_sim_agents_metrics import trajectory_features

from .config import DEFAULT_CONFIG, HetrodMetricConfig


@dataclass(frozen=True)
class HetrodFeatureBundle:
    object_ids: torch.Tensor
    object_types: torch.Tensor
    selected_mask: torch.Tensor
    gt_tracks: torch.Tensor
    simulated_future: torch.Tensor
    context_object_ids: torch.Tensor
    context_object_types: torch.Tensor
    context_anchor_mask: torch.Tensor
    context_gt_tracks: torch.Tensor
    context_simulated_future: torch.Tensor
    context_future_validity: torch.Tensor
    road_edges: list[torch.Tensor]
    future_validity: torch.Tensor
    speed_validity: torch.Tensor
    acceleration_validity: torch.Tensor
    log_kinematics: dict[str, torch.Tensor]
    sim_kinematics: dict[str, torch.Tensor]


def _prediction_indices_for_object_ids(
    prediction_agent_ids: torch.Tensor,
    object_ids: torch.Tensor,
) -> torch.Tensor:
    match = prediction_agent_ids.unsqueeze(1) == object_ids.unsqueeze(0)
    if not match.any(dim=0).all():
        missing = object_ids[~match.any(dim=0)].detach().cpu().tolist()
        raise KeyError(f"Rollout missing selected agent ids: {missing}")
    pred_pos, object_pos = torch.where(match)
    order = torch.argsort(object_pos)
    return pred_pos[order]


def validate_prediction(
    gt_scenario: dict,
    prediction: dict[str, torch.Tensor],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> None:
    required_keys = {"agent_id", "simulated_states"}
    missing_keys = required_keys - prediction.keys()
    if missing_keys:
        raise KeyError(f"Rollout missing required keys: {sorted(missing_keys)}")

    gt_object_ids = gt_scenario["object_ids"].int()
    prediction_agent_ids = torch.as_tensor(prediction["agent_id"]).int()
    simulated_states = torch.as_tensor(prediction["simulated_states"])
    if gt_object_ids.ndim != 1 or prediction_agent_ids.ndim != 1:
        raise ValueError("GT object_ids and rollout agent_id must be one-dimensional.")
    if torch.unique(gt_object_ids).numel() != gt_object_ids.numel():
        raise ValueError("GT object_ids must be unique.")
    if torch.unique(prediction_agent_ids).numel() != prediction_agent_ids.numel():
        raise ValueError("Rollout agent_id must be unique.")

    missing_agent_ids = gt_object_ids[~torch.isin(gt_object_ids, prediction_agent_ids)]
    extra_agent_ids = prediction_agent_ids[~torch.isin(prediction_agent_ids, gt_object_ids)]
    if missing_agent_ids.numel() or extra_agent_ids.numel():
        raise ValueError(
            "Rollout agent_id must exactly match all GT object_ids. "
            f"Missing: {missing_agent_ids.detach().cpu().tolist()}; "
            f"extra: {extra_agent_ids.detach().cpu().tolist()}."
        )

    expected_steps = gt_scenario["tracks"].shape[1] - config.future_start_index
    expected_shape = (
        prediction_agent_ids.numel(),
        expected_steps,
        4,
    )
    if simulated_states.ndim != 4:
        raise ValueError(
            "simulated_states must have shape [32, num_agents, num_steps, 4]."
        )
    if simulated_states.shape[0] != config.required_num_rollouts:
        raise ValueError(
            "simulated_states must contain exactly "
            f"{config.required_num_rollouts} rollouts, got {simulated_states.shape[0]}."
        )
    if tuple(simulated_states.shape[1:]) != expected_shape:
        raise ValueError(
            "simulated_states shape mismatch: expected "
            f"[{config.required_num_rollouts}, {expected_shape[0]}, {expected_shape[1]}, 4], "
            f"got {tuple(simulated_states.shape)}."
        )
    if not torch.isfinite(simulated_states).all():
        raise ValueError("simulated_states contains NaN or Inf.")


def build_feature_bundle(
    gt_scenario: dict,
    prediction: dict[str, torch.Tensor],
    selected_mask: torch.Tensor,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> HetrodFeatureBundle:
    """Build aligned GT/simulation kinematic features for selected HetroD agents."""
    validate_prediction(gt_scenario, prediction, config)
    selected_indices = torch.where(selected_mask)[0]
    if selected_indices.numel() == 0:
        raise ValueError("HetroD feature bundle requires at least one selected agent.")

    tracks = gt_scenario["tracks"]
    track_masks = gt_scenario["track_masks"]
    all_object_ids = gt_scenario["object_ids"].int()
    object_ids = all_object_ids[selected_indices]
    object_types = gt_scenario["object_types"].int()[selected_indices]
    pred_agent_ids = prediction["agent_id"].to(device=tracks.device).int()
    simulated_states = prediction["simulated_states"].to(device=tracks.device).float()

    context_object_ids = all_object_ids
    context_gt_tracks = tracks
    prediction_order = _prediction_indices_for_object_ids(pred_agent_ids, all_object_ids)
    context_simulated_future = simulated_states[:, prediction_order]
    context_anchor_mask = selected_mask.bool()
    simulated_future = context_simulated_future[:, context_anchor_mask]

    gt_selected = tracks[selected_indices]
    gt_xyzyaw = gt_selected[..., [0, 1, 2, 6]].float()
    gt_history = gt_xyzyaw[:, : config.future_start_index]
    sim_with_history = torch.cat(
        [
            gt_history.unsqueeze(0).expand(simulated_future.shape[0], -1, -1, -1),
            simulated_future,
        ],
        dim=2,
    )

    log_linear_speed, log_linear_accel, log_angular_speed, log_angular_accel = (
        trajectory_features.compute_kinematic_features(
            gt_xyzyaw.unsqueeze(0),
            seconds_per_step=config.seconds_per_step,
        )
    )
    sim_linear_speed, sim_linear_accel, sim_angular_speed, sim_angular_accel = (
        trajectory_features.compute_kinematic_features(
            sim_with_history,
            seconds_per_step=config.seconds_per_step,
        )
    )

    future_slice = slice(config.future_start_index, None)
    future_validity = track_masks[selected_indices, future_slice].bool()
    speed_validity, acceleration_validity = trajectory_features.compute_kinematic_validity(
        future_validity
    )

    return HetrodFeatureBundle(
        object_ids=object_ids,
        object_types=object_types,
        selected_mask=selected_mask,
        gt_tracks=gt_selected,
        simulated_future=simulated_future,
        context_object_ids=context_object_ids,
        context_object_types=gt_scenario["object_types"].int(),
        context_anchor_mask=context_anchor_mask,
        context_gt_tracks=context_gt_tracks,
        context_simulated_future=context_simulated_future,
        context_future_validity=track_masks[:, future_slice].bool(),
        road_edges=gt_scenario.get("road_edges", []),
        future_validity=future_validity,
        speed_validity=speed_validity,
        acceleration_validity=acceleration_validity,
        log_kinematics={
            "linear_speed": log_linear_speed[0, :, future_slice],
            "linear_acceleration": log_linear_accel[0, :, future_slice],
            "angular_speed": log_angular_speed[0, :, future_slice],
            "angular_acceleration": log_angular_accel[0, :, future_slice],
        },
        sim_kinematics={
            "linear_speed": sim_linear_speed[:, :, future_slice],
            "linear_acceleration": sim_linear_accel[:, :, future_slice],
            "angular_speed": sim_angular_speed[:, :, future_slice],
            "angular_acceleration": sim_angular_accel[:, :, future_slice],
        },
    )
