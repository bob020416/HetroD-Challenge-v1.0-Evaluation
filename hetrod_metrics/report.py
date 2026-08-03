"""Report aggregation helpers for HetroD Challenge metrics."""

from __future__ import annotations

from dataclasses import asdict
import math
import re
from typing import Any

import torch
from waymo_open_dataset.protos import sim_agents_metrics_pb2

from . import __version__
from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .coverage import compute_coverage
from .cross_type import compute_cross_type_interaction
from .features import build_feature_bundle
from .kinematic import compute_kinematic_realism, normalize_kinematic_realism
from .interaction_candidates import (
    competition_selection_audit,
)
from .interaction_selection import select_competition_agents
from .safety import compute_safety

METRIC_VERSION = f"hetrod-{__version__}"


class NoSelectedAgentsError(ValueError):
    """Raised when a scenario has no agents matching the HetroD selection spec."""


def validate_config(config: HetrodMetricConfig) -> None:
    if config.future_start_index != config.current_time_index + 1:
        raise ValueError("future_start_index must equal current_time_index + 1.")
    if config.required_num_rollouts <= 0:
        raise ValueError("required_num_rollouts must be positive.")
    if config.seconds_per_step <= 0.0:
        raise ValueError("seconds_per_step must be positive.")
    if config.min_future_valid_frames <= 0:
        raise ValueError("min_future_valid_frames must be positive.")
    if config.selection_min_history_valid_frames <= 0:
        raise ValueError(
            "selection_min_history_valid_frames must be positive."
        )
    if config.selection_min_anchor_path_length_m < 0.0:
        raise ValueError(
            "selection_min_anchor_path_length_m cannot be negative."
        )
    if config.selection_min_anchor_motion_extent_m < 0.0:
        raise ValueError(
            "selection_min_anchor_motion_extent_m cannot be negative."
        )
    if config.selection_max_arrival_gap_s < 0.0:
        raise ValueError("selection_max_arrival_gap_s cannot be negative.")
    if config.selection_max_synchronous_distance_m < 0.0:
        raise ValueError(
            "selection_max_synchronous_distance_m cannot be negative."
        )
    if config.selection_max_pairs <= 0:
        raise ValueError("selection_max_pairs must be positive.")
    if config.selection_max_pair_endpoints < 2:
        raise ValueError("selection_max_pair_endpoints must be at least two.")
    if config.selection_max_pairs_per_agent <= 0:
        raise ValueError("selection_max_pairs_per_agent must be positive.")
    if config.selection_num_fallback_agents < 0:
        raise ValueError(
            "selection_num_fallback_agents cannot be negative."
        )
    if not (
        0
        <= config.selection_min_fallback_agents
        <= config.selection_num_fallback_agents
    ):
        raise ValueError(
            "selection_min_fallback_agents must be between zero and "
            "selection_num_fallback_agents."
        )
    if config.cross_type_pair_chunk_size <= 0:
        raise ValueError("cross_type_pair_chunk_size must be positive.")
    if config.collision_pair_chunk_size <= 0:
        raise ValueError("collision_pair_chunk_size must be positive.")
    if config.valid_region_agent_chunk_size <= 0:
        raise ValueError("valid_region_agent_chunk_size must be positive.")
    if config.valid_region_query_chunk_size <= 0:
        raise ValueError("valid_region_query_chunk_size must be positive.")
    if config.coverage_grid_resolution_m <= 0.0:
        raise ValueError("coverage_grid_resolution_m must be positive.")
    weight_sum = (
        config.kinematic_weight
        + config.safety_weight
        + config.cross_type_weight
        + config.coverage_weight
    )
    if min(
        config.kinematic_weight,
        config.safety_weight,
        config.cross_type_weight,
        config.coverage_weight,
    ) < 0.0:
        raise ValueError("HetroD metric weights cannot be negative.")
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"HetroD metric weights must sum to 1.0, got {weight_sum}.")


def compute_overall_score(
    kinematic: dict[str, Any],
    safety: dict[str, Any],
    cross_type: dict[str, Any],
    coverage: dict[str, Any],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    validate_config(config)
    cross_type_applicable = cross_type.get("score") is not None
    base_target_weight = (
        config.kinematic_weight
        + config.safety_weight
        + config.cross_type_weight
    )
    active_quality_weight = (
        config.kinematic_weight
        + config.safety_weight
        + (
            config.cross_type_weight
            if cross_type_applicable
            else 0.0
        )
    )
    quality_scale = base_target_weight / active_quality_weight
    effective_weights = {
        "kinematic_realism": config.kinematic_weight * quality_scale,
        "safety": config.safety_weight * quality_scale,
        "cross_type_interaction": (
            config.cross_type_weight * quality_scale
            if cross_type_applicable
            else 0.0
        ),
    }
    coverage_bonus = (
        config.coverage_weight
        * coverage["score"]
        * kinematic["score"]
        * safety["score"]
    )
    weighted_components = {
        "kinematic_realism": (
            effective_weights["kinematic_realism"] * kinematic["score"]
        ),
        "safety": effective_weights["safety"] * safety["score"],
        "cross_type_interaction": (
            effective_weights["cross_type_interaction"]
            * (cross_type["score"] or 0.0)
        ),
        "coverage_bonus": coverage_bonus,
    }
    return {
        "score": sum(weighted_components.values()),
        "weighted_components": weighted_components,
        "coverage_bonus_gate": {
            "coverage": coverage["score"],
            "kinematic_realism": kinematic["score"],
            "safety": safety["score"],
        },
        "effective_quality_weights": effective_weights,
        "cross_type_applicable": cross_type_applicable,
        "formula": (
            "quality weights 0.30/0.35/0.25 (renormalized across "
            "applicable components) + "
            "0.10*coverage*kinematic*safety"
        ),
    }


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _, weight in values)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def _mean_present(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else 0.0


def _mean_present_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def evaluate_scenario(
    eval_config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    gt_scenario: dict[str, Any],
    prediction: dict[str, torch.Tensor],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Evaluate one scenario with all HetroD Challenge metric components."""
    validate_config(config)
    selection = select_competition_agents(gt_scenario, config)
    selected_mask = selection.anchor_mask
    if not selected_mask.any():
        raise NoSelectedAgentsError(
            "Scenario contains no agents selected by the HetroD filters."
        )

    features = build_feature_bundle(
        gt_scenario,
        prediction,
        selected_mask,
        config,
        interaction_pair_object_ids=selection.pair_object_ids,
    )
    num_rollouts = prediction["simulated_states"].shape[0]
    gt_future = gt_scenario["tracks"][
        :, config.future_start_index :, [0, 1, 2, 6]
    ].float()
    oracle_prediction = {
        "agent_id": gt_scenario["object_ids"],
        "simulated_states": gt_future.unsqueeze(0).repeat(
            num_rollouts, 1, 1, 1
        ),
    }
    oracle_features = build_feature_bundle(
        gt_scenario,
        oracle_prediction,
        selected_mask,
        config,
        interaction_pair_object_ids=selection.pair_object_ids,
    )
    oracle_kinematic = compute_kinematic_realism(
        eval_config,
        oracle_features,
        config,
    )
    kinematic = normalize_kinematic_realism(
        compute_kinematic_realism(eval_config, features, config),
        oracle_kinematic,
    )
    safety = compute_safety(features, config)
    cross_type = compute_cross_type_interaction(features, config)
    coverage = compute_coverage(features, config)
    overall = compute_overall_score(
        kinematic,
        safety,
        cross_type,
        coverage,
        config,
    )
    return {
        "metric_version": METRIC_VERSION,
        "score": overall["score"],
        "scenario_id": gt_scenario.get("scenario_id"),
        "num_selected_agents": int(selected_mask.sum().item()),
        "selection_audit": competition_selection_audit(
            gt_scenario,
            selection,
            config,
        ),
        "submission": {
            "required_agent_policy": "exact_match_all_gt_object_ids",
            "num_required_agents": int(gt_scenario["object_ids"].numel()),
            "num_submitted_agents": int(prediction["agent_id"].numel()),
            "num_rollouts": int(prediction["simulated_states"].shape[0]),
        },
        "kinematic_realism": kinematic,
        "safety": safety,
        "cross_type_interaction": cross_type,
        "coverage": coverage,
        "weighted_components": overall["weighted_components"],
        "overall_formula": overall["formula"],
        "coverage_bonus_gate": overall["coverage_bonus_gate"],
        "effective_quality_weights": overall["effective_quality_weights"],
        "metadata": {
            "valid_region_source": safety["valid_region_margin"][
                "valid_region_source"
            ],
            "valid_region_definition": (
                "pedestrian_core_crosswalk_gt_road_corridor_with_time_budget"
                if safety["valid_region_margin"]["valid_region_source"]
                == "type_specific_polygon_transition_aware"
                else (
                    "type_specific_buffered_polygon_full_footprint_on_gt_inside_frames"
                    if safety["valid_region_margin"]["valid_region_source"]
                    == "type_specific_polygon"
                    else "road_edge_margin_fallback"
                )
            ),
            "collision_definition": (
                "any_new_strict_pair_frame_box_overlap_relative_to_gt"
            ),
            "interaction_definition": (
                "reference_gt_selected_pair_minimum_distance_and_time"
            ),
            "coverage_definition": "normalized_incremental_box_union",
            "config": asdict(config),
        },
    }


def skipped_no_selected_agents_report(
    gt_scenario: dict[str, Any],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return a non-aggregated scenario report for GT with no selected agents."""
    validate_config(config)
    return {
        "metric_version": METRIC_VERSION,
        "scenario_id": gt_scenario.get("scenario_id"),
        "status": "skipped_no_selected_agents",
        "num_selected_agents": 0,
        "selection_audit": {
            "mode": "no_eligible_fallback_agent",
            "num_selected_anchors": 0,
        },
        "metadata": {
            "skip_reason": "No agents matched the HetroD selection filters.",
            "config": asdict(config),
        },
    }


def _aggregate_scenario_reports_within_location(
    scenario_reports: list[dict[str, Any]],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Aggregate sufficient statistics within one location."""
    scenario_reports = [
        report
        for report in scenario_reports
        if report.get("status") != "skipped_no_selected_agents"
    ]
    if not scenario_reports:
        raise ValueError("At least one scenario report is required.")
    validate_config(config)
    type_names = ("vehicle", "two_wheeler", "pedestrian")

    kinematic_metrics = {}
    for metric_name in (
        "linear_speed",
        "linear_acceleration",
        "angular_speed",
        "angular_acceleration",
    ):
        by_type = {}
        for type_name in type_names:
            log_ratio_values = []
            for report in scenario_reports:
                metric = report["kinematic_realism"]["metrics"][metric_name]
                log_ratio = metric["log_ratio_by_type"].get(type_name)
                weight = metric["num_samples_by_type"].get(type_name, 0)
                if log_ratio is not None and weight:
                    log_ratio_values.append((log_ratio, weight))
            mean_log_ratio = _weighted_mean(log_ratio_values)
            by_type[type_name] = (
                None
                if mean_log_ratio is None
                else min(math.exp(mean_log_ratio), 1.0)
            )
        kinematic_metrics[metric_name] = {
            "score": _mean_present(list(by_type.values())),
            "by_type": by_type,
        }
    kinematic = {
        "score": _mean_present(
            [metric["score"] for metric in kinematic_metrics.values()]
        ),
        "metrics": kinematic_metrics,
        "aggregation": "dataset_sample_weighted_then_agent_type_macro_average",
    }

    safety_components = {}
    for component_name in ("collision_rollout_rate", "valid_region_margin"):
        by_type = {}
        for type_name in type_names:
            type_reports = []
            for report in scenario_reports:
                type_report = report["safety"][component_name]["by_type"].get(type_name)
                if type_report is not None:
                    type_reports.append(type_report)
            if not type_reports:
                by_type[type_name] = None
                continue
            if component_name == "collision_rollout_rate":
                num_unsafe = sum(item["num_unsafe"] for item in type_reports)
                num_samples = sum(item["num_samples"] for item in type_reports)
                num_raw = sum(
                    item.get("num_raw_collided_agent_rollouts", item["num_unsafe"])
                    for item in type_reports
                )
                num_gt = sum(
                    item.get("num_gt_collided_agent_rollouts", 0)
                    for item in type_reports
                )
                rate = num_unsafe / num_samples
                by_type[type_name] = {
                    "score": 1.0 - rate,
                    "collision_rate": rate,
                    "new_collision_rate": rate,
                    "raw_sim_collision_rate": num_raw / num_samples,
                    "gt_collision_rate": num_gt / num_samples,
                    "num_collided_agent_rollouts": num_unsafe,
                    "num_raw_collided_agent_rollouts": num_raw,
                    "num_gt_collided_agent_rollouts": num_gt,
                    "num_valid_agent_rollouts": num_samples,
                    "num_unsafe": num_unsafe,
                    "num_samples": num_samples,
                }
            else:
                sim_outside = sum(item["num_unsafe"] for item in type_reports)
                sim_samples = sum(item["num_samples"] for item in type_reports)
                gt_outside = sum(item["gt_num_unsafe"] for item in type_reports)
                gt_samples = sum(item["gt_num_samples"] for item in type_reports)
                sim_rate = min(1.0, sim_outside / sim_samples)
                gt_rate = gt_outside / gt_samples
                by_type[type_name] = {
                    "score": 1.0 - sim_rate,
                    "outside_rate_on_gt_inside_frames": sim_rate,
                    "excess_outside_rate": sim_rate,
                    "sim_outside_rate": sim_rate,
                    "gt_outside_rate": gt_rate,
                    "num_unsafe": sim_outside,
                    "num_samples": sim_samples,
                    "gt_num_unsafe": gt_outside,
                    "gt_num_samples": gt_samples,
                }
                if any(
                    "num_transition_overstay_frames" in item
                    for item in type_reports
                ):
                    spatial = sum(
                        item.get("num_spatial_offroad_frames", 0)
                        for item in type_reports
                    )
                    overstay = sum(
                        item.get("num_transition_overstay_frames", 0)
                        for item in type_reports
                    )
                    by_type[type_name].update(
                        {
                            "num_spatial_offroad_frames": spatial,
                            "num_transition_overstay_frames": overstay,
                            "spatial_offroad_rate": spatial / sim_samples,
                            "transition_overstay_rate": overstay / sim_samples,
                            "excluded_map_unsupported_rate": gt_rate,
                        }
                    )
        safety_components[component_name] = {
            "score": _mean_present(
                [
                    value["score"] if value is not None else None
                    for value in by_type.values()
                ]
            ),
            "by_type": by_type,
        }
    safety = {
        "score": 0.5
        * (
            safety_components["collision_rollout_rate"]["score"]
            + safety_components["valid_region_margin"]["score"]
        ),
        **safety_components,
        "aggregation": "dataset_sample_weighted_then_agent_type_macro_average",
    }
    pair_type_names = (
        "vehicle_pedestrian",
        "vehicle_two_wheeler",
        "pedestrian_two_wheeler",
    )
    pair_type_scores = {}
    for pair_type_name in pair_type_names:
        distance_values = []
        time_values = []
        for report in scenario_reports:
            pair_report = report["cross_type_interaction"]["pair_type_scores"].get(
                pair_type_name
            )
            if pair_report is None:
                continue
            weight = pair_report["num_pairs"]
            distance_values.append((pair_report["distance_proximity_to_gt"], weight))
            time_values.append((pair_report["time_to_proximity_to_gt"], weight))
        distance = _weighted_mean(distance_values)
        time = _weighted_mean(time_values)
        if distance is None or time is None:
            continue
        pair_type_scores[pair_type_name] = {
            "score": 0.5 * (distance + time),
            "distance_proximity_to_gt": distance,
            "time_to_proximity_to_gt": time,
            "num_pairs": sum(weight for _, weight in distance_values),
        }
    cross_type = {
        "score": _mean_present_or_none(
            [value["score"] for value in pair_type_scores.values()]
        ),
        "distance_proximity_to_gt": _mean_present_or_none(
            [value["distance_proximity_to_gt"] for value in pair_type_scores.values()]
        ),
        "time_to_proximity_to_gt": _mean_present_or_none(
            [value["time_to_proximity_to_gt"] for value in pair_type_scores.values()]
        ),
        "applicable": bool(pair_type_scores),
        "pair_type_scores": pair_type_scores,
        "aggregation": "dataset_pair_weighted_then_pair_type_macro_average",
    }

    coverage_by_type = {}
    for type_name in type_names:
        values = []
        for report in scenario_reports:
            type_report = report["coverage"]["by_type"].get(type_name)
            if type_report is not None:
                values.append((type_report["score"], type_report["num_agents"]))
        score = _weighted_mean(values)
        coverage_by_type[type_name] = (
            None
            if score is None
            else {
                "score": score,
                "num_agents": sum(weight for _, weight in values),
            }
        )
    coverage = {
        "score": _mean_present(
            [
                value["score"] if value is not None else None
                for value in coverage_by_type.values()
            ]
        ),
        "by_type": coverage_by_type,
        "aggregation": "dataset_agent_weighted_then_agent_type_macro_average",
    }

    overall = compute_overall_score(
        kinematic,
        safety,
        cross_type,
        coverage,
        config,
    )
    return {
        "metric_version": METRIC_VERSION,
        "score": overall["score"],
        "num_scenarios": len(scenario_reports),
        "kinematic_realism": kinematic,
        "safety": safety,
        "cross_type_interaction": cross_type,
        "coverage": coverage,
        "weighted_components": overall["weighted_components"],
        "overall_formula": overall["formula"],
        "coverage_bonus_gate": overall["coverage_bonus_gate"],
        "effective_quality_weights": overall["effective_quality_weights"],
        "metadata": {
            "aggregation": "within_location_sufficient_statistics",
            "config": asdict(config),
        },
    }


def _location_from_scenario_id(scenario_id: Any) -> str:
    match = re.search(r"(?:^|_)loc(\d+)(?:_|$)", str(scenario_id))
    return f"loc{match.group(1)}" if match else "unknown"


def aggregate_scenario_reports(
    scenario_reports: list[dict[str, Any]],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Macro-average location-level scores after balanced within-location aggregation."""
    usable = [
        report
        for report in scenario_reports
        if report.get("status") != "skipped_no_selected_agents"
    ]
    if not usable:
        raise ValueError("At least one scenario report is required.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in usable:
        location = _location_from_scenario_id(report.get("scenario_id"))
        grouped.setdefault(location, []).append(report)
    by_location = {
        location: _aggregate_scenario_reports_within_location(reports, config)
        for location, reports in sorted(grouped.items())
    }

    # Use the pooled report as a detailed sufficient-statistics template, then
    # replace every score that contributes to the leaderboard with an equal
    # macro-average over locations.
    result = _aggregate_scenario_reports_within_location(usable, config)
    location_reports = list(by_location.values())
    result["score"] = _mean_present([item["score"] for item in location_reports])
    for key in ("kinematic_realism", "safety", "cross_type_interaction", "coverage"):
        scores = [item[key]["score"] for item in location_reports]
        result[key]["score"] = (
            _mean_present_or_none(scores)
            if key == "cross_type_interaction"
            else _mean_present(scores)
        )
        result[key]["aggregation"] = (
            "location_macro_average; pooled_by_type_details_are_diagnostic_only"
        )
    for component in ("collision_rollout_rate", "valid_region_margin"):
        result["safety"][component]["score"] = _mean_present(
            [item["safety"][component]["score"] for item in location_reports]
        )
    result["weighted_components"] = {
        name: _mean_present(
            [item["weighted_components"][name] for item in location_reports]
        )
        for name in (
            "kinematic_realism",
            "safety",
            "cross_type_interaction",
            "coverage_bonus",
        )
    }
    result["coverage_bonus_gate"] = {
        "aggregation": "location_macro_average_of_gated_bonus",
        "by_location": {
            location: item["coverage_bonus_gate"]
            for location, item in by_location.items()
        },
    }
    result["effective_quality_weights"] = {
        "aggregation": "reported_by_location",
        "by_location": {
            location: item["effective_quality_weights"]
            for location, item in by_location.items()
        },
    }
    result["num_scenarios"] = len(usable)
    result["num_locations"] = len(by_location)
    result["by_location"] = by_location
    result["metadata"]["aggregation"] = (
        "location_macro_average_after_within_location_type_balancing"
    )
    return result
