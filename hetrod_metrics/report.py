"""Report aggregation helpers for HetroD Challenge metrics."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import torch
from waymo_open_dataset.protos import sim_agents_metrics_pb2

from .agent_selection import select_agents, selection_audit
from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .coverage import compute_coverage
from .cross_type import compute_cross_type_interaction
from .features import build_feature_bundle
from .kinematic import compute_kinematic_realism, normalize_kinematic_realism
from .safety import compute_safety

METRIC_VERSION = "hetrod-0.1.0"


class NoSelectedAgentsError(ValueError):
    """Raised when a scenario has no agents matching the HetroD selection spec."""


def validate_config(config: HetrodMetricConfig) -> None:
    if config.future_start_index != config.current_time_index + 1:
        raise ValueError("future_start_index must equal current_time_index + 1.")
    if config.seconds_per_step <= 0.0:
        raise ValueError("seconds_per_step must be positive.")
    if config.coverage_grid_resolution_m <= 0.0:
        raise ValueError("coverage_grid_resolution_m must be positive.")
    if config.cross_type_distance_hist_bins <= 0 or config.cross_type_ttc_hist_bins <= 0:
        raise ValueError("Cross-type histogram bin counts must be positive.")
    if config.cross_type_distance_hist_max_m <= 0.0 or config.cross_type_ttc_hist_max_s <= 0.0:
        raise ValueError("Cross-type histogram ranges must be positive.")
    weight_sum = (
        config.kinematic_weight
        + config.safety_weight
        + config.cross_type_weight
        + config.coverage_weight
    )
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
    coverage_bonus = (
        config.coverage_weight
        * coverage["score"]
        * kinematic["score"]
        * safety["score"]
    )
    weighted_components = {
        "kinematic_realism": config.kinematic_weight * kinematic["score"],
        "safety": config.safety_weight * safety["score"],
        "cross_type_interaction": config.cross_type_weight * cross_type["score"],
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
        "formula": (
            "0.30*kinematic + 0.35*safety + 0.25*cross_type "
            "+ 0.10*coverage*kinematic*safety"
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


def evaluate_scenario(
    eval_config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    gt_scenario: dict[str, Any],
    prediction: dict[str, torch.Tensor],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Evaluate one scenario with all HetroD Challenge metric components."""
    validate_config(config)
    selected_mask = select_agents(gt_scenario, config)
    if not selected_mask.any():
        raise NoSelectedAgentsError(
            "Scenario contains no agents selected by the HetroD filters."
        )

    features = build_feature_bundle(gt_scenario, prediction, selected_mask, config)
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
        "selection_audit": selection_audit(gt_scenario, config),
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
        "metadata": {
            "valid_region_source": "road_edge_margin_fallback",
            "collision_definition": "box_overlap_depth_exceeds_annotation_tolerance",
            "interaction_definition": "constant_velocity_time_to_5m_proximity",
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
        "selection_audit": selection_audit(gt_scenario, config),
        "metadata": {
            "skip_reason": "No agents matched the HetroD selection filters.",
            "config": asdict(config),
        },
    }


def aggregate_scenario_reports(
    scenario_reports: list[dict[str, Any]],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Aggregate sufficient statistics across scenarios before macro averaging."""
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
    for component_name in (
        "collision_with_annotation_tolerance",
        "valid_region_margin",
    ):
        by_type = {}
        for type_name in type_names:
            values = []
            for report in scenario_reports:
                type_report = report["safety"][component_name]["by_type"].get(type_name)
                if type_report is not None:
                    values.append((type_report["score"], type_report["num_samples"]))
            score = _weighted_mean(values)
            by_type[type_name] = None if score is None else {"score": score}
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
            safety_components["collision_with_annotation_tolerance"]["score"]
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
        ttp_values = []
        for report in scenario_reports:
            pair_report = report["cross_type_interaction"]["pair_type_scores"].get(
                pair_type_name
            )
            if pair_report is None:
                continue
            weight = pair_report["num_pairs"]
            distance_values.append((pair_report["distance_proximity_to_gt"], weight))
            ttp_values.append((pair_report["time_to_proximity_to_gt"], weight))
        distance = _weighted_mean(distance_values)
        ttp = _weighted_mean(ttp_values)
        if distance is None or ttp is None:
            continue
        pair_type_scores[pair_type_name] = {
            "score": 0.5 * (distance + ttp),
            "distance_proximity_to_gt": distance,
            "time_to_proximity_to_gt": ttp,
            "num_pairs": sum(weight for _, weight in distance_values),
        }
    cross_type = {
        "score": _mean_present(
            [value["score"] for value in pair_type_scores.values()]
        ),
        "distance_proximity_to_gt": _mean_present(
            [value["distance_proximity_to_gt"] for value in pair_type_scores.values()]
        ),
        "time_to_proximity_to_gt": _mean_present(
            [value["time_to_proximity_to_gt"] for value in pair_type_scores.values()]
        ),
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
        "metadata": {
            "aggregation": "dataset_level_sufficient_statistics",
            "config": asdict(config),
        },
    }
