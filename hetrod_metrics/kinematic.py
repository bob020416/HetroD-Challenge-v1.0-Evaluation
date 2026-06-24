from __future__ import annotations

import math
from typing import Any

import torch
from waymo_open_dataset.protos import sim_agents_metrics_pb2

from wosac_fast_eval_tool.fast_sim_agents_metrics import estimators

from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .features import HetrodFeatureBundle


KINEMATIC_METRICS = (
    "linear_speed",
    "linear_acceleration",
    "angular_speed",
    "angular_acceleration",
)

TYPE_NAMES = {
    DEFAULT_CONFIG.vehicle_type: "vehicle",
    DEFAULT_CONFIG.two_wheeler_type: "two_wheeler",
    DEFAULT_CONFIG.pedestrian_type: "pedestrian",
}


def _valid_mean_exp(log_likelihood: torch.Tensor, validity: torch.Tensor) -> float | None:
    if log_likelihood.shape != validity.shape:
        raise ValueError(f"Shape mismatch: {log_likelihood.shape} vs {validity.shape}")
    if not validity.any():
        return None
    value = torch.exp(torch.mean(log_likelihood[validity]))
    if torch.isnan(value):
        return None
    return float(value.item())


def _mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and not math.isnan(value)]
    if not present:
        return None
    return float(sum(present) / len(present))


def _normalize_score(raw: float | None, oracle: float | None) -> float | None:
    if raw is None or oracle is None or oracle <= 1e-9:
        return None
    return float(min(raw / oracle, 1.0))


def _feature_config(
    eval_config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    metric_name: str,
) -> sim_agents_metrics_pb2.SimAgentMetricsConfig.FeatureConfig:
    return getattr(eval_config, metric_name)


def compute_kinematic_realism(
    eval_config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    features: HetrodFeatureBundle,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Compute HetroD type-balanced kinematic realism scores."""
    metric_scores: dict[str, Any] = {}
    for metric_name in KINEMATIC_METRICS:
        log_values = features.log_kinematics[metric_name]
        sim_values = features.sim_kinematics[metric_name]
        log_likelihood = estimators.log_likelihood_estimate_timeseries(
            feature_config=_feature_config(eval_config, metric_name),
            log_values=log_values,
            sim_values=sim_values,
        )
        validity = (
            features.acceleration_validity
            if "acceleration" in metric_name
            else features.speed_validity
        )

        per_type: dict[str, float | None] = {}
        num_agents_by_type: dict[str, int] = {}
        num_samples_by_type: dict[str, int] = {}
        for object_type in config.evaluated_object_types:
            type_mask = features.object_types == object_type
            type_name = TYPE_NAMES.get(object_type, f"type_{object_type}")
            num_agents_by_type[type_name] = int(type_mask.sum().item())
            if not type_mask.any():
                per_type[type_name] = None
                num_samples_by_type[type_name] = 0
                continue
            num_samples_by_type[type_name] = int(validity[type_mask].sum().item())
            per_type[type_name] = _valid_mean_exp(
                log_likelihood[type_mask],
                validity[type_mask],
            )
        metric_scores[metric_name] = {
            "score": _mean_present(list(per_type.values())),
            "by_type": per_type,
            "num_agents_by_type": num_agents_by_type,
            "num_samples_by_type": num_samples_by_type,
        }

    return {
        "score": _mean_present([metric_scores[name]["score"] for name in KINEMATIC_METRICS]),
        "metrics": metric_scores,
    }


def normalize_kinematic_realism(
    raw_report: dict[str, Any],
    oracle_report: dict[str, Any],
) -> dict[str, Any]:
    """Normalize raw WOSAC-style kinematic scores by the GT-as-rollout ceiling."""
    normalized_metrics: dict[str, Any] = {}
    for metric_name in KINEMATIC_METRICS:
        raw_metric = raw_report["metrics"][metric_name]
        oracle_metric = oracle_report["metrics"][metric_name]
        by_type = {
            type_name: _normalize_score(type_score, oracle_metric["by_type"].get(type_name))
            for type_name, type_score in raw_metric["by_type"].items()
        }
        log_ratio_by_type = {}
        for type_name, normalized_score in by_type.items():
            raw_type_score = raw_metric["by_type"].get(type_name)
            oracle_type_score = oracle_metric["by_type"].get(type_name)
            if (
                normalized_score is None
                or raw_type_score is None
                or oracle_type_score is None
                or raw_type_score <= 0.0
                or oracle_type_score <= 0.0
            ):
                log_ratio_by_type[type_name] = None
            else:
                log_ratio_by_type[type_name] = min(
                    math.log(raw_type_score) - math.log(oracle_type_score),
                    0.0,
                )
        normalized_metrics[metric_name] = {
            "score": _mean_present(list(by_type.values())),
            "raw_score": raw_metric["score"],
            "oracle_score": oracle_metric["score"],
            "by_type": by_type,
            "log_ratio_by_type": log_ratio_by_type,
            "raw_by_type": raw_metric["by_type"],
            "oracle_by_type": oracle_metric["by_type"],
            "num_agents_by_type": raw_metric["num_agents_by_type"],
            "num_samples_by_type": raw_metric["num_samples_by_type"],
        }

    return {
        "score": _mean_present([normalized_metrics[name]["score"] for name in KINEMATIC_METRICS]),
        "raw_score": raw_report["score"],
        "oracle_score": oracle_report["score"],
        "metrics": normalized_metrics,
    }
