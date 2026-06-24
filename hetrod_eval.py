from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hetrod_metrics.report import (
    NoSelectedAgentsError,
    aggregate_scenario_reports,
    evaluate_scenario,
    skipped_no_selected_agents_report,
)
from wosac_eval import (
    infer_scenario_id_from_name,
    load_eval_config,
    load_pickle,
    normalize_prediction,
)
from wosac_fast_eval_tool.scenario_gt_converter import gt_scenario_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HetroD Challenge rollout pickles."
    )
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--version", choices=("2024", "2025"), default="2025")
    parser.add_argument("--rollout-key", default="joint_future")
    return parser.parse_args()


def find_gt_path(gt_dir: Path, scenario_id: str) -> Path | None:
    candidates = (
        gt_dir / f"{scenario_id}.pkl",
        gt_dir / f"scenario_{scenario_id}.pkl",
    )
    return next((path for path in candidates if path.is_file()), None)


def resolve_rollout_files(rollout_dir: Path, gt_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[dict[str, str]]]:
    rollout_paths = sorted(rollout_dir.glob("*.pkl"))
    if not rollout_paths:
        raise FileNotFoundError(f"No rollout pickle files found in {rollout_dir}.")

    rollout_map: dict[str, Path] = {}
    errors = []
    for rollout_path in rollout_paths:
        scenario_id = infer_scenario_id_from_name(rollout_path.name)
        if scenario_id in rollout_map:
            errors.append(
                {
                    "scenario_id": scenario_id,
                    "rollout_file": str(rollout_path),
                    "error": f"Duplicate rollout file for scenario_id; first file is {rollout_map[scenario_id]}",
                }
            )
            continue
        rollout_map[scenario_id] = rollout_path

    gt_paths = sorted(gt_dir.glob("*.pkl"))
    if not gt_paths:
        raise FileNotFoundError(f"No GT pickle files found in {gt_dir}.")
    gt_map = {infer_scenario_id_from_name(path.name): path for path in gt_paths}

    matched = []
    for scenario_id, gt_path in gt_map.items():
        rollout_path = rollout_map.get(scenario_id)
        if rollout_path is None:
            errors.append(
                {
                    "scenario_id": scenario_id,
                    "gt_file": str(gt_path),
                    "error": "Missing rollout pickle.",
                }
            )
            continue
        matched.append((scenario_id, rollout_path, gt_path))

    for scenario_id, rollout_path in rollout_map.items():
        if scenario_id not in gt_map:
            errors.append(
                {
                    "scenario_id": scenario_id,
                    "rollout_file": str(rollout_path),
                    "error": "Missing GT pickle.",
                }
            )

    return matched, errors


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def to_jsonable_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: to_jsonable_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable_cpu(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable_cpu(item) for item in value]
    return value


def evaluate_directory(
    rollout_dir: Path,
    gt_dir: Path,
    *,
    device: str,
    version: str,
    rollout_key: str,
) -> dict[str, Any]:
    eval_config = load_eval_config(version)
    scenario_reports = []
    skipped = []
    matched_files, errors = resolve_rollout_files(rollout_dir, gt_dir)

    device = resolve_device(device)
    for scenario_id, rollout_path, gt_path in matched_files:
        try:
            gt_scenario = gt_scenario_to_device(load_pickle(gt_path), device=device)
            prediction = normalize_prediction(
                load_pickle(rollout_path),
                device=device,
                rollout_key=rollout_key,
            )
            scenario_reports.append(
                to_jsonable_cpu(evaluate_scenario(eval_config, gt_scenario, prediction))
            )
        except NoSelectedAgentsError:
            skipped.append(
                to_jsonable_cpu(skipped_no_selected_agents_report(gt_scenario))
            )
        except Exception as error:
            errors.append(
                {
                    "scenario_id": scenario_id,
                    "rollout_file": str(rollout_path),
                    "gt_file": str(gt_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    dataset_report = (
        aggregate_scenario_reports(scenario_reports)
        if scenario_reports
        else None
    )
    return {
        "dataset": dataset_report,
        "scenarios": scenario_reports,
        "skipped_scenarios": skipped,
        "errors": errors,
        "summary": {
            "num_rollout_files": len(list(rollout_dir.glob("*.pkl"))),
            "num_gt_files": len(list(gt_dir.glob("*.pkl"))),
            "num_matched_files": len(matched_files),
            "num_successful_scenarios": len(scenario_reports),
            "num_skipped_no_selected_agents": len(skipped),
            "num_errors": len(errors),
            "device": device,
        },
    }


def main() -> int:
    args = parse_args()
    report = evaluate_directory(
        args.rollout_dir,
        args.gt_dir,
        device=args.device,
        version=args.version,
        rollout_key=args.rollout_key,
    )
    output_path = args.output or args.rollout_dir / "hetrod_metrics_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {output_path}")
    if report["errors"]:
        print(f"Evaluation failed for {len(report['errors'])} scenario(s).")
        return 1
    if report["dataset"] is None:
        print("No evaluable scenarios were found.")
        return 1
    print(f"HetroD score: {report['dataset']['score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
