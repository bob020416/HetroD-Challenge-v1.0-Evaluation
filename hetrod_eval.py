from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hetrod_metrics.report import (
    NoSelectedAgentsError,
    aggregate_scenario_reports,
    evaluate_scenario,
    skipped_no_selected_agents_report,
)
from hetrod_metrics.selection_manifest import load_selection_manifest
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
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Organizer-only private target/exclusion manifest.",
    )
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N assigned scenarios; 0 disables it.",
    )
    return parser.parse_args()


def find_gt_path(gt_dir: Path, scenario_id: str) -> Path | None:
    candidates = (
        gt_dir / f"{scenario_id}.pkl",
        gt_dir / f"scenario_{scenario_id}.pkl",
    )
    return next((path for path in candidates if path.is_file()), None)


def resolve_rollout_files(rollout_dir: Path, gt_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[dict[str, str]]]:
    rollout_paths = sorted(
        path
        for path in rollout_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".npz", ".pkl"}
    )
    if not rollout_paths:
        raise FileNotFoundError(
            f"No .npz or .pkl rollout files found in {rollout_dir}."
        )

    rollout_map: dict[str, Path] = {}
    errors = []
    for rollout_path in rollout_paths:
        scenario_id = infer_scenario_id_from_name(rollout_path.name)
        if scenario_id in rollout_map:
            errors.append(
                {
                    "scenario_id": scenario_id,
                    "rollout_file": rollout_path.name,
                    "error": (
                        "Duplicate rollout file for scenario_id; first file is "
                        f"{rollout_map[scenario_id].name}"
                    ),
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
                    "gt_file": gt_path.name,
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
                    "rollout_file": rollout_path.name,
                    "error": "Missing GT pickle.",
                }
            )

    return matched, errors


def load_rollout(path: Path) -> Any:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    return load_pickle(path)


def rollout_file_count(rollout_dir: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in {".npz", ".pkl"}
        for path in rollout_dir.iterdir()
    )


def select_shard(
    items: list[tuple[str, Path, Path]],
    shard_id: int | None,
    num_shards: int | None,
) -> list[tuple[str, Path, Path]]:
    if shard_id is None and num_shards is None:
        return items
    if shard_id is None or num_shards is None:
        raise ValueError("shard_id and num_shards must be provided together.")
    if num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must be in [0, num_shards).")
    return [
        item
        for index, item in enumerate(items)
        if index % num_shards == shard_id
    ]


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
    selection_manifest: dict[str, Any] | None = None,
    selection_manifest_sha256: str | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative.")
    eval_config = load_eval_config(version)
    scenario_reports = []
    skipped = []
    excluded = []
    all_matched_files, resolution_errors = resolve_rollout_files(
        rollout_dir, gt_dir
    )
    matched_files = select_shard(all_matched_files, shard_id, num_shards)
    errors = (
        resolution_errors
        if shard_id is None or shard_id == 0
        else []
    )

    manifest_scenarios = None
    if selection_manifest is not None:
        manifest_scenarios = selection_manifest["scenarios"]
        gt_ids = {
            infer_scenario_id_from_name(path.name)
            for path in gt_dir.glob("*.pkl")
        }
        manifest_ids = set(manifest_scenarios)
        missing = sorted(gt_ids - manifest_ids)
        extra = sorted(manifest_ids - gt_ids)
        if (missing or extra) and (shard_id is None or shard_id == 0):
            errors.append(
                {
                    "error": "Selection manifest scenario coverage mismatch.",
                    "missing_scenario_ids": missing,
                    "extra_scenario_ids": extra,
                }
            )

    device = resolve_device(device)
    for position, (scenario_id, rollout_path, gt_path) in enumerate(
        matched_files, start=1
    ):
        selection_record = (
            manifest_scenarios.get(scenario_id)
            if manifest_scenarios is not None
            else None
        )
        if selection_record is None and manifest_scenarios is not None:
            continue
        if selection_record is not None and selection_record["status"] == "exclude":
            excluded.append(
                {
                    "scenario_id": scenario_id,
                    "status": "excluded_by_selection_manifest",
                    "reason": selection_record["reason"],
                }
            )
            continue
        try:
            gt_scenario = gt_scenario_to_device(load_pickle(gt_path), device=device)
            prediction = normalize_prediction(
                load_rollout(rollout_path),
                device=device,
                rollout_key=rollout_key,
                apply_sim_agent_mask=False,
            )
            scenario_reports.append(
                to_jsonable_cpu(
                    evaluate_scenario(
                        eval_config,
                        gt_scenario,
                        prediction,
                        selection_record=selection_record,
                    )
                )
            )
        except NoSelectedAgentsError:
            skipped.append(
                to_jsonable_cpu(skipped_no_selected_agents_report(gt_scenario))
            )
        except Exception as error:
            errors.append(
                {
                    "scenario_id": scenario_id,
                    "rollout_file": rollout_path.name,
                    "gt_file": gt_path.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if progress_every and (
            position % progress_every == 0 or position == len(matched_files)
        ):
            print(
                f"progress={position}/{len(matched_files)} "
                f"success={len(scenario_reports)} excluded={len(excluded)} "
                f"skipped={len(skipped)} errors={len(errors)}",
                flush=True,
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
        "excluded_scenarios": excluded,
        "errors": errors,
        "summary": {
            "num_rollout_files": rollout_file_count(rollout_dir),
            "num_gt_files": len(list(gt_dir.glob("*.pkl"))),
            "num_matched_files": len(matched_files),
            "num_global_matched_files": len(all_matched_files),
            "num_assigned_scenarios": len(matched_files),
            "num_successful_scenarios": len(scenario_reports),
            "num_skipped_no_selected_agents": len(skipped),
            "num_excluded_by_manifest": len(excluded),
            "num_errors": len(errors),
            "device": device,
            "selection_manifest_sha256": selection_manifest_sha256,
            "shard_id": shard_id,
            "num_shards": num_shards,
        },
    }


def main() -> int:
    args = parse_args()
    selection_manifest = None
    selection_manifest_sha256 = None
    if args.selection_manifest is not None:
        selection_manifest, selection_manifest_sha256 = load_selection_manifest(
            args.selection_manifest
        )
    report = evaluate_directory(
        args.rollout_dir,
        args.gt_dir,
        device=args.device,
        version=args.version,
        rollout_key=args.rollout_key,
        selection_manifest=selection_manifest,
        selection_manifest_sha256=selection_manifest_sha256,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        progress_every=args.progress_every,
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
