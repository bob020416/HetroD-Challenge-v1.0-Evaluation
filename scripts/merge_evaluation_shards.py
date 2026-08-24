#!/usr/bin/env python3
"""Merge complete HetroD evaluation shards into one official score report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hetrod_metrics.report import aggregate_scenario_reports


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one shard report is required.")
    expected_num_shards = len(reports)
    scenarios: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    assigned_total = 0
    shared: dict[str, Any] | None = None
    seen_shards = set()

    for report in reports:
        summary = report["summary"]
        shard_id = summary.get("shard_id")
        num_shards = summary.get("num_shards")
        if num_shards != expected_num_shards:
            raise ValueError(
                f"Shard {shard_id} reports num_shards={num_shards}, "
                f"expected {expected_num_shards}."
            )
        if not isinstance(shard_id, int) or not 0 <= shard_id < num_shards:
            raise ValueError(f"Invalid shard ID: {shard_id!r}.")
        if shard_id in seen_shards:
            raise ValueError(f"Duplicate shard report: {shard_id}.")
        seen_shards.add(shard_id)

        current_shared = {
            "num_rollout_files": summary["num_rollout_files"],
            "num_gt_files": summary["num_gt_files"],
            "num_global_matched_files": summary["num_global_matched_files"],
            "device": summary["device"],
            "selection_manifest_sha256": summary[
                "selection_manifest_sha256"
            ],
        }
        if shared is None:
            shared = current_shared
        elif current_shared != shared:
            raise ValueError("Shard reports disagree on frozen evaluation inputs.")

        assigned = int(summary["num_assigned_scenarios"])
        completed = (
            int(summary["num_successful_scenarios"])
            + int(summary["num_skipped_no_selected_agents"])
            + int(summary["num_excluded_by_manifest"])
            + int(summary["num_errors"])
        )
        if assigned != completed:
            raise ValueError(
                f"Shard {shard_id} is incomplete: assigned={assigned}, "
                f"completed={completed}."
            )
        assigned_total += assigned
        scenarios.extend(report["scenarios"])
        skipped.extend(report["skipped_scenarios"])
        excluded.extend(report["excluded_scenarios"])
        errors.extend(report["errors"])

    if seen_shards != set(range(expected_num_shards)):
        raise ValueError("Shard ID coverage is incomplete.")
    assert shared is not None
    if assigned_total != shared["num_global_matched_files"]:
        raise ValueError(
            "Merged shard assignment does not cover every matched scenario."
        )

    scenario_ids = [
        str(item["scenario_id"])
        for collection in (scenarios, skipped, excluded)
        for item in collection
    ]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("A scenario appears in more than one shard outcome.")
    scenarios.sort(key=lambda item: str(item["scenario_id"]))
    skipped.sort(key=lambda item: str(item["scenario_id"]))
    excluded.sort(key=lambda item: str(item["scenario_id"]))

    return {
        "dataset": aggregate_scenario_reports(scenarios) if scenarios else None,
        "scenarios": scenarios,
        "skipped_scenarios": skipped,
        "excluded_scenarios": excluded,
        "errors": errors,
        "summary": {
            **shared,
            "num_matched_files": assigned_total,
            "num_assigned_scenarios": assigned_total,
            "num_successful_scenarios": len(scenarios),
            "num_skipped_no_selected_agents": len(skipped),
            "num_excluded_by_manifest": len(excluded),
            "num_errors": len(errors),
            "num_shards": expected_num_shards,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for shard_id in range(args.num_shards):
        path = args.shard_dir / f"shard_{shard_id:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard report: {path}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))[
        "preflight"
    ]
    evaluation = merge_reports(reports)
    final = {"preflight": preflight, "evaluation": evaluation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    if evaluation["dataset"] is not None:
        print(f"HetroD score: {evaluation['dataset']['score']:.6f}")
    return 0 if preflight["status"] == "ok" and not evaluation["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
