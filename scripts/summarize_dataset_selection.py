#!/usr/bin/env python3
"""Summarize the official v0.8 agent selection over a HetroD data release."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch  # Import before installing the NumPy 1.x pickle compatibility aliases.

# Allow direct execution from a source checkout without installing the package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hetrod_metrics import __version__
from hetrod_metrics.config import DEFAULT_CONFIG
from hetrod_metrics.interaction_selection import select_competition_agents

# NumPy 2 pickles use numpy._core while older evaluation environments expose
# the same modules as numpy.core. Install aliases only after importing runtime
# dependencies, because SciPy performs its own NumPy ABI checks at import time.
if not hasattr(np, "_core"):
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)

METRIC_VERSION = f"hetrod-{__version__}"


LOCATION_PATTERN = re.compile(r"(?:^|_)loc(\d+)(?:_|$)")


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _scenario_record(task: tuple[str, str]) -> dict[str, Any]:
    split, raw_path = task
    gt = _load(Path(raw_path))
    result = select_competition_agents(gt)
    scenario_id = str(gt["scenario_id"])
    location_match = LOCATION_PATTERN.search(scenario_id)
    if location_match is None:
        raise ValueError(f"Cannot parse location from scenario id: {scenario_id}")
    object_types = gt["object_types"]
    selected_types = object_types[result.anchor_mask].detach().cpu().tolist()
    type_names = {
        DEFAULT_CONFIG.vehicle_type: "vehicle",
        DEFAULT_CONFIG.pedestrian_type: "pedestrian",
        DEFAULT_CONFIG.two_wheeler_type: "two_wheeler",
    }
    pair_types = []
    behavior_categories = []
    for pair in result.pairs:
        pair_types.append("_".join(sorted(pair["agent_types"])))
        behavior_categories.append(str(pair["behavior_category"]))
    return {
        "split": split,
        "scenario_id": scenario_id,
        "location": location_match.group(1),
        "mode": result.mode if result.anchor_mask.any() else "no_eligible_agent",
        "num_selected_anchors": int(result.anchor_mask.sum().item()),
        "num_interaction_pairs": len(result.pairs),
        "selected_types": [type_names[int(value)] for value in selected_types],
        "pair_types": pair_types,
        "behavior_categories": behavior_categories,
    }


def _manifest_ids(root: Path, split: str) -> list[str]:
    return [
        row.strip()
        for row in (root / "manifests" / f"{split}.txt").read_text().splitlines()
        if row.strip()
    ]


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    selected = [record["num_selected_anchors"] for record in records]
    pairs = [record["num_interaction_pairs"] for record in records]
    return {
        "num_scenarios": count,
        "num_evaluable": sum(value > 0 for value in selected),
        "num_no_eligible_agent": sum(value == 0 for value in selected),
        "location_counts": dict(sorted(Counter(
            record["location"] for record in records
        ).items())),
        "selection_mode_counts": dict(sorted(Counter(
            record["mode"] for record in records
        ).items())),
        "selected_anchor_type_counts": dict(sorted(Counter(
            value for record in records for value in record["selected_types"]
        ).items())),
        "selected_pair_type_counts": dict(sorted(Counter(
            value for record in records for value in record["pair_types"]
        ).items())),
        "behavior_counts": dict(sorted(Counter(
            value
            for record in records
            for value in record["behavior_categories"]
        ).items())),
        "mean_selected_anchors": sum(selected) / count if count else 0.0,
        "max_selected_anchors": max(selected, default=0),
        "mean_selected_pairs": sum(pairs) / count if count else 0.0,
        "max_selected_pairs": max(pairs, default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument(
        "--private-test-gt",
        type=Path,
        help="Organizer-only test GT directory; omit to summarize train/valid.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.public_root.resolve()
    split_ids = {split: _manifest_ids(root, split) for split in ("train", "valid", "test")}
    tasks: list[tuple[str, str]] = []
    for split in ("train", "valid"):
        tasks.extend(
            (split, str(root / split / "gt" / f"{scenario_id}.pkl"))
            for scenario_id in split_ids[split]
        )
    if args.private_test_gt is not None:
        test_root = args.private_test_gt.resolve()
        tasks.extend(
            ("test", str(test_root / f"{scenario_id}.pkl"))
            for scenario_id in split_ids["test"]
        )

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(_scenario_record, tasks, chunksize=8))
    by_split = {
        split: _summary([record for record in records if record["split"] == split])
        for split in ("train", "valid", "test")
        if any(record["split"] == split for record in records)
    }
    output = {
        "metric_version": METRIC_VERSION,
        "split_assignment": {
            "status": "frozen_from_hetrod_v1",
            "train": len(split_ids["train"]),
            "valid": len(split_ids["valid"]),
            "test": len(split_ids["test"]),
        },
        "selection": {
            "max_pairs_per_scenario": 4,
            "max_pair_endpoints_per_scenario": 8,
            "fallback_min_agents": 2,
            "fallback_target_agents": 4,
        },
        "total": _summary(records),
        "splits": by_split,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"SELECTION_SUMMARY_COMPLETE scenarios={len(records)} "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
