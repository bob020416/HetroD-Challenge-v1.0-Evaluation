#!/usr/bin/env python3
"""Validate the structure and privacy invariants of a public HetroD package."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch  # Import before installing the NumPy 1.x pickle compatibility aliases.

if not hasattr(np, "_core"):
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)


PREFIX = "sd_HetroD_1.0_"
EXPECTED_COUNTS = {"train": 5087, "valid": 955, "test": 955}
REQUIRED_REGIONS = {
    "vehicle",
    "cyclist",
    "pedestrian",
    "pedestrian_core",
    "pedestrian_crosswalk",
    "pedestrian_road",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _validate_definition(definition: dict[str, Any], label: str) -> None:
    if definition.get("schema_version") != "1.3":
        raise AssertionError(f"{label}: expected valid-region schema 1.3")
    if definition.get("policy") != "type-specific-v3-pedestrian-transition-aware":
        raise AssertionError(f"{label}: unexpected valid-region policy")


def _validate_regions(regions: dict[str, Any], label: str) -> None:
    if not REQUIRED_REGIONS.issubset(regions):
        raise AssertionError(f"{label}: incomplete valid-region layers")
    for name in REQUIRED_REGIONS:
        records = regions[name]
        if not isinstance(records, list):
            raise AssertionError(f"{label}: {name} is not a polygon list")
        for record in records:
            value = record["exterior"]
            exterior = (
                value.detach().cpu().numpy()
                if torch.is_tensor(value)
                else np.asarray(value)
            )
            if exterior.ndim != 2 or exterior.shape[0] < 4 or exterior.shape[1] < 2:
                raise AssertionError(f"{label}: malformed {name} polygon")
            if not np.isfinite(exterior).all():
                raise AssertionError(f"{label}: non-finite {name} polygon")


def _validate_scenario(task: tuple[str, bool]) -> str:
    raw_path, is_test = task
    scenario = _load(Path(raw_path))
    scenario_id = str(scenario["id"])
    metadata = scenario["metadata"]
    _validate_definition(metadata["hetrod_valid_region_definition"], scenario_id)
    _validate_regions(metadata["hetrod_valid_regions"], scenario_id)
    if is_test:
        if not scenario.get("public_test_input") or not metadata.get("future_states_removed"):
            raise AssertionError(f"{scenario_id}: public-test privacy markers missing")
        for track_id, track in scenario["tracks"].items():
            state = track["state"]
            if np.asarray(state["valid"], dtype=bool)[11:].any():
                raise AssertionError(f"{scenario_id}/{track_id}: future valid mask leaked")
            for key in ("position", "heading", "velocity", "length", "width", "height"):
                if np.any(np.asarray(state[key])[11:] != 0):
                    raise AssertionError(f"{scenario_id}/{track_id}: future {key} leaked")
    return scenario_id


def _validate_gt(raw_path: str) -> str:
    gt = _load(Path(raw_path))
    scenario_id = str(gt["scenario_id"])
    _validate_definition(gt["valid_region_definition"], scenario_id)
    _validate_regions(gt["valid_regions"], scenario_id)
    return scenario_id


def _manifest_ids(root: Path, split: str) -> list[str]:
    return [
        row.strip()
        for row in (root / "manifests" / f"{split}.txt").read_text().splitlines()
        if row.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.public_root.resolve()
    split_ids = {split: _manifest_ids(root, split) for split in ("train", "valid", "test")}
    counts = {split: len(ids) for split, ids in split_ids.items()}
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"Manifest counts changed: {counts}")
    all_ids = [scenario_id for ids in split_ids.values() for scenario_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Split manifests overlap or contain duplicate IDs")
    if (root / "test" / "gt").exists() or list((root / "test").rglob("*gt*.pkl")):
        raise AssertionError("Public test ground truth is present")

    scenario_tasks = [
        (str(root / split / "scenarionet" / f"{PREFIX}{scenario_id}.pkl"), False)
        for split in ("train", "valid")
        for scenario_id in split_ids[split]
    ] + [
        (str(root / "test" / "input" / f"{PREFIX}{scenario_id}.pkl"), True)
        for scenario_id in split_ids["test"]
    ]
    gt_paths = [
        str(root / split / "gt" / f"{scenario_id}.pkl")
        for split in ("train", "valid")
        for scenario_id in split_ids[split]
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        scenario_results = list(executor.map(_validate_scenario, scenario_tasks, chunksize=8))
        gt_results = list(executor.map(_validate_gt, gt_paths, chunksize=8))
    if set(scenario_results) != set(all_ids):
        raise AssertionError("ScenarioNet IDs do not match manifests")
    if set(gt_results) != set(split_ids["train"]) | set(split_ids["valid"]):
        raise AssertionError("GT IDs do not match manifests")
    print(
        f"PUBLIC_DATASET_OK scenarionet={len(scenario_results)} "
        f"gt={len(gt_results)} hidden_test={len(split_ids['test'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
