#!/usr/bin/env python3
"""Organizer-only consistency check across public test input, GT, and targets."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from hetrod_metrics.selection_manifest import (
    load_selection_manifest,
    manifest_selection_for_scenario,
)
from hetrod_metrics.submission import load_public_requirements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-gt-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    args = parser.parse_args()

    requirements = load_public_requirements(args.public_root)
    manifest, manifest_sha256 = load_selection_manifest(args.selection_manifest)
    gt_paths = {
        path.stem: path for path in args.private_gt_dir.glob("*.pkl")
    }
    expected_ids = set(requirements)
    if set(gt_paths) != expected_ids:
        raise ValueError("Private GT scenario IDs do not match public test manifest.")
    if set(manifest["scenarios"]) != expected_ids:
        raise ValueError("Selection scenario IDs do not match public test manifest.")

    scored = 0
    excluded = 0
    selected_agents = 0
    for scenario_id in sorted(expected_ids):
        with gt_paths[scenario_id].open("rb") as handle:
            gt = pickle.load(handle)
        if str(gt.get("scenario_id")) != scenario_id:
            raise ValueError(f"{scenario_id}: private GT ID mismatch.")
        gt_ids = torch.as_tensor(gt["object_ids"]).int().cpu()
        required_ids = torch.tensor(requirements[scenario_id], dtype=torch.int32)
        if gt_ids.numel() != required_ids.numel() or not torch.isin(
            gt_ids, required_ids
        ).all():
            raise ValueError(
                f"{scenario_id}: private GT object_ids do not exactly match "
                "public required_agent_ids."
            )
        record = manifest["scenarios"][scenario_id]
        if record["status"] == "exclude":
            excluded += 1
            continue
        selection = manifest_selection_for_scenario(gt, record)
        scored += 1
        selected_agents += int(selection.result.anchor_mask.sum())

    print(
        "PRIVATE_TEST_BUNDLE_OK "
        f"scenarios={len(expected_ids)} scored={scored} excluded={excluded} "
        f"selected_agents={selected_agents} "
        f"selection_sha256={manifest_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
