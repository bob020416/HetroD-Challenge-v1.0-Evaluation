#!/usr/bin/env python3
"""Organizer one-command preflight and private-test scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hetrod_eval import evaluate_directory
from hetrod_metrics.selection_manifest import load_selection_manifest
from hetrod_metrics.submission import (
    preflight_submission,
    resolve_submission_entries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-gt-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--version", choices=("2024", "2025"), default="2025")
    parser.add_argument("--rollout-key", default="joint_future")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow executable legacy .pkl input. Use only in a hardened sandbox.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Temporary extraction parent, preferably node-local scratch.",
    )
    return parser.parse_args()


def extract_submission_zip(submission: Path, destination: Path) -> Path:
    entries, errors = resolve_submission_entries(submission)
    if errors:
        raise ValueError(f"Cannot extract invalid submission: {errors[:5]}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(submission) as archive:
        for scenario_id, entry in entries.items():
            if entry.zip_member is None:
                raise ValueError("ZIP submission entry has no archive member.")
            with archive.open(entry.zip_member) as source:
                with (destination / f"{scenario_id}{entry.suffix}").open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    return destination


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    entries, resolution_errors = resolve_submission_entries(args.submission)
    unsafe_pickle_count = sum(entry.suffix == ".pkl" for entry in entries.values())
    if resolution_errors:
        write_report(
            args.output,
            {"preflight": {"status": "failed", "errors": resolution_errors}, "evaluation": None},
        )
        return 1
    if unsafe_pickle_count and not args.allow_unsafe_pickle:
        write_report(
            args.output,
            {
                "preflight": {
                    "status": "failed",
                    "errors": [
                        {
                            "error": (
                                f"Submission contains {unsafe_pickle_count} executable .pkl "
                                "files. Convert to .npz or explicitly use "
                                "--allow-unsafe-pickle inside a hardened sandbox."
                            )
                        }
                    ],
                },
                "evaluation": None,
            },
        )
        return 1
    preflight = preflight_submission(args.submission, args.public_root)
    if preflight["status"] != "ok" or args.preflight_only:
        write_report(args.output, {"preflight": preflight, "evaluation": None})
        return 0 if preflight["status"] == "ok" else 1

    selection_manifest, selection_sha256 = load_selection_manifest(
        args.selection_manifest
    )
    if args.submission.is_dir():
        rollout_dir = args.submission
        evaluation = evaluate_directory(
            rollout_dir,
            args.private_gt_dir,
            device=args.device,
            version=args.version,
            rollout_key=args.rollout_key,
            selection_manifest=selection_manifest,
            selection_manifest_sha256=selection_sha256,
        )
    else:
        temp_parent = str(args.work_dir) if args.work_dir is not None else None
        with tempfile.TemporaryDirectory(
            prefix="hetrod_submission_",
            dir=temp_parent,
        ) as directory:
            rollout_dir = extract_submission_zip(
                args.submission,
                Path(directory) / "rollouts",
            )
            evaluation = evaluate_directory(
                rollout_dir,
                args.private_gt_dir,
                device=args.device,
                version=args.version,
                rollout_key=args.rollout_key,
                selection_manifest=selection_manifest,
                selection_manifest_sha256=selection_sha256,
            )
    write_report(args.output, {"preflight": preflight, "evaluation": evaluation})
    if evaluation["errors"] or evaluation["dataset"] is None:
        return 1
    print(f"HetroD score: {evaluation['dataset']['score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
