#!/usr/bin/env python3
"""Preflight and safely normalize a participant ZIP for parallel scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hetrod_metrics.submission import extract_submission_zip, preflight_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    preflight = preflight_submission(args.submission, args.public_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"preflight": preflight}, indent=2) + "\n",
        encoding="utf-8",
    )
    if preflight["status"] != "ok":
        print(f"Preflight failed; see {args.report}")
        return 1
    extract_submission_zip(args.submission, args.output_dir)
    print(
        f"SUBMISSION_READY scenarios="
        f"{preflight['summary']['num_valid_scenarios']} directory={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
