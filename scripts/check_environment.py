#!/usr/bin/env python3
"""Fail fast when the HetroD evaluation environment is incompatible."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from hetrod_metrics import __version__
from wosac_eval import load_eval_config


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    load_eval_config("2025")
    waymo = package_version("waymo-open-dataset-tf-2-12-0")
    if waymo != "1.6.7":
        raise RuntimeError(
            "Expected waymo-open-dataset-tf-2-12-0==1.6.7, "
            f"found {waymo}."
        )
    print(f"HetroD metric: hetrod-{__version__}")
    print(f"Waymo Open Dataset: {waymo}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("ENVIRONMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
