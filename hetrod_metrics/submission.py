"""Public submission preflight for HetroD rollout archives/directories."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from collections import Counter
import pickle
from pathlib import Path, PurePosixPath
from typing import Any
import zipfile

import numpy as np
import torch

from .config import DEFAULT_CONFIG, HetrodMetricConfig


@dataclass(frozen=True)
class SubmissionEntry:
    scenario_id: str
    display_name: str
    path: Path | None = None
    zip_member: str | None = None
    suffix: str = ".npz"
    size_bytes: int = 0


MAX_SUBMISSION_FILE_BYTES = 512 * 1024 * 1024


def _manifest_rows(path: Path) -> list[str]:
    rows = [row.strip() for row in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate rows in manifest: {path}")
    return rows


def load_public_requirements(public_root: Path) -> dict[str, list[int]]:
    """Load public test required-agent IDs in frozen manifest order."""
    scenario_ids = _manifest_rows(public_root / "manifests" / "test.txt")
    input_paths = _manifest_rows(
        public_root / "manifests" / "test_input_paths.txt"
    )
    if len(scenario_ids) != len(input_paths):
        raise ValueError("test.txt and test_input_paths.txt have different lengths.")
    requirements = {}
    for scenario_id, relative_path in zip(scenario_ids, input_paths):
        input_path = public_root / relative_path
        with input_path.open("rb") as handle:
            scenario = pickle.load(handle)
        if str(scenario.get("id")) != scenario_id:
            raise ValueError(f"{scenario_id}: public test input ID mismatch.")
        required = [
            int(value)
            for value in scenario["metadata"]["required_agent_ids"]
        ]
        if not required or len(required) != len(set(required)):
            raise ValueError(
                f"{scenario_id}: required_agent_ids must be non-empty and unique."
            )
        requirements[scenario_id] = required
    return requirements


def resolve_submission_entries(
    submission: Path,
) -> tuple[dict[str, SubmissionEntry], list[dict[str, str]]]:
    """Resolve one pickle per scenario from a directory or ZIP archive."""
    entries: dict[str, SubmissionEntry] = {}
    errors: list[dict[str, str]] = []
    candidates: list[SubmissionEntry] = []
    if submission.is_dir():
        for path in sorted(
            candidate
            for candidate in submission.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".npz", ".pkl"}
        ):
            candidates.append(
                SubmissionEntry(
                    scenario_id=path.stem,
                    display_name=str(path.relative_to(submission)),
                    path=path,
                    suffix=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                )
            )
    elif submission.is_file() and submission.suffix.lower() == ".zip":
        with zipfile.ZipFile(submission) as archive:
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                if info.is_dir() or member.name.startswith("."):
                    continue
                if ".." in member.parts or member.is_absolute():
                    errors.append(
                        {"file": info.filename, "error": "Unsafe ZIP member path."}
                    )
                    continue
                if member.suffix.lower() not in {".npz", ".pkl"}:
                    continue
                candidates.append(
                    SubmissionEntry(
                        scenario_id=member.stem,
                        display_name=info.filename,
                        zip_member=info.filename,
                        suffix=member.suffix.lower(),
                        size_bytes=info.file_size,
                    )
                )
    else:
        raise ValueError("Submission must be a directory or .zip archive.")

    for entry in candidates:
        if entry.size_bytes > MAX_SUBMISSION_FILE_BYTES:
            errors.append(
                {
                    "scenario_id": entry.scenario_id,
                    "file": entry.display_name,
                    "error": "Submission member exceeds the 512 MiB safety limit.",
                }
            )
            continue
        if entry.scenario_id in entries:
            errors.append(
                {
                    "scenario_id": entry.scenario_id,
                    "file": entry.display_name,
                    "error": (
                        "Duplicate scenario file; first is "
                        f"{entries[entry.scenario_id].display_name}."
                    ),
                }
            )
            continue
        entries[entry.scenario_id] = entry
    return entries, errors


def load_submission_entry(submission: Path, entry: SubmissionEntry) -> Any:
    """Load one trusted participant pickle.

    Pickle is executable input. Organizer evaluation must run this operation in
    an isolated, no-network worker without credentials or unrelated mounts.
    """
    if entry.path is not None and entry.suffix == ".npz":
        with np.load(entry.path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    if entry.path is not None:
        with entry.path.open("rb") as handle:
            return pickle.load(handle)
    if entry.zip_member is None:
        raise ValueError("Submission entry has no backing file.")
    with zipfile.ZipFile(submission) as archive:
        with archive.open(entry.zip_member) as handle:
            raw = handle.read()
    if entry.suffix == ".npz":
        with np.load(BytesIO(raw), allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    return pickle.loads(raw)


def validate_rollout_payload(
    scenario_id: str,
    payload: Any,
    required_agent_ids: list[int],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise ValueError("Rollout pickle must contain a dictionary.")
    missing_keys = {"agent_id", "simulated_states"} - payload.keys()
    if missing_keys:
        raise KeyError(f"Missing keys: {sorted(missing_keys)}")
    extra_keys = payload.keys() - {"agent_id", "simulated_states"}
    if extra_keys:
        raise KeyError(f"Unexpected keys: {sorted(extra_keys)}")
    agent_ids = torch.as_tensor(payload["agent_id"]).int().cpu()
    states = torch.as_tensor(payload["simulated_states"]).cpu()
    if agent_ids.ndim != 1:
        raise ValueError("agent_id must be one-dimensional.")
    if torch.unique(agent_ids).numel() != agent_ids.numel():
        raise ValueError("agent_id must be unique.")
    required = torch.tensor(required_agent_ids, dtype=torch.int32)
    missing = required[~torch.isin(required, agent_ids)].tolist()
    extra = agent_ids[~torch.isin(agent_ids, required)].tolist()
    if missing or extra:
        raise ValueError(
            f"agent_id must exactly match required_agent_ids; missing={missing}, "
            f"extra={extra}."
        )
    expected_shape = (
        config.required_num_rollouts,
        len(required_agent_ids),
        80,
        4,
    )
    if tuple(states.shape) != expected_shape:
        raise ValueError(
            f"simulated_states shape must be {expected_shape}, got {tuple(states.shape)}."
        )
    if not states.is_floating_point():
        raise ValueError("simulated_states must use a floating-point dtype.")
    if not torch.isfinite(states).all():
        raise ValueError("simulated_states contains NaN or Inf.")
    return {
        "num_agents": len(required_agent_ids),
        "num_rollouts": int(states.shape[0]),
        "num_future_steps": int(states.shape[2]),
    }


def preflight_submission(
    submission: Path,
    public_root: Path,
) -> dict[str, Any]:
    requirements = load_public_requirements(public_root)
    entries, errors = resolve_submission_entries(submission)
    required_ids = set(requirements)
    submitted_ids = set(entries)
    for scenario_id in sorted(required_ids - submitted_ids):
        errors.append({"scenario_id": scenario_id, "error": "Missing rollout file."})
    for scenario_id in sorted(submitted_ids - required_ids):
        errors.append(
            {
                "scenario_id": scenario_id,
                "file": entries[scenario_id].display_name,
                "error": "Unexpected scenario ID.",
            }
        )

    successful = 0
    total_agents = 0
    for scenario_id in sorted(required_ids & submitted_ids):
        entry = entries[scenario_id]
        try:
            stats = validate_rollout_payload(
                scenario_id,
                load_submission_entry(submission, entry),
                requirements[scenario_id],
            )
            successful += 1
            total_agents += stats["num_agents"]
        except Exception as error:
            errors.append(
                {
                    "scenario_id": scenario_id,
                    "file": entry.display_name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "status": "ok" if not errors else "failed",
        "summary": {
            "num_required_scenarios": len(required_ids),
            "num_submission_files": len(entries),
            "num_valid_scenarios": successful,
            "num_errors": len(errors),
            "num_required_agent_trajectories": total_agents,
            "file_format_counts": dict(
                sorted(Counter(entry.suffix for entry in entries.values()).items())
            ),
        },
        "errors": errors,
        "security": {
            "safe_official_format": ".npz with allow_pickle=False",
            "pickle_warning": (
                "Legacy .pkl can execute code and is for trusted/local use only. "
                "Organizer scoring rejects it by default."
            ),
        },
    }
