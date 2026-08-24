"""Private selection-manifest support for organizer-side evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .interaction_candidates import AgentSelectionResult


SCHEMA_VERSION = "hetrod-selection-manifest-v1"


@dataclass(frozen=True)
class ManifestSelection:
    result: AgentSelectionResult
    interaction_pair_object_ids: list[list[int]] | None
    source: str


def load_selection_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Load and minimally validate a versioned selection manifest."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported selection manifest schema: "
            f"{manifest.get('schema_version')!r}."
        )
    expected_metric_version = f"hetrod-{__version__}"
    if manifest.get("metric_version") != expected_metric_version:
        raise ValueError(
            "Selection manifest metric version mismatch: expected "
            f"{expected_metric_version!r}, got {manifest.get('metric_version')!r}."
        )
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("Selection manifest must contain a non-empty scenarios map.")
    for scenario_id, record in scenarios.items():
        validate_manifest_record(scenario_id, record)
    return manifest, hashlib.sha256(raw).hexdigest()


def validate_manifest_record(scenario_id: str, record: Any) -> None:
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("Selection manifest scenario IDs must be non-empty strings.")
    if not isinstance(record, dict):
        raise ValueError(f"{scenario_id}: selection record must be an object.")
    status = record.get("status")
    if status not in {"score", "exclude"}:
        raise ValueError(f"{scenario_id}: status must be 'score' or 'exclude'.")
    if status == "exclude":
        if not str(record.get("reason", "")).strip():
            raise ValueError(f"{scenario_id}: excluded records require a reason.")
        return

    selected = record.get("selected_agent_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"{scenario_id}: scored records require selected_agent_ids.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
        raise ValueError(f"{scenario_id}: selected_agent_ids must contain integers.")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{scenario_id}: selected_agent_ids must be unique.")

    pairs = record.get("interaction_pair_object_ids")
    if pairs is None:
        return
    if not isinstance(pairs, list):
        raise ValueError(
            f"{scenario_id}: interaction_pair_object_ids must be a list or null."
        )
    normalized = []
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
            or pair[0] == pair[1]
        ):
            raise ValueError(
                f"{scenario_id}: each interaction pair must contain two distinct integers."
            )
        normalized.append(tuple(sorted(pair)))
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{scenario_id}: interaction pairs must be unique.")


def manifest_selection_for_scenario(
    gt_scenario: dict[str, Any],
    record: dict[str, Any],
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> ManifestSelection:
    """Resolve a scored manifest record to the evaluator's anchor mask."""
    scenario_id = str(gt_scenario.get("scenario_id", "unknown"))
    validate_manifest_record(scenario_id, record)
    if record["status"] != "score":
        raise ValueError(f"{scenario_id}: excluded record cannot be scored.")

    object_ids = gt_scenario["object_ids"].int()
    selected_ids = torch.tensor(
        record["selected_agent_ids"],
        dtype=object_ids.dtype,
        device=object_ids.device,
    )
    missing = selected_ids[~torch.isin(selected_ids, object_ids)]
    if missing.numel():
        raise ValueError(
            f"{scenario_id}: selection manifest contains target IDs absent from GT: "
            f"{missing.detach().cpu().tolist()}."
        )
    anchor_mask = torch.isin(object_ids, selected_ids)

    pair_ids = record.get("interaction_pair_object_ids")
    pairs: list[dict[str, Any]] = []
    if pair_ids is not None:
        id_to_type = {
            int(object_id): int(object_type)
            for object_id, object_type in zip(
                object_ids.detach().cpu().tolist(),
                gt_scenario["object_types"].detach().cpu().tolist(),
            )
        }
        missing_pair_ids = sorted(
            {
                int(value)
                for pair in pair_ids
                for value in pair
                if int(value) not in id_to_type
            }
        )
        if missing_pair_ids:
            raise ValueError(
                f"{scenario_id}: interaction pair IDs absent from GT: "
                f"{missing_pair_ids}."
            )
        selected_set = set(record["selected_agent_ids"])
        for pair in pair_ids:
            pair = [int(pair[0]), int(pair[1])]
            pairs.append(
                {
                    "agent_ids": pair,
                    "agent_types": [id_to_type[value] for value in pair],
                    "anchor_agent_ids": [
                        value for value in pair if value in selected_set
                    ],
                }
            )

    return ManifestSelection(
        result=AgentSelectionResult(
            anchor_mask=anchor_mask,
            pairs=pairs,
            mode=f"manifest_{record.get('source', 'private')}",
        ),
        interaction_pair_object_ids=pair_ids,
        source=str(record.get("source", "private")),
    )
