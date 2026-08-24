#!/usr/bin/env python3
"""Build an organizer-private test selection manifest from reviewed curation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hetrod_metrics import __version__
from hetrod_metrics.config import DEFAULT_CONFIG
from hetrod_metrics.selection_manifest import SCHEMA_VERSION, validate_manifest_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Union reviewed targets with frozen v0.8 selections."
    )
    parser.add_argument("--curations", type=Path, required=True)
    parser.add_argument("--automatic-selection", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--curator",
        action="append",
        dest="curators",
        help="Include only this curator; repeat as needed. Default: all except initial_seed.",
    )
    parser.add_argument(
        "--sanitize-human-targets",
        action="store_true",
        help=(
            "Drop human targets that are not official required/eligible agents, "
            "record an audit, and retain the v0.8 selection in every case."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_pairs(raw_pairs: Any) -> list[list[int]]:
    pairs = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in (raw_pairs or [])
        if isinstance(pair, list) and len(pair) == 2 and int(pair[0]) != int(pair[1])
    }
    return [list(pair) for pair in sorted(pairs)]


def curation_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("team_annotations"), list):
        return payload["team_annotations"]
    raise ValueError("Curations must be a Supabase row list or website team export.")


def human_selections(
    rows: list[dict[str, Any]],
    allowed_curators: set[str] | None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        curator = str(row.get("curator", ""))
        if curator == "initial_seed":
            continue
        if allowed_curators is not None and curator not in allowed_curators:
            continue
        annotation = row.get("annotation") or {}
        if annotation.get("decision") != "curated":
            continue
        agents = {int(value) for value in annotation.get("agents") or []}
        pairs = normalize_pairs(annotation.get("pairs"))
        agents.update(value for pair in pairs for value in pair)
        if not agents:
            continue
        grouped[str(row["scenario_id"])].append(
            {"curator": curator, "agents": agents, "pairs": pairs}
        )

    result = {}
    for scenario_id, selections in grouped.items():
        agents = sorted({value for item in selections for value in item["agents"]})
        pairs = normalize_pairs(
            [pair for item in selections for pair in item["pairs"]]
        )
        result[scenario_id] = {
            "selected_agent_ids": agents,
            # Null means the reviewer added targets but no additional explicit pair.
            # Frozen automatic pairs are merged later and always remain present.
            "interaction_pair_object_ids": pairs or None,
            "curators": sorted({item["curator"] for item in selections}),
        }
    return result


def load_gt(path: Path) -> dict[str, Any]:
    if not hasattr(np, "_core"):
        import numpy.core
        import numpy.core.multiarray
        import numpy.core.numeric

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
    with path.open("rb") as handle:
        return pickle.load(handle)


def merge_with_automatic_selection(
    automatic: dict[str, Any],
    human: dict[str, Any] | None,
) -> dict[str, Any]:
    """Union frozen v0.8 targets/pairs with optional reviewed additions."""
    automatic_agents = sorted({int(value) for value in automatic["anchors"]})
    automatic_pairs = normalize_pairs(
        [pair["ids"] for pair in automatic.get("pairs", [])]
    )
    human_agents = [] if human is None else human["selected_agent_ids"]
    human_pairs = (
        []
        if human is None or human.get("interaction_pair_object_ids") is None
        else human["interaction_pair_object_ids"]
    )
    record = {
        "status": "score",
        "source": (
            "automatic_v0.8+human_curated"
            if human is not None
            else "automatic_v0.8"
        ),
        "selected_agent_ids": sorted(set(automatic_agents) | set(human_agents)),
        "interaction_pair_object_ids": normalize_pairs(
            automatic_pairs + human_pairs
        ),
        "automatic_selected_agent_ids": automatic_agents,
        "human_selected_agent_ids": sorted(set(human_agents)),
    }
    if human is not None:
        record["curators"] = human["curators"]
    return record


def human_target_eligibility(
    scenario_id: str,
    selection: dict[str, Any],
    gt_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return a sanitized human selection and a transparent adjustment audit."""
    gt = load_gt(gt_dir / f"{scenario_id}.pkl")
    object_ids = torch.as_tensor(gt["object_ids"]).int()
    object_types = torch.as_tensor(gt["object_types"]).int()
    masks = torch.as_tensor(gt["track_masks"]).bool()
    valid_types = torch.tensor(DEFAULT_CONFIG.evaluated_object_types)
    history = masks[:, : DEFAULT_CONFIG.current_time_index + 1].sum(dim=1)
    future = masks[:, DEFAULT_CONFIG.future_start_index :].sum(dim=1)
    id_to_position = {
        int(object_id): index
        for index, object_id in enumerate(object_ids.tolist())
    }
    reasons: dict[int, str] = {}
    kept = []
    for agent_id in selection["selected_agent_ids"]:
        position = id_to_position.get(int(agent_id))
        if position is None:
            reasons[int(agent_id)] = "not_in_required_agent_ids"
        elif not bool(torch.isin(object_types[position], valid_types)):
            reasons[int(agent_id)] = "unsupported_agent_type"
        elif not bool(masks[position, DEFAULT_CONFIG.current_time_index]):
            reasons[int(agent_id)] = "current_frame_invalid"
        elif int(history[position]) < DEFAULT_CONFIG.selection_min_history_valid_frames:
            reasons[int(agent_id)] = "insufficient_history"
        elif int(future[position]) < DEFAULT_CONFIG.min_future_valid_frames:
            reasons[int(agent_id)] = "insufficient_future"
        else:
            kept.append(int(agent_id))
    kept_set = set(kept)
    original_pairs = selection.get("interaction_pair_object_ids")
    kept_pairs = (
        None
        if original_pairs is None
        else [
            pair
            for pair in original_pairs
            if int(pair[0]) in kept_set and int(pair[1]) in kept_set
        ]
    )
    removed_pairs = (
        []
        if original_pairs is None
        else [pair for pair in original_pairs if pair not in kept_pairs]
    )
    if original_pairs is not None and not kept_pairs:
        kept_pairs = None
    audit = None
    if reasons or removed_pairs:
        audit = {
            "removed_agent_ids": sorted(reasons),
            "removed_agent_reasons": {
                str(agent_id): reasons[agent_id] for agent_id in sorted(reasons)
            },
            "removed_interaction_pairs": removed_pairs,
            "all_human_targets_removed": not kept,
        }
    if not kept:
        return None, audit
    return {
        **selection,
        "selected_agent_ids": sorted(kept),
        "interaction_pair_object_ids": kept_pairs,
    }, audit


def validate_against_gt(
    scenario_id: str,
    record: dict[str, Any],
    gt_dir: Path,
) -> None:
    validate_manifest_record(scenario_id, record)
    gt_path = gt_dir / f"{scenario_id}.pkl"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing private GT: {gt_path}")
    gt = load_gt(gt_path)
    if str(gt.get("scenario_id")) != scenario_id:
        raise ValueError(f"{scenario_id}: private GT scenario_id mismatch.")
    if record["status"] == "exclude":
        return

    object_ids = torch.as_tensor(gt["object_ids"]).int()
    selected_ids = torch.tensor(record["selected_agent_ids"], dtype=torch.int32)
    missing = selected_ids[~torch.isin(selected_ids, object_ids)]
    if missing.numel():
        raise ValueError(
            f"{scenario_id}: selected IDs absent from GT: {missing.tolist()}."
        )

    object_types = torch.as_tensor(gt["object_types"]).int()
    masks = torch.as_tensor(gt["track_masks"]).bool()
    history = masks[:, : DEFAULT_CONFIG.current_time_index + 1].sum(dim=1)
    future = masks[:, DEFAULT_CONFIG.future_start_index :].sum(dim=1)
    valid_types = torch.tensor(DEFAULT_CONFIG.evaluated_object_types)
    eligible = (
        torch.isin(object_types, valid_types)
        & masks[:, DEFAULT_CONFIG.current_time_index]
        & (history >= DEFAULT_CONFIG.selection_min_history_valid_frames)
        & (future >= DEFAULT_CONFIG.min_future_valid_frames)
    )
    selected_positions = torch.where(torch.isin(object_ids, selected_ids))[0]
    if not eligible[selected_positions].all():
        invalid = object_ids[selected_positions[~eligible[selected_positions]]].tolist()
        raise ValueError(
            f"{scenario_id}: selected IDs violate competition eligibility: {invalid}."
        )
    pairs = record.get("interaction_pair_object_ids")
    if pairs is not None:
        pair_ids = torch.tensor(pairs, dtype=torch.int32).flatten()
        missing_pairs = pair_ids[~torch.isin(pair_ids, object_ids)]
        if missing_pairs.numel():
            raise ValueError(
                f"{scenario_id}: pair IDs absent from GT: "
                f"{sorted(set(missing_pairs.tolist()))}."
            )


def main() -> int:
    args = parse_args()
    scenario_ids = [
        row.strip()
        for row in args.scenario_manifest.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Scenario manifest contains duplicate IDs.")

    automatic = load_json(args.automatic_selection)
    exclusion_payload = load_json(args.exclusions)
    exclusions = exclusion_payload.get("exclusions") or {}
    rows = curation_rows(load_json(args.curations))
    allowed_curators = set(args.curators) if args.curators else None
    human = human_selections(rows, allowed_curators)
    if set(exclusions) - set(scenario_ids):
        raise ValueError("Exclusions contain IDs outside the scenario manifest.")
    if set(human) - set(scenario_ids):
        raise ValueError("Curations contain IDs outside the scenario manifest.")

    human_adjustments: dict[str, dict[str, Any]] = {}
    sanitized_human: dict[str, dict[str, Any]] = {}
    for scenario_id, selection in human.items():
        sanitized, audit = human_target_eligibility(
            scenario_id,
            selection,
            args.gt_dir,
        )
        if audit is not None:
            human_adjustments[scenario_id] = audit
        if sanitized is not None:
            sanitized_human[scenario_id] = sanitized
    if human_adjustments and not args.sanitize_human_targets:
        examples = list(human_adjustments.items())[:10]
        raise ValueError(
            f"Found {len(human_adjustments)} human selections requiring "
            f"sanitization; examples: {examples}. Re-run with "
            "--sanitize-human-targets after reviewing the audit."
        )
    human = sanitized_human

    scenarios: dict[str, dict[str, Any]] = {}
    sources = Counter()
    for scenario_id in scenario_ids:
        if scenario_id in exclusions:
            exclusion = exclusions[scenario_id]
            record = {
                "status": "exclude",
                "source": "organizer_exclusion",
                "reason": str(exclusion["reason"]),
                "notes": str(exclusion.get("notes", "")),
            }
        else:
            auto = automatic.get(scenario_id)
            if auto is None:
                raise ValueError(f"{scenario_id}: missing frozen automatic selection.")
            record = merge_with_automatic_selection(auto, human.get(scenario_id))
            if scenario_id in human_adjustments:
                record["curation_adjustment"] = human_adjustments[scenario_id]
        validate_against_gt(scenario_id, record, args.gt_dir)
        scenarios[scenario_id] = record
        sources[record["source"]] += 1

    summary = {
        "num_scenarios": len(scenarios),
        "num_scored": sum(record["status"] == "score" for record in scenarios.values()),
        "num_excluded": sum(
            record["status"] == "exclude" for record in scenarios.values()
        ),
        "selection_source_counts": dict(sorted(sources.items())),
        "num_selected_agents": sum(
            len(record.get("selected_agent_ids", [])) for record in scenarios.values()
        ),
        "num_explicit_pairs": sum(
            len(record.get("interaction_pair_object_ids") or [])
            for record in scenarios.values()
        ),
        "num_human_selection_adjustments": len(human_adjustments),
        "num_scenarios_with_human_additions": sum(
            bool(record.get("human_selected_agent_ids"))
            for record in scenarios.values()
        ),
        "num_automatic_selected_agents": sum(
            len(record.get("automatic_selected_agent_ids", []))
            for record in scenarios.values()
        ),
        "num_human_added_agent_entries": sum(
            len(record.get("human_selected_agent_ids", []))
            for record in scenarios.values()
        ),
        "num_human_selections_fully_removed": sum(
            value["all_human_targets_removed"]
            for value in human_adjustments.values()
        ),
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "metric_version": f"hetrod-{__version__}",
        "dataset_release": "schema1.3",
        "split": "test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "target_agents": "union_of_automatic_v0.8_and_human_curated",
            "interaction_pairs": "union_of_automatic_v0.8_and_human_curated",
            "human_target_without_pair": "retain_automatic_v0.8_pairs",
            "empty_human_target": "retain_automatic_v0.8_selection",
            "submission_completeness": "all_manifest_scenarios_including_exclusions",
            "scoring": "exclude_organizer_exclusions",
        },
        "inputs": {
            "curations_sha256": sha256(args.curations),
            "automatic_selection_sha256": sha256(args.automatic_selection),
            "exclusions_sha256": sha256(args.exclusions),
            "scenario_manifest_sha256": sha256(args.scenario_manifest),
        },
        "summary": summary,
        "human_selection_adjustments": human_adjustments,
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote private manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
