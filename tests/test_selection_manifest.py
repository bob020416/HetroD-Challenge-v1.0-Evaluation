from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

import torch

from hetrod_metrics.selection_manifest import (
    manifest_selection_for_scenario,
    load_selection_manifest,
    validate_manifest_record,
)


class SelectionManifestTests(unittest.TestCase):
    def setUp(self):
        self.gt = {
            "scenario_id": "sample",
            "object_ids": torch.tensor([10, 20, 30]),
            "object_types": torch.tensor([1, 2, 3]),
        }

    def test_resolves_targets_and_explicit_pairs(self):
        record = {
            "status": "score",
            "source": "human_curated",
            "selected_agent_ids": [20, 30],
            "interaction_pair_object_ids": [[10, 20]],
        }
        selection = manifest_selection_for_scenario(self.gt, record)
        self.assertEqual(selection.result.anchor_mask.tolist(), [False, True, True])
        self.assertEqual(selection.interaction_pair_object_ids, [[10, 20]])
        self.assertEqual(selection.result.context_only_object_ids, [10])

    def test_null_pairs_preserve_automatic_neighborhood_pairing(self):
        record = {
            "status": "score",
            "selected_agent_ids": [20],
            "interaction_pair_object_ids": None,
        }
        selection = manifest_selection_for_scenario(self.gt, record)
        self.assertIsNone(selection.interaction_pair_object_ids)

    def test_missing_target_is_rejected(self):
        record = {
            "status": "score",
            "selected_agent_ids": [999],
            "interaction_pair_object_ids": [],
        }
        with self.assertRaisesRegex(ValueError, "absent from GT"):
            manifest_selection_for_scenario(self.gt, record)

    def test_exclusion_requires_reason(self):
        with self.assertRaisesRegex(ValueError, "require a reason"):
            validate_manifest_record("sample", {"status": "exclude"})

    def test_load_rejects_metric_version_mismatch(self):
        manifest = {
            "schema_version": "hetrod-selection-manifest-v1",
            "metric_version": "hetrod-999.0.0",
            "scenarios": {
                "sample": {
                    "status": "score",
                    "selected_agent_ids": [10],
                    "interaction_pair_object_ids": [],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "metric version mismatch"):
                load_selection_manifest(path)


if __name__ == "__main__":
    unittest.main()
