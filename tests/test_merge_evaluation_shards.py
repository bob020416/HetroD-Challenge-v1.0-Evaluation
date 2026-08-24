from __future__ import annotations

import unittest

from scripts.merge_evaluation_shards import merge_reports


class MergeEvaluationShardsTests(unittest.TestCase):
    def shard(self, shard_id: int, scenario_id: str) -> dict:
        return {
            "dataset": None,
            "scenarios": [],
            "skipped_scenarios": [],
            "excluded_scenarios": [
                {
                    "scenario_id": scenario_id,
                    "status": "excluded_by_selection_manifest",
                    "reason": "test",
                }
            ],
            "errors": [],
            "summary": {
                "num_rollout_files": 2,
                "num_gt_files": 2,
                "num_matched_files": 1,
                "num_global_matched_files": 2,
                "num_assigned_scenarios": 1,
                "num_successful_scenarios": 0,
                "num_skipped_no_selected_agents": 0,
                "num_excluded_by_manifest": 1,
                "num_errors": 0,
                "device": "cuda",
                "selection_manifest_sha256": "abc",
                "shard_id": shard_id,
                "num_shards": 2,
            },
        }

    def test_merge_requires_and_combines_complete_shard_coverage(self):
        merged = merge_reports([self.shard(0, "a"), self.shard(1, "b")])
        self.assertEqual(merged["summary"]["num_matched_files"], 2)
        self.assertEqual(merged["summary"]["num_excluded_by_manifest"], 2)
        self.assertEqual(
            [row["scenario_id"] for row in merged["excluded_scenarios"]],
            ["a", "b"],
        )

    def test_merge_rejects_duplicate_shard_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate shard"):
            merge_reports([self.shard(0, "a"), self.shard(0, "b")])


if __name__ == "__main__":
    unittest.main()
