from __future__ import annotations

import unittest

from scripts.build_selection_manifest import merge_with_automatic_selection


class BuildSelectionManifestTests(unittest.TestCase):
    def test_human_targets_and_pairs_are_added_to_automatic_selection(self):
        automatic = {
            "anchors": [10, 20],
            "pairs": [{"ids": [10, 20]}],
        }
        human = {
            "selected_agent_ids": [20, 30],
            "interaction_pair_object_ids": [[20, 30]],
            "curators": ["reviewer"],
        }

        record = merge_with_automatic_selection(automatic, human)

        self.assertEqual(record["selected_agent_ids"], [10, 20, 30])
        self.assertEqual(
            record["interaction_pair_object_ids"], [[10, 20], [20, 30]]
        )
        self.assertEqual(record["automatic_selected_agent_ids"], [10, 20])
        self.assertEqual(record["human_selected_agent_ids"], [20, 30])

    def test_human_target_without_pair_retains_automatic_pairs(self):
        automatic = {
            "anchors": [10, 20],
            "pairs": [{"ids": [10, 20]}],
        }
        human = {
            "selected_agent_ids": [30],
            "interaction_pair_object_ids": None,
            "curators": ["reviewer"],
        }

        record = merge_with_automatic_selection(automatic, human)

        self.assertEqual(record["selected_agent_ids"], [10, 20, 30])
        self.assertEqual(record["interaction_pair_object_ids"], [[10, 20]])


if __name__ == "__main__":
    unittest.main()
