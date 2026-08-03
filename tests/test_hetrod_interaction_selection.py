from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.config import DEFAULT_CONFIG
from hetrod_metrics.interaction_candidates import FutureConflictSelectionConfig
from hetrod_metrics.interaction_ranking import _classify_and_score
from hetrod_metrics.interaction_selection import (
    OFFICIAL_SELECTION_CONFIG,
    select_competition_agents,
    select_interaction_agents,
)


class OfficialInteractionSelectionTests(unittest.TestCase):
    def test_local_static_context_is_selected_as_moving_static(self):
        scenario = self.make_base_scenario(2)
        scenario["tracks"][0, 11:, 0] = torch.cat(
            [torch.linspace(-8.0, 0.0, 40), torch.linspace(0.0, 4.0, 40)]
        )
        scenario["tracks"][1, 11:, 0] = 0.0
        scenario["tracks"][:, :11] = scenario["tracks"][:, 11:12]

        result = select_interaction_agents(scenario)

        self.assertEqual(len(result.pairs), 1)
        pair = result.pairs[0]
        self.assertEqual(pair["behavior_category"], "moving_static")
        self.assertGreater(pair["local_speeds_mps"][0], 0.3)
        self.assertLessEqual(pair["local_speeds_mps"][1], 0.3)

    def test_second_static_pair_is_allowed_only_for_new_anchor_type(self):
        scenario = self.make_base_scenario(4)
        moving = torch.cat(
            [torch.linspace(-8.0, 0.0, 40), torch.linspace(0.0, 4.0, 40)]
        )
        scenario["tracks"][0, 11:, 0] = moving
        scenario["tracks"][1, 11:, 0] = 0.0
        scenario["tracks"][2, 11:, 0] = moving
        scenario["tracks"][2:, 11:, 1] = 20.0
        scenario["tracks"][:, :11] = scenario["tracks"][:, 11:12]
        scenario["object_types"][:2] = (
            scenario_pb2.Track.ObjectType.TYPE_CYCLIST
        )

        result = select_interaction_agents(scenario)
        moving_static = [
            pair
            for pair in result.pairs
            if pair["behavior_category"] == "moving_static"
        ]

        self.assertEqual(len(moving_static), 2)
        self.assertEqual(
            {
                pair["agent_types"][0]
                for pair in moving_static
            },
            {"two_wheeler", "vehicle"},
        )

    def test_corridor_consistent_order_swap_is_tier_b_overtake(self):
        tracks = np.zeros((2, 80, 2), dtype=np.float32)
        tracks[0, :, 0] = np.linspace(0.0, 8.0, 80)
        tracks[1, :, 0] = np.linspace(-6.0, 14.0, 80)

        pair = self.classify(tracks)

        self.assertEqual(pair["behavior_category"], "overtake")
        self.assertTrue(pair["corridor_persistent"])
        self.assertEqual(pair["interaction_tier"], "B")

    def test_order_swap_across_separate_corridors_is_not_overtake(self):
        tracks = np.zeros((2, 80, 2), dtype=np.float32)
        tracks[0, :, 0] = np.linspace(0.0, 8.0, 80)
        tracks[1, :, 0] = np.linspace(-6.0, 14.0, 80)
        tracks[1, :, 1] = 4.0

        pair = self.classify(tracks)

        self.assertFalse(pair["corridor_persistent"])
        self.assertNotEqual(pair["behavior_category"], "overtake")

    def test_official_config_enables_all_semantic_safeguards(self):
        self.assertTrue(OFFICIAL_SELECTION_CONFIG.use_local_static_semantics)
        self.assertTrue(OFFICIAL_SELECTION_CONFIG.require_corridor_persistence)
        self.assertTrue(OFFICIAL_SELECTION_CONFIG.overtake_uses_dedicated_tier)
        self.assertTrue(OFFICIAL_SELECTION_CONFIG.reject_low_motion_pairs)
        self.assertTrue(OFFICIAL_SELECTION_CONFIG.prefer_pair_type_diversity)
        self.assertTrue(
            OFFICIAL_SELECTION_CONFIG.allow_static_cap_for_new_anchor_type
        )

    def test_competition_wrapper_honors_public_fallback_cap(self):
        scenario = self.make_base_scenario(4)
        config = replace(
            DEFAULT_CONFIG,
            selection_num_fallback_agents=2,
            selection_min_fallback_agents=2,
        )

        result = select_competition_agents(scenario, config)

        self.assertEqual(result.mode, "fallback_noninteractive")
        self.assertEqual(int(result.anchor_mask.sum().item()), 2)

    @staticmethod
    def classify(tracks: np.ndarray) -> dict:
        validity = np.ones((2, tracks.shape[1]), dtype=bool)
        candidate = {
            "agent_indices": [0, 1],
            "agent_ids": [10, 11],
            "agent_types": ["vehicle", "vehicle"],
            "pair_type": "vehicle_vehicle",
            "arrival_indices": [39, 39],
            "arrival_gap_s": 0.0,
            "timed_path_conflict": True,
            "direct_footprint_proximity": True,
            "future_path_only": False,
            "min_synchronous_distance_m": 0.0,
            "closing_distance_m": 2.0,
            "motion_response_score": 0.5,
            "responses": [{"score": 0.5}, {"score": 0.5}],
            "score": 0.5,
        }
        metric = DEFAULT_CONFIG
        selection = FutureConflictSelectionConfig(
            path_margin_m=metric.selection_path_margin_m,
            max_arrival_gap_s=metric.selection_max_arrival_gap_s,
            max_synchronous_distance_m=(
                metric.selection_max_synchronous_distance_m
            ),
            min_motion_response_score=(
                metric.selection_min_motion_response_score
            ),
            min_direct_closing_distance_m=(
                metric.selection_min_direct_closing_distance_m
            ),
            min_direct_motion_response_score=(
                metric.selection_min_direct_motion_response_score
            ),
        )
        return _classify_and_score(
            candidate,
            tracks,
            validity,
            {0, 1},
            selection,
            OFFICIAL_SELECTION_CONFIG,
            metric,
        )

    @staticmethod
    def make_base_scenario(num_agents: int) -> dict:
        tracks = torch.zeros(num_agents, 91, 9, dtype=torch.float32)
        masks = torch.ones(num_agents, 91, dtype=torch.bool)
        tracks[:, :, 3:6] = torch.tensor([4.0, 1.8, 1.5])
        return {
            "tracks": tracks,
            "track_masks": masks,
            "object_ids": torch.arange(10, 10 + num_agents),
            "object_types": torch.full(
                (num_agents,),
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                dtype=torch.int64,
            ),
        }


if __name__ == "__main__":
    unittest.main()
