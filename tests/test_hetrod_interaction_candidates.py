from __future__ import annotations

import unittest

import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.interaction_candidates import (
    FutureConflictSelectionConfig,
    select_competition_agents,
    select_future_conflict_agents,
)


class FutureConflictSelectionTests(unittest.TestCase):
    def test_selects_crossing_paths_with_a_safe_arrival_gap(self):
        scenario = self.make_scenario()
        selected, pairs = select_future_conflict_agents(scenario)

        self.assertEqual(selected.tolist(), [True, True, False])
        self.assertEqual(len(pairs), 1)
        self.assertLess(pairs[0]["path_distance_m"], 0.2)
        self.assertGreater(pairs[0]["min_synchronous_distance_m"], 1.0)

    def test_rejects_crossing_paths_with_a_large_arrival_gap(self):
        scenario = self.make_scenario(second_time_offset=40)
        selected, pairs = select_future_conflict_agents(
            scenario,
            FutureConflictSelectionConfig(
                max_arrival_gap_s=1.0,
                num_fallback_agents=0,
            ),
        )

        self.assertFalse(selected.any())
        self.assertEqual(pairs, [])

    def test_rejects_geometric_overlap_without_proximity_or_response(self):
        scenario = self.make_scenario()
        scenario["tracks"][1, 11:, 1] -= 3.0
        scenario["tracks"][1, :, 3:5] = torch.tensor([0.8, 0.8])
        selected, pairs = select_future_conflict_agents(
            scenario,
            FutureConflictSelectionConfig(
                max_synchronous_distance_m=1.0,
                min_motion_response_score=0.4,
                num_fallback_agents=0,
            ),
        )

        self.assertFalse(selected.any())
        self.assertEqual(pairs, [])

    def test_falls_back_to_active_type_diverse_agents_without_a_pair(self):
        scenario = self.make_scenario(second_time_offset=40)
        selected, pairs = select_future_conflict_agents(
            scenario,
            FutureConflictSelectionConfig(max_arrival_gap_s=1.0),
        )

        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(pairs, [])

    def test_selects_all_qualifying_pairs_without_an_agent_target(self):
        scenario = self.make_four_agent_scenario()
        selected, pairs = select_future_conflict_agents(scenario)

        self.assertGreaterEqual(len(pairs), 2)
        self.assertEqual(int(selected.sum()), 4)

    def test_partial_history_moving_anchor_uses_static_same_type_context(self):
        scenario = self.make_scenario()
        scenario["object_types"][1] = scenario["object_types"][0]
        scenario["tracks"][0, 11:, 0] = torch.linspace(-8.0, 0.0, 80)
        scenario["tracks"][0, 11:, 1] = 0.0
        scenario["tracks"][0, 11:, 6] = 0.0
        scenario["tracks"][1, 11:, :2] = torch.tensor([2.8, 0.0])
        scenario["tracks"][1, 11:, 6] = 0.0
        scenario["track_masks"][0, :2] = False
        scenario["tracks"][:, :11] = scenario["tracks"][:, 11:12]

        selected, pairs = select_future_conflict_agents(scenario)

        self.assertTrue(selected[0])
        self.assertFalse(selected[1])
        pair = next(pair for pair in pairs if pair["agent_ids"] == [10, 20])
        self.assertEqual(pair["anchor_agent_ids"], [10])
        self.assertEqual(pair["context_only_agent_ids"], [20])

    def test_competition_selector_preserves_explicit_pair_and_mode(self):
        scenario = self.make_scenario()
        result = select_competition_agents(scenario)

        self.assertEqual(result.mode, "interactive_pairs")
        self.assertTrue(result.pair_object_ids)
        self.assertEqual(result.anchor_mask.dtype, torch.bool)

    def test_competition_selector_caps_pairs_endpoints_and_degree(self):
        scenario = self.make_dense_crossing_scenario()
        result = select_competition_agents(scenario)

        self.assertLessEqual(len(result.pairs), 4)
        endpoint_ids = {
            agent_id
            for pair in result.pairs
            for agent_id in pair["agent_ids"]
        }
        self.assertLessEqual(len(endpoint_ids), 8)
        degrees = {
            agent_id: sum(
                agent_id in pair["agent_ids"] for pair in result.pairs
            )
            for agent_id in endpoint_ids
        }
        self.assertLessEqual(max(degrees.values()), 2)
        self.assertTrue(
            any(
                pair["agent_types"][0] != pair["agent_types"][1]
                for pair in result.pairs
            )
        )

    def test_motion_extent_rejects_jitter_as_an_anchor(self):
        scenario = self.make_scenario()
        scenario["tracks"][1, 11:, 0] = torch.linspace(0.0, 0.4, 80)
        scenario["tracks"][1, 11:, 1] = 0.0

        selected, pairs = select_future_conflict_agents(scenario)

        self.assertTrue(pairs)
        self.assertFalse(selected[1])
        self.assertIn(20, pairs[0]["context_only_agent_ids"])

    def make_scenario(self, second_time_offset: int = 15) -> dict:
        tracks = torch.zeros(3, 91, 9, dtype=torch.float32)
        masks = torch.ones(3, 91, dtype=torch.bool)
        object_types = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
            ],
            dtype=torch.int64,
        )
        tracks[:, :, 3:6] = torch.tensor([4.0, 1.8, 1.5])
        future_steps = 80
        tracks[0, 11:, 0] = torch.linspace(-10.0, 10.0, future_steps)
        tracks[1, 11:, 1] = torch.linspace(-10.0, 10.0, future_steps)
        tracks[1, 11:, 0] = 0.0
        if second_time_offset:
            tracks[1, 11:, 1] = torch.linspace(
                -10.0 - second_time_offset * 0.25,
                10.0 - second_time_offset * 0.25,
                future_steps,
            )
        tracks[2, 11:, 0] = 30.0
        tracks[:, :11] = tracks[:, 11:12]
        return {
            "tracks": tracks,
            "track_masks": masks,
            "object_ids": torch.tensor([10, 20, 30], dtype=torch.int32),
            "object_types": object_types,
        }

    def make_four_agent_scenario(self) -> dict:
        tracks = torch.zeros(4, 91, 9, dtype=torch.float32)
        masks = torch.ones(4, 91, dtype=torch.bool)
        tracks[:, :, 3:6] = torch.tensor([2.0, 1.0, 1.5])
        tracks[0, 11:, 0] = torch.linspace(-5.0, 5.0, 80)
        tracks[1, 11:, 1] = torch.linspace(-5.0, 5.0, 80)
        tracks[2, 11:, 0] = torch.linspace(-5.0, 5.0, 80)
        tracks[2, :, 1] = 10.0
        tracks[3, 11:, 1] = torch.linspace(5.0, 15.0, 80)
        tracks[:, :11] = tracks[:, 11:12]
        return {
            "tracks": tracks,
            "track_masks": masks,
            "object_ids": torch.tensor([10, 20, 30, 40], dtype=torch.int32),
            "object_types": torch.tensor(
                [
                    scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                    scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
                    scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                    scenario_pb2.Track.ObjectType.TYPE_CYCLIST,
                ]
            ),
        }

    def make_dense_crossing_scenario(self) -> dict:
        num_agents = 10
        tracks = torch.zeros(num_agents, 91, 9, dtype=torch.float32)
        masks = torch.ones(num_agents, 91, dtype=torch.bool)
        tracks[:, :, 3:6] = torch.tensor([2.0, 1.0, 1.5])
        for index in range(num_agents):
            offset = float(index % 3) * 0.25
            if index % 2:
                tracks[index, 11:, 0] = offset
                tracks[index, 11:, 1] = torch.linspace(-5.0, 5.0, 80)
            else:
                tracks[index, 11:, 0] = torch.linspace(-5.0, 5.0, 80)
                tracks[index, 11:, 1] = offset
        tracks[:, :11] = tracks[:, 11:12]
        object_types = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE
                if index % 2 == 0
                else scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN
                for index in range(num_agents)
            ]
        )
        return {
            "tracks": tracks,
            "track_masks": masks,
            "object_ids": torch.arange(100, 100 + num_agents),
            "object_types": object_types,
        }


if __name__ == "__main__":
    unittest.main()
