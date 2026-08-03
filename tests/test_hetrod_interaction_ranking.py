from __future__ import annotations

import unittest

import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.interaction_ranking import (
    select_diverse_interaction_agents,
)


class DiverseInteractionSelectionTests(unittest.TestCase):
    def test_crossing_pair_is_high_confidence_and_selected(self):
        scenario = self.make_base_scenario(2)
        scenario["object_types"][:] = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
            ]
        )
        scenario["tracks"][0, 11:, 0] = torch.linspace(-8.0, 8.0, 80)
        scenario["tracks"][1, 11:, 1] = torch.linspace(-8.0, 8.0, 80)
        scenario["tracks"][:, :11] = scenario["tracks"][:, 11:12]

        result = select_diverse_interaction_agents(scenario)

        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(
            result.pairs[0]["behavior_category"],
            "crossing_merge",
        )
        self.assertIn(result.pairs[0]["interaction_tier"], {"A", "B"})
        self.assertEqual(int(result.anchor_mask.sum()), 2)

    def test_unresponsive_static_proximity_is_not_an_interaction_pair(self):
        scenario = self.make_base_scenario(2)
        scenario["tracks"][0, 11:, 0] = torch.linspace(-8.0, 8.0, 80)
        scenario["tracks"][1, 11:, 0] = 0.0
        scenario["tracks"][:, :11] = scenario["tracks"][:, 11:12]

        result = select_diverse_interaction_agents(scenario)

        self.assertEqual(result.pairs, [])
        self.assertEqual(result.mode, "fallback_noninteractive")
        self.assertEqual(int(result.anchor_mask.sum()), 2)

    def test_following_behavior_is_capped_at_one_pair(self):
        scenario = self.make_base_scenario(4)
        for index, offset in enumerate([0.0, -4.0, -8.0, -12.0]):
            scenario["tracks"][index, 11:, 0] = torch.linspace(
                offset,
                offset + 8.0 + index,
                80,
            )
        scenario["tracks"][:, :11] = scenario["tracks"][:, 11:12]

        result = select_diverse_interaction_agents(scenario)

        following = [
            pair
            for pair in result.pairs
            if pair["behavior_category"] == "following"
        ]
        self.assertLessEqual(len(following), 1)

    def test_pair_and_endpoint_caps_are_preserved(self):
        scenario = self.make_base_scenario(10)
        for index in range(10):
            if index % 2:
                scenario["tracks"][index, 11:, 0] = float(index % 3) * 0.2
                scenario["tracks"][index, 11:, 1] = torch.linspace(
                    -5.0,
                    5.0,
                    80,
                )
                scenario["object_types"][index] = (
                    scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN
                )
            else:
                scenario["tracks"][index, 11:, 0] = torch.linspace(
                    -5.0,
                    5.0,
                    80,
                )
                scenario["tracks"][index, 11:, 1] = float(index % 3) * 0.2
        scenario["tracks"][:, :11] = scenario["tracks"][:, 11:12]

        result = select_diverse_interaction_agents(scenario)
        endpoints = {
            agent_id
            for pair in result.pairs
            for agent_id in pair["agent_ids"]
        }

        self.assertLessEqual(len(result.pairs), 4)
        self.assertLessEqual(len(endpoints), 8)

    @staticmethod
    def make_base_scenario(num_agents: int) -> dict:
        tracks = torch.zeros(num_agents, 91, 9, dtype=torch.float32)
        masks = torch.ones(num_agents, 91, dtype=torch.bool)
        tracks[:, :, 3:6] = torch.tensor([4.0, 1.8, 1.5])
        object_types = torch.full(
            (num_agents,),
            scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
            dtype=torch.int64,
        )
        return {
            "tracks": tracks,
            "track_masks": masks,
            "object_ids": torch.arange(10, 10 + num_agents),
            "object_types": object_types,
        }


if __name__ == "__main__":
    unittest.main()
