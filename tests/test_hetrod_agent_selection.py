from __future__ import annotations

import unittest

import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.agent_selection import (
    future_path_length,
    min_cross_type_distance,
    min_cross_type_ttc,
    select_agents,
    select_base_agents,
)


class HetrodAgentSelectionTests(unittest.TestCase):
    def test_future_path_length_keeps_agent_that_returns_to_start(self):
        tracks = torch.zeros(1, 91, 9)
        tracks[0, 10:51, 0] = torch.linspace(0.0, 3.0, 41)
        tracks[0, 51:, 0] = torch.linspace(3.0, 0.0, 40)

        self.assertGreater(float(future_path_length(tracks)[0]), 5.0)

    def test_select_base_agents_filters_history_future_type_and_motion(self):
        tracks = torch.zeros(5, 91, 9, dtype=torch.float32)
        track_masks = torch.ones(5, 91, dtype=torch.bool)
        object_types = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
                scenario_pb2.Track.ObjectType.TYPE_CYCLIST,
                scenario_pb2.Track.ObjectType.TYPE_OTHER,
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
            ],
            dtype=torch.int64,
        )

        tracks[0, -1, 0] = 2.0
        tracks[1, -1, 0] = 2.0
        tracks[2, -1, 0] = 2.0
        tracks[3, -1, 0] = 2.0
        tracks[4, -1, 0] = 0.5
        track_masks[1, 3] = False
        track_masks[2, 20] = False

        selected = select_base_agents(
            {
                "tracks": tracks,
                "track_masks": track_masks,
                "object_types": object_types,
            }
        )

        self.assertEqual(selected.tolist(), [True, False, False, False, False])

    def test_cross_type_distance_uses_different_type_pairs(self):
        scenario = self.make_scenario()

        min_distance = min_cross_type_distance(scenario)

        self.assertLess(min_distance[0].item(), 5.0)
        self.assertLess(min_distance[1].item(), 5.0)
        self.assertGreater(min_distance[2].item(), 5.0)

    def test_cross_type_ttc_detects_future_proximity(self):
        scenario = self.make_scenario()
        scenario["tracks"][0, 10:, 0] = 0.0
        scenario["tracks"][1, 10:, 0] = 20.0
        scenario["tracks"][0, 10:, 7] = 5.0
        scenario["tracks"][1, 10:, 7] = -5.0

        min_ttc = min_cross_type_ttc(scenario)

        self.assertLess(min_ttc[0].item(), 4.0)
        self.assertLess(min_ttc[1].item(), 4.0)

    def test_select_agents_applies_full_spec_filters(self):
        scenario = self.make_scenario()

        selected = select_agents(scenario)

        self.assertEqual(selected.tolist(), [True, True, False])

    def make_scenario(self) -> dict:
        tracks = torch.zeros(3, 91, 9, dtype=torch.float32)
        track_masks = torch.ones(3, 91, dtype=torch.bool)
        object_types = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
                scenario_pb2.Track.ObjectType.TYPE_CYCLIST,
            ],
            dtype=torch.int64,
        )
        tracks[:, :, 3] = torch.tensor([4.8, 1.0, 1.8])[:, None]
        tracks[:, :, 4] = torch.tensor([2.0, 1.0, 0.8])[:, None]
        tracks[:, :, 5] = 1.5
        tracks[0, :, 0] = torch.linspace(0.0, 2.0, 91)
        tracks[1, :, 0] = torch.linspace(3.0, 5.0, 91)
        tracks[2, :, 0] = torch.linspace(20.0, 22.0, 91)
        road_edge = torch.tensor(
            [
                [-10.0, -10.0, 0.0],
                [30.0, -10.0, 0.0],
                [30.0, 10.0, 0.0],
                [-10.0, 10.0, 0.0],
                [-10.0, -10.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return {
            "tracks": tracks,
            "track_masks": track_masks,
            "object_types": object_types,
            "road_edges": [road_edge],
        }


if __name__ == "__main__":
    unittest.main()
