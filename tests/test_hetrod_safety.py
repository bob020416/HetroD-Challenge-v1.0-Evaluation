from __future__ import annotations

import unittest

import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.features import build_feature_bundle
from hetrod_metrics.safety import collision_with_safe_margin, compute_safety


class HetrodSafetyTests(unittest.TestCase):
    def test_safety_scores_clean_rollouts_high(self):
        gt = self.make_gt()
        features = build_feature_bundle(gt, self.make_prediction(gt), torch.tensor([True, True]))

        report = compute_safety(features)

        self.assertGreater(report["score"], 0.99)
        self.assertGreater(report["collision_with_safe_margin"]["score"], 0.99)
        self.assertGreater(report["valid_region_margin"]["score"], 0.99)

    def test_collision_with_safe_margin_penalizes_overlap(self):
        gt = self.make_gt()
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1, :, 0] = prediction["simulated_states"][:, 0, :, 0]
        prediction["simulated_states"][:, 1, :, 1] = prediction["simulated_states"][:, 0, :, 1]
        features = build_feature_bundle(gt, prediction, torch.tensor([True, True]))

        report = collision_with_safe_margin(features)

        self.assertLess(report["score"], 0.5)
        self.assertGreater(report["unsafe_rate"], 0.5)

    def test_collision_checks_selected_anchor_against_non_selected_context(self):
        gt = self.make_gt(num_agents=3)
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1, :, 0] = prediction["simulated_states"][:, 0, :, 0]
        prediction["simulated_states"][:, 1, :, 1] = prediction["simulated_states"][:, 0, :, 1]
        features = build_feature_bundle(gt, prediction, torch.tensor([True, False, False]))

        report = collision_with_safe_margin(features)

        self.assertEqual(report["num_context_agents"], 3)
        self.assertLess(report["score"], 0.5)

    def test_safety_macro_averages_agent_types(self):
        gt = self.make_gt_with_imbalanced_types()
        prediction = self.make_prediction(gt)
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True, True, True, True]),
        )

        report = collision_with_safe_margin(features)

        self.assertAlmostEqual(report["by_type"]["vehicle"]["score"], 0.0, places=6)
        self.assertAlmostEqual(report["by_type"]["pedestrian"]["score"], 1.0, places=6)
        self.assertAlmostEqual(report["by_type"]["two_wheeler"]["score"], 1.0, places=6)
        self.assertAlmostEqual(report["score"], 2.0 / 3.0, places=6)

    def make_gt(self, num_agents: int = 2) -> dict:
        tracks = torch.zeros(num_agents, 91, 9, dtype=torch.float32)
        track_masks = torch.ones(num_agents, 91, dtype=torch.bool)
        object_ids = torch.arange(10, 10 + 10 * num_agents, 10, dtype=torch.int32)
        base_types = [
            scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
            scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
            scenario_pb2.Track.ObjectType.TYPE_CYCLIST,
        ]
        object_types = torch.tensor(
            base_types[:num_agents],
            dtype=torch.int64,
        )
        tracks[:, :, 3] = 2.0
        tracks[:, :, 4] = 1.0
        tracks[:, :, 5] = 1.5
        for i in range(num_agents):
            tracks[i, :, 0] = torch.linspace(-5.0, 5.0, 91)
            tracks[i, :, 1] = float(i) * 8.0
        road_edge = torch.tensor(
            [
                [-20.0, -20.0, 0.0],
                [20.0, -20.0, 0.0],
                [20.0, 20.0, 0.0],
                [-20.0, 20.0, 0.0],
                [-20.0, -20.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return {
            "tracks": tracks,
            "track_masks": track_masks,
            "object_ids": object_ids,
            "object_types": object_types,
            "road_edges": [road_edge],
        }

    def make_prediction(self, gt: dict) -> dict:
        future = gt["tracks"][:, 11:, [0, 1, 2, 6]]
        return {
            "agent_id": gt["object_ids"],
            "simulated_states": future.unsqueeze(0).repeat(32, 1, 1, 1),
        }

    def make_gt_with_imbalanced_types(self) -> dict:
        tracks = torch.zeros(5, 91, 9, dtype=torch.float32)
        track_masks = torch.ones(5, 91, dtype=torch.bool)
        object_ids = torch.arange(10, 60, 10, dtype=torch.int32)
        object_types = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
                scenario_pb2.Track.ObjectType.TYPE_CYCLIST,
            ]
        )
        tracks[:, :, 3:6] = torch.tensor([2.0, 1.0, 1.5])
        tracks[:3, :, 0:2] = 0.0
        tracks[3, :, 1] = 10.0
        tracks[4, :, 1] = -10.0
        road_edge = torch.tensor(
            [
                [-20.0, -20.0, 0.0],
                [20.0, -20.0, 0.0],
                [20.0, 20.0, 0.0],
                [-20.0, 20.0, 0.0],
                [-20.0, -20.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return {
            "tracks": tracks,
            "track_masks": track_masks,
            "object_ids": object_ids,
            "object_types": object_types,
            "road_edges": [road_edge],
        }


if __name__ == "__main__":
    unittest.main()
