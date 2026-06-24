from __future__ import annotations

import unittest

import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.coverage import compute_coverage
from hetrod_metrics.features import build_feature_bundle


class HetrodCoverageTests(unittest.TestCase):
    def test_identical_rollouts_have_zero_incremental_coverage(self):
        gt = self.make_gt()
        features = build_feature_bundle(
            gt,
            self.make_prediction(gt),
            torch.tensor([True, True, True]),
        )

        report = compute_coverage(features)

        self.assertAlmostEqual(report["score"], 0.0, places=6)
        for type_report in report["by_type"].values():
            self.assertAlmostEqual(type_report["score"], 0.0, places=6)

    def test_diverse_valid_rollouts_increase_coverage(self):
        gt = self.make_gt()
        prediction = self.make_prediction(gt)
        offsets = torch.linspace(-3.0, 3.0, prediction["simulated_states"].shape[0])
        prediction["simulated_states"][:, :, :, 1] += offsets[:, None, None]
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True, True]),
        )

        report = compute_coverage(features)

        self.assertGreater(report["score"], 0.05)
        self.assertEqual(
            set(report["by_type"]),
            {"vehicle", "two_wheeler", "pedestrian"},
        )
        expected = sum(
            type_report["score"] for type_report in report["by_type"].values()
        ) / 3.0
        self.assertAlmostEqual(report["score"], expected, places=6)

    def test_outside_region_modes_do_not_increase_coverage(self):
        gt = self.make_gt()
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][1:, :, :, 0] += 100.0
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True, True]),
        )

        report = compute_coverage(features)

        self.assertAlmostEqual(report["score"], 0.0, places=6)
        for type_report in report["by_type"].values():
            self.assertLessEqual(type_report["mean_valid_occupancy_fraction"], 0.13)

    def test_missing_pedestrian_dimensions_use_default_footprint(self):
        gt = self.make_gt()
        gt["tracks"][2, :, 3:5] = 0.0
        prediction = self.make_prediction(gt)
        offsets = torch.linspace(-1.0, 1.0, prediction["simulated_states"].shape[0])
        prediction["simulated_states"][:, 2, :, 1] += offsets[:, None]
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True, True]),
        )

        report = compute_coverage(features)

        self.assertIsNotNone(report["by_type"]["pedestrian"])
        self.assertGreater(report["by_type"]["pedestrian"]["score"], 0.0)

    def make_gt(self) -> dict:
        tracks = torch.zeros(3, 13, 9, dtype=torch.float32)
        track_masks = torch.ones(3, 13, dtype=torch.bool)
        object_ids = torch.tensor([10, 20, 30], dtype=torch.int32)
        object_types = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_CYCLIST,
                scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
            ],
            dtype=torch.int64,
        )
        tracks[:, :, 3] = torch.tensor([4.0, 2.0, 0.8])[:, None]
        tracks[:, :, 4] = torch.tensor([2.0, 0.8, 0.8])[:, None]
        tracks[:, :, 5] = 1.5
        tracks[:, :, 0] = torch.linspace(-2.0, 2.0, 13)
        tracks[:, :, 1] = torch.tensor([-5.0, 0.0, 5.0])[:, None]
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
            "simulated_states": future.unsqueeze(0).repeat(8, 1, 1, 1),
        }


if __name__ == "__main__":
    unittest.main()
