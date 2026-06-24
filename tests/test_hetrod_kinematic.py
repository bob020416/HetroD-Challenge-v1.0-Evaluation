from __future__ import annotations

import unittest

import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.features import build_feature_bundle
from hetrod_metrics.kinematic import compute_kinematic_realism, normalize_kinematic_realism
from wosac_eval import load_eval_config


class HetrodKinematicTests(unittest.TestCase):
    def test_kinematic_realism_scores_selected_agents_by_type(self):
        gt = self.make_gt()
        selected = torch.tensor([True, True, True])
        prediction = self.make_prediction(gt)

        features = build_feature_bundle(gt, prediction, selected)
        report = compute_kinematic_realism(load_eval_config("2025"), features)

        self.assertIsNotNone(report["score"])
        self.assertGreaterEqual(report["score"], 0.0)
        self.assertLessEqual(report["score"], 1.0)
        for metric in report["metrics"].values():
            self.assertIsNotNone(metric["score"])
            self.assertEqual(set(metric["by_type"].keys()), {"vehicle", "two_wheeler", "pedestrian"})

    def test_gt_as_rollout_oracle_scores_near_one(self):
        gt = self.make_gt()
        selected = torch.tensor([True, True, True])
        prediction = self.make_prediction(gt)

        features = build_feature_bundle(gt, prediction, selected)
        report = compute_kinematic_realism(load_eval_config("2025"), features)

        # WOSAC's histogram estimators use binning and smoothing, so even an
        # exact GT-as-rollout oracle is high but not exactly 1.0.
        self.assertGreater(report["score"], 0.97)
        for metric_name, metric in report["metrics"].items():
            self.assertGreater(metric["score"], 0.97, metric_name)
            for type_name, type_score in metric["by_type"].items():
                self.assertIsNotNone(type_score, type_name)
                self.assertGreater(type_score, 0.97, f"{metric_name}:{type_name}")

    def test_oracle_normalized_kinematic_scores_gt_as_one(self):
        gt = self.make_gt()
        selected = torch.tensor([True, True, True])
        prediction = self.make_prediction(gt)

        features = build_feature_bundle(gt, prediction, selected)
        raw = compute_kinematic_realism(load_eval_config("2025"), features)
        normalized = normalize_kinematic_realism(raw, raw)

        self.assertAlmostEqual(normalized["score"], 1.0, places=6)
        for metric in normalized["metrics"].values():
            self.assertAlmostEqual(metric["score"], 1.0, places=6)
            for type_score in metric["by_type"].values():
                self.assertAlmostEqual(type_score, 1.0, places=6)

    def make_gt(self) -> dict:
        tracks = torch.zeros(3, 91, 9, dtype=torch.float32)
        track_masks = torch.ones(3, 91, dtype=torch.bool)
        object_ids = torch.tensor([10, 20, 30], dtype=torch.int32)
        object_types = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_CYCLIST,
                scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
            ],
            dtype=torch.int64,
        )
        for i in range(3):
            tracks[i, :, 0] = torch.linspace(0.0, 9.0 + i, 91)
            tracks[i, :, 1] = float(i) * 3.0
            tracks[i, :, 3] = 4.0
            tracks[i, :, 4] = 2.0
            tracks[i, :, 5] = 1.5
            tracks[i, :, 7] = 1.0 + 0.1 * i
        return {
            "tracks": tracks,
            "track_masks": track_masks,
            "object_ids": object_ids,
            "object_types": object_types,
        }

    def make_prediction(self, gt: dict) -> dict:
        future = gt["tracks"][:, 11:, [0, 1, 2, 6]]
        simulated_states = future.unsqueeze(0).repeat(32, 1, 1, 1)
        return {
            "agent_id": gt["object_ids"],
            "simulated_states": simulated_states,
        }


if __name__ == "__main__":
    unittest.main()
