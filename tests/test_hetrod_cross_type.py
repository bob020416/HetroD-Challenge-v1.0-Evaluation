from __future__ import annotations

import unittest

import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.cross_type import compute_cross_type_interaction
from hetrod_metrics.features import build_feature_bundle


class HetrodCrossTypeTests(unittest.TestCase):
    def test_gt_as_rollout_scores_one_with_non_selected_context(self):
        gt = self.make_gt()
        selected = torch.tensor([True, False, False])
        features = build_feature_bundle(gt, self.make_prediction(gt), selected)

        report = compute_cross_type_interaction(features)

        self.assertEqual(report["num_cross_type_pairs"], 2)
        self.assertAlmostEqual(report["score"], 1.0, places=6)
        self.assertAlmostEqual(report["distance_proximity_to_gt"], 1.0, places=6)
        self.assertAlmostEqual(report["ttc_proximity_to_gt"], 1.0, places=6)

    def test_cross_type_penalizes_shifted_context_distribution(self):
        gt = self.make_gt()
        selected = torch.tensor([True, False, False])
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1:, :, 0] += 20.0
        features = build_feature_bundle(gt, prediction, selected)

        report = compute_cross_type_interaction(features)

        self.assertLess(report["distance_proximity_to_gt"], 1.0)
        self.assertLess(
            report["pair_type_scores"]["vehicle_pedestrian"]["distance_event_similarity"],
            1.0,
        )
        self.assertEqual(
            report["scoring_method"],
            "pairwise_type_balanced_histogram_similarity",
        )
        self.assertAlmostEqual(
            report["pair_type_scores"]["vehicle_pedestrian"][
                "distance_proximity_to_gt"
            ],
            report["pair_type_scores"]["vehicle_pedestrian"][
                "distance_distribution_similarity"
            ],
            places=6,
        )

    def test_pair_types_receive_equal_weight(self):
        gt = self.make_gt()
        selected = torch.tensor([True, False, False])
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1, :, 0] += 20.0
        features = build_feature_bundle(gt, prediction, selected)

        report = compute_cross_type_interaction(features)
        type_scores = report["pair_type_scores"]

        self.assertEqual(
            set(type_scores),
            {"vehicle_pedestrian", "vehicle_two_wheeler"},
        )
        expected_distance = sum(
            item["distance_proximity_to_gt"] for item in type_scores.values()
        ) / len(type_scores)
        expected_ttc = sum(
            item["ttc_proximity_to_gt"] for item in type_scores.values()
        ) / len(type_scores)
        self.assertAlmostEqual(
            report["distance_proximity_to_gt"],
            expected_distance,
            places=6,
        )
        self.assertAlmostEqual(
            report["ttc_proximity_to_gt"],
            expected_ttc,
            places=6,
        )

    def test_selected_anchor_pairs_are_counted_once(self):
        gt = self.make_gt()
        selected = torch.tensor([True, True, True])
        features = build_feature_bundle(gt, self.make_prediction(gt), selected)

        report = compute_cross_type_interaction(features)

        self.assertEqual(report["num_cross_type_pairs"], 3)
        self.assertEqual(report["num_included_pairs"], 3)
        self.assertEqual(
            sum(item["num_pairs"] for item in report["pair_type_scores"].values()),
            3,
        )
        self.assertAlmostEqual(report["score"], 1.0, places=6)

    def test_larger_interaction_error_scores_lower(self):
        gt = self.make_gt()
        selected = torch.tensor([True, False, False])
        mild_prediction = self.make_prediction(gt)
        severe_prediction = self.make_prediction(gt)
        mild_prediction["simulated_states"][:, 1, :, 0] += 0.5
        severe_prediction["simulated_states"][:, 1, :, 0] += 20.0

        mild_report = compute_cross_type_interaction(
            build_feature_bundle(gt, mild_prediction, selected)
        )
        severe_report = compute_cross_type_interaction(
            build_feature_bundle(gt, severe_prediction, selected)
        )

        self.assertGreater(mild_report["score"], severe_report["score"])

    def test_missing_non_selected_context_is_rejected(self):
        gt = self.make_gt()
        selected = torch.tensor([True, False, False])
        prediction = self.make_prediction(gt)
        prediction = {
            "agent_id": prediction["agent_id"][:1],
            "simulated_states": prediction["simulated_states"][:, :1],
        }

        with self.assertRaisesRegex(ValueError, "exactly match all GT object_ids"):
            build_feature_bundle(gt, prediction, selected)

    def make_gt(self) -> dict:
        tracks = torch.zeros(3, 91, 9, dtype=torch.float32)
        track_masks = torch.ones(3, 91, dtype=torch.bool)
        object_ids = torch.tensor([10, 20, 30], dtype=torch.int32)
        object_types = torch.tensor(
            [
                scenario_pb2.Track.ObjectType.TYPE_VEHICLE,
                scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN,
                scenario_pb2.Track.ObjectType.TYPE_CYCLIST,
            ],
            dtype=torch.int64,
        )
        for i in range(3):
            tracks[i, :, 0] = torch.linspace(0.0, 8.0, 91)
            tracks[i, :, 1] = float(i + 1) * 2.0
            tracks[i, :, 3] = 2.0
            tracks[i, :, 4] = 1.0
            tracks[i, :, 5] = 1.5
        return {
            "tracks": tracks,
            "track_masks": track_masks,
            "object_ids": object_ids,
            "object_types": object_types,
        }

    def make_prediction(self, gt: dict) -> dict:
        future = gt["tracks"][:, 11:, [0, 1, 2, 6]]
        return {
            "agent_id": gt["object_ids"],
            "simulated_states": future.unsqueeze(0).repeat(32, 1, 1, 1),
        }


if __name__ == "__main__":
    unittest.main()
