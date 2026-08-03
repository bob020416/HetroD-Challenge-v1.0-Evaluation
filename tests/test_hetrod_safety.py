from __future__ import annotations

import unittest

import torch
from waymo_open_dataset.protos import scenario_pb2

from hetrod_metrics.features import build_feature_bundle
from hetrod_metrics.safety import collision_rollout_rate, compute_safety


class HetrodSafetyTests(unittest.TestCase):
    def test_safety_scores_clean_rollouts_high(self):
        gt = self.make_gt()
        features = build_feature_bundle(gt, self.make_prediction(gt), torch.tensor([True, True]))

        report = compute_safety(features)

        self.assertGreater(report["score"], 0.99)
        self.assertGreater(report["collision_rollout_rate"]["score"], 0.99)
        self.assertGreater(report["valid_region_margin"]["score"], 0.99)

    def test_collision_rollout_rate_penalizes_overlap(self):
        gt = self.make_gt()
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1, :, 0] = prediction["simulated_states"][:, 0, :, 0]
        prediction["simulated_states"][:, 1, :, 1] = prediction["simulated_states"][:, 0, :, 1]
        features = build_feature_bundle(gt, prediction, torch.tensor([True, True]))

        report = collision_rollout_rate(features)

        self.assertLess(report["score"], 0.5)
        self.assertGreater(report["unsafe_rate"], 0.5)

    def test_one_colliding_frame_marks_the_agent_rollout_collided(self):
        gt = self.make_gt()
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1, 25, 0:2] = prediction[
            "simulated_states"
        ][:, 0, 25, 0:2]
        features = build_feature_bundle(gt, prediction, torch.tensor([True, True]))

        report = collision_rollout_rate(features)

        self.assertGreater(report["collision_rate"], 0.99)
        self.assertEqual(report["num_collided_agent_rollouts"], 64)

    def test_gt_pair_overlap_does_not_lower_replay_score(self):
        gt = self.make_gt()
        gt["tracks"][1, 36, 0:2] = gt["tracks"][0, 36, 0:2]
        features = build_feature_bundle(
            gt,
            self.make_prediction(gt),
            torch.tensor([True, True]),
        )

        report = collision_rollout_rate(features)

        self.assertAlmostEqual(report["score"], 1.0, places=6)
        self.assertEqual(report["num_collided_agent_rollouts"], 0)
        self.assertGreater(report["num_raw_collided_agent_rollouts"], 0)
        self.assertGreater(report["num_gt_collided_agent_rollouts"], 0)

    def test_touching_box_edges_are_not_a_collision(self):
        gt = self.make_gt()
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1, :, 0] = prediction[
            "simulated_states"
        ][:, 0, :, 0]
        # Both boxes are 1 m wide, so a 1 m center separation only touches.
        prediction["simulated_states"][:, 1, :, 1] = 1.0
        prediction["simulated_states"][:, 0, :, 1] = 0.0
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True]),
        )

        report = collision_rollout_rate(features)

        self.assertAlmostEqual(report["score"], 1.0, places=6)
        self.assertEqual(report["num_collided_agent_rollouts"], 0)

    def test_valid_region_scores_gt_replay_one_even_when_gt_is_outside(self):
        gt = self.make_gt()
        gt["tracks"][0, 11:, 1] = 30.0
        features = build_feature_bundle(
            gt, self.make_prediction(gt), torch.tensor([True, True])
        )

        report = compute_safety(features)["valid_region_margin"]

        self.assertAlmostEqual(report["score"], 1.0, places=6)
        self.assertGreater(report["gt_outside_rate"], 0.0)

    def test_type_specific_polygon_takes_precedence_over_road_edges(self):
        gt = self.make_gt()
        gt["valid_regions"] = self.make_valid_regions()
        gt["valid_region_definition"] = {
            "schema_version": "1.2",
            "boundary_margin_m_by_agent_type": {
                "vehicle": 0.75,
                "cyclist": 0.75,
                "pedestrian": 2.0,
            },
        }
        prediction = self.make_prediction(gt)
        # This remains inside the broad road edge but leaves the pedestrian region.
        prediction["simulated_states"][:, 1, :, 1] = 0.0
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True]),
        )

        report = compute_safety(features)["valid_region_margin"]

        self.assertEqual(report["valid_region_source"], "type_specific_polygon")
        self.assertAlmostEqual(
            report["by_type"]["vehicle"]["score"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            report["by_type"]["pedestrian"]["score"],
            0.0,
            places=6,
        )
        self.assertEqual(report["polygon_schema_version"], "1.2")

    def test_gt_outside_frames_cannot_cancel_another_agents_error(self):
        gt = self.make_gt()
        gt["object_types"][:] = scenario_pb2.Track.ObjectType.TYPE_VEHICLE
        gt["valid_regions"] = self.make_valid_regions()
        gt["valid_region_definition"] = {
            "schema_version": "1.2",
            "boundary_margin_m_by_agent_type": {
                "vehicle": 0.75,
                "cyclist": 0.75,
                "pedestrian": 2.0,
            },
        }
        # Agent 0 is valid and inside in GT; agent 1 is outside in GT.
        gt["tracks"][0, :, 1] = 0.0
        gt["tracks"][1, :, 1] = 8.0
        prediction = self.make_prediction(gt)
        # Swap their future lateral positions. The aggregate outside rate is
        # unchanged, but the GT-inside agent is now wrong.
        prediction["simulated_states"][:, 0, :, 1] = 8.0
        prediction["simulated_states"][:, 1, :, 1] = 0.0
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True]),
        )

        report = compute_safety(features)["valid_region_margin"]

        self.assertAlmostEqual(report["by_type"]["vehicle"]["score"], 0.0, places=6)
        self.assertAlmostEqual(
            report["by_type"]["vehicle"]["outside_rate_on_gt_inside_frames"],
            1.0,
            places=6,
        )
        self.assertGreater(report["by_type"]["vehicle"]["gt_outside_rate"], 0.0)

    def test_transition_aware_gt_replay_scores_one_and_excludes_unsupported(self):
        gt = self.make_gt()
        gt["valid_regions"] = self.make_transition_valid_regions()
        gt["valid_region_definition"] = self.make_transition_definition()
        # One valid pedestrian frame is outside every mapped semantic layer.
        gt["tracks"][1, 50, 0:2] = torch.tensor([30.0, 30.0])
        features = build_feature_bundle(
            gt,
            self.make_prediction(gt),
            torch.tensor([True, True]),
        )

        report = compute_safety(features)["valid_region_margin"]
        pedestrian = report["by_type"]["pedestrian"]

        self.assertEqual(
            report["valid_region_source"],
            "type_specific_polygon_transition_aware",
        )
        self.assertAlmostEqual(pedestrian["score"], 1.0, places=6)
        self.assertGreater(pedestrian["excluded_map_unsupported_rate"], 0.0)
        self.assertEqual(pedestrian["num_spatial_offroad_frames"], 0)
        self.assertEqual(pedestrian["num_transition_overstay_frames"], 0)

    def test_transition_aware_region_penalizes_spatial_detour(self):
        gt = self.make_gt()
        gt["valid_regions"] = self.make_transition_valid_regions()
        gt["valid_region_definition"] = self.make_transition_definition()
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1, :, 1] = 20.0
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True]),
        )

        pedestrian = compute_safety(features)["valid_region_margin"]["by_type"][
            "pedestrian"
        ]

        self.assertLess(pedestrian["score"], 0.1)
        self.assertGreater(pedestrian["spatial_offroad_rate"], 0.9)

    def test_transition_aware_region_penalizes_transition_overstay(self):
        gt = self.make_gt()
        gt["valid_regions"] = self.make_transition_valid_regions()
        gt["valid_region_definition"] = self.make_transition_definition()
        # GT reaches the permanent core after crossing the road.
        gt["tracks"][1, 31:, 0] = 4.0
        prediction = self.make_prediction(gt)
        # The rollout remains in the crosswalk transition for the full horizon.
        prediction["simulated_states"][:, 1, :, 0] = 0.0
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True]),
        )

        pedestrian = compute_safety(features)["valid_region_margin"]["by_type"][
            "pedestrian"
        ]

        self.assertGreater(pedestrian["num_transition_overstay_frames"], 0)
        self.assertGreater(pedestrian["transition_overstay_rate"], 0.0)
        self.assertLess(pedestrian["score"], 1.0)

    def test_collision_checks_selected_anchor_against_non_selected_context(self):
        gt = self.make_gt(num_agents=3)
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, 1, :, 0] = prediction["simulated_states"][:, 0, :, 0]
        prediction["simulated_states"][:, 1, :, 1] = prediction["simulated_states"][:, 0, :, 1]
        features = build_feature_bundle(gt, prediction, torch.tensor([True, False, False]))

        report = collision_rollout_rate(features)

        self.assertEqual(report["num_context_agents"], 3)
        self.assertLess(report["score"], 0.5)

    def test_safety_macro_averages_agent_types(self):
        gt = self.make_gt_with_imbalanced_types()
        prediction = self.make_prediction(gt)
        prediction["simulated_states"][:, :3, :, 0:2] = 0.0
        features = build_feature_bundle(
            gt,
            prediction,
            torch.tensor([True, True, True, True, True]),
        )

        report = collision_rollout_rate(features)

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

    def make_valid_regions(self) -> dict:
        def record(min_x: float, min_y: float, max_x: float, max_y: float):
            return {
                "exterior": torch.tensor(
                    [
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y],
                        [min_x, min_y],
                    ],
                    dtype=torch.float32,
                ),
                "holes": [],
            }

        return {
            "vehicle": [record(-20.0, -2.0, 20.0, 2.0)],
            "cyclist": [record(-20.0, -12.0, 20.0, -6.0)],
            "pedestrian": [record(-20.0, 6.0, 20.0, 10.0)],
        }

    def make_transition_valid_regions(self) -> dict:
        regions = self.make_valid_regions()

        def record(min_x: float, min_y: float, max_x: float, max_y: float):
            return {
                "exterior": torch.tensor(
                    [
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y],
                        [min_x, min_y],
                    ],
                    dtype=torch.float32,
                ),
                "holes": [],
            }

        regions.update(
            {
                "pedestrian_core": [record(2.0, 6.0, 20.0, 10.0)],
                "pedestrian_crosswalk": [record(-1.0, 6.0, 1.0, 10.0)],
                "pedestrian_road": [record(-20.0, 5.0, 2.0, 11.0)],
            }
        )
        return regions

    def make_transition_definition(self) -> dict:
        return {
            "schema_version": "1.3",
            "policy": "type-specific-v3-pedestrian-transition-aware",
            "pedestrian_gt_road_corridor_margin_m": 1.5,
            "pedestrian_transition_time_slack_s": 1.0,
            "boundary_margin_m_by_region": {
                "vehicle": 0.75,
                "cyclist": 0.75,
                "pedestrian_core": 0.75,
                "pedestrian_crosswalk": 1.5,
                "pedestrian_road": 0.0,
            },
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
        tracks[0, :, 1] = 0.0
        tracks[1, :, 1] = 3.0
        tracks[2, :, 1] = 6.0
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
