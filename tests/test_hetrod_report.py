from __future__ import annotations

import math
import unittest

from hetrod_metrics.report import aggregate_scenario_reports, compute_overall_score


class HetrodReportTests(unittest.TestCase):
    def test_overall_uses_quality_gated_coverage_bonus(self):
        report = compute_overall_score(
            {"score": 0.8},
            {"score": 0.9},
            {"score": 0.7},
            {"score": 0.6},
        )

        self.assertAlmostEqual(
            report["score"],
            0.30 * 0.8
            + 0.35 * 0.9
            + 0.25 * 0.7
            + 0.10 * 0.6 * 0.8 * 0.9,
            places=6,
        )
        self.assertAlmostEqual(
            report["weighted_components"]["coverage_bonus"],
            0.10 * 0.6 * 0.8 * 0.9,
            places=6,
        )

    def test_dataset_aggregation_weights_within_type_then_macro_averages_types(self):
        first = self.make_report(
            vehicle_score=1.0,
            vehicle_samples=100,
            pedestrian_score=0.25,
            pedestrian_samples=1,
        )
        second = self.make_report(
            vehicle_score=0.25,
            vehicle_samples=100,
            pedestrian_score=1.0,
            pedestrian_samples=1,
        )

        report = aggregate_scenario_reports([first, second])

        kinematic = report["kinematic_realism"]["metrics"]["linear_speed"]
        self.assertAlmostEqual(kinematic["by_type"]["vehicle"], 0.5, places=6)
        self.assertAlmostEqual(kinematic["by_type"]["pedestrian"], 0.5, places=6)
        self.assertAlmostEqual(kinematic["score"], 0.5, places=6)
        self.assertEqual(
            report["metadata"]["aggregation"],
            "dataset_level_sufficient_statistics",
        )

    def test_dataset_aggregation_ignores_skipped_no_selected_agents(self):
        valid = self.make_report(
            vehicle_score=1.0,
            vehicle_samples=10,
            pedestrian_score=1.0,
            pedestrian_samples=10,
        )
        skipped = {
            "status": "skipped_no_selected_agents",
            "scenario_id": "empty_selection",
        }

        report = aggregate_scenario_reports([skipped, valid])

        self.assertEqual(report["num_scenarios"], 1)
        self.assertAlmostEqual(report["score"], 1.0, places=6)

    def make_report(
        self,
        *,
        vehicle_score: float,
        vehicle_samples: int,
        pedestrian_score: float,
        pedestrian_samples: int,
    ) -> dict:
        type_scores = {
            "vehicle": vehicle_score,
            "two_wheeler": None,
            "pedestrian": pedestrian_score,
        }
        type_samples = {
            "vehicle": vehicle_samples,
            "two_wheeler": 0,
            "pedestrian": pedestrian_samples,
        }
        kinematic_metrics = {
            name: {
                "score": 0.5 * (vehicle_score + pedestrian_score),
                "by_type": type_scores,
                "log_ratio_by_type": {
                    "vehicle": math.log(max(vehicle_score, 1e-6)),
                    "two_wheeler": None,
                    "pedestrian": math.log(max(pedestrian_score, 1e-6)),
                },
                "num_samples_by_type": type_samples,
            }
            for name in (
                "linear_speed",
                "linear_acceleration",
                "angular_speed",
                "angular_acceleration",
            )
        }
        safety_type = {
            "vehicle": {
                "score": vehicle_score,
                "num_samples": vehicle_samples,
            },
            "two_wheeler": None,
            "pedestrian": {
                "score": pedestrian_score,
                "num_samples": pedestrian_samples,
            },
        }
        coverage_type = {
            "vehicle": {"score": vehicle_score, "num_agents": 10},
            "two_wheeler": None,
            "pedestrian": {"score": pedestrian_score, "num_agents": 1},
        }
        return {
            "kinematic_realism": {
                "score": 0.5 * (vehicle_score + pedestrian_score),
                "metrics": kinematic_metrics,
            },
            "safety": {
                "collision_with_annotation_tolerance": {"by_type": safety_type},
                "valid_region_margin": {"by_type": safety_type},
            },
            "cross_type_interaction": {
                "pair_type_scores": {
                    "vehicle_pedestrian": {
                        "distance_proximity_to_gt": vehicle_score,
                        "time_to_proximity_to_gt": pedestrian_score,
                        "num_pairs": 2,
                    }
                }
            },
            "coverage": {"by_type": coverage_type},
        }


if __name__ == "__main__":
    unittest.main()
