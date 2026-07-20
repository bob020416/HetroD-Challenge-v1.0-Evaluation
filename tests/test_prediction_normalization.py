from __future__ import annotations

import unittest

import torch

from wosac_eval import normalize_prediction


class PredictionNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.rollout = {
            "agents_id": [10, 20, 30],
            "sim_agent_mask": [True, False, True],
            "model_rollouts": {
                "baseline": {
                    "rollouts": torch.zeros(32, 3, 80, 4),
                }
            },
        }

    def test_wosac_default_applies_sim_agent_mask(self):
        prediction = normalize_prediction(
            self.rollout,
            device="cpu",
            rollout_key="baseline",
        )

        self.assertEqual(prediction["agent_id"].tolist(), [10, 30])
        self.assertEqual(tuple(prediction["simulated_states"].shape), (32, 2, 80, 4))

    def test_hetrod_can_keep_all_saved_agents(self):
        prediction = normalize_prediction(
            self.rollout,
            device="cpu",
            rollout_key="baseline",
            apply_sim_agent_mask=False,
        )

        self.assertEqual(prediction["agent_id"].tolist(), [10, 20, 30])
        self.assertEqual(tuple(prediction["simulated_states"].shape), (32, 3, 80, 4))


if __name__ == "__main__":
    unittest.main()
