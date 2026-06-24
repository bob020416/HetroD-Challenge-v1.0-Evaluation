from __future__ import annotations

import unittest

import torch

from hetrod_metrics.features import validate_prediction


class HetrodValidationTests(unittest.TestCase):
    def setUp(self):
        self.gt = {
            "tracks": torch.zeros(3, 91, 9),
            "object_ids": torch.tensor([10, 20, 30], dtype=torch.int32),
        }
        self.prediction = {
            "agent_id": self.gt["object_ids"].clone(),
            "simulated_states": torch.zeros(32, 3, 80, 4),
        }

    def test_complete_submission_is_accepted(self):
        validate_prediction(self.gt, self.prediction)

    def test_missing_agent_is_rejected(self):
        prediction = {
            "agent_id": self.prediction["agent_id"][:-1],
            "simulated_states": self.prediction["simulated_states"][:, :-1],
        }
        with self.assertRaisesRegex(ValueError, "exactly match all GT object_ids"):
            validate_prediction(self.gt, prediction)

    def test_duplicate_agent_id_is_rejected(self):
        prediction = {
            **self.prediction,
            "agent_id": torch.tensor([10, 10, 30]),
        }
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_prediction(self.gt, prediction)

    def test_non_finite_state_is_rejected(self):
        prediction = {
            **self.prediction,
            "simulated_states": self.prediction["simulated_states"].clone(),
        }
        prediction["simulated_states"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            validate_prediction(self.gt, prediction)

    def test_wrong_future_length_is_rejected(self):
        prediction = {
            **self.prediction,
            "simulated_states": torch.zeros(32, 3, 79, 4),
        }
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_prediction(self.gt, prediction)

    def test_wrong_num_rollouts_is_rejected(self):
        prediction = {
            **self.prediction,
            "simulated_states": torch.zeros(31, 3, 80, 4),
        }
        with self.assertRaisesRegex(ValueError, "exactly 32 rollouts"):
            validate_prediction(self.gt, prediction)


if __name__ == "__main__":
    unittest.main()
