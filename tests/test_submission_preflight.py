from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from hetrod_metrics.submission import (
    preflight_submission,
    validate_rollout_payload,
)


class SubmissionPreflightTests(unittest.TestCase):
    def payload(self):
        return {
            "agent_id": torch.tensor([10, 20]),
            "simulated_states": torch.zeros(32, 2, 80, 4),
        }

    def test_payload_contract(self):
        stats = validate_rollout_payload("scene", self.payload(), [20, 10])
        self.assertEqual(stats["num_rollouts"], 32)

    def test_payload_rejects_nan(self):
        payload = self.payload()
        payload["simulated_states"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            validate_rollout_payload("scene", payload, [10, 20])

    def test_payload_allows_additional_metadata_keys(self):
        payload = self.payload()
        payload["scenario_id"] = "scene"
        stats = validate_rollout_payload("scene", payload, [10, 20])
        self.assertEqual(stats["num_agents"], 2)

    def test_complete_npz_zip_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            (public / "manifests").mkdir(parents=True)
            (public / "test" / "input").mkdir(parents=True)
            (public / "manifests" / "test.txt").write_text("scene\n")
            (public / "manifests" / "test_input_paths.txt").write_text(
                "test/input/scene.pkl\n"
            )
            with (public / "test" / "input" / "scene.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "id": "scene",
                        "metadata": {"required_agent_ids": [10, 20]},
                    },
                    handle,
                )
            payload_path = root / "scene.npz"
            import numpy as np
            np.savez(
                payload_path,
                agent_id=self.payload()["agent_id"].numpy(),
                simulated_states=self.payload()["simulated_states"].numpy(),
            )
            submission = root / "submission.zip"
            with zipfile.ZipFile(submission, "w") as archive:
                archive.write(payload_path, "team/scene.npz")

            report = preflight_submission(submission, public)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["summary"]["num_valid_scenarios"], 1)

    def test_original_pkl_zip_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            (public / "manifests").mkdir(parents=True)
            (public / "test" / "input").mkdir(parents=True)
            (public / "manifests" / "test.txt").write_text("scene\n")
            (public / "manifests" / "test_input_paths.txt").write_text(
                "test/input/scene.pkl\n"
            )
            with (public / "test" / "input" / "scene.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "id": "scene",
                        "metadata": {"required_agent_ids": [10, 20]},
                    },
                    handle,
                )
            payload_path = root / "scene.pkl"
            with payload_path.open("wb") as handle:
                pickle.dump(self.payload(), handle)
            submission = root / "submission.zip"
            with zipfile.ZipFile(submission, "w") as archive:
                archive.write(payload_path, "team/scene.pkl")

            report = preflight_submission(submission, public)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["summary"]["num_valid_scenarios"], 1)
        self.assertEqual(report["summary"]["file_format_counts"], {".pkl": 1})


if __name__ == "__main__":
    unittest.main()
