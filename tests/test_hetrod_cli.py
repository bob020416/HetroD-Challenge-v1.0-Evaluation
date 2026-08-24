from __future__ import annotations

from contextlib import chdir
from pathlib import Path
import tempfile
import unittest

from hetrod_eval import resolve_rollout_files
from wosac_eval import load_eval_config


class HetrodCliTests(unittest.TestCase):
    def test_metric_config_loads_outside_repository_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with chdir(directory):
                config = load_eval_config("2025")
        self.assertIsNotNone(config)

    def test_resolution_reports_portable_filenames_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir = root / "rollouts"
            gt_dir = root / "gt"
            rollout_dir.mkdir()
            gt_dir.mkdir()
            (rollout_dir / "scenario_a.pkl").touch()
            (rollout_dir / "scenario_extra.pkl").touch()
            (gt_dir / "scenario_a.pkl").touch()
            (gt_dir / "scenario_missing.pkl").touch()

            matched, errors = resolve_rollout_files(rollout_dir, gt_dir)

        self.assertEqual(len(matched), 1)
        self.assertEqual(
            {item["scenario_id"] for item in errors},
            {"extra", "missing"},
        )
        for error in errors:
            for key in ("rollout_file", "gt_file"):
                if key in error:
                    self.assertEqual(error[key], Path(error[key]).name)


if __name__ == "__main__":
    unittest.main()
