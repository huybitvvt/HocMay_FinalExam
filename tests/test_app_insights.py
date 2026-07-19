from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from app_insights import read_model_summary


class AppInsightsTests(unittest.TestCase):
    def test_dataset_summary_falls_back_to_repo_reports(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models_dir = root / "models"
            runtime_reports = root / "runtime" / "reports"
            repo_reports = root / "repo" / "reports"
            models_dir.mkdir(parents=True)
            runtime_reports.mkdir(parents=True)
            repo_reports.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "model": "efficientnetb0",
                        "accuracy": 0.961,
                        "macro_f1": 0.9456,
                        "weighted_f1": 0.961,
                    }
                ]
            ).to_csv(models_dir / "model_comparison.csv", index=False)
            (models_dir / "class_names.json").write_text(
                json.dumps(["battery", "plastic"]),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {"class": "battery", "count": 10},
                    {"class": "plastic", "count": 20},
                ]
            ).to_csv(repo_reports / "dataset_distribution.csv", index=False)

            summary = read_model_summary(
                models_dir=models_dir,
                reports_dir=runtime_reports,
                fallback_reports_dir=repo_reports,
            )

            self.assertEqual(summary["total_images"], 30)
            self.assertEqual(summary["dataset"]["class"].tolist(), ["battery", "plastic"])


if __name__ == "__main__":
    unittest.main()
