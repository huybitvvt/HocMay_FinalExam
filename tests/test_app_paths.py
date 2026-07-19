from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app_paths import PROJECT_ROOT, get_app_paths


PATH_ENV_NAMES = [
    "APP_STORAGE_DIR",
    "APP_MODELS_DIR",
    "APP_MODEL_PATH",
    "APP_YOLO_MODEL_PATH",
    "APP_CLEANLINESS_MODEL_PATH",
    "APP_COLLECTION_POINTS_PATH",
]


class AppPathsTests(unittest.TestCase):
    def test_defaults_to_project_paths(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in PATH_ENV_NAMES:
                os.environ.pop(name, None)
            paths = get_app_paths()

        self.assertEqual(paths.storage_root, PROJECT_ROOT)
        self.assertEqual(paths.feedback_dir, PROJECT_ROOT / "feedback")
        self.assertEqual(paths.history_path, PROJECT_ROOT / "data" / "user_history.csv")
        self.assertFalse(paths.uses_external_storage)

    def test_external_storage_routes_runtime_outputs(self):
        with TemporaryDirectory() as temp_dir:
            storage = Path(temp_dir)
            model = storage / "models" / "custom.keras"
            with patch.dict(
                os.environ,
                {
                    "APP_STORAGE_DIR": str(storage),
                    "APP_MODEL_PATH": str(model),
                },
                clear=False,
            ):
                paths = get_app_paths()
                paths.ensure_runtime_dirs()

            self.assertEqual(paths.model_path, model)
            self.assertEqual(paths.reports_dir, storage / "reports")
            self.assertEqual(paths.feedback_dir, storage / "feedback")
            self.assertEqual(paths.history_path, storage / "data" / "user_history.csv")
            self.assertTrue(paths.reports_dir.exists())
            self.assertTrue(paths.feedback_dir.exists())
            self.assertTrue(paths.history_path.parent.exists())
            self.assertTrue(paths.uses_external_storage)


if __name__ == "__main__":
    unittest.main()
