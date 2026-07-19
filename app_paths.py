from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


def _prefer_existing(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    storage_root: Path
    models_dir: Path
    reports_dir: Path
    feedback_dir: Path
    history_path: Path
    collection_points_path: Path
    model_path: Path
    yolo_model_path: Path
    cleanliness_model_path: Path

    @property
    def uses_external_storage(self) -> bool:
        return self.storage_root.resolve() != self.project_root.resolve()

    def ensure_runtime_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)


def get_app_paths() -> AppPaths:
    storage_root = _env_path("APP_STORAGE_DIR") or PROJECT_ROOT
    external_models_dir = storage_root / "models"
    models_dir = _env_path("APP_MODELS_DIR") or _prefer_existing(
        external_models_dir,
        PROJECT_ROOT / "models",
    )
    external_points = storage_root / "data" / "collection_points.csv"
    collection_points_path = _env_path("APP_COLLECTION_POINTS_PATH") or _prefer_existing(
        external_points,
        PROJECT_ROOT / "data" / "collection_points.csv",
    )

    model_path = _env_path("APP_MODEL_PATH") or _prefer_existing(
        models_dir / "best_model.keras",
        PROJECT_ROOT / "models" / "best_model.keras",
    )
    yolo_model_path = _env_path("APP_YOLO_MODEL_PATH") or _prefer_existing(
        models_dir / "waste_detector.pt",
        PROJECT_ROOT / "models" / "waste_detector.pt",
    )
    cleanliness_model_path = _env_path("APP_CLEANLINESS_MODEL_PATH") or _prefer_existing(
        models_dir / "cleanliness_model.keras",
        PROJECT_ROOT / "models" / "cleanliness_model.keras",
    )

    return AppPaths(
        project_root=PROJECT_ROOT,
        storage_root=storage_root,
        models_dir=models_dir,
        reports_dir=storage_root / "reports",
        feedback_dir=storage_root / "feedback",
        history_path=storage_root / "data" / "user_history.csv",
        collection_points_path=collection_points_path,
        model_path=model_path,
        yolo_model_path=yolo_model_path,
        cleanliness_model_path=cleanliness_model_path,
    )
