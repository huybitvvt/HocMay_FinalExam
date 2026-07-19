from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CLASS_NAMES = ["clean", "dirty", "uncertain"]

CLEANLINESS_LABELS_VI = {
    "clean": "Sạch",
    "dirty": "Có thể bị bẩn",
    "uncertain": "Không chắc chắn",
}

RECYCLABLE_CLASSES = {
    "paper",
    "cardboard",
    "plastic",
    "metal",
    "brown-glass",
    "green-glass",
    "white-glass",
}


def load_cleanliness_class_names(model_path: str | Path) -> list[str]:
    path = Path(model_path)
    sidecar = path.with_suffix(".classes.json")
    if not sidecar.exists():
        return DEFAULT_CLASS_NAMES.copy()
    names = json.loads(sidecar.read_text(encoding="utf-8"))
    return [str(name) for name in names]


def predict_cleanliness(
    model,
    image: Image.Image,
    class_names: list[str] | None = None,
) -> dict:
    names = class_names or DEFAULT_CLASS_NAMES
    input_shape = getattr(model, "input_shape", (None, 224, 224, 3))
    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)
    resized = image.convert("RGB").resize((width, height))
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    probabilities = np.asarray(model.predict(batch, verbose=0))[0].astype(float)

    if len(probabilities) != len(names):
        raise ValueError(
            f"Model trả về {len(probabilities)} lớp nhưng file nhãn có {len(names)} lớp."
        )

    order = np.argsort(probabilities)[::-1]
    ranking = [
        {
            "class_name": names[int(index)],
            "label": CLEANLINESS_LABELS_VI.get(names[int(index)], names[int(index)]),
            "probability": float(probabilities[int(index)]),
        }
        for index in order
    ]
    best = ranking[0]
    return {
        "class_name": best["class_name"],
        "label": best["label"],
        "confidence": best["probability"],
        "ranking": ranking,
    }


def cleanliness_advice(
    waste_class: str,
    result: dict | None,
    confidence_threshold: float = 0.65,
) -> str:
    if waste_class not in RECYCLABLE_CLASSES:
        return "Không áp dụng model độ sạch cho loại rác này."
    if result is None:
        return "Chưa có model độ sạch; chỉ có thể nhắc người dùng rửa và giữ khô trước khi tái chế."
    if result["confidence"] < confidence_threshold or result["class_name"] == "uncertain":
        return "Model chưa xác định rõ độ sạch. Nên kiểm tra trực tiếp và rửa sơ nếu có thức ăn hoặc dầu mỡ."
    if result["class_name"] == "dirty":
        return "Vật thể có dấu hiệu bị bẩn. Cần loại bỏ thức ăn, rửa sơ và để khô trước khi tái chế."
    return "Vật thể có dấu hiệu đủ sạch để phân loại; vẫn cần kiểm tra không còn thức ăn, dầu mỡ hoặc vật liệu lẫn."
