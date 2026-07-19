from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from waste_rules import VI_LABELS


def load_yolo_model(model_path: str | Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài Ultralytics. Chạy `pip install -r requirements-extended.txt`."
        ) from exc
    return YOLO(str(model_path))


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_yolo_result(result) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = _to_numpy(boxes.xyxy)
    confidences = _to_numpy(boxes.conf)
    class_ids = _to_numpy(boxes.cls).astype(int)
    names = getattr(result, "names", {})

    detections = []
    for xyxy, confidence, class_id in zip(coordinates, confidences, class_ids):
        if isinstance(names, dict):
            class_name = str(names.get(int(class_id), class_id))
        else:
            class_name = str(names[int(class_id)])
        detections.append(
            {
                "bbox": [float(value) for value in xyxy.tolist()],
                "class_id": int(class_id),
                "class_name": class_name,
                "label": VI_LABELS.get(class_name, class_name),
                "confidence": float(confidence),
            }
        )
    return detections


def detect_objects(
    model,
    image: Image.Image,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.50,
) -> list[dict]:
    results = model.predict(
        source=np.asarray(image.convert("RGB")),
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    return parse_yolo_result(results[0]) if results else []


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    line_width = max(2, round(min(output.size) / 180))

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline="#16a34a", width=line_width)
        text = f"{detection['label']} {detection['confidence'] * 100:.1f}%"
        text_box = draw.textbbox((x1, y1), text)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            (x1, text_y, x1 + text_width + 8, text_y + text_height + 6),
            fill="#16a34a",
        )
        draw.text((x1 + 4, text_y + 3), text, fill="white")
    return output
