from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def assess_image_quality(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    issues = []
    if blur_score < 80:
        issues.append("Ảnh có thể bị mờ")
    if brightness < 55:
        issues.append("Ảnh hơi tối")
    if brightness > 205:
        issues.append("Ảnh quá sáng")
    if contrast < 35:
        issues.append("Độ tương phản thấp")

    if not issues:
        verdict = "Ảnh đủ rõ để dự đoán"
    else:
        verdict = "; ".join(issues)

    return {
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "verdict": verdict,
        "has_warning": bool(issues),
    }


def recycling_cleanliness_hint(class_name: str, quality: dict) -> str:
    recyclable = {"paper", "cardboard", "plastic", "metal", "brown-glass", "green-glass", "white-glass"}
    if class_name not in recyclable:
        return "Không áp dụng chấm điểm độ sạch tái chế cho nhóm này."
    if quality["has_warning"]:
        return "Ảnh chưa đủ rõ để đánh giá sơ bộ độ sạch. Khi demo, nên chụp vật thể rõ nền hơn."
    if quality["contrast"] < 45 or quality["brightness"] < 70:
        return "Cần kiểm tra lại: vật thể có thể bị bẩn hoặc ánh sáng làm khó nhận biết. Nên rửa/làm sạch trước khi tái chế nếu dính thức ăn."
    return "Có thể tái chế nếu vật thể khô, ít dính thức ăn/dầu mỡ và đã tách khỏi vật liệu khác."
