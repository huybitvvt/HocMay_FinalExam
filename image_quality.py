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


def multi_object_hint(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("RGB"))
    small = cv2.resize(arr, (320, 320))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = small.shape[0] * small.shape[1]
    large_contours = [
        contour
        for contour in contours
        if 0.015 <= cv2.contourArea(contour) / image_area <= 0.45
    ]
    object_like_count = len(large_contours)
    has_warning = object_like_count >= 4
    if has_warning:
        message = "Ảnh có thể chứa nhiều vật thể. Kết quả chỉ áp dụng cho vật thể nổi bật nhất."
    else:
        message = "Không phát hiện dấu hiệu rõ ràng của nhiều vật thể."
    return {
        "object_like_count": object_like_count,
        "has_warning": has_warning,
        "message": message,
    }
