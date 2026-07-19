from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


RISK_BY_GROUP = {
    "Nguy hại": ("Cao", "Không bỏ vào rác sinh hoạt; ưu tiên đưa tới điểm thu gom chuyên biệt."),
    "Hữu cơ": ("Thấp", "Có thể tách riêng để ủ compost hoặc bỏ vào thùng rác hữu cơ."),
    "Tái chế": ("Trung bình", "Cần làm sạch, giữ khô và tách khỏi rác bẩn trước khi tái chế."),
    "Tái sử dụng": ("Thấp", "Ưu tiên quyên góp, tái sử dụng hoặc đưa tới điểm thu hồi phù hợp."),
    "Rác khác": ("Trung bình", "Kiểm tra lại trước khi bỏ vào rác còn lại để tránh lẫn pin, hóa chất hoặc vật sắc nhọn."),
}


def prediction_status(ranking: list[dict], confidence_threshold: float, margin_threshold: float = 0.15) -> dict:
    top1 = ranking[0]["probability"] if ranking else 0.0
    top2 = ranking[1]["probability"] if len(ranking) > 1 else 0.0
    margin = top1 - top2
    reasons = []
    if top1 < confidence_threshold:
        reasons.append("độ tin cậy thấp")
    if margin < margin_threshold:
        reasons.append("hai nhãn đầu quá gần nhau")

    if reasons:
        return {
            "status": "Cần kiểm tra",
            "message": "Không đủ chắc chắn: " + ", ".join(reasons) + ". Nên chụp lại hoặc kiểm tra thủ công.",
            "margin": margin,
            "is_uncertain": True,
        }
    return {
        "status": "Tin cậy",
        "message": "Dự đoán đủ tin cậy để tham khảo gợi ý xử lý.",
        "margin": margin,
        "is_uncertain": False,
    }


def disposal_conclusion(advice: dict, confidence: float, status: dict) -> dict:
    risk, risk_note = RISK_BY_GROUP.get(advice["group"], RISK_BY_GROUP["Rác khác"])
    if status["is_uncertain"]:
        action = "Chưa nên ra quyết định xử lý chỉ dựa trên ảnh này. Hãy chụp lại rõ hơn hoặc đối chiếu top-3."
    else:
        action = advice["action"]
    return {
        "waste_type": advice["label"],
        "group": advice["group"],
        "risk": risk,
        "risk_note": risk_note,
        "destination": advice["bin"],
        "action": action,
        "confidence_percent": f"{confidence * 100:.2f}%",
        "status": status["status"],
    }


def read_model_summary(
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    fallback_reports_dir: str | Path | None = None,
) -> dict:
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    comparison_path = models_path / "model_comparison.csv"
    classes_path = models_path / "class_names.json"
    if not classes_path.exists():
        classes_path = models_path / "best_model.classes.json"
    dataset_candidates = [
        reports_path / "dataset_distribution.csv",
        models_path / "dataset_distribution.csv",
    ]
    if fallback_reports_dir is not None:
        dataset_candidates.append(Path(fallback_reports_dir) / "dataset_distribution.csv")
    dataset_path = next((path for path in dataset_candidates if path.exists()), None)

    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else []
    dataset = pd.read_csv(dataset_path) if dataset_path else pd.DataFrame()
    if dataset.empty:
        local_data_dir = Path("garbage_classification")
        if local_data_dir.exists():
            rows = []
            for class_dir in sorted(path for path in local_data_dir.iterdir() if path.is_dir()):
                count = sum(
                    1
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTS
                )
                rows.append({"class": class_dir.name, "count": count})
            dataset = pd.DataFrame(rows)
    best = comparison.sort_values("macro_f1", ascending=False).iloc[0].to_dict() if not comparison.empty else {}
    total_images = int(dataset["count"].sum()) if "count" in dataset else None
    return {
        "comparison": comparison,
        "classes": classes,
        "dataset": dataset,
        "best": best,
        "total_images": total_images,
    }


def read_feedback_stats(feedback_dir: str | Path = "feedback") -> dict:
    feedback_path = Path(feedback_dir)
    log_path = feedback_path / "feedback_log.csv"
    if not log_path.exists():
        return {"log": pd.DataFrame(), "total": 0, "top_corrected": pd.DataFrame(), "top_pairs": pd.DataFrame()}

    log = pd.read_csv(log_path)
    top_corrected = log["corrected_class"].value_counts().reset_index()
    top_corrected.columns = ["Nhãn đúng", "Số ảnh"]
    pair_df = log.groupby(["predicted_class", "corrected_class"]).size().reset_index(name="Số lần")
    top_pairs = pair_df.sort_values("Số lần", ascending=False).head(10)
    top_pairs.columns = ["Model dự đoán", "Nhãn đúng", "Số lần"]
    return {
        "log": log,
        "total": len(log),
        "top_corrected": top_corrected,
        "top_pairs": top_pairs,
    }
