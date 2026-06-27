from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default="reports")
    return parser.parse_args()


def load_report(models_dir: Path, model_name: str) -> pd.DataFrame:
    path = models_dir / f"{model_name}_classification_report.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file report: {path}")
    return pd.read_csv(path, index_col=0)


def load_confusion_matrix(models_dir: Path, model_name: str) -> pd.DataFrame:
    path = models_dir / f"{model_name}_confusion_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không thấy confusion matrix: {path}")
    return pd.read_csv(path, index_col=0)


def hardest_classes(report: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    metric_rows = {"accuracy", "macro avg", "weighted avg"}
    class_rows = report.loc[[idx for idx in report.index if idx not in metric_rows]].copy()
    class_rows["support"] = class_rows["support"].astype(int)
    return class_rows.sort_values(["f1-score", "support"], ascending=[True, False]).head(top_n)


def top_confusions(cm: pd.DataFrame, top_n: int = 8) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for true_label in cm.index:
        for pred_label in cm.columns:
            if true_label == pred_label:
                continue
            value = int(cm.loc[true_label, pred_label])
            if value > 0:
                pairs.append((true_label, pred_label, value))
    return sorted(pairs, key=lambda item: item[2], reverse=True)[:top_n]


def quality_comment(row: pd.Series) -> list[str]:
    comments = []
    accuracy = float(row["accuracy"])
    macro_f1 = float(row["macro_f1"])
    weighted_f1 = float(row["weighted_f1"])

    if accuracy >= 0.9:
        comments.append("Accuracy cao, mô hình phân loại tốt trên tập validation.")
    elif accuracy >= 0.8:
        comments.append("Accuracy khá, có thể bảo vệ được nhưng nên phân tích lỗi rõ.")
    else:
        comments.append("Accuracy còn thấp, cần tăng epoch, kiểm tra dữ liệu hoặc augmentation.")

    if macro_f1 >= 0.85:
        comments.append("Macro-F1 tốt, hiệu năng tương đối đều giữa các lớp.")
    elif macro_f1 >= 0.70:
        comments.append("Macro-F1 chấp nhận được, vẫn cần phân tích các lớp yếu.")
    else:
        comments.append("Macro-F1 thấp, không nên chỉ báo cáo accuracy; cần kiểm tra confusion matrix và split dữ liệu.")

    if weighted_f1 - macro_f1 > 0.15:
        comments.append("Weighted-F1 cao hơn macro-F1 nhiều, dấu hiệu mô hình thiên về lớp nhiều ảnh hoặc validation chưa cân bằng.")

    return comments


def build_markdown(models_dir: Path) -> str:
    comparison_path = models_dir / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Không thấy {comparison_path}. Hãy train xong trước.")

    comparison = pd.read_csv(comparison_path).sort_values("macro_f1", ascending=False)
    best = comparison.iloc[0]
    best_model = str(best["model"])
    best_report = load_report(models_dir, best_model)
    best_cm = load_confusion_matrix(models_dir, best_model)
    hard = hardest_classes(best_report)
    confusions = top_confusions(best_cm)

    lines = [
        "# Đánh giá kết quả thực nghiệm",
        "",
        "## Bảng so sánh model",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Model nên chọn: `{best_model}`",
        "",
        f"- Accuracy: `{float(best['accuracy']):.4f}`",
        f"- Macro-F1: `{float(best['macro_f1']):.4f}`",
        f"- Weighted-F1: `{float(best['weighted_f1']):.4f}`",
        "",
        "## Nhận xét nhanh",
        "",
    ]
    lines.extend(f"- {item}" for item in quality_comment(best))

    lines.extend(
        [
            "",
            "## Các lớp cần phân tích kỹ",
            "",
            hard[["precision", "recall", "f1-score", "support"]].to_markdown(floatfmt=".4f"),
            "",
            "## Các nhầm lẫn lớn nhất",
            "",
        ]
    )
    if confusions:
        lines.extend(f"- `{true}` bị dự đoán thành `{pred}`: {count} ảnh" for true, pred, count in confusions)
    else:
        lines.append("- Không có nhầm lẫn ngoài đường chéo trong confusion matrix.")

    lines.extend(
        [
            "",
            "## Cách trình bày khi bảo vệ",
            "",
            "- Chọn model theo macro-F1 vì dataset lệch lớp.",
            "- Dùng accuracy để nói hiệu năng tổng thể, dùng macro-F1 để nói độ công bằng giữa các lớp.",
            "- Mở confusion matrix để giải thích các lớp dễ nhầm.",
            "- Demo thêm Grad-CAM và cảnh báo độ tin cậy thấp để cho thấy hệ thống có khả năng giải thích.",
            "- Nếu macro-F1 thấp bất thường, không đưa kết quả đó làm kết quả cuối; hãy train lại bằng bản code đã split validation theo từng lớp.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(models_dir)
    out_path = out_dir / "danh_gia_ket_qua.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
