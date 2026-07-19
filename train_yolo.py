from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO cho bài toán phát hiện nhiều vật thể rác.")
    parser.add_argument("--data", required=True, help="File YAML mô tả dataset YOLO.")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="0", help="GPU id, `cpu` hoặc để trống cho tự động.")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--run-name", default="waste_detection")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Thiếu Ultralytics. Chạy `pip install -r requirements-extended.txt` trước."
        ) from exc

    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Không tìm thấy file dataset YAML: {data_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_dir = out_dir / "yolo_runs"
    device = args.device if args.device else None

    model = YOLO(args.base_model)
    train_result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.image_size,
        device=device,
        project=str(project_dir),
        name=args.run_name,
        exist_ok=True,
    )

    save_dir = Path(train_result.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise SystemExit(f"Không tìm thấy weights tốt nhất tại: {best_weights}")

    final_weights = out_dir / "waste_detector.pt"
    shutil.copy2(best_weights, final_weights)

    best_model = YOLO(str(final_weights))
    metrics = best_model.val(
        data=str(data_path),
        imgsz=args.image_size,
        device=device,
    )
    summary = {
        "model": str(final_weights),
        "base_model": args.base_model,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    metrics_path = out_dir / "waste_detector_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
