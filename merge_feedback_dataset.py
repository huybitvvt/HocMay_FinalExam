from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="garbage_classification")
    parser.add_argument("--feedback-dir", default="feedback")
    parser.add_argument("--out-dir", default="garbage_classification_with_feedback")
    return parser.parse_args()


def copy_images(src_class_dir: Path, dst_class_dir: Path, prefix: str) -> int:
    count = 0
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    for image_path in src_class_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        target = dst_class_dir / f"{prefix}_{image_path.name}"
        if not target.exists():
            shutil.copy2(image_path, target)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    feedback_dir = Path(args.feedback_dir)
    out_dir = Path(args.out_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Không thấy base dataset: {base_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    summary = []
    for class_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        copied = copy_images(class_dir, out_dir / class_dir.name, "base")
        summary.append((class_dir.name, copied, 0))

    if feedback_dir.exists():
        summary_map = {name: [base_count, feedback_count] for name, base_count, feedback_count in summary}
        for class_dir in sorted(path for path in feedback_dir.iterdir() if path.is_dir()):
            copied = copy_images(class_dir, out_dir / class_dir.name, "feedback")
            if class_dir.name not in summary_map:
                summary_map[class_dir.name] = [0, 0]
            summary_map[class_dir.name][1] += copied
        summary = [(name, counts[0], counts[1]) for name, counts in sorted(summary_map.items())]

    print("Dataset đã trộn:", out_dir)
    print("class,base,feedback,total")
    for name, base_count, feedback_count in summary:
        print(f"{name},{base_count},{feedback_count},{base_count + feedback_count}")


if __name__ == "__main__":
    main()
