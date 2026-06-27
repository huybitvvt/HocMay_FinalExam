from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="garbage_classification")
    parser.add_argument("--out-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        count = sum(1 for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
        rows.append({"class": class_dir.name, "count": count})

    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    total = int(df["count"].sum())
    df["ratio"] = df["count"] / total
    df.to_csv(out_dir / "dataset_distribution.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["class"], df["count"], color="#1f8a5b")
    ax.set_title("Phân bố số ảnh theo lớp")
    ax.set_xlabel("Lớp")
    ax.set_ylabel("Số ảnh")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_distribution.png", dpi=160)
    plt.close(fig)

    print(df)
    print(f"Total images: {total}")
    print(f"Saved: {out_dir / 'dataset_distribution.csv'}")
    print(f"Saved: {out_dir / 'dataset_distribution.png'}")


if __name__ == "__main__":
    main()
