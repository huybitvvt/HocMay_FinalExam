from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import tensorflow as tf


CLASS_NAMES = ["clean", "dirty", "uncertain"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model nhận biết rác tái chế sạch/bẩn.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_dataset(data_dir: Path) -> None:
    missing = []
    for split in ["train", "val"]:
        for class_name in CLASS_NAMES:
            class_dir = data_dir / split / class_name
            if not class_dir.exists():
                missing.append(str(class_dir))
    if missing:
        raise SystemExit(
            "Dataset độ sạch phải có các thư mục train/val cho clean, dirty, uncertain. "
            "Đang thiếu:\n- " + "\n- ".join(missing)
        )


def make_dataset(
    directory: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    ).prefetch(tf.data.AUTOTUNE)


def calculate_class_weights(train_dir: Path) -> dict[int, float]:
    counts = []
    for class_name in CLASS_NAMES:
        count = sum(
            1
            for path in (train_dir / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if count == 0:
            raise SystemExit(f"Lớp `{class_name}` không có ảnh train.")
        counts.append(count)
    total = sum(counts)
    return {
        index: total / (len(CLASS_NAMES) * count)
        for index, count in enumerate(counts)
    }


def build_model(image_size: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="cleanliness_mobilenetv2"), base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    out_dir: Path,
) -> dict:
    probabilities = model.predict(dataset, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = np.concatenate([labels.numpy() for _, labels in dataset])
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=list(range(len(CLASS_NAMES))),
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Greens")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Dự đoán",
        ylabel="Nhãn thật",
        title="Confusion matrix - độ sạch",
    )
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "cleanliness_confusion_matrix.png", dpi=160)
    plt.close(fig)

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    validate_dataset(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(
        data_dir / "train",
        args.image_size,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_dataset = make_dataset(
        data_dir / "val",
        args.image_size,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    test_dir = data_dir / "test"
    evaluation_dataset = (
        make_dataset(
            test_dir,
            args.image_size,
            args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        if test_dir.exists()
        else validation_dataset
    )
    class_weights = calculate_class_weights(data_dir / "train")
    model, base_model = build_model(args.image_size)
    compile_model(model, 1e-3)
    model_path = out_dir / "cleanliness_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )

    best_model = tf.keras.models.load_model(model_path, compile=False)
    metrics = evaluate_model(best_model, evaluation_dataset, out_dir)
    model_path.with_suffix(".classes.json").write_text(
        json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "cleanliness_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
