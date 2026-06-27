from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="garbage_classification")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=["mobilenetv2", "efficientnetb0"])
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    return parser.parse_args()


def build_datasets(data_dir: str, batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    class_names = train_ds.class_names
    class_counts = []
    for class_name in class_names:
        class_dir = Path(data_dir) / class_name
        count = sum(1 for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
        class_counts.append(count)
    total = sum(class_counts)
    class_weight = {
        idx: float(total / (len(class_names) * count))
        for idx, count in enumerate(class_counts)
        if count > 0
    }

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )

    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names, class_weight


def create_model(name: str, num_classes: int) -> tf.keras.Model:
    name = name.lower()
    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unsupported model: {name}")

    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune(model: tf.keras.Model, fine_tune_layers: int, learning_rate: float = 1e-5) -> None:
    backbone = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    if backbone is None:
        return
    backbone.trainable = True
    for layer in backbone.layers[:-fine_tune_layers]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(model_name: str, out_dir: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / f"{model_name}_checkpoint.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]


def save_history(history_parts: list[tf.keras.callbacks.History], out_dir: Path, model_name: str) -> None:
    frames = []
    offset = 0
    for history in history_parts:
        frame = pd.DataFrame(history.history)
        frame.insert(0, "epoch", np.arange(offset + 1, offset + len(frame) + 1))
        frames.append(frame)
        offset += len(frame)
    if not frames:
        return
    history_df = pd.concat(frames, ignore_index=True)
    history_df.to_csv(out_dir / f"{model_name}_history.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_df["epoch"], history_df["accuracy"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_accuracy"], label="validation")
    axes[0].set_title(f"Accuracy - {model_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["loss"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_loss"], label="validation")
    axes[1].set_title(f"Loss - {model_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_history.png", dpi=160)
    plt.close(fig)


def evaluate_and_save(model, val_ds, class_names, out_dir: Path, model_name: str) -> dict:
    y_true = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)
    probs = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(report).transpose().to_csv(out_dir / f"{model_name}_classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / f"{model_name}_confusion_matrix.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_title(f"Confusion matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    return {
        "model": model_name,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, class_names, class_weight = build_datasets(args.data_dir, args.batch_size)
    (out_dir / "class_names.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    results = []
    best_score = -1.0
    best_model_path = out_dir / "best_model.keras"

    for model_name in args.models:
        model = create_model(model_name, len(class_names))
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(1, args.epochs // 2),
            class_weight=class_weight,
            callbacks=make_callbacks(f"{model_name}_warmup", out_dir),
        )
        fine_tune(model, args.fine_tune_layers)
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=len(history.history["loss"]),
            class_weight=class_weight,
            callbacks=make_callbacks(f"{model_name}_finetune", out_dir),
        )
        save_history([history, fine_history], out_dir, model_name)
        model.save(out_dir / f"{model_name}.keras")
        metrics = evaluate_and_save(model, val_ds, class_names, out_dir, model_name)
        results.append(metrics)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            model.save(best_model_path)
            best_model_path.with_suffix(".classes.json").write_text(
                json.dumps(class_names, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    pd.DataFrame(results).sort_values("macro_f1", ascending=False).to_csv(out_dir / "model_comparison.csv", index=False)
    print(pd.DataFrame(results).sort_values("macro_f1", ascending=False))


if __name__ == "__main__":
    main()
