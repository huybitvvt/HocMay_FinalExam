from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from waste_rules import CLASS_NAMES


IMG_SIZE = (224, 224)


def load_class_names(model_path: str | Path) -> list[str]:
    model_path = Path(model_path)
    sidecar = model_path.with_suffix(".classes.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return CLASS_NAMES


def preprocess_image(image: Image.Image, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    image = image.convert("RGB").resize(img_size)
    arr = np.asarray(image).astype("float32")
    return np.expand_dims(arr, axis=0)


def predict_image(model: tf.keras.Model, image: Image.Image, class_names: list[str]) -> dict:
    batch = preprocess_image(image)
    probs = model.predict(batch, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    ranking = sorted(
        [{"class_name": class_names[i], "probability": float(probs[i])} for i in range(len(class_names))],
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "class_name": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "ranking": ranking,
    }


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            output_shape = tuple(layer.output.shape)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Không tìm thấy lớp Conv2D để tạo Grad-CAM.")


def _locate_layer(model: tf.keras.Model, layer_name: str) -> tf.keras.layers.Layer:
    try:
        return model.get_layer(layer_name)
    except ValueError:
        for layer in model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    return layer.get_layer(layer_name)
                except ValueError:
                    continue
    raise


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image: Image.Image,
    class_index: int | None = None,
    layer_name: str | None = None,
) -> np.ndarray:
    batch = preprocess_image(image)
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    target_layer = _locate_layer(model, layer_name)
    grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    base = np.asarray(image.convert("RGB"))
    heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(colored * alpha + base * (1 - alpha))
    return Image.fromarray(overlay)


def uncertainty_message(confidence: float, threshold: float = 0.65) -> str:
    if confidence >= threshold:
        return "Dự đoán đủ tự tin để tham khảo gợi ý xử lý."
    return "Độ tin cậy thấp: nên chụp lại ảnh rõ hơn hoặc kiểm tra top-3 dự đoán trước khi xử lý rác."
