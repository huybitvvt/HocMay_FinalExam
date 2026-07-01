from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from image_quality import assess_image_quality, recycling_cleanliness_hint
from model_utils import load_class_names, make_gradcam_heatmap, overlay_gradcam, predict_image, uncertainty_message
from reporting import append_prediction_log, build_html_report, save_html_report
from waste_rules import VI_LABELS, get_waste_advice


st.set_page_config(page_title="Phân loại rác thải", page_icon="♻", layout="wide")


@st.cache_resource
def load_model(model_file: str):
    try:
        return tf.keras.models.load_model(model_file, compile=False)
    except TypeError:
        # Backward-compatible loader for older models saved with Lambda(preprocess_input).
        # The current best model is EfficientNetB0, whose preprocess_input is identity-like.
        return tf.keras.models.load_model(
            model_file,
            custom_objects={"preprocess_input": tf.keras.applications.efficientnet.preprocess_input},
            compile=False,
            safe_mode=False,
        )


def render_advice(class_name: str, confidence: float, confidence_threshold: float) -> None:
    advice = get_waste_advice(class_name)
    st.markdown(
        f"""
        <div style="border-left: 6px solid {advice['color']}; padding: 0.75rem 1rem; background: #f7f7f7;">
            <h3 style="margin: 0 0 .25rem 0;">{advice['label']}</h3>
            <div><b>Nhóm xử lý:</b> {advice['group']}</div>
            <div><b>Nơi bỏ:</b> {advice['bin']}</div>
            <div><b>Cách xử lý:</b> {advice['action']}</div>
            <div><b>Ý nghĩa môi trường:</b> {advice['impact']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if confidence < confidence_threshold:
        st.warning(uncertainty_message(confidence))
    else:
        st.success(uncertainty_message(confidence))


def save_feedback(uploaded_file, predicted_class: str, corrected_class: str, confidence: float) -> Path:
    feedback_dir = Path("feedback") / corrected_class
    feedback_dir.mkdir(parents=True, exist_ok=True)
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha1(file_bytes).hexdigest()[:10]
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    image_path = feedback_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{digest}{suffix}"
    image_path.write_bytes(file_bytes)

    log_path = Path("feedback") / "feedback_log.csv"
    row = pd.DataFrame(
        [
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "file": str(image_path),
                "predicted_class": predicted_class,
                "corrected_class": corrected_class,
                "confidence": confidence,
            }
        ]
    )
    row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False, encoding="utf-8-sig")
    return image_path


st.title("Hệ thống phân loại rác thải qua ảnh")
st.caption("Dự đoán loại rác, gợi ý xử lý bằng tiếng Việt và giải thích vùng ảnh model chú ý bằng Grad-CAM.")

with st.sidebar:
    st.header("Cấu hình")
    model_path = st.text_input("Đường dẫn model", value="models/best_model.keras")
    confidence_threshold = st.slider("Ngưỡng cảnh báo độ tin cậy", 0.30, 0.95, 0.65, 0.05)
    show_gradcam = st.checkbox("Hiển thị Grad-CAM", value=True)
    save_session_log = st.checkbox("Lưu nhật ký kiểm thử", value=True)
    st.divider()
    st.markdown("**Gợi ý demo bảo vệ**")
    st.markdown("- Dùng ảnh rõ và ảnh nhiễu để so sánh độ tin cậy.")
    st.markdown("- Chỉ ra Grad-CAM có tập trung vào vật thể hay nền.")
    st.markdown("- Mở bảng top-3 để nói về sai số giữa các lớp giống nhau.")
    st.markdown("- Sửa nhãn sai để chứng minh vòng lặp cải thiện dữ liệu.")

uploaded_files = st.file_uploader(
    "Tải ảnh rác tự chụp hoặc nhiều ảnh để kiểm thử",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

model_file = Path(model_path)
if not model_file.exists():
    st.info("Chưa có model. Hãy chạy notebook Colab để tạo `models/best_model.keras`, sau đó tải về cùng thư mục app.")
    st.stop()

model = load_model(str(model_file))
class_names = load_class_names(model_file)

if not uploaded_files:
    st.stop()

summary_rows = []
for idx, file in enumerate(uploaded_files):
    image = Image.open(file)
    quality = assess_image_quality(image)
    result = predict_image(model, image, class_names)
    advice = get_waste_advice(result["class_name"])
    cleanliness_hint = recycling_cleanliness_hint(result["class_name"], quality)
    summary_rows.append(
        {
            "Ảnh": file.name,
            "Dự đoán": advice["label"],
            "Nhóm": advice["group"],
            "Độ tin cậy": round(result["confidence"], 4),
            "Chất lượng ảnh": quality["verdict"],
            "Gợi ý độ sạch": cleanliness_hint,
            "Cảnh báo": "Thấp" if result["confidence"] < confidence_threshold else "OK",
        }
    )

    left, right = st.columns([1, 1.1], vertical_alignment="top")
    with left:
        st.image(image, caption=file.name, use_container_width=True)
        if show_gradcam:
            try:
                heatmap = make_gradcam_heatmap(model, image)
                st.image(overlay_gradcam(image, heatmap), caption="Grad-CAM: vùng model chú ý", use_container_width=True)
            except Exception as exc:
                st.warning(f"Không tạo được Grad-CAM cho model này: {exc}")

    with right:
        st.metric("Kết quả", advice["label"], f"{result['confidence'] * 100:.1f}%")
        render_advice(result["class_name"], result["confidence"], confidence_threshold)
        if quality["has_warning"]:
            st.warning(f"Chất lượng ảnh: {quality['verdict']}")
        else:
            st.info(f"Chất lượng ảnh: {quality['verdict']}")
        st.caption(
            f"Blur: {quality['blur_score']} | Sáng: {quality['brightness']} | Tương phản: {quality['contrast']}"
        )
        st.info(cleanliness_hint)
        top3 = pd.DataFrame(result["ranking"][:3])
        top3["Tên tiếng Việt"] = top3["class_name"].map(lambda name: get_waste_advice(name)["label"])
        top3["Xác suất"] = top3["probability"].map(lambda value: f"{value * 100:.2f}%")
        st.dataframe(top3[["Tên tiếng Việt", "Xác suất"]], hide_index=True, use_container_width=True)

        with st.expander("Phản hồi nếu model dự đoán sai"):
            corrected_label = st.selectbox(
                "Nhãn đúng",
                options=class_names,
                index=class_names.index(result["class_name"]),
                format_func=lambda name: VI_LABELS.get(name, name),
                key=f"correct_{idx}_{file.name}",
            )
            if st.button("Lưu ảnh vào dữ liệu phản hồi", key=f"save_{idx}_{file.name}"):
                saved_path = save_feedback(file, result["class_name"], corrected_label, result["confidence"])
                st.success(f"Đã lưu: {saved_path}")
    st.divider()

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    if save_session_log:
        session_key = hashlib.sha1(json.dumps(summary_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if st.session_state.get("last_logged_session") != session_key:
            log_path = append_prediction_log(summary_rows)
            st.session_state["last_logged_session"] = session_key
            st.session_state["last_log_path"] = str(log_path)
        st.caption(f"Nhật ký kiểm thử: {st.session_state.get('last_log_path', 'chưa lưu')}")

    st.subheader("Tổng hợp kiểm thử")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Số ảnh", len(summary_df))
    col_b.metric("Cảnh báo thấp", int((summary_df["Cảnh báo"] == "Thấp").sum()))
    col_c.metric("Số nhóm rác", summary_df["Nhóm"].nunique())

    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    group_df = summary_df["Nhóm"].value_counts().reset_index()
    group_df.columns = ["Nhóm", "Số lượng"]
    st.bar_chart(group_df, x="Nhóm", y="Số lượng")

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
    html_report = build_html_report(summary_df)
    st.download_button(
        "Tải CSV kết quả kiểm thử",
        csv_bytes,
        file_name="ket_qua_kiem_thu_rac.csv",
        mime="text/csv",
    )
    st.download_button(
        "Tải báo cáo HTML",
        html_report.encode("utf-8"),
        file_name="bao_cao_kiem_thu_rac.html",
        mime="text/html",
    )
    if st.button("Lưu báo cáo HTML vào thư mục reports"):
        report_path = save_html_report(summary_df)
        st.success(f"Đã lưu báo cáo: {report_path}")
