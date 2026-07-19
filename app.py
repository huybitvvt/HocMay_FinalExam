from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from app_insights import disposal_conclusion, prediction_status, read_feedback_stats, read_model_summary
from app_paths import AppPaths, get_app_paths
from cleanliness_model import (
    RECYCLABLE_CLASSES,
    cleanliness_advice,
    load_cleanliness_class_names,
    predict_cleanliness,
)
from collection_points import (
    google_maps_url,
    load_collection_points,
    merge_collection_points,
    rank_collection_points,
    save_collection_points,
    validate_collection_points,
)
from image_quality import assess_image_quality, multi_object_hint, recycling_cleanliness_hint
from model_utils import load_class_names, make_gradcam_heatmap, overlay_gradcam, predict_image
from object_detection import detect_objects, draw_detections, load_yolo_model
from reporting import append_prediction_log, build_html_report, save_html_report
from session_planner import build_session_plan
from user_history import (
    append_user_history,
    build_history_stats,
    filter_history_period,
    normalize_user_id,
    read_user_history,
)
from waste_rules import VI_LABELS, get_waste_advice


st.set_page_config(page_title="Phân loại rác thải", page_icon="♻", layout="wide")
APP_PATHS = get_app_paths()


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


@st.cache_resource
def load_detection_model(model_file: str):
    return load_yolo_model(model_file)


@st.cache_resource
def load_secondary_model(model_file: str):
    return tf.keras.models.load_model(model_file, compile=False)


def render_advice(class_name: str) -> None:
    advice = get_waste_advice(class_name)
    st.markdown(
        f"""
        <div style="border-left: 6px solid {advice['color']}; padding: 0.75rem 1rem; background: rgba(255,255,255,0.06); color: inherit; border-radius: 8px;">
            <h3 style="margin: 0 0 .25rem 0;">{advice['label']}</h3>
            <div><b>Nhóm xử lý:</b> {advice['group']}</div>
            <div><b>Nơi bỏ:</b> {advice['bin']}</div>
            <div><b>Cách xử lý:</b> {advice['action']}</div>
            <div><b>Ý nghĩa môi trường:</b> {advice['impact']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_conclusion(conclusion: dict) -> None:
    st.markdown(
        f"""
        <div style="border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 0.9rem 1rem; background: rgba(31,138,91,0.10);">
            <div><b>Loại rác:</b> {conclusion['waste_type']}</div>
            <div><b>Nhóm xử lý:</b> {conclusion['group']}</div>
            <div><b>Mức rủi ro:</b> {conclusion['risk']}</div>
            <div><b>Nơi xử lý:</b> {conclusion['destination']}</div>
            <div><b>Hành động:</b> {conclusion['action']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_information(paths: AppPaths) -> None:
    info = read_model_summary(models_dir=paths.models_dir, reports_dir=paths.reports_dir)
    best = info["best"]
    st.subheader("Thông tin mô hình")
    if best:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Model đang chọn", str(best.get("model", "")).upper())
        col_b.metric("Accuracy", f"{float(best.get('accuracy', 0)) * 100:.2f}%")
        col_c.metric("Macro-F1", f"{float(best.get('macro_f1', 0)) * 100:.2f}%")
        col_d.metric("Weighted-F1", f"{float(best.get('weighted_f1', 0)) * 100:.2f}%")
    else:
        st.info("Chưa tìm thấy `models/model_comparison.csv`.")

    cols = st.columns(2)
    cols[0].metric("Số lớp", len(info["classes"]) if info["classes"] else 0)
    cols[1].metric("Số ảnh dataset", info["total_images"] or "Chưa có")
    if info["total_images"] is None:
        st.caption("Chưa thấy `reports/dataset_distribution.csv`. Copy thư mục `reports` từ Drive vào runtime để hiện số ảnh dataset.")

    if not info["comparison"].empty:
        st.dataframe(info["comparison"], hide_index=True, use_container_width=True)
    if not info["dataset"].empty:
        st.bar_chart(info["dataset"], x="class", y="count")


def render_feedback_statistics(paths: AppPaths) -> None:
    stats = read_feedback_stats(paths.feedback_dir)
    st.subheader("Phản hồi dữ liệu")
    st.metric("Số ảnh đã lưu phản hồi", stats["total"])
    if stats["total"] == 0:
        st.info("Chưa có phản hồi. Khi model dự đoán sai, mở mục phản hồi trong kết quả và lưu nhãn đúng.")
        return
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Nhãn được sửa nhiều nhất**")
        st.dataframe(stats["top_corrected"], hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("**Cặp nhầm lẫn thường gặp**")
        st.dataframe(stats["top_pairs"], hide_index=True, use_container_width=True)
    st.markdown("**Nhật ký phản hồi**")
    st.dataframe(stats["log"], hide_index=True, use_container_width=True)


def render_session_plan(summary_df: pd.DataFrame) -> None:
    plan = build_session_plan(summary_df)
    st.subheader("Kế hoạch xử lý phiên này")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Điểm phân loại", f"{plan['score']}/100")
    col_b.metric("Tỷ lệ tái chế/tái sử dụng", f"{plan['recyclable_ratio'] * 100:.1f}%")
    col_c.metric("Số nhóm cần chuẩn bị", len(plan["bucket_plan"]))

    st.markdown("**Việc cần làm trước**")
    for action in plan["priority_actions"]:
        st.markdown(f"- {action}")

    if not plan["bucket_plan"].empty:
        st.markdown("**Phân thùng / tuyến xử lý**")
        st.dataframe(plan["bucket_plan"], hide_index=True, use_container_width=True)

    st.markdown("**Gợi ý bổ sung dữ liệu**")
    for suggestion in plan["data_suggestions"]:
        st.markdown(f"- {suggestion}")


def render_user_history(user_id: str, paths: AppPaths) -> None:
    history = read_user_history(user_id=user_id, history_path=paths.history_path)
    st.subheader("Lịch sử phân loại của người dùng")
    st.caption(
        "Thống kê dựa trên số lượt ảnh đã nhận diện, không phải khối lượng rác theo kg. "
        "Phiên không lưu lịch sử sẽ không xuất hiện tại đây."
    )
    if history.empty:
        st.info("Chưa có lịch sử cho người dùng này. Hãy phân loại ảnh và bật lưu lịch sử trong thanh bên.")
        return

    full_stats = build_history_stats(history)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Tổng lượt ghi nhận", full_stats["total"])
    metric_cols[1].metric("Tuần hiện tại", full_stats["this_week"])
    metric_cols[2].metric("Tháng hiện tại", full_stats["this_month"])

    detail_cols = st.columns(3)
    detail_cols[0].metric("Tái chế / tái sử dụng", f"{full_stats['recyclable_ratio'] * 100:.1f}%")
    detail_cols[1].metric("Rác nguy hại", full_stats["hazardous_count"])
    detail_cols[2].metric("Cần kiểm tra lại", f"{full_stats['uncertain_rate'] * 100:.1f}%")

    period_label = st.selectbox(
        "Khoảng thời gian hiển thị",
        ["7 ngày gần nhất", "30 ngày gần nhất", "Toàn bộ"],
        index=1,
        key=f"history_period_{user_id}",
    )
    period = {
        "7 ngày gần nhất": "7_days",
        "30 ngày gần nhất": "30_days",
        "Toàn bộ": "all",
    }[period_label]
    selected = filter_history_period(history, period)
    if selected.empty:
        st.info(f"Không có lượt phân loại trong khoảng: {period_label.lower()}.")
        return

    selected_stats = build_history_stats(selected)
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Số lượt nhận diện theo ngày**")
        st.line_chart(selected_stats["daily"], x="Ngày", y="Số lượt")
    with chart_right:
        st.markdown("**Phân bố theo nhóm xử lý**")
        st.bar_chart(selected_stats["groups"], x="Nhóm", y="Số lượt")

    st.markdown("**Các loại rác được ghi nhận nhiều nhất**")
    st.dataframe(selected_stats["classes"].head(12), hide_index=True, use_container_width=True)

    display_history = selected.copy()
    display_history["recorded_at"] = pd.to_datetime(
        display_history["recorded_at"],
        errors="coerce",
        format="mixed",
    ).dt.strftime("%d/%m/%Y %H:%M")
    display_history["confidence"] = pd.to_numeric(
        display_history["confidence"],
        errors="coerce",
    ).map(lambda value: f"{value * 100:.2f}%" if pd.notna(value) else "")
    display_history = display_history.rename(
        columns={
            "recorded_at": "Thời gian",
            "image_name": "Ảnh",
            "prediction": "Dự đoán",
            "group": "Nhóm",
            "confidence": "Độ tin cậy",
            "status": "Trạng thái",
        }
    )
    st.markdown("**Chi tiết lịch sử**")
    st.dataframe(
        display_history[["Thời gian", "Ảnh", "Dự đoán", "Nhóm", "Độ tin cậy", "Trạng thái"]],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Tải lịch sử CSV",
        selected.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"lich_su_phan_loai_{user_id}.csv",
        mime="text/csv",
        key=f"download_history_{user_id}_{period}",
    )


def render_collection_point_map(paths: AppPaths) -> None:
    st.subheader("Điểm thu gom pin và rác điện tử")
    st.caption(
        "Danh sách ban đầu được đối chiếu từ Việt Nam Tái Chế. "
        "Nên mở liên kết nguồn và xác nhận điểm còn hoạt động trước khi mang rác đến."
    )
    try:
        points = load_collection_points(paths.collection_points_path)
    except ValueError as exc:
        st.error(f"Dữ liệu điểm thu gom chưa đúng định dạng: {exc}")
        return

    with st.expander("Bổ sung dữ liệu điểm thu gom toàn quốc"):
        st.caption(
            "File CSV mới phải giữ đúng các cột của file mẫu. "
            "Hệ thống loại bản ghi trùng theo tên và địa chỉ."
        )
        st.download_button(
            "Tải file CSV mẫu",
            points.head(0).to_csv(index=False).encode("utf-8-sig"),
            file_name="mau_diem_thu_gom.csv",
            mime="text/csv",
        )
        uploaded_points = st.file_uploader(
            "Nhập CSV điểm thu gom đã xác minh",
            type=["csv"],
            key="collection_points_upload",
        )
        if uploaded_points is not None:
            try:
                incoming_points = validate_collection_points(pd.read_csv(uploaded_points))
                points = merge_collection_points(points, incoming_points)
                st.success(f"Đã đọc {len(incoming_points)} điểm hợp lệ từ file.")
                if st.button("Lưu danh sách vào thư mục dữ liệu"):
                    output_path = paths.storage_root / "data" / "collection_points.csv"
                    saved_path = save_collection_points(points, output_path)
                    st.success(f"Đã lưu {len(points)} điểm tại: {saved_path}")
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"Không đọc được file điểm thu gom: {exc}")

    if points.empty:
        st.info(f"Chưa có dữ liệu điểm thu gom tại `{paths.collection_points_path}`.")
        return

    city = st.selectbox(
        "Khu vực",
        ["Tất cả", *sorted(points["city"].dropna().unique().tolist())],
    )
    selected = points if city == "Tất cả" else points[points["city"] == city]
    selected = selected.reset_index(drop=True)

    coordinate_mode = st.radio(
        "Tìm điểm gần vị trí",
        ["Không dùng vị trí", "GPS tự động", "Nhập tọa độ"],
        horizontal=True,
    )
    ranked = selected.copy()
    latitude = None
    longitude = None
    if coordinate_mode == "GPS tự động":
        try:
            from streamlit_geolocation import streamlit_geolocation

            location = streamlit_geolocation()
            if isinstance(location, dict):
                latitude = location.get("latitude")
                longitude = location.get("longitude")
                accuracy = location.get("accuracy")
                if latitude is not None and longitude is not None:
                    accuracy_note = f", sai số khoảng {accuracy:.0f} m" if accuracy else ""
                    st.success(f"Đã nhận vị trí từ trình duyệt{accuracy_note}.")
                else:
                    st.info("Nhấn nút định vị và cho phép trình duyệt truy cập vị trí.")
        except ImportError:
            st.warning(
                "Chưa cài thành phần GPS. Chạy `pip install -r requirements-extended.txt` "
                "hoặc chọn nhập tọa độ."
            )
        except Exception as exc:
            st.warning(f"Không lấy được GPS: {exc}. Có thể chuyển sang nhập tọa độ.")
    elif coordinate_mode == "Nhập tọa độ":
        default_latitude = float(selected["latitude"].mean())
        default_longitude = float(selected["longitude"].mean())
        location_cols = st.columns(2)
        latitude = location_cols[0].number_input(
            "Vĩ độ",
            min_value=-90.0,
            max_value=90.0,
            value=default_latitude,
            format="%.6f",
        )
        longitude = location_cols[1].number_input(
            "Kinh độ",
            min_value=-180.0,
            max_value=180.0,
            value=default_longitude,
            format="%.6f",
        )

    if latitude is not None and longitude is not None:
        ranked = rank_collection_points(selected, latitude, longitude)
        if not ranked.empty:
            nearest = ranked.iloc[0]
            st.success(
                f"Gần nhất theo đường chim bay: {nearest['name']} "
                f"({nearest['distance_km']:.2f} km)."
            )
            st.caption("Khoảng cách này không phải quãng đường di chuyển thực tế.")

    map_frame = ranked.rename(columns={"latitude": "lat", "longitude": "lon"})
    st.map(map_frame[["lat", "lon"]], use_container_width=True)

    display_points = ranked.copy()
    display_points["Bản đồ"] = display_points["address"].map(google_maps_url)
    if "distance_km" in display_points:
        display_points["Khoảng cách"] = display_points["distance_km"].map(lambda value: f"{value:.2f} km")
    display_points = display_points.rename(
        columns={
            "name": "Điểm thu gom",
            "city": "Thành phố",
            "address": "Địa chỉ",
            "accepted_waste": "Loại tiếp nhận",
            "source_url": "Nguồn",
            "verified_date": "Ngày đối chiếu",
        }
    )
    visible_columns = ["Điểm thu gom", "Thành phố", "Địa chỉ", "Loại tiếp nhận"]
    if "Khoảng cách" in display_points:
        visible_columns.append("Khoảng cách")
    visible_columns.extend(["Bản đồ", "Nguồn", "Ngày đối chiếu"])
    st.dataframe(
        display_points[visible_columns],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Bản đồ": st.column_config.LinkColumn("Mở bản đồ", display_text="Mở"),
            "Nguồn": st.column_config.LinkColumn("Nguồn", display_text="Kiểm tra"),
        },
    )


def save_feedback(
    uploaded_file,
    predicted_class: str,
    corrected_class: str,
    confidence: float,
    feedback_root: str | Path,
) -> Path:
    feedback_dir = Path(feedback_root) / corrected_class
    feedback_dir.mkdir(parents=True, exist_ok=True)
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha1(file_bytes).hexdigest()[:10]
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    image_path = feedback_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{digest}{suffix}"
    image_path.write_bytes(file_bytes)

    log_path = Path(feedback_root) / "feedback_log.csv"
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
    model_path = st.text_input("Đường dẫn model", value=str(APP_PATHS.model_path))
    confidence_threshold = st.slider("Ngưỡng cảnh báo độ tin cậy", 0.30, 0.95, 0.65, 0.05)
    show_gradcam = st.checkbox("Hiển thị Grad-CAM", value=True)
    save_session_log = st.checkbox("Lưu nhật ký kiểm thử", value=True)
    st.divider()
    st.header("Lịch sử người dùng")
    user_name = st.text_input("Tên người dùng", value="Khách")
    user_id = normalize_user_id(user_name)
    save_history = st.checkbox("Lưu lịch sử phân loại", value=True)
    st.caption(f"Mã lịch sử: {user_id}")
    st.divider()
    st.header("Model mở rộng")
    yolo_available = APP_PATHS.yolo_model_path.exists()
    use_yolo = st.checkbox(
        "Phát hiện nhiều vật thể bằng YOLO",
        value=yolo_available,
        disabled=not yolo_available,
    )
    if yolo_available:
        yolo_confidence = st.slider("Ngưỡng YOLO", 0.10, 0.90, 0.35, 0.05)
    else:
        yolo_confidence = 0.35
        st.caption(f"Chưa có weights: `{APP_PATHS.yolo_model_path}`")

    cleanliness_available = APP_PATHS.cleanliness_model_path.exists()
    use_cleanliness_model = st.checkbox(
        "Đánh giá sạch/bẩn bằng model",
        value=cleanliness_available,
        disabled=not cleanliness_available,
    )
    if cleanliness_available:
        cleanliness_confidence = st.slider("Ngưỡng độ sạch", 0.40, 0.95, 0.65, 0.05)
    else:
        cleanliness_confidence = 0.65
        st.caption(f"Chưa có model: `{APP_PATHS.cleanliness_model_path}`")
    st.divider()
    st.subheader("Lưu trữ")
    storage_label = "Google Drive / thư mục ngoài" if APP_PATHS.uses_external_storage else "Thư mục project"
    st.caption(f"{storage_label}: `{APP_PATHS.storage_root}`")
    st.caption("Đặt biến môi trường `APP_STORAGE_DIR` trước khi chạy app để đổi nơi lưu.")

classify_tab, collection_tab, history_tab, model_tab, feedback_tab = st.tabs(
    ["Phân loại", "Điểm thu gom", "Lịch sử & thống kê", "Thông tin mô hình", "Phản hồi dữ liệu"]
)

with model_tab:
    render_model_information(APP_PATHS)

with feedback_tab:
    render_feedback_statistics(APP_PATHS)

with collection_tab:
    render_collection_point_map(APP_PATHS)

with classify_tab:
    upload_tab, camera_tab = st.tabs(["Tải ảnh", "Chụp ảnh"])
    with upload_tab:
        uploaded_files = st.file_uploader(
            "Tải ảnh rác tự chụp hoặc nhiều ảnh để kiểm thử",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
    with camera_tab:
        camera_file = st.camera_input("Chụp ảnh rác bằng camera")

    input_files = list(uploaded_files or [])
    if camera_file is not None:
        input_files.append(camera_file)

    model_file = Path(model_path)
    if not model_file.exists():
        st.info("Chưa có model. Hãy tạo hoặc copy `models/best_model.keras` vào cùng thư mục app.")
    elif not input_files:
        st.info("Tải ảnh lên hoặc chụp ảnh để bắt đầu phân loại.")
    else:
        model = load_model(str(model_file))
        class_names = load_class_names(model_file)
        detection_model = None
        if use_yolo:
            try:
                detection_model = load_detection_model(str(APP_PATHS.yolo_model_path))
            except Exception as exc:
                st.warning(f"Không tải được YOLO, app tiếp tục dùng phân loại ảnh: {exc}")

        cleanliness_model = None
        cleanliness_classes = []
        if use_cleanliness_model:
            try:
                cleanliness_model = load_secondary_model(str(APP_PATHS.cleanliness_model_path))
                cleanliness_classes = load_cleanliness_class_names(APP_PATHS.cleanliness_model_path)
            except Exception as exc:
                st.warning(f"Không tải được model sạch/bẩn, app tiếp tục dùng gợi ý sơ bộ: {exc}")

        summary_rows = []

        for idx, file in enumerate(input_files):
            image = Image.open(file)
            quality = assess_image_quality(image)
            multi_object = multi_object_hint(image)
            result = predict_image(model, image, class_names)
            status = prediction_status(result["ranking"], confidence_threshold)
            advice = get_waste_advice(result["class_name"])
            conclusion = disposal_conclusion(advice, result["confidence"], status)
            detections = []
            if detection_model is not None:
                try:
                    detections = detect_objects(
                        detection_model,
                        image,
                        confidence_threshold=yolo_confidence,
                    )
                    detected_count = len(detections)
                    multi_object = {
                        "object_like_count": detected_count,
                        "has_warning": detected_count >= 2,
                        "message": (
                            f"YOLO phát hiện {detected_count} vật thể trong ảnh."
                            if detected_count
                            else "YOLO chưa phát hiện được vật thể phù hợp trong ảnh."
                        ),
                    }
                except Exception as exc:
                    st.warning(f"YOLO không xử lý được ảnh `{file.name}`: {exc}")

            cleanliness_result = None
            if cleanliness_model is not None and result["class_name"] in RECYCLABLE_CLASSES:
                try:
                    cleanliness_result = predict_cleanliness(
                        cleanliness_model,
                        image,
                        cleanliness_classes,
                    )
                except Exception as exc:
                    st.warning(f"Model sạch/bẩn không xử lý được ảnh `{file.name}`: {exc}")
            if cleanliness_result is not None:
                cleanliness_hint = cleanliness_advice(
                    result["class_name"],
                    cleanliness_result,
                    cleanliness_confidence,
                )
                cleanliness_status = (
                    f"{cleanliness_result['label']} "
                    f"({cleanliness_result['confidence'] * 100:.1f}%)"
                )
            else:
                cleanliness_hint = recycling_cleanliness_hint(result["class_name"], quality)
                cleanliness_status = "Chưa đánh giá bằng model"

            warning_label = "Thấp" if status["is_uncertain"] else "OK"
            summary_rows.append(
                {
                    "Ảnh": file.name,
                    "Dự đoán": advice["label"],
                    "Nhóm": advice["group"],
                    "Độ tin cậy": round(result["confidence"], 4),
                    "Trạng thái": status["status"],
                    "Chênh lệch top-2": round(status["margin"], 4),
                    "Mức rủi ro": conclusion["risk"],
                    "Kết luận xử lý": f"{conclusion['destination']} - {conclusion['action']}",
                    "Chất lượng ảnh": quality["verdict"],
                    "Nhiều vật thể": "Có thể" if multi_object["has_warning"] else "Không",
                    "Số vật thể YOLO": len(detections) if detection_model is not None else "",
                    "Độ sạch model": cleanliness_status,
                    "Gợi ý độ sạch": cleanliness_hint,
                    "Cảnh báo": warning_label,
                }
            )

            left, right = st.columns([1, 1.1], vertical_alignment="top")
            with left:
                st.image(image, caption=file.name, use_container_width=True)
                if detection_model is not None:
                    if detections:
                        st.image(
                            draw_detections(image, detections),
                            caption=f"YOLO: phát hiện {len(detections)} vật thể",
                            use_container_width=True,
                        )
                    else:
                        st.caption("YOLO chưa tìm thấy vật thể thuộc 12 lớp rác trong ảnh.")
                if show_gradcam:
                    try:
                        heatmap = make_gradcam_heatmap(model, image)
                        st.image(overlay_gradcam(image, heatmap), caption="Grad-CAM: vùng model chú ý", use_container_width=True)
                        st.caption("Grad-CAM dùng để kiểm tra model đang tập trung vào vật thể hay nền ảnh.")
                    except Exception as exc:
                        st.warning(f"Không tạo được Grad-CAM cho model này: {exc}")

            with right:
                st.metric("Kết quả", advice["label"], f"{result['confidence'] * 100:.1f}%")
                render_conclusion(conclusion)
                if status["is_uncertain"]:
                    st.warning(status["message"])
                else:
                    st.success(status["message"])
                render_advice(result["class_name"])
                if quality["has_warning"]:
                    st.warning(f"Chất lượng ảnh: {quality['verdict']}")
                else:
                    st.info(f"Chất lượng ảnh: {quality['verdict']}")
                st.caption(
                    f"Blur: {quality['blur_score']} | Sáng: {quality['brightness']} | Tương phản: {quality['contrast']}"
                )
                if multi_object["has_warning"]:
                    st.warning(multi_object["message"])
                else:
                    st.caption(multi_object["message"])
                if detections:
                    detection_rows = []
                    for detection in detections:
                        detection_advice = get_waste_advice(detection["class_name"])
                        detection_rows.append(
                            {
                                "Vật thể": detection["label"],
                                "Tin cậy": f"{detection['confidence'] * 100:.2f}%",
                                "Nhóm": detection_advice["group"],
                                "Nơi xử lý": detection_advice["bin"],
                            }
                        )
                    st.markdown("**Các vật thể YOLO phát hiện**")
                    st.dataframe(
                        pd.DataFrame(detection_rows),
                        hide_index=True,
                        use_container_width=True,
                    )
                if cleanliness_result is not None:
                    st.metric(
                        "Độ sạch tái chế",
                        cleanliness_result["label"],
                        f"{cleanliness_result['confidence'] * 100:.1f}%",
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
                        saved_path = save_feedback(
                            file,
                            result["class_name"],
                            corrected_label,
                            result["confidence"],
                            APP_PATHS.feedback_dir,
                        )
                        st.success(f"Đã lưu: {saved_path}")
            st.divider()

        summary_df = pd.DataFrame(summary_rows)
        session_payload = {"user_id": user_id, "rows": summary_rows}
        session_key = hashlib.sha1(
            json.dumps(session_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if save_session_log:
            if st.session_state.get("last_prediction_log_session") != session_key:
                log_path = append_prediction_log(summary_rows, APP_PATHS.reports_dir)
                st.session_state["last_prediction_log_session"] = session_key
                st.session_state["prediction_log_path"] = str(log_path)
            st.caption(f"Nhật ký kiểm thử: {st.session_state.get('prediction_log_path', 'chưa lưu')}")
        if save_history:
            if st.session_state.get("last_history_session") != session_key:
                history_path, saved_count = append_user_history(
                    summary_rows,
                    user_name=user_name,
                    session_id=session_key,
                    history_path=APP_PATHS.history_path,
                )
                st.session_state["last_history_session"] = session_key
                st.session_state["history_path"] = str(history_path)
                st.session_state["history_saved_count"] = saved_count
            st.caption(
                f"Lịch sử người dùng: {st.session_state.get('history_path', 'chưa lưu')} "
                f"({st.session_state.get('history_saved_count', 0)} lượt mới)"
            )

        st.subheader("Tổng hợp kiểm thử")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Số ảnh", len(summary_df))
        col_b.metric("Cần kiểm tra", int((summary_df["Trạng thái"] == "Cần kiểm tra").sum()))
        col_c.metric("Số nhóm rác", summary_df["Nhóm"].nunique())
        col_d.metric("Có thể nhiều vật thể", int((summary_df["Nhiều vật thể"] == "Có thể").sum()))

        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        group_df = summary_df["Nhóm"].value_counts().reset_index()
        group_df.columns = ["Nhóm", "Số lượng"]
        st.bar_chart(group_df, x="Nhóm", y="Số lượng")
        render_session_plan(summary_df)

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
            report_path = save_html_report(summary_df, APP_PATHS.reports_dir)
            st.success(f"Đã lưu báo cáo: {report_path}")

with history_tab:
    render_user_history(user_id, APP_PATHS)
