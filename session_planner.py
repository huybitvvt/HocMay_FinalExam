from __future__ import annotations

import pandas as pd


GROUP_ORDER = ["Nguy hại", "Tái chế", "Hữu cơ", "Tái sử dụng", "Rác khác"]

GROUP_ACTIONS = {
    "Nguy hại": "Gom riêng, giữ khô ráo, đưa tới điểm thu gom pin/rác điện tử.",
    "Tái chế": "Làm sạch sơ, giữ khô, tách khỏi rác hữu cơ và rác bẩn.",
    "Hữu cơ": "Tách riêng để ủ compost hoặc bỏ vào thùng rác hữu cơ.",
    "Tái sử dụng": "Kiểm tra còn dùng được không; ưu tiên quyên góp hoặc thu hồi.",
    "Rác khác": "Bỏ vào thùng rác còn lại sau khi kiểm tra không lẫn rác nguy hại.",
}

KNOWN_WEAK_CLASSES = {
    "Nhựa": "Nên bổ sung thêm ảnh nhựa trong nhiều điều kiện ánh sáng và nền khác nhau.",
    "Kim loại": "Nên bổ sung ảnh lon/hộp kim loại bị móp, phản sáng hoặc có nhãn màu.",
    "Thủy tinh trắng": "Nên bổ sung ảnh thủy tinh trong suốt, phản chiếu và chụp trên nền sáng.",
    "Giấy": "Nên bổ sung ảnh giấy bị nhàu, giấy mỏng và giấy có chữ/in màu.",
    "Bìa carton": "Nên bổ sung ảnh carton bị gấp, rách hoặc dính băng keo.",
}


def build_session_plan(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty:
        return {
            "score": 0,
            "priority_actions": [],
            "bucket_plan": pd.DataFrame(),
            "data_suggestions": [],
            "recyclable_ratio": 0.0,
        }

    total = len(summary_df)
    uncertain_count = int((summary_df["Trạng thái"] == "Cần kiểm tra").sum())
    multi_count = int((summary_df["Nhiều vật thể"] == "Có thể").sum())
    hazardous_count = int((summary_df["Nhóm"] == "Nguy hại").sum())
    recyclable_count = int(summary_df["Nhóm"].isin(["Tái chế", "Tái sử dụng"]).sum())
    quality_warning_count = int(summary_df["Chất lượng ảnh"].str.contains("mờ|tối|sáng|tương phản", case=False, na=False).sum())
    score = max(0, 100 - uncertain_count * 15 - multi_count * 10 - quality_warning_count * 5)

    bucket_rows = []
    for group in GROUP_ORDER:
        group_df = summary_df[summary_df["Nhóm"] == group]
        if group_df.empty:
            continue
        bucket_rows.append(
            {
                "Nhóm": group,
                "Số ảnh": len(group_df),
                "Ảnh liên quan": ", ".join(group_df["Ảnh"].astype(str).tolist()[:5]),
                "Việc cần làm": GROUP_ACTIONS[group],
            }
        )

    priority_actions = []
    if hazardous_count:
        priority_actions.append(f"Ưu tiên xử lý {hazardous_count} ảnh thuộc nhóm nguy hại; không bỏ chung rác sinh hoạt.")
    if uncertain_count:
        priority_actions.append(f"Có {uncertain_count} ảnh cần kiểm tra lại vì model chưa đủ chắc chắn.")
    if multi_count:
        priority_actions.append(f"Có {multi_count} ảnh có thể chứa nhiều vật thể; nên chụp riêng từng món rác.")
    if recyclable_count:
        priority_actions.append(f"Có {recyclable_count} ảnh thuộc nhóm tái chế/tái sử dụng; cần làm sạch hoặc kiểm tra tình trạng trước khi bỏ.")
    if not priority_actions:
        priority_actions.append("Các ảnh trong phiên này có thể xử lý theo gợi ý của từng nhóm rác.")

    data_suggestions = []
    weak_labels = [label for label in summary_df["Dự đoán"].unique() if label in KNOWN_WEAK_CLASSES]
    for label in weak_labels[:4]:
        data_suggestions.append(f"{label}: {KNOWN_WEAK_CLASSES[label]}")
    uncertain_labels = summary_df.loc[summary_df["Trạng thái"] == "Cần kiểm tra", "Dự đoán"].value_counts()
    for label, count in uncertain_labels.head(3).items():
        suggestion = f"{label}: có {count} ảnh chưa chắc chắn trong phiên này, nên chụp thêm ảnh rõ hơn để kiểm thử."
        if suggestion not in data_suggestions:
            data_suggestions.append(suggestion)
    if not data_suggestions:
        data_suggestions.append("Chưa có dấu hiệu cần bổ sung dữ liệu từ phiên kiểm thử này.")

    return {
        "score": score,
        "priority_actions": priority_actions,
        "bucket_plan": pd.DataFrame(bucket_rows),
        "data_suggestions": data_suggestions,
        "recyclable_ratio": recyclable_count / total if total else 0.0,
    }
