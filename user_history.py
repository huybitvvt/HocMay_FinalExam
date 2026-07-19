from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

import pandas as pd


DEFAULT_HISTORY_PATH = Path("data") / "user_history.csv"

HISTORY_COLUMNS = [
    "recorded_at",
    "user_id",
    "user_name",
    "session_id",
    "image_name",
    "prediction",
    "group",
    "confidence",
    "status",
    "quality",
    "multi_object",
    "disposal",
]

RECYCLABLE_GROUPS = {"Tái chế", "Tái sử dụng"}


def normalize_user_id(user_name: str) -> str:
    """Tạo mã người dùng đơn giản để lọc lịch sử, không thay thế đăng nhập."""
    normalized = unicodedata.normalize("NFKD", (user_name or "").strip())
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    user_id = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    return user_id[:50] or "khach"


def append_user_history(
    rows: list[dict],
    user_name: str,
    session_id: str | None = None,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    recorded_at: datetime | None = None,
) -> tuple[Path, int]:
    """Ghi một phiên phân loại và bỏ qua nếu session_id đã tồn tại."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    session_id = session_id or uuid4().hex
    timestamp = (recorded_at or datetime.now()).isoformat(timespec="seconds")
    clean_name = (user_name or "").strip()[:80] or "Khách"
    user_id = normalize_user_id(clean_name)

    if path.exists():
        try:
            existing_sessions = pd.read_csv(path, usecols=["session_id"], dtype=str)
            if session_id in set(existing_sessions["session_id"].dropna()):
                return path, 0
        except (ValueError, pd.errors.EmptyDataError):
            pass

    history_rows = []
    for row in rows:
        history_rows.append(
            {
                "recorded_at": timestamp,
                "user_id": user_id,
                "user_name": clean_name,
                "session_id": session_id,
                "image_name": row.get("Ảnh", ""),
                "prediction": row.get("Dự đoán", ""),
                "group": row.get("Nhóm", ""),
                "confidence": row.get("Độ tin cậy", 0.0),
                "status": row.get("Trạng thái", ""),
                "quality": row.get("Chất lượng ảnh", ""),
                "multi_object": row.get("Nhiều vật thể", ""),
                "disposal": row.get("Kết luận xử lý", ""),
            }
        )

    if not history_rows:
        return path, 0

    frame = pd.DataFrame(history_rows, columns=HISTORY_COLUMNS)
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )
    return path, len(frame)


def read_user_history(
    user_id: str | None = None,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
) -> pd.DataFrame:
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    for column in HISTORY_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[HISTORY_COLUMNS]
    if user_id:
        frame = frame[frame["user_id"].astype(str) == user_id]
    return frame.reset_index(drop=True)


def _with_parsed_time(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["_recorded_at"] = pd.to_datetime(
        prepared.get("recorded_at", pd.Series(dtype=str)),
        errors="coerce",
        format="mixed",
    )
    return prepared.dropna(subset=["_recorded_at"])


def filter_history_period(
    frame: pd.DataFrame,
    period: str,
    now: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    prepared = _with_parsed_time(frame)
    if prepared.empty or period == "all":
        return prepared.drop(columns=["_recorded_at"], errors="ignore").reset_index(drop=True)

    current = pd.Timestamp(now or datetime.now())
    days = {"7_days": 7, "30_days": 30}.get(period)
    if days is None:
        raise ValueError(f"Khoảng thời gian không hợp lệ: {period}")

    start = current.normalize() - pd.Timedelta(days=days - 1)
    selected = prepared[
        (prepared["_recorded_at"] >= start)
        & (prepared["_recorded_at"] <= current)
    ]
    return selected.drop(columns=["_recorded_at"]).reset_index(drop=True)


def build_history_stats(
    frame: pd.DataFrame,
    now: datetime | pd.Timestamp | None = None,
) -> dict:
    prepared = _with_parsed_time(frame)
    current = pd.Timestamp(now or datetime.now())
    prepared = prepared[prepared["_recorded_at"] <= current]

    if prepared.empty:
        return {
            "total": 0,
            "this_week": 0,
            "this_month": 0,
            "recyclable_ratio": 0.0,
            "hazardous_count": 0,
            "uncertain_rate": 0.0,
            "average_confidence": 0.0,
            "daily": pd.DataFrame(columns=["Ngày", "Số lượt"]),
            "monthly": pd.DataFrame(columns=["Tháng", "Số lượt"]),
            "groups": pd.DataFrame(columns=["Nhóm", "Số lượt"]),
            "classes": pd.DataFrame(columns=["Loại rác", "Số lượt"]),
        }

    today = current.normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    total = len(prepared)
    recyclable_count = int(prepared["group"].isin(RECYCLABLE_GROUPS).sum())
    confidence = pd.to_numeric(prepared["confidence"], errors="coerce").fillna(0.0)

    daily = (
        prepared.assign(Ngày=prepared["_recorded_at"].dt.date)
        .groupby("Ngày")
        .size()
        .reset_index(name="Số lượt")
        .sort_values("Ngày")
    )
    monthly = (
        prepared.assign(Tháng=prepared["_recorded_at"].dt.to_period("M").astype(str))
        .groupby("Tháng")
        .size()
        .reset_index(name="Số lượt")
        .sort_values("Tháng")
    )
    groups = prepared["group"].fillna("Chưa xác định").value_counts().reset_index()
    groups.columns = ["Nhóm", "Số lượt"]
    classes = prepared["prediction"].fillna("Chưa xác định").value_counts().reset_index()
    classes.columns = ["Loại rác", "Số lượt"]

    return {
        "total": total,
        "this_week": int((prepared["_recorded_at"] >= week_start).sum()),
        "this_month": int((prepared["_recorded_at"] >= month_start).sum()),
        "recyclable_ratio": recyclable_count / total,
        "hazardous_count": int((prepared["group"] == "Nguy hại").sum()),
        "uncertain_rate": float((prepared["status"] == "Cần kiểm tra").mean()),
        "average_confidence": float(confidence.mean()),
        "daily": daily,
        "monthly": monthly,
        "groups": groups,
        "classes": classes,
    }
