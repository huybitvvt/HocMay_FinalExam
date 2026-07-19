from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd


DEFAULT_POINTS_PATH = Path("data") / "collection_points.csv"

POINT_COLUMNS = [
    "name",
    "city",
    "address",
    "latitude",
    "longitude",
    "accepted_waste",
    "source_url",
    "verified_date",
]


def validate_collection_points(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in POINT_COLUMNS if column not in frame]
    if missing:
        raise ValueError("Thiếu cột dữ liệu điểm thu gom: " + ", ".join(missing))

    frame = frame[POINT_COLUMNS].copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    valid_coordinates = (
        frame["latitude"].between(-90, 90)
        & frame["longitude"].between(-180, 180)
    )
    return frame[valid_coordinates].reset_index(drop=True)


def load_collection_points(
    points_path: str | Path = DEFAULT_POINTS_PATH,
) -> pd.DataFrame:
    path = Path(points_path)
    if not path.exists():
        return pd.DataFrame(columns=POINT_COLUMNS)
    return validate_collection_points(pd.read_csv(path))


def merge_collection_points(
    current: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.concat(
        [
            validate_collection_points(current),
            validate_collection_points(incoming),
        ],
        ignore_index=True,
    )
    merged["_dedupe_name"] = merged["name"].astype(str).str.strip().str.casefold()
    merged["_dedupe_address"] = merged["address"].astype(str).str.strip().str.casefold()
    merged = merged.drop_duplicates(
        subset=["_dedupe_name", "_dedupe_address"],
        keep="last",
    )
    return merged.drop(columns=["_dedupe_name", "_dedupe_address"]).reset_index(drop=True)


def save_collection_points(
    frame: pd.DataFrame,
    points_path: str | Path,
) -> Path:
    path = Path(points_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_collection_points(frame)
    validated.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def distance_km(
    latitude: float,
    longitude: float,
    target_latitudes: pd.Series,
    target_longitudes: pd.Series,
) -> pd.Series:
    """Khoảng cách đường chim bay theo công thức Haversine."""
    earth_radius_km = 6371.0088
    lat1 = np.radians(float(latitude))
    lon1 = np.radians(float(longitude))
    lat2 = np.radians(pd.to_numeric(target_latitudes, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(target_longitudes, errors="coerce"))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return pd.Series(
        2 * earth_radius_km * np.arcsin(np.sqrt(value)),
        index=target_latitudes.index,
    )


def rank_collection_points(
    frame: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    ranked = frame.copy()
    if ranked.empty:
        ranked["distance_km"] = pd.Series(dtype=float)
        return ranked
    ranked["distance_km"] = distance_km(
        latitude,
        longitude,
        ranked["latitude"],
        ranked["longitude"],
    )
    return ranked.sort_values("distance_km").reset_index(drop=True)


def google_maps_url(address: str) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(address)}"
