from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from collection_points import (
    distance_km,
    google_maps_url,
    load_collection_points,
    merge_collection_points,
    rank_collection_points,
    save_collection_points,
)


class CollectionPointTests(unittest.TestCase):
    def test_distance_to_same_coordinate_is_zero(self):
        distances = distance_km(
            10.0,
            106.0,
            pd.Series([10.0]),
            pd.Series([106.0]),
        )
        self.assertAlmostEqual(float(distances.iloc[0]), 0.0, places=6)

    def test_rank_collection_points_returns_nearest_first(self):
        points = pd.DataFrame(
            {
                "name": ["Xa", "Gần"],
                "latitude": [21.0, 10.01],
                "longitude": [105.0, 106.01],
            }
        )
        ranked = rank_collection_points(points, 10.0, 106.0)
        self.assertEqual(ranked.iloc[0]["name"], "Gần")

    def test_loader_drops_invalid_coordinates(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points.csv"
            pd.DataFrame(
                [
                    {
                        "name": "Hợp lệ",
                        "city": "TP. Hồ Chí Minh",
                        "address": "Địa chỉ A",
                        "latitude": 10.0,
                        "longitude": 106.0,
                        "accepted_waste": "Pin",
                        "source_url": "https://example.com",
                        "verified_date": "2026-07-19",
                    },
                    {
                        "name": "Sai tọa độ",
                        "city": "TP. Hà Nội",
                        "address": "Địa chỉ B",
                        "latitude": 200.0,
                        "longitude": 105.0,
                        "accepted_waste": "Pin",
                        "source_url": "https://example.com",
                        "verified_date": "2026-07-19",
                    },
                ]
            ).to_csv(path, index=False)
            loaded = load_collection_points(path)
            self.assertEqual(loaded["name"].tolist(), ["Hợp lệ"])

    def test_google_maps_url_encodes_address(self):
        url = google_maps_url("82 Bà Huyện Thanh Quan")
        self.assertIn("google.com/maps/search", url)
        self.assertNotIn(" ", url)

    def test_merge_prefers_newer_duplicate_and_can_save(self):
        columns = {
            "city": "TP. Hà Nội",
            "latitude": 21.0,
            "longitude": 105.0,
            "accepted_waste": "Pin",
            "source_url": "https://example.com",
            "verified_date": "2026-07-19",
        }
        current = pd.DataFrame(
            [{"name": "Điểm A", "address": "Địa chỉ A", **columns}]
        )
        incoming = pd.DataFrame(
            [
                {
                    "name": "điểm a",
                    "address": "ĐỊA CHỈ A",
                    **{**columns, "verified_date": "2026-07-20"},
                }
            ]
        )

        merged = merge_collection_points(current, incoming)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["verified_date"], "2026-07-20")

        with TemporaryDirectory() as temp_dir:
            path = save_collection_points(merged, Path(temp_dir) / "points.csv")
            self.assertTrue(path.exists())
            self.assertEqual(len(load_collection_points(path)), 1)


if __name__ == "__main__":
    unittest.main()
