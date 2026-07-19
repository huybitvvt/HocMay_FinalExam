from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from user_history import (
    append_user_history,
    build_history_stats,
    filter_history_period,
    normalize_user_id,
    read_user_history,
)


SAMPLE_ROWS = [
    {
        "Ảnh": "pin.jpg",
        "Dự đoán": "Pin / rác nguy hại",
        "Nhóm": "Nguy hại",
        "Độ tin cậy": 0.91,
        "Trạng thái": "Tin cậy",
        "Chất lượng ảnh": "Ảnh đủ rõ để dự đoán",
        "Nhiều vật thể": "Không",
        "Kết luận xử lý": "Điểm thu gom pin",
    },
    {
        "Ảnh": "chai.jpg",
        "Dự đoán": "Nhựa",
        "Nhóm": "Tái chế",
        "Độ tin cậy": 0.62,
        "Trạng thái": "Cần kiểm tra",
        "Chất lượng ảnh": "Ảnh hơi tối",
        "Nhiều vật thể": "Có thể",
        "Kết luận xử lý": "Thùng nhựa tái chế",
    },
]


class UserHistoryTests(unittest.TestCase):
    def test_normalize_user_id_supports_vietnamese_name(self):
        self.assertEqual(normalize_user_id("Nguyễn Doãn Huy"), "nguyen-doan-huy")
        self.assertEqual(normalize_user_id(""), "khach")

    def test_append_is_idempotent_per_session(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.csv"
            _, first_count = append_user_history(
                SAMPLE_ROWS,
                "Nguyễn Doãn Huy",
                session_id="session-01",
                history_path=path,
                recorded_at=datetime(2026, 7, 19, 10, 0),
            )
            _, second_count = append_user_history(
                SAMPLE_ROWS,
                "Nguyễn Doãn Huy",
                session_id="session-01",
                history_path=path,
                recorded_at=datetime(2026, 7, 19, 10, 0),
            )

            history = read_user_history("nguyen-doan-huy", path)
            self.assertEqual(first_count, 2)
            self.assertEqual(second_count, 0)
            self.assertEqual(len(history), 2)

    def test_build_history_stats(self):
        frame = pd.DataFrame(
            [
                {
                    "recorded_at": "2026-07-13T10:00:00",
                    "group": "Tái chế",
                    "prediction": "Nhựa",
                    "confidence": 0.8,
                    "status": "Tin cậy",
                },
                {
                    "recorded_at": "2026-07-19T10:00:00",
                    "group": "Nguy hại",
                    "prediction": "Pin / rác nguy hại",
                    "confidence": 0.6,
                    "status": "Cần kiểm tra",
                },
                {
                    "recorded_at": "2026-06-20T10:00:00",
                    "group": "Tái sử dụng",
                    "prediction": "Quần áo / vải",
                    "confidence": 0.9,
                    "status": "Tin cậy",
                },
            ]
        )
        stats = build_history_stats(frame, now=datetime(2026, 7, 19, 12, 0))

        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["this_week"], 2)
        self.assertEqual(stats["this_month"], 2)
        self.assertEqual(stats["hazardous_count"], 1)
        self.assertAlmostEqual(stats["recyclable_ratio"], 2 / 3)
        self.assertAlmostEqual(stats["uncertain_rate"], 1 / 3)

    def test_filter_history_period(self):
        frame = pd.DataFrame(
            {
                "recorded_at": [
                    "2026-07-19T08:00:00",
                    "2026-07-12T08:00:00",
                    "2026-06-01T08:00:00",
                ]
            }
        )
        selected = filter_history_period(
            frame,
            "7_days",
            now=datetime(2026, 7, 19, 12, 0),
        )
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
