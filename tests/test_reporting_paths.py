from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from reporting import append_prediction_log, save_html_report


class ReportingPathTests(unittest.TestCase):
    def test_outputs_can_be_routed_to_external_directory(self):
        rows = [
            {
                "Ảnh": "chai.jpg",
                "Dự đoán": "Nhựa",
                "Nhóm": "Tái chế",
                "Độ tin cậy": 0.9,
                "Trạng thái": "Tin cậy",
                "Kết luận xử lý": "Thùng nhựa",
                "Chất lượng ảnh": "Ảnh đủ rõ để dự đoán",
                "Nhiều vật thể": "Không",
                "Cảnh báo": "OK",
            }
        ]
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "drive" / "reports"
            log_path = append_prediction_log(rows, output_dir)
            report_path = save_html_report(pd.DataFrame(rows), output_dir)

            self.assertEqual(log_path.parent, output_dir)
            self.assertEqual(report_path.parent, output_dir)
            self.assertTrue(log_path.exists())
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
