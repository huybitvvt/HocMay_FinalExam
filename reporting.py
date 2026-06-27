from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd


REPORT_DIR = Path("reports")


def append_prediction_log(rows: list[dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORT_DIR / "prediction_log.csv"
    df = pd.DataFrame(rows)
    df.insert(0, "time", datetime.now().isoformat(timespec="seconds"))
    df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False, encoding="utf-8-sig")
    return log_path


def build_html_report(summary_df: pd.DataFrame, title: str = "Báo cáo kiểm thử phân loại rác") -> str:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(summary_df)
    low_conf = int((summary_df["Cảnh báo"] == "Thấp").sum()) if total else 0
    group_counts = summary_df["Nhóm"].value_counts().to_dict() if total else {}

    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['Ảnh']))}</td>"
            f"<td>{html.escape(str(row['Dự đoán']))}</td>"
            f"<td>{html.escape(str(row['Nhóm']))}</td>"
            f"<td>{float(row['Độ tin cậy']) * 100:.2f}%</td>"
            f"<td>{html.escape(str(row.get('Chất lượng ảnh', '')))}</td>"
            f"<td>{html.escape(str(row['Cảnh báo']))}</td>"
            "</tr>"
        )

    group_items = "".join(f"<li>{html.escape(group)}: {count}</li>" for group, count in group_counts.items())
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; }}
    h1 {{ font-size: 26px; margin-bottom: 4px; }}
    .meta {{ color: #5f6368; margin-bottom: 24px; }}
    .cards {{ display: flex; gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #dadce0; border-radius: 8px; padding: 12px 16px; min-width: 160px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #dadce0; padding: 8px; text-align: left; }}
    th {{ background: #f1f3f4; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">Tạo lúc: {created_at}</div>
  <div class="cards">
    <div class="card"><div>Tổng ảnh</div><div class="value">{total}</div></div>
    <div class="card"><div>Cảnh báo thấp</div><div class="value">{low_conf}</div></div>
    <div class="card"><div>Tỷ lệ cần kiểm tra</div><div class="value">{(low_conf / total * 100 if total else 0):.1f}%</div></div>
  </div>
  <h2>Phân bố nhóm rác</h2>
  <ul>{group_items}</ul>
  <h2>Chi tiết dự đoán</h2>
  <table>
    <thead>
      <tr><th>Ảnh</th><th>Dự đoán</th><th>Nhóm</th><th>Độ tin cậy</th><th>Chất lượng ảnh</th><th>Cảnh báo</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>"""


def save_html_report(summary_df: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"bao_cao_kiem_thu_{datetime.now():%Y%m%d_%H%M%S}.html"
    report_path.write_text(build_html_report(summary_df), encoding="utf-8")
    return report_path
