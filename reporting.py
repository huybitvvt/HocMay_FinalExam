from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from session_planner import build_session_plan


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
    uncertain = int((summary_df.get("Trạng thái", pd.Series(dtype=str)) == "Cần kiểm tra").sum()) if total else 0
    multi_object = int((summary_df.get("Nhiều vật thể", pd.Series(dtype=str)) == "Có thể").sum()) if total else 0
    group_counts = summary_df["Nhóm"].value_counts().to_dict() if total else {}
    conclusion = "Tất cả ảnh đều đủ tin cậy để tham khảo hướng xử lý." if uncertain == 0 else "Có ảnh cần kiểm tra lại trước khi xử lý rác."
    plan = build_session_plan(summary_df)

    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['Ảnh']))}</td>"
            f"<td>{html.escape(str(row['Dự đoán']))}</td>"
            f"<td>{html.escape(str(row['Nhóm']))}</td>"
            f"<td>{float(row['Độ tin cậy']) * 100:.2f}%</td>"
            f"<td>{html.escape(str(row.get('Trạng thái', '')))}</td>"
            f"<td>{html.escape(str(row.get('Kết luận xử lý', '')))}</td>"
            f"<td>{html.escape(str(row.get('Chất lượng ảnh', '')))}</td>"
            f"<td>{html.escape(str(row.get('Nhiều vật thể', '')))}</td>"
            f"<td>{html.escape(str(row['Cảnh báo']))}</td>"
            "</tr>"
        )

    group_items = "".join(f"<li>{html.escape(group)}: {count}</li>" for group, count in group_counts.items())
    action_items = "".join(f"<li>{html.escape(action)}</li>" for action in plan["priority_actions"])
    bucket_rows = []
    for _, row in plan["bucket_plan"].iterrows():
        bucket_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['Nhóm']))}</td>"
            f"<td>{html.escape(str(row['Số ảnh']))}</td>"
            f"<td>{html.escape(str(row['Ảnh liên quan']))}</td>"
            f"<td>{html.escape(str(row['Việc cần làm']))}</td>"
            "</tr>"
        )
    data_items = "".join(f"<li>{html.escape(item)}</li>" for item in plan["data_suggestions"])
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
    <div class="card"><div>Cần kiểm tra</div><div class="value">{uncertain}</div></div>
    <div class="card"><div>Có thể nhiều vật thể</div><div class="value">{multi_object}</div></div>
    <div class="card"><div>Tỷ lệ cần kiểm tra</div><div class="value">{(low_conf / total * 100 if total else 0):.1f}%</div></div>
  </div>
  <h2>Kết luận phiên kiểm thử</h2>
  <p>{html.escape(conclusion)}</p>
  <h2>Kế hoạch xử lý phiên này</h2>
  <div class="cards">
    <div class="card"><div>Điểm phân loại</div><div class="value">{plan['score']}/100</div></div>
    <div class="card"><div>Tỷ lệ tái chế/tái sử dụng</div><div class="value">{plan['recyclable_ratio'] * 100:.1f}%</div></div>
    <div class="card"><div>Số nhóm cần chuẩn bị</div><div class="value">{len(plan['bucket_plan'])}</div></div>
  </div>
  <h3>Việc cần làm trước</h3>
  <ul>{action_items}</ul>
  <h3>Phân thùng / tuyến xử lý</h3>
  <table>
    <thead><tr><th>Nhóm</th><th>Số ảnh</th><th>Ảnh liên quan</th><th>Việc cần làm</th></tr></thead>
    <tbody>{''.join(bucket_rows)}</tbody>
  </table>
  <h3>Gợi ý bổ sung dữ liệu</h3>
  <ul>{data_items}</ul>
  <h2>Phân bố nhóm rác</h2>
  <ul>{group_items}</ul>
  <h2>Chi tiết dự đoán</h2>
  <table>
    <thead>
      <tr><th>Ảnh</th><th>Dự đoán</th><th>Nhóm</th><th>Độ tin cậy</th><th>Trạng thái</th><th>Kết luận xử lý</th><th>Chất lượng ảnh</th><th>Nhiều vật thể</th><th>Cảnh báo</th></tr>
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
