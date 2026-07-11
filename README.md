# Hệ thống phân loại rác thải qua ảnh

Đề tài này xây dựng hệ thống phân loại rác thải từ ảnh và gợi ý cách xử lý rác bằng tiếng Việt. Ngoài phần dự đoán nhãn, nhóm em làm thêm giao diện Streamlit để thử ảnh thực tế, xem top-3 kết quả, xem Grad-CAM và lưu phản hồi nếu model dự đoán sai.

## Mục tiêu

- Phân loại ảnh rác vào 12 lớp.
- So sánh MobileNetV2 và EfficientNetB0.
- Chọn model tốt hơn dựa trên accuracy, macro-F1 và weighted-F1.
- Gợi ý cách xử lý rác sau khi dự đoán.
- Có giao diện để upload ảnh hoặc chụp ảnh bằng camera.
- Có lưu feedback để bổ sung dữ liệu cho lần train sau.

## Dataset

Dataset gồm 12 lớp:

```text
battery
biological
brown-glass
cardboard
clothes
green-glass
metal
paper
plastic
shoes
trash
white-glass
```

Tổng số ảnh hiện tại: 15,515 ảnh.

Dataset bị lệch lớp, ví dụ lớp `clothes` nhiều hơn khá nhiều so với các lớp như `brown-glass`, `green-glass`, `trash`. Vì vậy khi train nhóm em dùng `class_weight`, và khi đánh giá không chỉ nhìn accuracy mà còn dùng thêm macro-F1.

## Mô hình sử dụng

Nhóm em thử hai mô hình transfer learning:

- MobileNetV2
- EfficientNetB0

Cả hai đều dùng pretrained ImageNet, sau đó fine-tune lại cho bài toán phân loại 12 lớp rác.

Kết quả tốt nhất hiện tại:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| EfficientNetB0 | 0.9610 | 0.9456 | 0.9610 |
| MobileNetV2 | 0.9465 | 0.9225 | 0.9466 |

Model được chọn cho app là `EfficientNetB0` vì có macro-F1 và accuracy cao hơn.

## Chức năng trong app

- Upload một hoặc nhiều ảnh.
- Chụp ảnh trực tiếp bằng camera trình duyệt.
- Hiển thị nhãn dự đoán và độ tin cậy.
- Hiển thị top-3 nhãn có xác suất cao nhất.
- Gợi ý xử lý rác theo nhóm: tái chế, hữu cơ, nguy hại, tái sử dụng, rác khác.
- Cảnh báo khi model chưa đủ chắc chắn.
- Cảnh báo ảnh mờ, tối, quá sáng hoặc tương phản thấp.
- Cảnh báo nếu ảnh có thể chứa nhiều vật thể.
- Hiển thị Grad-CAM để xem vùng ảnh model chú ý.
- Lưu feedback khi model dự đoán sai.
- Xuất kết quả kiểm thử ra CSV/HTML.
- Có tab xem thông tin model và thống kê feedback.
- Tạo kế hoạch xử lý cho cả phiên ảnh: điểm phân loại, nhóm thùng cần chuẩn bị, việc cần làm trước và gợi ý bổ sung dữ liệu.

## Cấu trúc project

```text
HocMay_FinalExam/
  app.py
  train_colab.py
  analyze_dataset.py
  evaluate_results.py
  merge_feedback_dataset.py
  model_utils.py
  image_quality.py
  app_insights.py
  session_planner.py
  reporting.py
  waste_rules.py
  requirements.txt
  garbage_waste_colab.ipynb
  docs/
  models/
  reports/
```

Dataset ảnh gốc không đưa trực tiếp vào repo vì dung lượng lớn.

## Chạy trên Google Colab

Bật GPU:

```text
Runtime -> Change runtime type -> T4 GPU
```

Clone code:

```bash
!git clone https://github.com/huybitvvt/HocMay_FinalExam.git
%cd HocMay_FinalExam
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Khai báo đường dẫn dataset:

```python
DATA_DIR = '/content/drive/MyDrive/HocMay_FinalExam/hocmayfinalexam/garbage_classification'
```

Cài thư viện:

```bash
!pip -q install -r requirements.txt
```

Phân tích dataset:

```bash
!python analyze_dataset.py --data-dir "$DATA_DIR" --out-dir reports
```

Train model:

```bash
!python train_colab.py --data-dir "$DATA_DIR" --epochs 12 --batch-size 32 --models mobilenetv2 efficientnetb0
```

Đánh giá kết quả:

```bash
!python evaluate_results.py --models-dir models --out-dir reports
```

## Chạy app Streamlit

Chạy local:

```bash
streamlit run app.py
```

Chạy trên Colab:

```bash
!pip -q install -r requirements.txt
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

## Train lại với dữ liệu feedback

Khi test app, nếu model dự đoán sai thì có thể sửa nhãn và lưu ảnh vào `feedback/`.

Tạo dataset mới có trộn feedback:

```bash
python merge_feedback_dataset.py --base-dir garbage_classification --feedback-dir feedback --out-dir garbage_classification_with_feedback
```

Train lại:

```bash
python train_colab.py --data-dir garbage_classification_with_feedback --epochs 12 --batch-size 32 --models mobilenetv2 efficientnetb0
```

## Kết quả lưu trong repo

Các file kết quả chính:

```text
models/best_model.keras
models/best_model.classes.json
models/model_comparison.csv
models/efficientnetb0_confusion_matrix.png
models/efficientnetb0_history.png
reports/dataset_distribution.csv
reports/dataset_distribution.png
reports/danh_gia_ket_qua.md
```

## Ghi chú

- Model cuối cùng là EfficientNetB0.
- Dataset có lệch lớp nên nhóm em dùng `class_weight`.
- Khi đánh giá model, nhóm em dùng thêm macro-F1 thay vì chỉ dựa vào accuracy.
- Một số lớp dễ nhầm là `plastic`, `metal`, `white-glass`, `paper`, `cardboard`.
- Điểm khác với app chỉ phân loại ảnh là có thêm phần tổng hợp cả phiên kiểm thử thành kế hoạch xử lý rác.
