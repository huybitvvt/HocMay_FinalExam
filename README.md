# Hệ thống phân loại rác thải qua ảnh

Project này dùng mô hình học sâu để phân loại ảnh rác thải và đưa ra gợi ý xử lý rác bằng tiếng Việt. Ngoài phần dự đoán nhãn, mình làm thêm giao diện Streamlit để thử ảnh thực tế, xem top-3 kết quả, xem Grad-CAM và lưu phản hồi nếu model dự đoán sai.

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

Dataset bị lệch lớp, ví dụ lớp `clothes` nhiều hơn khá nhiều so với các lớp như `brown-glass`, `green-glass`, `trash`. Vì vậy khi train mình dùng `class_weight`, và khi đánh giá không chỉ nhìn accuracy mà còn dùng thêm macro-F1.

## Mô hình sử dụng

Mình thử hai mô hình transfer learning:

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

## Cấu trúc project

```text
hocmayfinalexam/
  app.py
  train_colab.py
  analyze_dataset.py
  evaluate_results.py
  merge_feedback_dataset.py
  model_utils.py
  image_quality.py
  app_insights.py
  reporting.py
  waste_rules.py
  requirements.txt
  garbage_waste_colab.ipynb
  docs/
```

Thư mục dataset `garbage_classification/` không đưa lên GitHub vì có nhiều ảnh. Dataset để trên Google Drive.

## Chạy trên Google Colab

Bật GPU trước:

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

Nếu không nhớ đường dẫn:

```bash
!find "/content/drive/MyDrive" -maxdepth 6 -type d -name "garbage_classification"
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

Đánh giá kết quả sau train:

```bash
!python evaluate_results.py --models-dir models --out-dir reports
```

Lưu model và report vào Drive:

```bash
!mkdir -p "/content/drive/MyDrive/HocMay_FinalExam/results"
!cp -r models reports "/content/drive/MyDrive/HocMay_FinalExam/results/"
```

## Chạy app Streamlit trên Colab

Nếu đã train xong và đã lưu model vào Drive, lần sau không cần train lại. Chỉ cần clone code, copy model từ Drive rồi chạy app.

```bash
!git clone https://github.com/huybitvvt/HocMay_FinalExam.git || true
%cd HocMay_FinalExam
!git pull
!pip -q install -r requirements.txt
!cp -r "/content/drive/MyDrive/HocMay_FinalExam/results/models" .
```

Tải `cloudflared` để mở app:

```bash
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
```

Chạy Streamlit:

```bash
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true > streamlit.log 2>&1 &
```

Mở tunnel:

```bash
!./cloudflared tunnel --url http://localhost:8501
```

Colab sẽ in ra link dạng `trycloudflare.com`, mở link đó để dùng app.

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

## File kết quả cần dùng cho báo cáo

Sau khi train xong, các file chính nằm trong `models/` và `reports/`:

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

## Ghi chú khi viết báo cáo

- Nên nói rõ dataset bị lệch lớp, nên dùng `class_weight`.
- Không nên chỉ báo cáo accuracy, cần thêm macro-F1.
- Model cuối cùng là EfficientNetB0.
- Một số lớp dễ nhầm: `plastic`, `metal`, `white-glass`, `paper`, `cardboard`.
- App có thêm phần gợi ý xử lý rác, cảnh báo độ tin cậy thấp, Grad-CAM và feedback.
