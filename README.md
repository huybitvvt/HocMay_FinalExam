# Hệ thống phân loại rác thải qua ảnh và gợi ý xử lý rác

Đây là bản hoàn chỉnh phục vụ bảo vệ học phần Học máy và Khai phá dữ liệu. Hệ thống không chỉ phân loại ảnh rác bằng deep learning, mà còn đưa ra gợi ý xử lý rác bằng tiếng Việt, giải thích dự đoán bằng Grad-CAM, kiểm thử ảnh thực tế và lưu phản hồi để cải thiện dữ liệu.

## Chức năng đã có

- Phân loại 12 lớp rác: `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.
- So sánh `MobileNetV2` và `EfficientNetB0` bằng `accuracy`, `macro_f1`, `weighted_f1`.
- Xử lý lệch lớp bằng `class_weight`.
- Fine-tuning transfer learning.
- Xuất `classification_report`, `confusion_matrix`, biểu đồ accuracy/loss và bảng so sánh model.
- Giao diện Streamlit tiếng Việt.
- Upload một hoặc nhiều ảnh rác tự chụp.
- Gợi ý xử lý rác: tái chế, hữu cơ, nguy hại, tái sử dụng, rác khác.
- Grad-CAM để giải thích vùng ảnh model chú ý.
- Cảnh báo khi độ tin cậy thấp.
- Đánh giá chất lượng ảnh: mờ, quá tối, quá sáng, tương phản thấp.
- Gợi ý độ sạch tái chế cho giấy, nhựa, kim loại, thủy tinh.
- Lưu nhật ký kiểm thử vào `reports/prediction_log.csv`.
- Xuất CSV và HTML báo cáo kiểm thử.
- Người dùng sửa nhãn sai, app lưu ảnh vào `feedback/` để tái huấn luyện.
- Script trộn dữ liệu feedback vào dataset cho vòng train tiếp theo.

## Cấu trúc file

```text
hocmayfinalexam/
  garbage_classification/          # dataset gốc
  app.py                           # giao diện Streamlit
  train_colab.py                   # train và so sánh model
  analyze_dataset.py               # phân tích phân bố dataset
  merge_feedback_dataset.py        # trộn dataset gốc với feedback
  model_utils.py                   # inference và Grad-CAM
  image_quality.py                 # đánh giá chất lượng ảnh
  reporting.py                     # log và báo cáo HTML
  waste_rules.py                   # luật xử lý rác tiếng Việt
  garbage_waste_colab.ipynb        # notebook Colab
  requirements.txt
```

## Quy trình chạy trên Google Colab

Khuyến nghị: code lấy từ GitHub, dataset lớn vẫn để Google Drive.

1. Upload dataset lên Google Drive, ví dụ:

```text
/content/drive/MyDrive/HocMay_FinalExam/hocmayfinalexam/garbage_classification
```

2. Trong Colab bật GPU: `Runtime` -> `Change runtime type` -> `T4 GPU`.
3. Clone code từ GitHub:

```bash
!git clone https://github.com/huybitvvt/HocMay_FinalExam.git
%cd HocMay_FinalExam
```

4. Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. Khai báo đường dẫn dataset:

```python
DATA_DIR = '/content/drive/MyDrive/HocMay_FinalExam/hocmayfinalexam/garbage_classification'
```

Nếu không chắc đường dẫn dataset:

```bash
!find "/content/drive/MyDrive" -maxdepth 5 -type d -name "garbage_classification"
```

Hoặc chạy bằng lệnh:

```bash
pip install -r requirements.txt
python analyze_dataset.py --data-dir "$DATA_DIR" --out-dir reports
python train_colab.py --data-dir "$DATA_DIR" --epochs 12 --batch-size 32 --models mobilenetv2 efficientnetb0
```

Kết quả quan trọng nằm trong `models/` và `reports/`:

- `models/best_model.keras`
- `models/best_model.classes.json`
- `models/model_comparison.csv`
- `models/*_classification_report.csv`
- `models/*_confusion_matrix.png`
- `models/*_history.png`
- `reports/dataset_distribution.csv`
- `reports/dataset_distribution.png`
- `reports/danh_gia_ket_qua.md`

Sau khi train xong, chạy đánh giá tự động:

```bash
python evaluate_results.py --models-dir models --out-dir reports
```

Script này chọn model tốt nhất theo macro-F1, liệt kê lớp yếu, các cặp lớp dễ nhầm và cảnh báo nếu kết quả có dấu hiệu bất thường.

## Chạy giao diện Streamlit

```bash
streamlit run app.py
```

Trên Colab có thể chạy:

```bash
pip install pyngrok
streamlit run app.py --server.port 8501 --server.headless true
```

## Train lại với dữ liệu feedback

Sau khi test bằng ảnh tự chụp, nếu model đoán sai thì sửa nhãn trong app. Ảnh sẽ được lưu vào `feedback/<class>/`.

Tạo dataset mới đã trộn feedback:

```bash
python merge_feedback_dataset.py --base-dir garbage_classification --feedback-dir feedback --out-dir garbage_classification_with_feedback
```

Train lại:

```bash
python train_colab.py --data-dir garbage_classification_with_feedback --epochs 12 --batch-size 32 --models mobilenetv2 efficientnetb0
```

## Kịch bản demo khi bảo vệ

1. Mở bảng phân bố dataset để chỉ ra dữ liệu lệch lớp và cách xử lý bằng `class_weight`.
2. Mở bảng so sánh MobileNetV2/EfficientNetB0, chọn model theo `macro_f1`.
3. Upload ảnh tự chụp rõ nét, trình bày dự đoán + top-3 xác suất.
4. Mở Grad-CAM để giải thích model tập trung vào vật thể hay nền.
5. Chỉ ra gợi ý xử lý rác: pin là nguy hại, giấy/nhựa/kim loại/thủy tinh là tái chế, hữu cơ có thể ủ compost.
6. Upload ảnh mờ/tối để chứng minh hệ thống có cảnh báo chất lượng ảnh và độ tin cậy thấp.
7. Cố tình chọn một ảnh model đoán sai, sửa nhãn và lưu vào `feedback/`.
8. Trình bày script trộn feedback để train lại, thể hiện vòng lặp cải thiện dữ liệu.
9. Xuất báo cáo HTML/CSV của phiên kiểm thử.

## Điểm khác biệt để nhấn mạnh

- Có tính ứng dụng sau dự đoán: hệ thống gợi ý cách xử lý rác cụ thể, không dừng ở nhãn phân loại.
- Có khả năng giải thích bằng Grad-CAM, giúp mô hình minh bạch hơn.
- Có đánh giá chất lượng ảnh đầu vào, phù hợp tình huống người dùng chụp ảnh thực tế.
- Có vòng lặp phản hồi dữ liệu: sửa nhãn sai, lưu ảnh, trộn vào dataset, train lại.
- Có báo cáo kiểm thử tự động bằng CSV/HTML, phù hợp trình bày kết quả thực nghiệm.
- Dùng `macro_f1` để đánh giá công bằng hơn khi dataset lệch lớp.

## Lưu ý khi viết báo cáo

Dataset hiện lệch lớp mạnh, ví dụ `clothes` nhiều hơn các lớp như `brown-glass`, `green-glass`, `trash`. Vì vậy không nên chỉ báo cáo accuracy. Cần đưa thêm macro-F1, confusion matrix và phân tích các cặp lớp dễ nhầm như `brown-glass`/`green-glass`/`white-glass`, `paper`/`cardboard`, `plastic`/`trash`.
