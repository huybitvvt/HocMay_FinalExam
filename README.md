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
- Lưu lịch sử theo tên người dùng và thống kê số lượt rác theo tuần, tháng, nhóm xử lý.
- Cho phép lọc và tải lịch sử của từng người dùng ra CSV.
- Có bản đồ điểm thu gom pin/rác điện tử, GPS tự động hoặc nhập tọa độ và xếp điểm gần nhất.
- Cho phép nhập CSV để mở rộng dữ liệu điểm thu gom mà không sửa code.
- Hỗ trợ tùy chọn YOLO đa vật thể khi có `waste_detector.pt`.
- Hỗ trợ tùy chọn model đánh giá sạch/bẩn khi có `cleanliness_model.keras`.
- Có thể lưu feedback, lịch sử và báo cáo thẳng vào Google Drive qua `APP_STORAGE_DIR`.

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
  app_paths.py
  object_detection.py
  cleanliness_model.py
  collection_points.py
  session_planner.py
  user_history.py
  reporting.py
  waste_rules.py
  train_yolo.py
  train_cleanliness.py
  requirements.txt
  requirements-extended.txt
  Dockerfile
  configs/
  garbage_waste_colab.ipynb
  docs/
  models/
  reports/
  data/
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

Nếu dùng YOLO và GPS tự động:

```bash
!pip -q install -r requirements-extended.txt
```

Cấu hình để feedback, lịch sử và báo cáo được lưu trực tiếp vào Google Drive:

```python
import os

os.environ["APP_STORAGE_DIR"] = "/content/drive/MyDrive/HocMay_FinalExam/runtime"
os.environ["APP_MODEL_PATH"] = "/content/drive/MyDrive/HocMay_FinalExam/runtime/models/best_model.keras"
```

Hướng dẫn đầy đủ: [`docs/COLAB_DRIVE_VA_TRIEN_KHAI.md`](docs/COLAB_DRIVE_VA_TRIEN_KHAI.md).

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

## Lịch sử và thống kê người dùng

Người dùng nhập tên ở thanh bên và bật `Lưu lịch sử phân loại`. Mỗi phiên ảnh chỉ được ghi một lần vào:

```text
data/user_history.csv
```

Tab `Lịch sử & thống kê` hiển thị:

- Tổng số lượt ảnh đã nhận diện.
- Số lượt trong tuần hiện tại và tháng hiện tại.
- Tỷ lệ thuộc nhóm tái chế hoặc tái sử dụng.
- Số lượt rác nguy hại và tỷ lệ dự đoán cần kiểm tra lại.
- Biểu đồ theo ngày, nhóm xử lý và danh sách loại rác.

Các số liệu trên là số lượt nhận diện từ ảnh, chưa phải khối lượng rác theo kg. Tên người dùng hiện chỉ dùng để tách lịch sử, chưa phải hệ thống tài khoản có mật khẩu.

## Bản đồ điểm thu gom

Dữ liệu ban đầu nằm trong:

```text
data/collection_points.csv
```

App cho phép lọc theo thành phố, lấy GPS từ trình duyệt hoặc nhập tọa độ và xếp các điểm theo khoảng cách đường chim bay. Người dùng có thể nhập thêm file CSV điểm thu gom đã xác minh. Danh sách mặc định hiện chưa bao phủ toàn quốc. Địa chỉ được đối chiếu từ trang [Việt Nam Tái Chế](https://vietnamrecycles.com/gioi-thieu/dia-diem-thu-hoi); người dùng vẫn cần kiểm tra nguồn hoặc liên hệ đơn vị vận hành trước khi đến.

## YOLO và model sạch/bẩn

Code train và phần tích hợp vào Streamlit đã có sẵn, nhưng hai chức năng chỉ được bật khi có weights thật:

```text
models/waste_detector.pt
models/cleanliness_model.keras
models/cleanliness_model.classes.json
```

Train YOLO:

```bash
python train_yolo.py --data configs/waste_detection.yaml --out-dir models
```

Train model sạch/bẩn:

```bash
python train_cleanliness.py --data-dir /duong-dan/cleanliness_dataset --out-dir models
```

YOLO cần dataset bounding box. Model sạch/bẩn cần dataset riêng gồm `clean`, `dirty` và `uncertain`. Không thể dùng kết quả của hai chức năng này để báo cáo trước khi train và đánh giá weights.

## Triển khai Docker

```bash
docker build -t waste-classifier .
docker run --rm -p 8501:8501 -v /duong-dan/runtime:/app/runtime waste-classifier
```

Bản mở rộng có YOLO và GPS:

```bash
docker build --build-arg INSTALL_EXTENDED=true -t waste-classifier-extended .
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

## Hạn chế hiện tại

- Dataset chưa có nhiều ảnh rác được chụp trong điều kiện thực tế tại Việt Nam.
- Pipeline YOLO đã có nhưng chưa có weights được train từ dataset bounding box của đề tài; khi thiếu weights app vẫn dùng cảnh báo contour.
- Pipeline sạch/bẩn đã có nhưng chưa có dataset và weights đã đánh giá; khi thiếu model app chỉ đưa ra lưu ý sơ bộ.
- Bản đồ mới có một số điểm thu gom ban đầu tại Hà Nội và TP. Hồ Chí Minh, chưa bao phủ toàn quốc.
- Lịch sử có thể lưu vào Google Drive nhưng vẫn dùng CSV, chưa có đăng nhập và chưa phù hợp cho nhiều người dùng ghi đồng thời.
- Thống kê hiện đếm số lượt nhận diện, chưa đo được khối lượng rác thực tế.

## Hướng phát triển tiếp

1. Thu thập và gắn nhãn thêm ảnh rác thực tế tại Việt Nam, tập trung vào nhựa, kim loại và thủy tinh.
2. Gắn nhãn dataset bounding box, train YOLO và báo cáo precision, recall, mAP@50, mAP@50–95.
3. Mở rộng CSV điểm thu gom đã xác minh và bổ sung quãng đường di chuyển thực tế.
4. Thu thập nhãn sạch/bẩn, train model và báo cáo macro-F1 cùng confusion matrix.
5. Triển khai Docker trên máy chủ, tách phần suy luận thành API và chuyển lịch sử sang cơ sở dữ liệu.
6. Sau khi có dữ liệu trọng lượng hoặc số lượng vật thể đáng tin cậy, mở rộng thống kê theo tuần/tháng thành báo cáo lượng rác.
