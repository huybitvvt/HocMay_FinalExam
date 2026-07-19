# Chạy app với Google Drive và triển khai web

## 1. Cấu trúc lưu trên Google Drive

Nhóm em dùng một thư mục lưu trữ riêng để dữ liệu không mất khi Colab reset:

```text
MyDrive/HocMay_FinalExam/runtime/
  models/
    best_model.keras
    best_model.classes.json
    waste_detector.pt                 # chỉ có sau khi train YOLO
    cleanliness_model.keras           # chỉ có sau khi train model sạch/bẩn
    cleanliness_model.classes.json
  reports/
  feedback/
  data/
    user_history.csv
    collection_points.csv             # tùy chọn, nếu muốn ghi đè dữ liệu trong repo
```

## 2. Cấu hình Colab

Mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone đúng nhánh đang phát triển:

```bash
!git clone --branch agent/add-history-collection-map --single-branch \
  https://github.com/huybitvvt/HocMay_FinalExam.git
%cd HocMay_FinalExam
```

Khai báo nơi lưu trên Drive trước khi chạy Streamlit:

```python
import os
from pathlib import Path

STORAGE_DIR = Path("/content/drive/MyDrive/HocMay_FinalExam/runtime")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["APP_STORAGE_DIR"] = str(STORAGE_DIR)
os.environ["APP_MODEL_PATH"] = str(STORAGE_DIR / "models" / "best_model.keras")
os.environ["STORAGE_DIR"] = str(STORAGE_DIR)

# Chỉ bật sau khi đã train và copy weights tương ứng.
os.environ["APP_YOLO_MODEL_PATH"] = str(STORAGE_DIR / "models" / "waste_detector.pt")
os.environ["APP_CLEANLINESS_MODEL_PATH"] = str(
    STORAGE_DIR / "models" / "cleanliness_model.keras"
)
```

Cài bản cơ bản:

```bash
!pip -q install -r requirements.txt
```

Cài bản mở rộng nếu dùng YOLO và GPS tự động:

```bash
!pip -q install -r requirements-extended.txt
```

Chạy app:

```bash
!streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```

Khi `APP_STORAGE_DIR` đã được đặt, app sẽ lưu các phần sau vào Drive:

- `feedback/`: ảnh và nhật ký người dùng sửa nhãn.
- `reports/`: prediction log và báo cáo HTML.
- `data/user_history.csv`: lịch sử và thống kê người dùng.
- `data/collection_points.csv`: danh sách điểm thu gom nhập thêm từ giao diện.

## 3. Train YOLO trên Colab

YOLO cần dataset bounding box, không dùng trực tiếp dataset classification hiện tại. Mỗi ảnh phải có file nhãn YOLO tương ứng.

```text
waste_detection_dataset/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

Sửa `path` trong `configs/waste_detection.yaml`, sau đó chạy:

```bash
!python train_yolo.py \
  --data configs/waste_detection.yaml \
  --epochs 50 \
  --batch-size 16 \
  --image-size 640 \
  --out-dir "$STORAGE_DIR/models"
```

Kết quả chính:

```text
models/waste_detector.pt
models/waste_detector_metrics.json
```

Cần báo cáo tối thiểu precision, recall, mAP@50 và mAP@50–95.

## 4. Train model sạch/bẩn

Dataset cần tách riêng:

```text
cleanliness_dataset/
  train/
    clean/
    dirty/
    uncertain/
  val/
    clean/
    dirty/
    uncertain/
  test/
    clean/
    dirty/
    uncertain/
```

Chạy:

```bash
!python train_cleanliness.py \
  --data-dir "/content/drive/MyDrive/HocMay_FinalExam/cleanliness_dataset" \
  --out-dir "$STORAGE_DIR/models" \
  --epochs 12 \
  --fine-tune-epochs 5 \
  --batch-size 32
```

Kết quả chính:

```text
models/cleanliness_model.keras
models/cleanliness_model.classes.json
models/cleanliness_metrics.json
models/cleanliness_confusion_matrix.png
```

## 5. Chạy bằng Docker

Build bản cơ bản:

```bash
docker build -t waste-classifier .
```

Build bản có YOLO và GPS:

```bash
docker build --build-arg INSTALL_EXTENDED=true -t waste-classifier-extended .
```

Chạy và gắn thư mục lưu trữ:

```bash
docker run --rm -p 8501:8501 \
  -v /duong-dan/runtime:/app/runtime \
  waste-classifier
```

Docker giúp app chạy ổn định ngoài Colab, nhưng CSV cục bộ chưa phù hợp khi có nhiều tiến trình ghi đồng thời. Nếu triển khai cho nhiều người dùng thật, cần chuyển lịch sử và feedback metadata sang cơ sở dữ liệu.
