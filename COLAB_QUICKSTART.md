# Colab Quickstart

Chạy các cell dưới đây trong Google Colab.

## 1. Bật GPU

`Runtime` -> `Change runtime type` -> `T4 GPU` -> `Save`.

Kiểm tra:

```bash
!nvidia-smi
```

## 2. Clone code

```bash
!git clone https://github.com/huybitvvt/HocMay_FinalExam.git
%cd HocMay_FinalExam
```

## 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4. Tìm dataset

```bash
!find "/content/drive/MyDrive" -maxdepth 6 -type d -name "garbage_classification"
```

Copy đường dẫn in ra, ví dụ:

```python
DATA_DIR = '/content/drive/MyDrive/HocMay_FinalExam/hocmayfinalexam/garbage_classification'
```

## 5. Cài thư viện và train

```bash
!pip -q install -r requirements.txt
!python analyze_dataset.py --data-dir "$DATA_DIR" --out-dir reports
!python train_colab.py --data-dir "$DATA_DIR" --epochs 12 --batch-size 32 --models mobilenetv2 efficientnetb0
```

Test nhanh trước khi train đầy đủ:

```bash
!python train_colab.py --data-dir "$DATA_DIR" --epochs 3 --batch-size 32 --models mobilenetv2
```

## 6. Xem kết quả

```python
import pandas as pd
pd.read_csv('models/model_comparison.csv')
```

Đánh giá tự động để lấy nhận xét đưa vào báo cáo:

```bash
!python evaluate_results.py --models-dir models --out-dir reports
```

File nhận xét được lưu tại:

```text
reports/danh_gia_ket_qua.md
```

## 7. Chạy Streamlit

```bash
!pip -q install pyngrok
```

```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```

```bash
!streamlit run app.py --server.port 8501 --server.headless true
```
