# Đánh giá kết quả thực nghiệm

## Bảng so sánh model

| model          |   accuracy |   macro_f1 |   weighted_f1 |
|:---------------|-----------:|-----------:|--------------:|
| efficientnetb0 |     0.9610 |     0.9456 |        0.9610 |
| mobilenetv2    |     0.9465 |     0.9225 |        0.9466 |

## Model nên chọn: `efficientnetb0`

- Accuracy: `0.9610`
- Macro-F1: `0.9456`
- Weighted-F1: `0.9610`

## Nhận xét nhanh

- Accuracy cao, mô hình phân loại tốt trên tập validation.
- Macro-F1 tốt, hiệu năng tương đối đều giữa các lớp.

## Các lớp cần phân tích kỹ

|             |   precision |   recall |   f1-score |   support |
|:------------|------------:|---------:|-----------:|----------:|
| plastic     |      0.9255 |   0.8613 |     0.8922 |  173.0000 |
| metal       |      0.8623 |   0.9351 |     0.8972 |  154.0000 |
| white-glass |      0.9133 |   0.8839 |     0.8984 |  155.0000 |
| green-glass |      0.8714 |   0.9683 |     0.9173 |  126.0000 |
| paper       |      0.9598 |   0.9095 |     0.9340 |  210.0000 |

## Các nhầm lẫn lớn nhất

- `clothes` bị dự đoán thành `shoes`: 11 ảnh
- `plastic` bị dự đoán thành `green-glass`: 8 ảnh
- `white-glass` bị dự đoán thành `metal`: 8 ảnh
- `plastic` bị dự đoán thành `metal`: 7 ảnh
- `cardboard` bị dự đoán thành `paper`: 5 ảnh
- `paper` bị dự đoán thành `biological`: 5 ảnh
- `paper` bị dự đoán thành `cardboard`: 5 ảnh
- `plastic` bị dự đoán thành `white-glass`: 5 ảnh

## Cách trình bày khi bảo vệ

- Chọn model theo macro-F1 vì dataset lệch lớp.
- Dùng accuracy để nói hiệu năng tổng thể, dùng macro-F1 để nói độ công bằng giữa các lớp.
- Mở confusion matrix để giải thích các lớp dễ nhầm.
- Demo thêm Grad-CAM và cảnh báo độ tin cậy thấp để cho thấy hệ thống có khả năng giải thích.
- Nếu macro-F1 thấp bất thường, không đưa kết quả đó làm kết quả cuối; hãy train lại bằng bản code đã split validation theo từng lớp.
