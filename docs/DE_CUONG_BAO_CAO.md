# Đề cương báo cáo

## 1. Tên đề tài

Hệ thống phân loại rác thải qua ảnh và gợi ý xử lý rác bằng học sâu.

## 2. Lý do chọn đề tài

Phân loại rác tại nguồn là bước quan trọng để tăng tỷ lệ tái chế và giảm rác chôn lấp. Tuy nhiên người dùng thường khó xác định rác thuộc nhóm nào, đặc biệt với các vật liệu dễ nhầm như giấy/carton, các loại thủy tinh, nhựa/rác khác. Đề tài dùng mô hình học sâu để nhận diện ảnh rác và đưa ra hướng xử lý phù hợp.

## 3. Mục tiêu

- Xây dựng mô hình phân loại ảnh rác.
- So sánh ít nhất hai kiến trúc CNN transfer learning.
- Gợi ý xử lý rác bằng tiếng Việt theo nhóm: tái chế, hữu cơ, nguy hại, tái sử dụng, rác khác.
- Giải thích dự đoán bằng Grad-CAM.
- Kiểm thử bằng ảnh tự chụp và lưu phản hồi để cải thiện dữ liệu.

## 4. Dữ liệu

Dataset gồm 12 lớp:

`battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.

Cần trình bày:

- Số ảnh mỗi lớp từ `reports/dataset_distribution.csv`.
- Dataset bị lệch lớp, vì vậy dùng `class_weight`.
- Chia train/validation theo tỷ lệ 80/20.

## 5. Phương pháp

- Tiền xử lý ảnh về kích thước 224x224.
- Tăng cường dữ liệu: lật ngang, xoay nhẹ, zoom, thay đổi tương phản.
- Transfer learning với MobileNetV2 và EfficientNetB0.
- Huấn luyện hai giai đoạn:
  1. Đóng băng backbone, train head classifier.
  2. Mở một phần các lớp cuối để fine-tune.
- Đánh giá bằng accuracy, macro-F1, weighted-F1 và confusion matrix.

## 6. Chức năng hệ thống

- Upload ảnh hoặc nhiều ảnh.
- Dự đoán top-1 và top-3 xác suất.
- Gợi ý xử lý rác theo nhãn dự đoán.
- Cảnh báo độ tin cậy thấp.
- Chấm chất lượng ảnh: mờ, tối, sáng, tương phản thấp.
- Grad-CAM để giải thích vùng ảnh model chú ý.
- Lưu phản hồi nếu model dự đoán sai.
- Xuất báo cáo CSV/HTML cho phiên kiểm thử.

## 7. Kết quả thực nghiệm

Điền sau khi train xong:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| MobileNetV2 | ... | ... | ... |
| EfficientNetB0 | ... | ... | ... |

Chọn model tốt nhất theo macro-F1 vì dataset lệch lớp.

## 8. Phân tích lỗi

Các cặp lớp dễ nhầm cần kiểm tra trong confusion matrix:

- `paper` và `cardboard`.
- `brown-glass`, `green-glass`, `white-glass`.
- `plastic` và `trash`.
- `clothes` và `shoes` nếu ảnh chỉ có vật liệu vải.

Nguyên nhân có thể gồm:

- Ảnh thiếu sáng, bị mờ, vật thể quá nhỏ.
- Nền ảnh giống vật thể.
- Các lớp có đặc trưng hình dạng/màu sắc gần nhau.
- Dataset lệch lớp.

## 9. Điểm mới của đề tài

- Không chỉ phân loại, mà đưa ra hành động xử lý rác.
- Có giải thích bằng Grad-CAM.
- Có kiểm tra chất lượng ảnh đầu vào.
- Có vòng lặp phản hồi dữ liệu để cải thiện model.
- Có báo cáo kiểm thử tự động.

## 10. Hướng phát triển

- Thu thập thêm ảnh rác tại Việt Nam.
- Tích hợp bản đồ điểm thu gom pin/rác điện tử.
- Triển khai trên điện thoại.
- Thêm phát hiện nhiều vật thể rác trong một ảnh bằng object detection.
- Tự động gợi ý làm sạch rác tái chế bằng mô hình phụ.
