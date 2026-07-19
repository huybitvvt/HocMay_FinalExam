# Kịch bản bảo vệ – phiên bản cập nhật

## 1. Mở đầu

Nhóm em xây dựng hệ thống phân loại rác thải qua ảnh bằng học sâu. Phần cốt lõi là phân loại ảnh vào 12 lớp. Sau dự đoán, hệ thống hiển thị top-3, Grad-CAM, cảnh báo độ tin cậy và gợi ý người dùng nên xử lý rác theo hướng tái chế, hữu cơ, nguy hại, tái sử dụng hoặc rác còn lại.

## 2. Trình bày dữ liệu

Dataset có 15.515 ảnh thuộc 12 lớp. Dữ liệu bị lệch rõ rệt, trong đó `clothes` chiếm khoảng 34,32%, còn `brown-glass` chỉ khoảng 3,91%.

Nhóm em thực hiện:

- Resize ảnh về 224×224.
- Augmentation bằng lật ngang, xoay nhẹ, zoom và thay đổi tương phản.
- Chia train/validation theo từng lớp với seed cố định.
- Dùng `class_weight` trong quá trình train.

Vì dataset lệch lớp, nhóm em không chỉ dùng accuracy mà còn dùng macro-F1 và weighted-F1.

## 3. Trình bày mô hình

Nhóm em so sánh MobileNetV2 và EfficientNetB0. Cả hai đều pretrained trên ImageNet. Đầu tiên nhóm em đóng băng backbone để train phần phân loại, sau đó fine-tune 30 lớp cuối với learning rate thấp hơn.

Kết quả:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| EfficientNetB0 | 0,9610 | 0,9456 | 0,9610 |
| MobileNetV2 | 0,9465 | 0,9225 | 0,9466 |

Nhóm em chọn EfficientNetB0 vì tốt hơn ở cả accuracy và macro-F1.

## 4. Trình bày lỗi

Một số nhầm lẫn đáng chú ý:

- `clothes` thành `shoes`: 11 ảnh.
- `plastic` thành `green-glass`: 8 ảnh.
- `white-glass` thành `metal`: 8 ảnh.
- `plastic` thành `metal`: 7 ảnh.
- `cardboard` thành `paper`: 5 ảnh.

Nguyên nhân chính là vật liệu gần nhau, hiện tượng phản sáng, màu sắc bị ảnh hưởng bởi ánh sáng và hình dạng vật thể không rõ.

## 5. Kịch bản demo

1. Nhập tên người dùng để tách lịch sử.
2. Upload một ảnh rõ, chỉ có một vật thể chính.
3. Chỉ ra nhãn, confidence, top-3 và nhóm xử lý.
4. Mở Grad-CAM để giải thích vùng ảnh model tập trung.
5. Upload một ảnh mờ hoặc tối để demo cảnh báo chất lượng.
6. Upload nhiều ảnh để demo kế hoạch xử lý theo phiên.
7. Nếu model đoán sai, sửa nhãn và lưu feedback.
8. Mở lịch sử để xem số lượt nhận diện theo tuần/tháng.
9. Mở điểm thu gom để minh họa khoảng cách đường chim bay.
10. Xuất CSV hoặc HTML.

Không nên bật YOLO hoặc model sạch/bẩn khi chưa có weights đã train và đánh giá.

## 6. Câu hỏi phản biện thường gặp

### Vì sao dùng macro-F1?

Dataset bị lệch lớp. Accuracy có thể cao nếu model học tốt các lớp nhiều ảnh. Macro-F1 tính F1 riêng từng lớp rồi lấy trung bình đều nên phản ánh công bằng hơn giữa 12 lớp.

### Vì sao vẫn báo accuracy?

Accuracy thể hiện tỷ lệ dự đoán đúng tổng thể. Nhóm em dùng accuracy để nhìn hiệu năng chung, macro-F1 để đánh giá đều giữa các lớp và weighted-F1 để tính đến số lượng ảnh của từng lớp.

### Vì sao dùng class_weight?

Lớp nhiều ảnh sẽ xuất hiện trong quá trình tối ưu nhiều hơn. `class_weight` làm lỗi ở các lớp ít ảnh có ảnh hưởng lớn hơn, giúp hạn chế model chỉ ưu tiên lớp đông.

### Vì sao chọn EfficientNetB0?

EfficientNetB0 đạt accuracy 96,10% và macro-F1 94,56%, đều cao hơn MobileNetV2. Vì vậy nhóm em chọn EfficientNetB0 làm model chính cho app.

### Grad-CAM có chứng minh model hiểu đúng ảnh không?

Không. Grad-CAM chỉ cho biết vùng ảnh đóng góp mạnh vào dự đoán. Nó hỗ trợ kiểm tra model đang chú ý vào vật thể hay nền, nhưng không phải bằng chứng tuyệt đối model hiểu đúng.

### Hệ thống có nhận diện nhiều vật thể không?

Chưa. Model hiện phân loại toàn ảnh và phù hợp nhất khi có một vật thể chính. Cảnh báo nhiều vật thể hiện dựa trên contour, chỉ là heuristic. Hướng phát triển là gắn bounding box và train YOLO.

### Hệ thống đã nhận biết rác sạch/bẩn chưa?

Chưa. Hiện app chỉ đưa ra lưu ý chung dựa trên loại rác và chất lượng ảnh. Muốn kết luận sạch/bẩn cần dataset riêng và model được đánh giá bằng macro-F1 cùng confusion matrix.

### Bản đồ có bao phủ toàn quốc không?

Không. App chỉ cung cấp một số điểm thu gom pin/rác điện tử tham khảo tại Hà Nội và TP. Hồ Chí Minh. Khoảng cách được tính theo đường chim bay bằng công thức Haversine, không phải quãng đường giao thông.

### Lịch sử có phải tài khoản người dùng không?

Không. Tên chỉ được dùng để tách lịch sử kiểm thử. Hệ thống chưa có mật khẩu hoặc xác thực người dùng.

### Thống kê có phải lượng rác thực tế không?

Không. Hệ thống thống kê số lượt ảnh được nhận diện. Nhóm em không gọi đây là khối lượng rác vì app chưa có dữ liệu cân nặng.

### Vì sao chạy bằng Colab?

Colab hỗ trợ GPU và thuận tiện cho việc train, thử model. Code được lấy từ GitHub; model, feedback, lịch sử và báo cáo có thể lưu trên Google Drive. Hạn chế là app phụ thuộc phiên runtime nên chưa phải web production.

### Feedback có làm model học ngay không?

Không. Ảnh và nhãn sửa được lưu lại để nhóm em kiểm tra. Sau đó dùng `merge_feedback_dataset.py` để trộn dữ liệu và train lại. Việc này tránh đưa phản hồi chưa kiểm duyệt trực tiếp vào model.

## 7. Cách kết thúc

Nhóm em đã hoàn thành hệ thống phân loại 12 lớp và chọn EfficientNetB0 làm model chính. Ngoài kết quả thực nghiệm, hệ thống còn có top-3, Grad-CAM, kiểm tra chất lượng ảnh, gợi ý xử lý, feedback và kế hoạch xử lý theo phiên. Các phần YOLO, nhận biết sạch/bẩn và triển khai web ổn định được giữ ở hướng phát triển vì chưa có đủ dữ liệu và kết quả đánh giá.
