# Kịch bản bảo vệ

## Mở đầu

Em xây dựng hệ thống phân loại rác thải qua ảnh bằng học sâu. Điểm chính của hệ thống là không chỉ trả về nhãn rác, mà còn gợi ý người dùng nên xử lý rác như thế nào, ví dụ tái chế, hữu cơ, nguy hại hoặc rác còn lại.

## Trình bày dữ liệu

Dataset có 12 lớp. Trước khi train, em phân tích phân bố lớp bằng `analyze_dataset.py`. Dữ liệu bị lệch lớp, ví dụ một số lớp có nhiều ảnh hơn hẳn các lớp khác. Vì vậy em dùng `class_weight` và đánh giá bằng macro-F1 thay vì chỉ dùng accuracy.

## Trình bày mô hình

Em dùng transfer learning với MobileNetV2 và EfficientNetB0. Quy trình train gồm hai bước: đầu tiên đóng băng backbone để train lớp phân loại, sau đó fine-tune một số lớp cuối. Hai mô hình được so sánh bằng accuracy, macro-F1, weighted-F1 và confusion matrix.

## Demo hệ thống

1. Upload ảnh rác rõ nét.
2. Chỉ ra kết quả dự đoán, top-3 xác suất và nhóm xử lý.
3. Mở Grad-CAM để giải thích vùng ảnh model tập trung.
4. Upload ảnh mờ hoặc tối để hệ thống cảnh báo chất lượng ảnh.
5. Nếu model đoán sai, sửa nhãn và lưu vào feedback.
6. Xuất báo cáo HTML/CSV của phiên kiểm thử.

## Câu hỏi phản biện thường gặp

### Vì sao dùng macro-F1?

Vì dataset bị lệch lớp. Accuracy có thể cao nếu model học tốt lớp nhiều ảnh, nhưng macro-F1 tính trung bình đều trên các lớp nên phản ánh công bằng hơn.

### Grad-CAM có tác dụng gì?

Grad-CAM giúp quan sát vùng ảnh đóng góp mạnh vào dự đoán. Nếu heatmap tập trung vào vật thể rác, dự đoán đáng tin hơn. Nếu heatmap tập trung vào nền, có thể model đang học sai đặc trưng.

### Vì sao cần cảnh báo độ tin cậy thấp?

Trong bài toán thực tế, dự đoán sai có thể dẫn đến xử lý rác sai. Vì vậy hệ thống cần cảnh báo khi xác suất thấp và hiển thị top-3 để người dùng kiểm tra.

### Feedback dùng để làm gì?

Khi người dùng sửa nhãn sai, ảnh được lưu vào `feedback/`. Sau đó có thể trộn feedback vào dataset bằng `merge_feedback_dataset.py` và train lại. Đây là vòng lặp cải thiện dữ liệu.

### Hạn chế hiện tại là gì?

Hệ thống hiện phân loại một vật thể chính trong ảnh. Nếu ảnh có nhiều loại rác cùng lúc, mô hình phân loại ảnh đơn có thể không đủ. Hướng phát triển là dùng object detection để phát hiện nhiều vật thể.
