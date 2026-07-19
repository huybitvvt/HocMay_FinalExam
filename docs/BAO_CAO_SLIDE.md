# Nội dung bộ slide bảo vệ – phiên bản cập nhật

Bộ slide gồm 14 trang, bám đúng chức năng hiện có. Phần YOLO và nhận biết sạch/bẩn chỉ xuất hiện trong hướng phát triển, không trình bày như kết quả đã hoàn thành.

---

## Slide 1 – Tên đề tài

### Nội dung trên slide

**HỆ THỐNG PHÂN LOẠI RÁC THẢI QUA ẢNH<br>
VÀ GỢI Ý XỬ LÝ RÁC BẰNG HỌC SÂU**

*Từ nhận diện ảnh đến hướng xử lý thực tế*

Học phần: Học máy và Khai phá dữ liệu

### Lời trình bày

Nhóm em xây dựng hệ thống nhận diện ảnh rác thải vào 12 lớp. Sau khi dự đoán, hệ thống không chỉ trả về nhãn mà còn giải thích vùng model chú ý, cảnh báo ảnh chưa tốt và gợi ý cách xử lý rác bằng tiếng Việt.

### Gợi ý hình ảnh

- Một ảnh rác đầu vào.
- Mũi tên đi qua model.
- Kết quả gồm nhãn, xác suất và gợi ý xử lý.

---

## Slide 2 – Thành viên và phân công

### Nội dung trên slide

| Thành viên | Phân công chính |
| --- | --- |
| Nguyễn Doãn Huy | Trưởng nhóm, train model, so sánh MobileNetV2 và EfficientNetB0 |
| Bùi Quốc Việt | Chuẩn bị dataset, phân tích phân bố và hỗ trợ Colab |
| Đỗ Văn Đạt | Giao diện Streamlit, upload/camera và top-3 |
| Vũ Thị Thu Hường | Luật gợi ý xử lý rác bằng tiếng Việt |
| Trương Đức Đàm | Grad-CAM, thống kê kết quả và hỗ trợ báo cáo |

### Lời trình bày

Nhóm em chia công việc theo bốn phần: dữ liệu, mô hình, giao diện và giải thích kết quả. Các thành viên cùng kiểm thử app và hoàn thiện nội dung báo cáo.

---

## Slide 3 – Bài toán và mục tiêu

### Nội dung trên slide

**Vấn đề**

- Người dùng dễ nhầm giấy với carton, nhựa với kim loại hoặc thủy tinh.
- Biết tên loại rác chưa đủ, người dùng còn cần biết nên xử lý thế nào.

**Mục tiêu**

1. Phân loại ảnh rác thành 12 lớp.
2. So sánh hai mô hình transfer learning.
3. Gợi ý xử lý rác sau dự đoán.
4. Tăng khả năng giải thích và kiểm tra kết quả.

**Phạm vi hiện tại:** phân loại vật thể rác nổi bật trong ảnh.

### Lời trình bày

Trọng tâm của đề tài là bài toán phân loại ảnh. Phần gợi ý xử lý, Grad-CAM và kiểm tra ảnh được bổ sung để kết quả phân loại có ý nghĩa sử dụng rõ hơn.

---

## Slide 4 – Dataset và quy trình chuẩn bị dữ liệu

### Nội dung trên slide

**Dataset**

- 15.515 ảnh.
- 12 lớp rác.
- Lệch lớp rõ rệt: `clothes` chiếm khoảng 34,32%, trong khi `brown-glass` khoảng 3,91%.

**Chuẩn bị dữ liệu**

```text
Ảnh gốc
→ Resize 224×224
→ Augmentation
→ Chia train/validation theo từng lớp
→ Tính class_weight
```

Augmentation gồm: lật ngang, xoay nhẹ, zoom và thay đổi tương phản.

### Lời trình bày

Dataset bị lệch lớp khá mạnh nên nhóm em không chia dữ liệu một cách ngẫu nhiên trên toàn bộ ảnh. Nhóm em chia validation theo từng lớp và sử dụng `class_weight` để hạn chế model thiên về lớp có nhiều ảnh.

### Gợi ý hình ảnh

- Biểu đồ `reports/dataset_distribution.png`.
- Làm nổi bật lớp `clothes`, `brown-glass`, `green-glass` và `trash`.

---

## Slide 5 – Mô hình và kết quả thực nghiệm

### Nội dung trên slide

**Hai mô hình**

- MobileNetV2.
- EfficientNetB0.
- Pretrained ImageNet.
- Train head trước, sau đó fine-tune 30 lớp cuối.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| **EfficientNetB0** | **0,9610** | **0,9456** | **0,9610** |
| MobileNetV2 | 0,9465 | 0,9225 | 0,9466 |

**Model được chọn: EfficientNetB0**

### Lời trình bày

EfficientNetB0 tốt hơn MobileNetV2 ở cả ba chỉ số. Nhóm em đặc biệt quan tâm macro-F1 vì chỉ số này tính F1 đều giữa 12 lớp, phù hợp hơn với dataset lệch lớp.

---

## Slide 6 – Phân tích lỗi và các lớp dễ nhầm

### Nội dung trên slide

**Các lớp có F1 thấp hơn**

| Lớp | F1-score |
| --- | ---: |
| `plastic` | 0,8922 |
| `metal` | 0,8972 |
| `white-glass` | 0,8984 |
| `green-glass` | 0,9173 |
| `paper` | 0,9340 |

**Một số nhầm lẫn lớn**

- `clothes → shoes`: 11 ảnh.
- `plastic → green-glass`: 8 ảnh.
- `white-glass → metal`: 8 ảnh.
- `plastic → metal`: 7 ảnh.
- `cardboard → paper`: 5 ảnh.

### Lời trình bày

Nhựa, kim loại và thủy tinh dễ nhầm khi vật thể phản sáng. Giấy và carton gần nhau về vật liệu. Quần áo và giày dép có thể cùng xuất hiện đặc trưng vải. Những kết quả này cho thấy accuracy cao nhưng vẫn cần xem confusion matrix và chỉ số từng lớp.

### Gợi ý hình ảnh

- `models/efficientnetb0_confusion_matrix.png`.
- Chỉ đánh dấu 4–5 ô nhầm lẫn lớn, không đưa toàn bộ bảng classification report lên slide.

---

## Slide 7 – Quy trình hoạt động của hệ thống

### Nội dung trên slide

```text
Upload ảnh / Camera
        ↓
Kiểm tra chất lượng ảnh
        ↓
EfficientNetB0 phân loại 12 lớp
        ↓
Nhãn + Confidence + Top-3
        ↓
Grad-CAM + Gợi ý xử lý
        ↓
Feedback + Báo cáo + Lịch sử
```

- **Đầu vào:** một hoặc nhiều ảnh.
- **Đầu ra:** kết quả từng ảnh và kế hoạch xử lý cho cả phiên.

### Lời trình bày

App được xây dựng bằng Streamlit. Người dùng có thể tải nhiều ảnh hoặc chụp bằng camera. Mỗi ảnh được kiểm tra chất lượng, phân loại và đưa ra top-3. Sau đó hệ thống hiển thị Grad-CAM, hướng xử lý và lưu kết quả nếu người dùng cho phép.

---

## Slide 8 – Giải thích và cơ chế cảnh báo

### Nội dung trên slide

**Grad-CAM**

- Cho biết vùng ảnh đóng góp mạnh vào dự đoán.
- Giúp kiểm tra model chú ý vào vật thể hay nền ảnh.

**Cảnh báo**

- Confidence thấp.
- Top-1 và top-2 quá gần nhau.
- Ảnh mờ, tối, quá sáng hoặc tương phản thấp.
- Ảnh có dấu hiệu chứa nhiều vật thể.

> Cảnh báo nhiều vật thể hiện là heuristic dựa trên contour, chưa phải YOLO.

### Lời trình bày

Grad-CAM không chứng minh model hiểu ảnh giống con người, nhưng giúp quan sát vùng model đang sử dụng. Hệ thống cũng không ép mọi dự đoán đều đáng tin mà cảnh báo khi ảnh hoặc xác suất chưa đủ tốt.

---

## Slide 9 – Từ nhãn dự đoán đến hành động xử lý

### Nội dung trên slide

| Nhóm xử lý | Lớp tiêu biểu | Gợi ý |
| --- | --- | --- |
| Nguy hại | Pin | Gom riêng, đưa đến điểm thu gom pin/rác điện tử |
| Hữu cơ | Rác sinh học | Tách riêng, có thể ủ compost |
| Tái chế | Giấy, carton, nhựa, kim loại, thủy tinh | Làm sạch sơ, giữ khô, phân loại riêng |
| Tái sử dụng | Quần áo, giày dép | Làm sạch, quyên góp hoặc thu hồi |
| Rác khác | `trash` | Kiểm tra không lẫn pin, hóa chất hoặc vật sắc nhọn |

### Lời trình bày

Phần gợi ý được xây dựng bằng luật tiếng Việt theo từng lớp. Đây không phải một model ML thứ hai. Mục tiêu là chuyển kết quả phân loại thành hành động dễ hiểu, đồng thời cảnh báo riêng đối với rác nguy hại.

---

## Slide 10 – Các chức năng hỗ trợ đã hoàn thành

### Nội dung trên slide

**Kế hoạch theo phiên**

- Tổng hợp nhiều ảnh theo nhóm xử lý.
- Ưu tiên rác nguy hại.
- Gợi ý thùng hoặc tuyến xử lý cần chuẩn bị.

**Lịch sử**

- Tách lịch sử theo tên người dùng.
- Thống kê số lượt nhận diện theo tuần/tháng.
- Xuất lịch sử ra CSV.

**Điểm thu gom tham khảo**

- Một số điểm tại Hà Nội và TP. Hồ Chí Minh.
- Tính khoảng cách đường chim bay bằng công thức Haversine.

### Lời trình bày

Thống kê của nhóm em là số lượt ảnh được nhận diện, không phải khối lượng rác. Tên chỉ dùng để tách lịch sử, chưa phải tài khoản đăng nhập. Bản đồ là danh sách điểm tham khảo, không được trình bày là bản đồ toàn quốc.

---

## Slide 11 – Feedback loop và cải thiện dữ liệu

### Nội dung trên slide

```text
1. Dự đoán
      ↓
2. Người dùng phát hiện sai
      ↓
3. Chọn lại nhãn đúng
      ↓
4. Lưu ảnh và feedback log
      ↓
5. Trộn vào dataset
      ↓
6. Train và đánh giá lại
```

**File hỗ trợ:** `merge_feedback_dataset.py`

Feedback, lịch sử và báo cáo có thể lưu vào Google Drive khi chạy Colab.

### Lời trình bày

Feedback không làm model thay đổi ngay lập tức. Ảnh được lưu lại cùng nhãn người dùng sửa, sau đó nhóm em kiểm tra, trộn vào dataset và train lại. Cách này tránh việc đưa dữ liệu chưa kiểm duyệt trực tiếp vào model.

---

## Slide 12 – Hạn chế hiện tại

### Nội dung trên slide

1. Model phân loại toàn ảnh, phù hợp nhất khi ảnh có một vật thể chính.
2. Chưa có bộ kiểm thử riêng gồm ảnh rác thực tế tại Việt Nam.
3. Một số lớp gần nhau về màu sắc hoặc vật liệu vẫn còn nhầm.
4. Cảnh báo nhiều vật thể hiện mới là heuristic.
5. Gợi ý xử lý dựa trên luật chung, chưa cá nhân hóa theo quy định từng địa phương.
6. App chạy thử nghiệm trên Colab nên phụ thuộc vào phiên runtime.
7. Bản đồ mới có một số điểm tham khảo và chỉ tính khoảng cách đường chim bay.

### Lời trình bày

Nhóm em tách rõ kết quả đã đo được và phần chưa làm. Accuracy 96,10% là kết quả trên tập validation của dataset hiện tại, chưa có nghĩa model sẽ đạt tương tự với mọi ảnh rác ngoài thực tế.

---

## Slide 13 – Hướng phát triển

### Nội dung trên slide

**1. Dữ liệu thực tế tại Việt Nam**

- Thu thập thêm ảnh nền phức tạp, ánh sáng khác nhau.
- Tập trung vào nhựa, kim loại, thủy tinh, giấy và carton.

**2. Phát hiện nhiều vật thể**

- Gắn nhãn bounding box.
- Train YOLO và đánh giá bằng precision, recall, mAP.

**3. Nhận biết sạch/bẩn**

- Xây dựng dataset `clean`, `dirty`, `uncertain`.
- Chỉ tích hợp sau khi có macro-F1 và confusion matrix.

**4. Web app ổn định**

- Triển khai ngoài Colab để không phụ thuộc phiên runtime.

### Lời trình bày

Nhóm em đã xác định kiến trúc cho YOLO và model sạch/bẩn, nhưng chưa xem đây là kết quả hoàn thành vì chưa có dataset và weights được đánh giá. Ưu tiên đầu tiên vẫn là dữ liệu thực tế và kiểm thử ngoài dataset hiện tại.

---

## Slide 14 – Kết luận

### Nội dung trên slide

**EfficientNetB0**

- Accuracy: **96,10%**
- Macro-F1: **94,56%**
- Weighted-F1: **96,10%**

**Hệ thống hướng tới ba giá trị**

1. **Nhận diện được** – phân loại 12 lớp và hiển thị top-3.
2. **Giải thích được** – Grad-CAM và cảnh báo độ tin cậy.
3. **Hành động được** – gợi ý xử lý, feedback và kế hoạch theo phiên.

**Cảm ơn thầy/cô và các bạn đã lắng nghe.**

### Lời trình bày

Nhóm em đã hoàn thành phần cốt lõi là phân loại rác qua ảnh và gợi ý xử lý. Kết quả thực nghiệm cho thấy EfficientNetB0 tốt hơn MobileNetV2. Điểm khác biệt của hệ thống là kết hợp kết quả ML với khả năng giải thích, cảnh báo và hướng xử lý thực tế.
