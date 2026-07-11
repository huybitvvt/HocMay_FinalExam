# Báo cáo tóm tắt đề tài

## 1. Tên đề tài

Hệ thống phân loại rác thải qua ảnh và gợi ý xử lý rác.

Đề tài tập trung vào việc nhận diện loại rác từ ảnh, sau đó đưa ra hướng xử lý phù hợp như tái chế, rác hữu cơ, rác nguy hại, tái sử dụng hoặc rác còn lại. Ngoài phần phân loại ảnh, hệ thống còn có giao diện thử nghiệm để upload ảnh, chụp ảnh bằng camera, xem độ tin cậy, xem Grad-CAM và tổng hợp kế hoạch xử lý cho nhiều ảnh rác trong cùng một phiên.

## 2. Lý do chọn đề tài

Phân loại rác tại nguồn là việc cần thiết nhưng trong thực tế không phải lúc nào người dùng cũng biết một loại rác nên bỏ vào nhóm nào. Một số loại như giấy và carton, các loại thủy tinh, nhựa và rác khác khá dễ nhầm. Vì vậy nhóm em chọn đề tài này để thử áp dụng học sâu vào bài toán phân loại rác qua ảnh, đồng thời bổ sung phần gợi ý xử lý để kết quả dự đoán có tính ứng dụng hơn.

Nếu chỉ dừng ở việc model trả về nhãn, hệ thống sẽ khá giống nhiều bài phân loại ảnh khác. Vì vậy nhóm em phát triển thêm phần giao diện và phần tổng hợp xử lý rác sau dự đoán, để hệ thống gần với tình huống sử dụng thực tế hơn.

## 3. Mục tiêu thực hiện

- Xây dựng mô hình phân loại ảnh rác thành 12 lớp.
- So sánh hai mô hình transfer learning là MobileNetV2 và EfficientNetB0.
- Đánh giá model bằng accuracy, macro-F1 và weighted-F1.
- Xây dựng giao diện Streamlit để thử ảnh thực tế.
- Gợi ý cách xử lý rác sau khi dự đoán.
- Hiển thị Grad-CAM để xem vùng ảnh model chú ý.
- Cảnh báo khi ảnh đầu vào chưa tốt hoặc model chưa đủ chắc chắn.
- Lưu phản hồi khi model dự đoán sai để có thể bổ sung dữ liệu cho lần train sau.
- Tổng hợp nhiều ảnh thành một kế hoạch xử lý rác cho cả phiên kiểm thử.

## 4. Dataset

Dataset hiện có 15,515 ảnh, gồm 12 lớp:

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

Một điểm cần chú ý là dataset bị lệch lớp. Ví dụ lớp `clothes` có số ảnh nhiều hơn đáng kể so với các lớp như `brown-glass`, `green-glass`, `trash`. Nếu chỉ dùng accuracy thì kết quả có thể bị ảnh hưởng bởi các lớp nhiều ảnh. Vì vậy trong quá trình train nhóm em có dùng `class_weight`, và khi đánh giá có dùng thêm macro-F1 để xem model học có đều giữa các lớp hay không.

## 5. Tiền xử lý và chia dữ liệu

Ảnh được resize về kích thước 224x224 để phù hợp với MobileNetV2 và EfficientNetB0. Trong quá trình train, nhóm em dùng thêm augmentation như lật ảnh, xoay nhẹ, zoom và thay đổi tương phản để model đỡ phụ thuộc vào một kiểu ảnh cố định.

Dữ liệu được chia thành tập train và validation. Phần chia validation được làm theo từng lớp để tránh trường hợp một số lớp bị thiếu hoặc quá ít trong tập validation. Đây là điểm quan trọng vì dataset bị lệch lớp, nếu chia không cẩn thận thì kết quả macro-F1 có thể không phản ánh đúng chất lượng model.

## 6. Mô hình sử dụng

Nhóm em thử hai mô hình:

- MobileNetV2
- EfficientNetB0

Cả hai đều dùng transfer learning với trọng số pretrained trên ImageNet. Quy trình train gồm hai phần: đầu tiên train phần phân loại phía trên backbone, sau đó fine-tune một số lớp cuối để model thích nghi tốt hơn với ảnh rác trong dataset.

EfficientNetB0 được chọn làm model chính cho app vì cho kết quả tốt hơn MobileNetV2 trên tập validation.

## 7. Kết quả hiện tại

Kết quả so sánh model:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| EfficientNetB0 | 0.9610 | 0.9456 | 0.9610 |
| MobileNetV2 | 0.9465 | 0.9225 | 0.9466 |

EfficientNetB0 đạt accuracy 96.10%, macro-F1 94.56% và weighted-F1 96.10%. Kết quả này tốt hơn MobileNetV2, nên nhóm em dùng EfficientNetB0 làm model chính.

Macro-F1 đạt 94.56% là kết quả quan trọng vì chỉ số này tính trung bình đều trên các lớp. Với dataset lệch lớp, macro-F1 giúp đánh giá công bằng hơn so với chỉ nhìn accuracy.

## 8. Các lớp còn dễ nhầm

Một số lớp vẫn có thể nhầm với nhau:

- `paper` và `cardboard`: vì đều là vật liệu giấy, hình dạng trong ảnh có thể gần giống nhau.
- `brown-glass`, `green-glass`, `white-glass`: khác nhau chủ yếu ở màu sắc, dễ bị ảnh hưởng bởi ánh sáng.
- `plastic`, `metal`, `white-glass`: có thể bị nhầm khi vật thể phản sáng hoặc nền ảnh quá sáng.
- `clothes` và `shoes`: một số ảnh có vật liệu vải hoặc hình dạng không rõ.

Những lỗi này là hợp lý với bài toán ảnh thực tế, vì ảnh rác thường có nền phức tạp, ánh sáng không ổn định và vật thể có thể bị che khuất hoặc bị bẩn.

## 9. Giao diện ứng dụng

Giao diện được làm bằng Streamlit. Các chức năng chính:

- Tải một hoặc nhiều ảnh.
- Chụp ảnh trực tiếp bằng camera.
- Hiển thị nhãn dự đoán, độ tin cậy và top-3 xác suất.
- Gợi ý xử lý rác theo từng nhóm.
- Hiển thị Grad-CAM.
- Cảnh báo ảnh mờ, tối, quá sáng hoặc tương phản thấp.
- Cảnh báo khi model chưa chắc chắn.
- Cảnh báo ảnh có thể chứa nhiều vật thể.
- Lưu phản hồi nếu model dự đoán sai.
- Xuất kết quả kiểm thử ra CSV/HTML.

Giao diện không chỉ phục vụ việc thử model, mà còn giúp kiểm tra nhanh ảnh thực tế và lưu lại kết quả kiểm thử.

## 10. Gợi ý xử lý rác

Sau khi dự đoán, hệ thống không chỉ hiện nhãn mà còn đưa ra hướng xử lý. Ví dụ:

- Pin được xếp vào nhóm rác nguy hại, cần đưa tới điểm thu gom pin hoặc rác điện tử.
- Rác hữu cơ có thể tách riêng để ủ compost hoặc bỏ vào thùng hữu cơ.
- Giấy, carton, nhựa, kim loại, thủy tinh được gợi ý theo hướng tái chế nếu đủ sạch.
- Quần áo và giày dép được gợi ý tái sử dụng hoặc quyên góp nếu còn dùng được.
- Rác khác được đưa vào nhóm còn lại, nhưng vẫn cần kiểm tra để tránh lẫn pin, hóa chất hoặc vật sắc nhọn.

Phần này giúp kết quả dự đoán có tính thực tế hơn, vì người dùng không chỉ biết ảnh thuộc lớp nào mà còn biết nên xử lý như thế nào.

## 11. Grad-CAM

Nhóm em thêm Grad-CAM để xem vùng ảnh mà model chú ý khi dự đoán. Nếu vùng heatmap tập trung vào vật thể rác thì dự đoán có vẻ hợp lý hơn. Nếu heatmap tập trung vào nền hoặc vùng không liên quan, có thể ảnh đó chưa tốt hoặc model đang học đặc trưng chưa đúng.

Grad-CAM không làm tăng accuracy trực tiếp, nhưng giúp mô hình dễ giải thích hơn. Đây là điểm hữu ích khi trình bày kết quả của mô hình học sâu, vì CNN thường bị xem là khó giải thích.

## 12. Cảnh báo độ tin cậy và chất lượng ảnh

Trong thực tế, không nên ép model luôn phải đưa ra một kết luận chắc chắn. Vì vậy nhóm em thêm phần cảnh báo khi:

- Độ tin cậy thấp.
- Top-1 và top-2 quá gần nhau.
- Ảnh bị mờ.
- Ảnh quá tối hoặc quá sáng.
- Độ tương phản thấp.
- Ảnh có thể chứa nhiều vật thể.

Khi gặp các trường hợp này, hệ thống vẫn đưa ra kết quả tham khảo nhưng sẽ báo người dùng nên kiểm tra lại hoặc chụp lại ảnh rõ hơn.

## 13. Phần khác biệt chính của đề tài

Điểm khác biệt lớn nhất hiện tại là hệ thống không chỉ phân loại từng ảnh riêng lẻ. Sau khi người dùng đưa nhiều ảnh vào, hệ thống sẽ tổng hợp thành một kế hoạch xử lý cho cả phiên ảnh.

Phần kế hoạch xử lý gồm:

- Điểm phân loại của phiên kiểm thử.
- Tỷ lệ rác tái chế hoặc tái sử dụng.
- Số nhóm rác cần chuẩn bị.
- Việc cần làm trước, ví dụ rác nguy hại cần gom riêng.
- Bảng phân thùng hoặc tuyến xử lý theo từng nhóm rác.
- Gợi ý lớp nào nên bổ sung dữ liệu nếu phiên đó có nhiều ảnh chưa chắc chắn.

Nhờ vậy, app không chỉ trả lời “ảnh này là gì”, mà còn hỗ trợ trả lời “với nhóm ảnh rác này thì nên xử lý như thế nào”.

## 14. Feedback và cải thiện dữ liệu

Trong app có phần phản hồi nếu model dự đoán sai. Người dùng có thể chọn lại nhãn đúng, sau đó hệ thống lưu ảnh vào thư mục `feedback/`. Các ảnh này có thể được trộn lại vào dataset bằng script `merge_feedback_dataset.py` để train lại ở lần sau.

Phần này giúp hệ thống có một vòng lặp cải thiện dữ liệu:

```text
Dự đoán -> phát hiện sai -> lưu feedback -> trộn dữ liệu -> train lại
```

Đây là cách đơn giản nhưng thực tế để cải thiện model dần theo dữ liệu mới.

## 15. Tiến độ hiện tại

Đến hiện tại project đã hoàn thành các phần chính:

- Đã chuẩn bị dataset 12 lớp.
- Đã train và so sánh MobileNetV2 với EfficientNetB0.
- Đã chọn EfficientNetB0 làm model chính.
- Đã có kết quả đánh giá model.
- Đã có giao diện Streamlit.
- Đã có upload ảnh và chụp ảnh bằng camera.
- Đã có gợi ý xử lý rác.
- Đã có Grad-CAM.
- Đã có cảnh báo ảnh chưa tốt và model chưa chắc chắn.
- Đã có lưu feedback.
- Đã có xuất báo cáo CSV/HTML.
- Đã có phần kế hoạch xử lý cho cả phiên ảnh.

## 16. Hạn chế hiện tại

Project vẫn còn một số hạn chế:

- Model đang phân loại theo ảnh, chưa phát hiện nhiều vật thể riêng biệt trong cùng một ảnh.
- Nếu ảnh có nhiều loại rác cùng lúc, hệ thống chỉ dự đoán theo vật thể nổi bật nhất.
- Dataset chưa chắc đã phản ánh đầy đủ rác trong điều kiện thực tế ở Việt Nam.
- Một số lớp gần nhau về vật liệu hoặc màu sắc vẫn có thể nhầm.
- Grad-CAM chỉ hỗ trợ giải thích vùng chú ý, chưa phải là bằng chứng chắc chắn model hiểu đúng vật thể.
- Phần gợi ý xử lý rác đang dựa trên luật được thiết kế thủ công, chưa cá nhân hóa theo từng địa phương.

## 17. Hướng phát triển tiếp

Nếu tiếp tục phát triển, nhóm em sẽ ưu tiên các hướng sau:

1. Thu thập thêm ảnh rác thực tế tại Việt Nam, đặc biệt là các lớp model còn dễ nhầm như nhựa, kim loại, thủy tinh và giấy/carton.
2. Thêm object detection để phát hiện nhiều vật thể rác trong một ảnh, thay vì chỉ phân loại toàn ảnh.
3. Thêm bản đồ hoặc danh sách điểm thu gom pin, rác điện tử theo khu vực.
4. Cải thiện phần đánh giá độ sạch của rác tái chế, ví dụ phát hiện chai/hộp bị dính thức ăn hoặc dầu mỡ.
5. Triển khai app ổn định hơn trên một nền tảng web thay vì chạy tạm qua Colab.
6. Thêm cơ chế lưu lịch sử người dùng và thống kê lượng rác theo ngày/tuần.
7. Tối ưu model để chạy nhẹ hơn trên điện thoại hoặc thiết bị cấu hình thấp.

## 18. Kết luận

Đề tài đã xây dựng được một hệ thống phân loại rác thải qua ảnh dựa trên mô hình học sâu. EfficientNetB0 cho kết quả tốt nhất với accuracy 96.10% và macro-F1 94.56%. Ngoài phần phân loại, hệ thống còn có giao diện thử nghiệm, gợi ý xử lý rác, Grad-CAM, cảnh báo độ tin cậy, lưu feedback và tổng hợp kế hoạch xử lý cho nhiều ảnh.

Điểm nhóm em muốn nhấn mạnh là hệ thống không chỉ dừng ở nhận diện nhãn rác, mà cố gắng đưa kết quả dự đoán vào một quy trình xử lý rác thực tế hơn.
