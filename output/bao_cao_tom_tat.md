# Báo cáo tóm tắt hệ thống phát hiện phương tiện vượt đèn đỏ

## 1 Bài toán và cách hoạt động

Hệ thống phân tích video giao thông từ camera cố định để phát hiện xe đi qua vạch dừng khi đèn đỏ. Kết quả gồm video có khung xe và ID, thời điểm sự kiện, ảnh toàn cảnh và ảnh cắt phương tiện để kiểm tra lại.

Đầu vào gồm video và cấu hình của góc camera: vạch dừng, phía xe đi tới, vùng đèn điều khiển làn và vùng làn nếu có. Cần hiệu chỉnh khi đổi góc quay; hệ thống chưa tự hiểu đúng mọi giao lộ.

**Luồng xử lý:** Video → nhận diện xe và theo dõi ID → đọc trạng thái đèn → kiểm tra quỹ đạo cắt vạch → xác nhận và lưu bằng chứng.

| Thành phần | Thuật toán hoặc mô hình | Công việc |
|---|---|---|
| Nhận diện phương tiện | YOLO11x pretrained COCO | Tìm khung bao và phân loại xe đạp, ô tô, xe máy, xe buýt, xe tải |
| Theo dõi phương tiện | ByteTrack | Liên kết các lần phát hiện để duy trì ID xe qua nhiều khung hình |
| Tìm đèn giao thông | YOLOv8s pretrained COCO | Tìm đèn trong vùng giám sát đã chọn |
| Nhận biết trạng thái đèn | MobileNetV2 của Autoware và HSV | Đọc đỏ, vàng, xanh hoặc không rõ; lọc trạng thái theo thời gian |
| Xác định xe qua vạch | Hình học và quản lý trạng thái | Kiểm tra hướng di chuyển, đoạn vạch bị cắt và màu đèn |

YOLO cho biết xe nằm ở đâu trong từng ảnh; ByteTrack giúp biết đó vẫn là cùng một xe. Với đèn, MobileNetV2 được ưu tiên khi đủ tin cậy; HSV hỗ trợ đọc màu khi kết quả mạng chưa rõ. Vạch dừng ở chế độ mặc định do người dùng xác nhận. Chế độ tự đề xuất vạch dùng CLAHE, ngưỡng thích nghi và HoughLinesP chỉ phục vụ thử nghiệm.

### Điều kiện ghi nhận sự kiện

Mỗi xe được đại diện bằng một điểm, mặc định là trung điểm cạnh đáy khung bao. Hệ thống tính khoảng cách có dấu từ điểm này đến vạch: dấu cho biết xe ở phía nào, còn hướng được xác định từ cấu hình camera.

Một sự kiện được xác nhận khi đã thấy xe ở trước vạch, quỹ đạo cắt đoạn vạch đúng hướng trong lúc đèn đỏ và xe tiếp tục được quan sát phía sau vạch đủ số lần xác nhận, mặc định là 3. Một ID chỉ được ghi nhận một lần trong lượt chạy. Xe xuất hiện sẵn sau vạch chưa đủ điều kiện; điểm đáy khung bao cũng chưa phải vị trí bánh hoặc đầu xe chính xác.

## 2 Huấn luyện và kết quả thực nghiệm

**Project có fine-tune, còn pipeline video mặc định hiện dùng pretrained.** Hai phần cần được trình bày riêng để tránh gán kết quả của checkpoint này cho checkpoint khác.

- **Pretrained:** mô hình đã được bên phát triển huấn luyện trước; project tải trọng số để sử dụng.
- **Fine-tune:** tiếp tục huấn luyện từ trọng số pretrained trên dữ liệu của thí nghiệm.
- **Train từ đầu:** khởi tạo trọng số mới rồi huấn luyện; đây không phải cách thí nghiệm hiện tại sử dụng.

Notebook đã fine-tune SSD300-VGG16, YOLOv8s và YOLO11x trong 80 epoch mỗi model, seed 42, trên tập con COCO gồm 1.000 ảnh train, 200 validation và 200 test với 5 lớp phương tiện. Checkpoint tốt nhất được chọn trên validation, sau đó đánh giá trên test. SSD dùng ảnh 300 × 300; hai YOLO dùng ảnh 640 × 640. Batch của SSD và YOLOv8s là 8, YOLO11x là 4.

Pipeline video hiện dùng YOLO11x pretrained COCO để nhận diện xe, YOLOv8s pretrained COCO để tìm đèn và MobileNetV2 pretrained để đọc trạng thái. ByteTrack, luật cắt vạch và thao tác hiệu chỉnh camera không được huấn luyện trong project.

| Mô hình đã fine-tune | mAP50 | mAP50–95 | F1 | FPS trên H100 |
|---|---:|---:|---:|---:|
| SSD300-VGG16 | 0,433 | 0,257 | 0,445 | 166,8 |
| YOLOv8s | 0,519 | 0,318 | 0,529 | 119,9 |
| YOLO11x | 0,513 | 0,309 | 0,514 | 65,2 |

Trong lần thử này, YOLOv8s có mAP và F1 cao nhất; SSD nhanh nhất. Đây là kết quả detector phương tiện trên COCO subset, chưa phải chất lượng phát hiện vượt đèn đỏ. Các model khởi tạo từ pretrained COCO nên tập test COCO subset cũng chưa chứng minh khả năng tổng quát trên camera mới.

### Hiểu FPS và mAP

**FPS** là số khung hình xử lý trong một giây, dùng để đo tốc độ. Với video 30 FPS, hệ thống cần duy trì khoảng 30 FPS toàn pipeline để theo kịp đầu vào khi xử lý đủ frame. FPS trong bảng chỉ đo detector trên H100, batch 1, FP32; khi thêm tracking, đọc đèn và lưu video, tốc độ toàn hệ thống cần đo riêng.

**Vì sao YOLO11x chậm hơn SSD300-VGG16?** YOLO11x là bản lớn, có khoảng 56,9 triệu tham số so với 24,3 triệu của SSD. Đầu vào 640 × 640 cũng có khoảng 4,55 lần số pixel của 300 × 300. Quy mô tính toán và kích thước ảnh góp phần giải thích chênh lệch; tốc độ còn phụ thuộc kiến trúc và cách thực thi. Model mới hơn không mặc định nhanh hơn. SSD nhanh hơn nhưng mAP thấp hơn YOLO11x trong bảng này.

**mAP** là trung bình AP của các lớp, đánh giá khả năng phát hiện và định vị đối tượng. AP tổng hợp quan hệ Precision–Recall khi thay đổi ngưỡng tin cậy. **IoU** là diện tích giao chia diện tích hợp giữa khung dự đoán và khung nhãn; càng gần 1 càng khớp. mAP50 dùng ngưỡng IoU 0,50; mAP50–95 lấy trung bình ở các ngưỡng 0,50 đến 0,95, bước 0,05, nên chặt chẽ hơn. mAP 0,318 không có nghĩa chỉ nhận đúng 31,8% số xe.

## 3 Đánh giá hệ thống và cách trình bày

### Đánh giá sự kiện vượt đèn đỏ

Để đo toàn hệ thống, cần gán nhãn thủ công đầy đủ các sự kiện trên video test, gồm cả xe hệ thống bỏ sót và video không có vi phạm. Mỗi nhãn ghi vạch hoặc làn và thời điểm cắt theo cùng quy ước điểm đại diện của xe. Dữ liệu camera nên chia theo camera hoặc nhóm video cùng góc, tránh tách các frame sát nhau vào train và test.

- **TP:** hệ thống báo đúng một sự kiện có thật.
- **FP:** báo nhầm hoặc báo trùng sự kiện.
- **FN:** bỏ sót sự kiện có thật.
- **Precision = TP / (TP + FP):** trong các sự kiện báo ra, bao nhiêu là đúng.
- **Recall = TP / (TP + FN):** trong các sự kiện thực tế, tìm được bao nhiêu.
- **F1 = 2PR / (P + R):** kết hợp Precision và Recall.

Project có `evaluate_events.py` để ghép sự kiện một-một theo vạch và thời gian, mặc định dung sai 0,5 giây. Với nhiều xe qua gần nhau, cần kiểm tra thêm đúng chiếc xe. Hiện chưa có kết quả định lượng sự kiện đã được kiểm chứng trong workspace; không nên công bố một tỷ lệ chính xác chung cho toàn hệ thống. Cần bổ sung FPS toàn pipeline, lỗi đổi ID, nhầm màu đèn và các trường hợp che khuất, ban đêm, ngược sáng.

### Cách trình bày ngắn gọn khi bảo vệ

Trình bày theo thứ tự: bài toán và đầu ra → sơ đồ xử lý → YOLO và ByteTrack → đọc đèn và điều kiện cắt vạch → pretrained và fine-tune → bảng kết quả và giải thích FPS, mAP → demo → giới hạn và hướng đánh giá tiếp.

Đoạn giới thiệu có thể dùng: “Em xây dựng hệ thống phát hiện xe vượt vạch khi đèn đỏ từ camera cố định. Hệ thống kết hợp YOLO nhận diện xe, ByteTrack theo dõi, mô hình đọc trạng thái đèn và luật hình học kiểm tra cắt vạch. Em đã fine-tune và so sánh ba detector phương tiện. Pipeline video hiện dùng model pretrained; chất lượng sự kiện cần tiếp tục đánh giá trên video có nhãn.”

Nếu được hỏi vì sao YOLOv8s tốt nhất nhưng pipeline dùng YOLO11x, cần giải thích: bảng đo checkpoint fine-tune trên COCO subset, còn pipeline dùng YOLO11x pretrained dựa trên kiểm tra trực quan ở camera. Chưa có benchmark camera đủ để khẳng định lựa chọn đó tối ưu. Đóng góp của project là tích hợp các thành phần, xây dựng logic cắt vạch có hướng và lưu bằng chứng truy vết.

**Nguồn trong project:** `config.yaml`, `main.py`, `src/crossing_tracker.py`, `src/traffic_light_detector.py`, `src/light_classifier.py`, `evaluate_events.py` và output của `notebooks/train-traffic-3-models-modal-h100.ipynb`. Số liệu phản ánh thí nghiệm đã lưu trong notebook.
