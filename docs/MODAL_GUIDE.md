# Notebook và kết quả Modal

Lượt train chính thức đã chạy trên Modal H100 bằng
[`train-traffic-3-models-modal-h100.ipynb`](../notebooks/train-traffic-3-models-modal-h100.ipynb):
80 epoch mỗi model, dataset COCO subset 1.000 train / 200 val / 200 test.
Notebook trong dự án có output của lượt chạy và được giữ nguyên.

Kết quả đầy đủ được xuất trên Volume `traffic-training` tại
`/mnt/traffic-training/exports/traffic_h100_v1_full.zip`. Bản ZIP đã tải về nên được
lưu ngoài repo để dự phòng. Trong ZIP, `yolo11x/weights/best.pt` là trọng số đã được
chép vào dự án thành `weights/yolo11x.pt`. `yolov8s/weights/best.pt` trong ZIP là
model phương tiện 5 lớp; `weights/yolov8s.pt` của dự án là checkpoint COCO 80 lớp
đã tải sẵn để tìm đèn giao thông.

Các file dùng cho báo cáo nằm trong ZIP: `test_benchmark.csv`, `test_benchmark.json`
và `paper_artifacts/EXPERIMENTAL_RESULTS.md` cùng các biểu đồ. Xem
[PAPER_GUIDE.md](PAPER_GUIDE.md) để đọc đúng phạm vi các số đo.

Nếu chạy một thí nghiệm mới:

1. Upload notebook ở [Modal Notebooks](https://modal.com/notebooks), chọn GPU trong
   sidebar và gắn Volume ở Files. Volume hiện tại dùng đường dẫn `/mnt/traffic-training`.
2. Đặt `RUN_NAME` mới trước khi chạy để không ghi vào lượt train đã hoàn tất.
   `PILOT=True` chạy 3 epoch/model để kiểm tra; `PILOT=False` dùng `FULL_EPOCHS`.
3. Nếu dùng dataset camera riêng, xem [DATASET_GUIDE.md](DATASET_GUIDE.md), upload
   ZIP vào Volume và chọn `DATA_SOURCE="zip"`.
4. Sau khi tải kết quả, tắt kernel trong giao diện Modal để ngừng tính phí compute.
   Dataset và checkpoint trên Volume vẫn tồn tại qua lần khởi động kernel.

Chỉ resume một run bị ngắt khi có `last` checkpoint và giữ nguyên dataset, seed,
epochs, imgsz và tên run. Không cần chạy lại lượt `traffic_h100_v1_full` đã xong.
Giao diện và cách gắn Volume: [tài liệu Modal Notebooks](https://modal.com/docs/guide/notebooks).
