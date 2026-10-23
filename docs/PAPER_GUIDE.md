# Kết quả thực nghiệm nhận diện phương tiện

Notebook Modal đã train SSD300-VGG16, YOLOv8s và YOLO11x trong 80 epoch/model trên
tập con COCO gồm 1.000 ảnh train, 200 val và 200 test, cùng 5 lớp phương tiện.
Checkpoint tốt nhất được chọn trên val; bảng sau là kết quả test hiển thị trong
[`notebook đã chạy`](../notebooks/train-traffic-3-models-modal-h100.ipynb).

| Model | mAP50–95 | mAP50 | FPS đo trên H100 |
|---|---:|---:|---:|
| SSD300-VGG16 | 0,257 | 0,433 | 166,8 |
| YOLOv8s | **0,318** | **0,519** | 119,9 |
| YOLO11x | 0,309 | 0,513 | 65,2 |

Trong lần chạy này, YOLOv8s có mAP cao nhất và SSD có FPS detector cao nhất.
FPS được đo batch 1, FP32 trên cùng H100; đầu vào SSD là 300×300, YOLO là 640×640.
Đó là phép so sánh cấu hình thực dụng, không cô lập riêng ảnh hưởng của kiến trúc.
YOLO11x hiện vẫn là model được chọn trong `config.yaml` để chạy video. Nếu mục tiêu
là chọn model triển khai theo độ chính xác, cần đánh giá lại cả hai YOLO trên video
camera thực tế trước khi quyết định.

Khi viết phần thực nghiệm, lấy bảng chính xác và hình từ ZIP Modal:

```text
test_benchmark.csv
test_benchmark.json
experiment.json
paper_artifacts/
  EXPERIMENTAL_RESULTS.md
  dataset_distribution.png
  training_curves.png
  accuracy_speed_complexity.png
  per_class_ap.png
  per_class_metrics.csv
  confusion_ssd300_vgg16.png
  confusion_yolov8s.png
  confusion_yolo11x.png
  pr_ssd300_vgg16.png
  pr_yolov8s.png
  pr_yolo11x.png
  qualitative_comparison.png
```

Nên trình bày cách chia dữ liệu, pretrained COCO, batch/epoch/seed, mAP và chỉ số
từng lớp, tốc độ, đường học và vài ảnh thành công/thất bại. Kết quả hiện chỉ có
một seed và test trên COCO subset; chưa đo chất lượng tracking, vạch dừng, màu đèn
hay sự kiện vượt đèn đỏ trên bộ camera độc lập. Không dùng kết quả này để khẳng định
độ chính xác của toàn hệ thống ngoài thực tế.
