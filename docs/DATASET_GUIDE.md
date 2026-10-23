# Chuẩn bị dữ liệu phương tiện

Notebook `notebooks/train-traffic-3-models-modal-h100.ipynb` đã dùng tập con COCO
gồm 1.000 ảnh train, 200 val và 200 test có bounding box. Không cần API key.
Các model fine-tune từ pretrained COCO, không train từ trọng số ngẫu nhiên.
Tập con này dùng kiểm tra quy trình, không đại diện camera Việt Nam và không phải COCO test chính thức.

## Dữ liệu riêng cho kết quả đồ án

1. Thu thập video từ nhiều camera/góc nhìn, ngày/đêm, nắng/mưa, giao thông thưa/đông.
2. **Chia camera/video trước khi trích frame**: khoảng 70% train, 15% val, 15% test.
   Giữ một camera hoặc nhóm video hoàn toàn riêng cho test; các đoạn cắt từ cùng video gốc
   phải ở cùng tập. Nếu chỉ có một camera, tách theo phiên/ngày quay và ghi rõ giới hạn tổng quát hóa.
3. Trích khoảng một ảnh mỗi 1–3 giây tùy mức chuyển động, bỏ ảnh gần trùng. Có thể dùng FFmpeg:
   `ffmpeg -i video.mp4 -vf fps=1 frames/cam01_%06d.jpg` (tạo thư mục `frames` trước).
4. Dùng CVAT hoặc Roboflow Annotate để vẽ **bounding box**, không vẽ polygon.
5. Thống nhất 5 lớp: `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
   Quy định nhất quán cách xử lý xe che khuất, xe nhỏ, xe đạp điện. Gán đầy đủ mọi xe thuộc lớp đã chọn.
6. Giữ thêm ảnh âm tính không có xe; file nhãn `.txt` rỗng. Không để nhãn thiếu.
7. Xuất **YOLO Detection** (YOLOv8 format dùng được cho cả YOLO11 và bộ chuyển đổi SSD).
8. Nén dataset thành ZIP, upload lên Modal Volume. Trong notebook chọn `DATA_SOURCE = 'zip'`
   và sửa `DATASET_ZIP`. Dùng `RUN_NAME` mới khi thay dataset.

Có thể bắt đầu vài nghìn ảnh đa dạng và tăng dữ liệu ở các tình huống model còn sai.
Không có số ảnh tối thiểu bảo đảm chất lượng. Không nhân bản ảnh để bù lớp hiếm vào val/test.

## Cấu trúc ZIP

```text
traffic_dataset/
  data.yaml
  images/
    train/cam01_000001.jpg
    val/cam07_000001.jpg
    test/cam09_000001.jpg
  labels/
    train/cam01_000001.txt
    val/cam07_000001.txt
    test/cam09_000001.txt
```

`data.yaml`:

```yaml
train: images/train
val: images/val
test: images/test
names: [bicycle, car, motorcycle, bus, truck]
```

Một dòng nhãn: `2 0.50 0.60 0.20 0.30` nghĩa là xe máy, tâm ở (50%, 60%), rộng 20%, cao 30% ảnh.
Class IDs bắt đầu từ 0; SSD tự cộng 1 để dành ID 0 cho background.
Notebook kiểm tra tọa độ và ảnh trùng byte giữa các tập, không tự phát hiện frame cùng camera hay ảnh chỉnh sáng.

## Đánh giá

- Chọn checkpoint trên val; chỉ đánh giá test sau khi chốt cấu hình.
- Cùng dữ liệu và COCO evaluator cho cả ba model; không suy mAP từ precision/recall.
- SSD đầu vào 300, YOLO mặc định 640; augmentation và optimizer khác nhau.
  Đây là so sánh cấu hình thực dụng, không chứng minh khác biệt chỉ đến từ kiến trúc.
- Kết quả đo hiện tại trên COCO subset nằm trong [PAPER_GUIDE.md](PAPER_GUIDE.md);
  không suy từ tên model rằng YOLO11x chắc chắn chính xác nhất.
- Giai đoạn sau cần nhãn sự kiện vi phạm để kiểm tra chất lượng toàn hệ thống vượt đèn đỏ.

Nguồn: [COCO](https://cocodataset.org/#download), [định dạng YOLO](https://docs.ultralytics.com/datasets/detect/).
Nguồn ảnh và license IDs được lưu trong `sources_and_licenses.json` của dataset COCO tải về.
