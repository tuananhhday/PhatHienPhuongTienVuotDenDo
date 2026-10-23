# Nâng độ chính xác cho camera cố định

## Quyết định triển khai

Chọn pipeline có hiệu chỉnh góc quay một lần. Vạch, vùng xe và quan hệ đèn–làn được người
dùng xác nhận; xe và màu đèn được nhận diện tự động trong toàn video. Không có bằng chứng
hiện tại cho lời hứa “video nào cũng đúng”. Nếu không thấy đèn điều khiển làn đó thì chỉ
có thể xác định xe qua vạch, không thể kết luận vượt đèn đỏ từ video này.

Lỗi cũ đã xử lý: ROI red1 dùng cho các video khác; đoán hướng xe theo độ cao của vạch;
loại ID vĩnh viễn vì từng ở sau một vạch khi không đỏ; hủy ứng viên vì mất màu sau thời điểm
cắt; kiểm tra đoạn vạch chỉ bằng tọa độ x. Cấu hình mới gắn với dấu vân tay nội dung video,
lưu từng làn và hướng bằng một điểm phía trước vạch, dùng giao đoạn thẳng thực sự.

ByteTrack nhận bbox xuống conf 0.10 để giữ quan sát yếu cho liên kết, đầu vào xe tăng lên 960.
Đây là lựa chọn cần đánh giá trade-off false positive/tốc độ, không tự bảo đảm recall cao hơn.
Ứng viên phải có vị trí trước vạch và quan sát sau vạch; mất track ngắn tối đa 0.2 giây được
cho phép khi cả hai đầu đều đỏ. Mất ID hoàn toàn vẫn có thể bỏ sót hoặc đếm lại một xe.
Điểm mặc định là giữa đáy bbox, không phải bánh/mũi xe thực; có thể chỉnh `anchor: [x,y]`
trong profile theo tỉ lệ bbox nhưng phải dùng cùng quy ước khi gán ground truth.

## Model đã chuẩn bị và nguồn

| Thành phần | Dùng hiện tại | Giới hạn |
|---|---|---|
| Xe | YOLO11x pretrained COCO, `weights/coco/yolo11x.pt` | Chưa fine-tune camera mục tiêu; chưa có mAP camera thực |
| Đèn | Autoware MobileNetV2 ONNX, nhận crop 224×224 | Dữ liệu Nhật; chuyển miền có thể sai, đặc biệt mũi tên và ban đêm |
| Vạch/làn | Xác nhận hình học từng góc quay | Cần làm lại khi đổi góc, zoom, cắt cảnh; chưa hỗ trợ camera di động |
| Vạch tự động | CV/Hough chỉ chạy thử | Không phải model đã train để phân biệt stop line/crosswalk |

Nguồn chính thức: [YOLO11](https://docs.ultralytics.com/models/yolo11/),
[Autoware classifier](https://huggingface.co/AutowareFoundation/traffic_light_classifier),
[Ultralytics tracking](https://docs.ultralytics.com/modes/track/).
`download_models.py` pin Autoware revision `527c4905afd8f1bfcc03253529a105f7fa5a6d36`
và asset Ultralytics `v8.3.0`, kiểm tra SHA256; không ghi đè model fine-tune của dự án.
Autoware dùng Apache-2.0; pretrained YOLO11 theo AGPL-3.0/Enterprise của nhà phát hành.

Đã kiểm tra thêm [TLD-READY](https://github.com/KASTEL-MobilityLab/traffic-light-detection):
nhóm tác giả công bố detector đèn trên DTLD và nghiên cứu gán đèn cho hướng đi.
Các lớp road-marking trong `custom_rm.yaml` là mũi tên đi thẳng/trái/phải, **không phải
vạch dừng**. Không thay bằng model đó rồi gọi là detector stop-line.

## Thử nghiệm cục bộ, không phải benchmark

Trên `red3`, đã kiểm tra ảnh trống và chạy cả clip với cùng hình học: bản fine-tune hiện tại
cho nhiều bbox lớn lên mặt đường trống; COCO gốc không có các bbox đó ở frame đã kiểm tra.
Cả hai luồng ghi một ứng viên xe đỏ cắt vạch. Xe vàng từ đường ngang đi vào sau vạch không
được coi là xe vượt vạch đang giám sát. Đây là kiểm tra trực quan, không chứng minh độ chính
xác tổng quát hoặc cho phép báo một con số Precision/Recall.
Classifier Autoware nhận hai crop đèn đỏ của red1 với điểm khoảng 0.99; đó là confidence
của model trên hai crop, không phải accuracy 99% của dự án.

## Fine-tune phần nào trước

1. **Đèn:** gán nhãn crop từ camera thật: đỏ/vàng/xanh/tắt-không rõ, tách đèn tròn và mũi tên.
   Đánh giá confusion matrix, recall đỏ, tỷ lệ UNKNOWN và độ trễ đổi trạng thái. Ưu tiên
   các trường hợp cháy sáng, ngược sáng, đêm và đèn nhỏ. Với classifier hiện tại có thể dùng
   quy trình retrain mmpretrain của Autoware; hoặc train detector đèn theo trạng thái từ
   YOLO pretrained nếu cần tự đề xuất ROI. Không trộn thứ tự lớp giữa các checkpoint.
2. **Xe:** so sánh COCO gốc và checkpoint custom trên cùng ảnh đã gán bbox; fine-tune
   checkpoint tốt hơn bằng video camera mục tiêu và các trường hợp nhỏ/che khuất/xe máy.
   Không train tiếp chỉ trên COCO subset nhỏ rồi mặc định coi là tốt hơn pretrained.
3. **Vạch:** nếu cần giảm thao tác thiết lập, gán polygon `stop_line` và `crosswalk`,
   fine-tune YOLO11-seg pretrained để *đề xuất* vùng vạch. Cần thêm xe/làn/đèn để chọn
   đúng vạch; segmentation tự nó chưa suy ra vi phạm. `dataset_stopline` hiện chưa có
   ảnh/nhãn nên chưa thể train hoặc báo chất lượng model này.

Mốc khởi đầu đề xuất: vài chục góc camera, vài nghìn ảnh đa dạng; lấy mẫu tăng dần theo lỗi,
không coi số lượng này là bảo đảm chất lượng. Chia train/val/test **theo camera**, cả các
clip cùng góc phải chung split; không chia ngẫu nhiên các frame sát nhau.

Lấy ảnh để gán nhãn, ví dụ:

```powershell
python prepare_scene_dataset.py --video dataset_videos/red1.mp4 --split train --interval 1
```

Script tạo `scene_dataset/images/<split>/`, manifest và mẫu ground-truth sự kiện. Ảnh chưa
gán nhãn được đánh dấu `unlabeled`; không tạo label rỗng giả làm ảnh âm. Đưa ảnh vào công cụ
gán nhãn, kiểm tra thủ công, xuất YOLO detection/segmentation phù hợp rồi mới train trên
Modal. Chưa chạy train mới trong thay đổi này vì chưa có bộ nhãn camera thật.

## Đánh giá đúng mục tiêu

Gán thủ công **toàn bộ** sự kiện trên video test, kể cả xe model bỏ sót; gồm gate ID và
thời gian qua vạch theo một quy ước điểm xe nhất quán. Thêm video không có vi phạm để
đo báo sai. Lấy `video_identity` từ `scene.json` của run, tạo JSON:

```json
{
  "video_identity": "COPY_FROM_SCENE_JSON",
  "reviewed": true,
  "events": [
    {"gate_id": "gate_1", "video_time_seconds": 2.7}
  ]
}
```

Thời gian trong ví dụ chỉ minh họa định dạng, không phải nhãn kiểm chứng.

```powershell
python evaluate_events.py --run violations/run_... --truth annotations/red3_events.json --tolerance 0.5
```

Xuất `event_metrics.json`: TP/FP/FN, Precision, Recall, F1, các sự kiện báo nhầm/bỏ sót.
Ghép một-một theo làn và thời gian; dự đoán trùng bị tính FP. Đây là metric theo cửa sổ
thời gian: với nhiều xe cùng làn qua gần nhau, cần thêm ghép quỹ đạo/bbox và kiểm tra thủ công.
Giữ tolerance cố định giữa các model. Không coi test logic hoặc video chạy xong là accuracy.

## Xem lỗi còn lại

- `frame_diagnostics.csv`: đèn UNKNOWN ở đâu, trạng thái từng làn, số track.
- `crossings.json`: mọi lần cắt quan sát được; `review_unknown_light` không cộng vi phạm;
  `red_candidate` chưa đủ xác nhận; `rejected_*` bị hủy do quay lại; `confirmed_red_crossing`
  đã qua điều kiện. Xe hoàn toàn không detect sẽ chưa có bản ghi cắt, phải đối chiếu video.
- `events.json/csv`: ID, thời gian video, frame cắt và frame xác nhận. Ảnh chứng cứ hiện tại
  ở frame xác nhận; `scene.json` giúp tái hiện đúng cấu hình.
- `run_metadata.json`: cấu hình, checkpoint/hash và trạng thái đã xử lý hết video. Công cụ
  đánh giá từ chối run dừng giữa chừng để không so dự đoán một đoạn với nhãn cả video.

Nếu cần thử checkpoint đã train: thêm `--vehicle-weights weights/yolo11x.pt` vào cùng lệnh
chạy. Dùng kết quả sự kiện và nhãn thật để chọn model, không chỉ đếm số bbox.
