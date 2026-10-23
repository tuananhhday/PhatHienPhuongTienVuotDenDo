# Phát hiện phương tiện vượt đèn đỏ

Dự án đọc video giao thông, phát hiện và theo dõi phương tiện, xác định trạng thái đèn và ghi nhận xe cắt vạch dừng khi đèn đỏ. Các video đã hiệu chỉnh đi kèm cấu hình vạch dừng và vùng đèn, nên có thể chạy ngay sau khi tải model; không cần gán nhãn hay huấn luyện lại để xem các video này.

## Báo cáo HTML

Mở [báo cáo](output/html/bao_cao_de_hieu.html) bằng trình duyệt sau khi clone hoặc tải repository. Báo cáo có năm phần: bài toán và input/output, dataset, pretrained và fine-tune, mô hình và thuật toán, các chỉ số đánh giá. Ảnh, biểu đồ và sơ đồ được nhúng sẵn trong file để đọc ngoại tuyến.

## Cài đặt sau khi clone

Trên Windows PowerShell, cần Python 3.10 trở lên và kết nối Internet cho lần tải model đầu tiên:

```powershell
git clone https://github.com/tuananhhday/PhatHienPhuongTienVuotDenDo.git
cd PhatHienPhuongTienVuotDenDo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_models.py
```

`download_models.py` tải YOLO11x pretrained COCO cùng classifier đèn và kiểm tra SHA-256. Model YOLOv8s dùng để tìm đèn đã có trong repo. Nếu PowerShell chặn kích hoạt venv, có thể gọi trực tiếp `.\.venv\Scripts\python.exe` thay cho `python` ở các lệnh sau.

## Chạy video đã gán vạch và vùng đèn

Từ thư mục gốc của dự án, chạy video mẫu `red1.mp4`:

```powershell
python main.py --video dataset_videos/red1.mp4 --output output_red1.mp4
```

Lệnh trên mở cửa sổ xem video khi xử lý, đồng thời lưu `output_red1.mp4` ở thư mục gốc của dự án. **Chờ đến khi terminal báo `Video kết quả đã hoàn tất`** rồi mới mở file MP4; đóng terminal giữa chừng có thể để lại file chưa phát được. Muốn chỉ xuất file, không mở cửa sổ, thêm `--headless`. Thiết bị mặc định là GPU NVIDIA khi PyTorch nhận CUDA, hoặc CPU nếu không có GPU. Có thể chọn rõ bằng `--device 0` hay `--device cpu`.

Các video **đã có cấu hình được commit** là `red1`, `red2`, `red3`, `red5`, `red6`, `red7`, `red8`, `red9`, `red10`. Để chạy tất cả trong PowerShell:

```powershell
foreach ($n in 1,2,3,5,6,7,8,9,10) {
    python main.py --video "dataset_videos/red$n.mp4" --headless --output "output_red$n.mp4"
}
```

Mỗi cấu hình trong `configs/scenes/` lưu tọa độ vạch dừng, vùng đèn và hướng xe cho đúng video tương ứng. Chương trình kiểm tra dấu vân tay nội dung video và kích thước khung hình, nên vẫn tìm đúng cấu hình sau khi clone sang máy hoặc thư mục khác. `red4.mp4` hiện **chưa có cấu hình đã xác nhận**; chạy nó ở chế độ mặc định sẽ báo cần hiệu chỉnh, thay vì âm thầm dùng vạch sai.

Kết quả gồm video được vẽ khung xe, ID, đèn và vạch theo đường dẫn `--output`. Mỗi lượt chạy tạo thư mục `violations/run_.../` chứa `events.json`, `events.csv`, `events.md` và ảnh bằng chứng. Kết quả cần được xem lại trước khi kết luận có vi phạm thực tế.

## Video mới chưa có cấu hình

Để gán vạch dừng và vùng đèn cho video mới, chạy có cửa sổ:

```powershell
python main.py --video dataset_videos/red4.mp4 --calibrate --output output_red4.mp4
```

Trong cửa sổ hiệu chỉnh, bấm hai đầu vạch dừng rồi hai góc bao vùng đèn điều khiển làn xe đó. Nhấn `S` để lưu, `U` để bỏ điểm gần nhất, `R` để làm lại, `Q` để hủy. Sau khi lưu, chạy lại lệnh bình thường với `--headless`. Cấu hình mới được lưu trong `configs/scenes/` và phải được commit nếu muốn người clone sau sử dụng được.

Có thể thử thuật toán tự dò vạch bằng `--experimental-auto`, nhưng kết quả chỉ nên dùng để chẩn đoán cho video chưa hiệu chỉnh. Luồng mặc định luôn ưu tiên cấu hình đã gán.

## Model và dữ liệu

Pipeline video mặc định dùng `weights/coco/yolo11x.pt` pretrained COCO để nhận diện xe, `weights/yolov8s.pt` để phát hiện đèn và classifier Autoware để đọc trạng thái đèn. Muốn thử checkpoint xe khác mà không đổi model mặc định:

```powershell
python main.py --video dataset_videos/red1.mp4 --vehicle-weights weights/model_khac.pt --headless --output output_thu.mp4
```

`dataset_stopline/` là bộ ảnh và nhãn dùng cho thí nghiệm nhận diện vạch; ảnh và nhãn của bộ này không cần để chạy các video đã hiệu chỉnh. Huấn luyện lại cũng không phải bước cài đặt bắt buộc.

## Nhận diện phương tiện trong ảnh

Đặt ảnh `.jpg`, `.png`, `.jpeg`, `.bmp` hoặc `.webp` vào `images/input/`, rồi chạy:

```powershell
python detect_image.py --source images/input --device 0
```

Ảnh kết quả nằm trong `images/output/`. Có thể thay `--source` bằng đường dẫn một ảnh và bỏ `--device 0` để tự chọn GPU hoặc CPU.

## Thành phần chính

| Đường dẫn | Vai trò |
|---|---|
| `main.py` | Điều phối pipeline video và tham số dòng lệnh |
| `calibrate_scene.py` | Giao diện hiệu chỉnh vạch và vùng đèn |
| `src/stopline_detector.py` | Tự đề xuất vạch cho video mới khi dùng `--experimental-auto` |
| `src/traffic_light_detector.py` | Phát hiện đèn và đọc màu đèn |
| `src/crossing_tracker.py` | Theo dõi xe cắt vạch |
| `src/violation_engine.py` | Ghi nhận sự kiện nghi vượt đèn đỏ |

## Kiểm thử

```powershell
python -m pip install pytest
python -m pytest tests -q
```
