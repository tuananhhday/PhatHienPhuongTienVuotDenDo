import argparse
import csv
import json
import hashlib
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Optional, Sequence

import cv2
import numpy as np
import yaml

from src.camera_stabilizer import CameraStabilizer
from src.report_generator import ReportGenerator
from src.stopline_detector import StopLineDetector
from src.traffic_light_detector import TrafficLightDetector
from src.violation_engine import ViolationEngine
from src.scene_config import scene_path, load_scene
from src.light_classifier import LightClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("TrafficSystem")


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_inference_device(requested: str = "auto"):
    """Chọn GPU NVIDIA khi có CUDA; cho phép ép CPU hoặc một GPU cụ thể."""
    import torch

    value = str(requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            logger.info("Dùng GPU: %s", torch.cuda.get_device_name(0))
            return 0
        logger.warning("Không thấy CUDA; pipeline sẽ chạy bằng CPU và cửa sổ xem có thể chậm.")
        return "cpu"

    if value == "cpu":
        logger.info("Dùng CPU theo cấu hình.")
        return "cpu"

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Đã yêu cầu GPU '{requested}' nhưng PyTorch không nhận CUDA. "
            "Hãy kiểm tra nvidia-smi và cài PyTorch bản CUDA."
        )
    normalized = value.removeprefix("cuda:")
    if not normalized.isdigit():
        raise ValueError("--device phải là auto, cpu hoặc số GPU như 0, 1")
    gpu_index = int(normalized)
    if gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {gpu_index} không tồn tại; máy chỉ nhận {torch.cuda.device_count()} GPU."
        )
    logger.info("Dùng GPU %s: %s", gpu_index, torch.cuda.get_device_name(gpu_index))
    return gpu_index


def draw_hud(
    frame: np.ndarray,
    traffic_state: str,
    total_violations: int,
    violation_ids: Sequence[int],
    fps: float,
    stop_lines_info: str,
    camera_id: str
):
    """Vẽ trạng thái đèn, ID vi phạm và tổng bộ đếm lên video."""
    h, w = frame.shape[:2]

    panel_h = min(76, h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    state_colors = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
        "GREEN": (0, 255, 0),
        "UNKNOWN": (160, 160, 160),
    }
    cur_color = state_colors.get(traffic_state, state_colors["UNKNOWN"])
    cv2.circle(frame, (25, 23), 10, cur_color, -1)
    cv2.putText(frame, f"DEN: {traffic_state}", (44, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, cur_color, 2)
    cv2.putText(
        frame, f"CAM: {camera_id} | {stop_lines_info} | FPS: {fps:.1f}",
        (230, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1,
    )
    ids_text = ", ".join(map(str, violation_ids[-8:])) if violation_ids else "-"
    violation_color = (0, 0, 255) if total_violations else (255, 255, 255)
    cv2.putText(
        frame, f"VI PHAM: {total_violations} | ID DA GHI: {ids_text}",
        (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, violation_color, 2,
    )


def run_pipeline(
    video_source: str,
    camera_id: Optional[str] = None,
    redetect: bool = False,
    max_lines: Optional[int] = None,
    output_video: Optional[str] = "output_violation.mp4",
    stabilize: bool = False,
    headless: bool = False,
    debug: bool = False,
    device: Optional[str] = None,
    scene_config: Optional[str] = None,
    calibrate: bool = False,
    experimental_auto: bool = False,
    vehicle_weights: Optional[str] = None,
):
    cfg = load_yaml_config("config.yaml")
    # Each unknown input gets its own cache by default. Reuse a cache across
    # clips only when --camera-id is explicitly provided.
    if not camera_id:
        camera_id = Path(str(video_source)).stem or "cam_01"
    inference_device = resolve_inference_device(
        device if device is not None else cfg.get("runtime", {}).get("device", "auto")
    )

    # Mở nguồn video
    cap = cv2.VideoCapture(int(video_source) if str(video_source).isdigit() else str(video_source))
    if not cap.isOpened():
        logger.error(f"Không thể mở nguồn video: {video_source}")
        return

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError('Không đọc được frame đầu video')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    scene = None
    if calibrate or not experimental_auto:
        profile_path = Path(scene_config) if scene_config else scene_path(video_source)
        if calibrate:
            if headless:
                cap.release()
                raise ValueError('Cần cấu hình camera trước: python calibrate_scene.py --video "' + str(video_source) + '"')
            from calibrate_scene import calibrate as setup_scene
            scene = setup_scene(video_source, profile_path)
            if scene is None:
                cap.release()
                return
        elif not profile_path.is_file():
            cap.release()
            raise ValueError(
                'Chưa có cấu hình cho video này. Chạy lại với --calibrate để chọn '
                'vạch, phía xe đi tới, đèn và vùng làn trước khi phát hiện.'
            )
        else:
            scene = load_scene(profile_path, video_source, first_frame.shape)
        stop_lines = scene['gates']
    else:
        logger.warning('EXPERIMENTAL: vạch tự đề xuất chưa được xác nhận, kết quả chỉ dùng chẩn đoán.')
        line_detector = StopLineDetector(config_path="config.yaml")
        stop_lines = line_detector.detect(video_source, camera_id=camera_id,
                                          redetect=redetect, max_lines=max_lines, debug=debug)

    if not stop_lines:
        cap.release()
        logger.error(f"[LỖI] Không thể phát hiện hoặc đọc vạch dừng cho video: {video_source}")
        return

    # 2. KHỞI TẠO BỘ CHỐNG RUNG CAMERA
    stabilizer = CameraStabilizer(
        enabled=stabilize or cfg.get("camera", {}).get("stabilization_enabled", False),
        method=cfg.get("stabilization", {}).get("method", "ORB"),
        max_features=cfg.get("stabilization", {}).get("max_features", 1000)
    )

    ret, first_frame = cap.read()
    if ret:
        stabilizer.set_reference_frame(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 3. KHỞI TẠO MODEL YOLO VÀ TRACKER (BYTETRACK)
    from ultralytics import YOLO
    model_weights = vehicle_weights or cfg.get("models", {}).get("vehicle_detector", {}).get("weights", "weights/yolo11x.pt")
    if not Path(model_weights).is_file():
        raise FileNotFoundError(f"Không tìm thấy trọng số xe: {model_weights}")
    logger.info(f"Đang tải model YOLO ({model_weights}) kèm ByteTrack...")
    yolo_model = YOLO(model_weights)

    # 4. KHỞI TẠO BỘ NHẬN DIỆN ĐÈN TÍN HIỆU
    light_cfg = cfg.get("models", {}).get("traffic_light", {})
    classifier = None
    if light_cfg.get('classifier', 'hsv') == 'autoware':
        classifier_weights = light_cfg['classifier_weights']
        if not Path(classifier_weights).is_file():
            raise FileNotFoundError('Chạy python download_models.py trước để tải classifier đèn.')
        classifier = LightClassifier(classifier_weights, light_cfg['classifier_labels'],
                                     light_cfg.get('classifier_conf', 0.8))
    # Always load the light detector. For calibrated scenes it finds a tight
    # lamp box inside the user-selected signal region before color inference.
    light_weights = light_cfg.get("weights", "weights/yolov8s.pt")
    if not Path(light_weights).is_file():
        raise FileNotFoundError(f"Cannot find traffic-light weights: {light_weights}")
    light_model = YOLO(light_weights)
    if light_model.names.get(9) != "traffic light":
        raise ValueError("Traffic-light model must expose COCO class 9")
    detectors = {}
    for gate in stop_lines:
        detectors[gate['id']] = TrafficLightDetector(yolo_model=light_model,
                                       conf_thresh=light_cfg.get("conf", 0.25),
                                       max_missing_frames=light_cfg.get("max_missing_frames", 10),
                                       state_confirm_frames=light_cfg.get("state_confirm_frames", 3),
                                       min_color_pixels=light_cfg.get("min_color_pixels", 3),
                                       min_color_ratio=light_cfg.get("min_color_ratio", 0.002),
                                       color_margin=light_cfg.get("color_margin", 1.15),
                                       traffic_roi=None,
                                       device=inference_device,
                                       fixed_boxes=gate.get('signal_boxes'),
                                       classifier=classifier,
                                       movement=gate.get('movement', 'straight'))

    # 5. KHỞI TẠO LOGIC VI PHẠM (VIOLATION ENGINE)
    output_dir = Path(cfg.get("general", {}).get("output_dir", "violations")) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    violation_engine = ViolationEngine(
        stop_lines=stop_lines,
        output_dir=output_dir,
        min_violation_frames=cfg.get("violation_logic", {}).get("min_violation_frames", 3),
        allow_right_turn=cfg.get("violation_logic", {}).get("allow_right_turn", False),
        right_turn_x_ratio=cfg.get("violation_logic", {}).get("right_turn_x_ratio", 0.75),
        crossing_margin_px=cfg.get('violation_logic', {}).get('crossing_margin_px', 2.0),
        max_track_gap_frames=max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25) * cfg.get('violation_logic', {}).get('max_track_gap_seconds', .2))),
        fps=cap.get(cv2.CAP_PROP_FPS) or 25,
    )
    if scene:
        (output_dir / 'scene.json').write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding='utf-8')
    report_gen = ReportGenerator(output_dir=output_dir)
    report_gen.export_reports([])
    trace_file = (output_dir / 'frame_diagnostics.csv').open('w', newline='', encoding='utf-8')
    trace = csv.writer(trace_file)
    trace.writerow(['frame', 'video_seconds', 'gate_id', 'raw_light', 'stable_light', 'tracks', 'violations'])

    # Thiết lập VideoWriter nếu cần lưu video
    writer = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    run_metadata = {'source': str(video_source), 'vehicle_weights': str(model_weights),
                    'vehicle_sha256': hashlib.sha256(Path(model_weights).read_bytes()).hexdigest(),
                    'config': cfg, 'fps': fps_in, 'expected_frames': expected_frames,
                    'processed_frames': 0, 'completed': False}
    (output_dir / 'run_metadata.json').write_text(json.dumps(run_metadata, indent=2), encoding='utf-8')
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps_in, (w_in, h_in))
        if not writer.isOpened():
            cap.release()
            trace_file.close()
            raise RuntimeError(f"Không mở được file video kết quả: {Path(output_video).resolve()}")
        logger.info(f"Đang lưu video kết quả ra: {Path(output_video).resolve()}")

    logger.info("\n" + "=" * 55)
    logger.info(" HỆ THỐNG PHÁT HIỆN VƯỢT ĐÈN ĐỎ ĐANG HOẠT ĐỘNG")
    logger.info(" Màu đèn được nhận diện tự động. Phím: [q]: Thoát | [space]: Tạm dừng")
    logger.info("=" * 55 + "\n")

    frame_idx = 0
    paused = False
    prev_time = time.time()
    fps = 0.0
    reached_end = False

    class_aliases = {"person": "Nguoi di bo", "bicycle": "Xe dap", "car": "O to",
                     "motorcycle": "Xe may", "bus": "Xe buyt", "truck": "Xe tai"}
    vehicle_cfg = cfg.get("models", {}).get("vehicle_detector", {})
    selected_names = vehicle_cfg.get("track_names", list(class_aliases))
    track_classes = [i for i, name in yolo_model.names.items() if name in selected_names]
    if not track_classes:
        raise ValueError(f"No selected classes in checkpoint: {yolo_model.names}")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    reached_end = expected_frames > 0 and frame_idx >= expected_frames
                    logger.info("Đã xử lý hết video.")
                    break
                frame_idx += 1

                # Ổn định hóa vạch dừng nếu camera rung lắc
                current_stop_lines = stabilizer.stabilize_lines(stop_lines, frame)
                violation_engine.update_stop_lines(current_stop_lines)

                # Chạy YOLO Tracking với ByteTrack
                results = yolo_model.track(
                    frame,
                    persist=True,
                    tracker=cfg.get("models", {}).get("vehicle_detector", {}).get("tracker", "bytetrack.yaml"),
                    classes=track_classes,
                    conf=cfg.get("models", {}).get("vehicle_detector", {}).get("conf", 0.35),
                    imgsz=vehicle_cfg.get('imgsz', 960),
                    device=inference_device,
                    verbose=False
                )

                # Thu thập tracks
                tracks = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes_data = results[0].boxes
                    for i in range(len(boxes_data)):
                        box = boxes_data.xyxy[i].cpu().numpy().astype(int)
                        cls_id = int(boxes_data.cls[i].cpu().numpy())
                        conf = float(boxes_data.conf[i].cpu().numpy())
                        track_id = int(boxes_data.id[i].cpu().numpy()) if boxes_data.id is not None and boxes_data.id[i] is not None else None

                        if track_id is None:
                            continue
                        tracks.append({
                            "id": track_id,
                            "box": box,
                            "class_name": class_aliases.get(yolo_model.names[cls_id], yolo_model.names[cls_id]),
                            "conf": conf
                        })

                # Nhận diện trạng thái đèn giao thông
                # Run the detector once; each calibrated gate keeps only
                # tight lamp boxes inside its selected signal region.
                light_results = light_model.predict(
                    frame, conf=0.10, classes=[9], device=inference_device, verbose=False
                )
                light_detections = []
                if light_results and light_results[0].boxes is not None:
                    light_boxes = light_results[0].boxes
                    for light_index, raw_box in enumerate(light_boxes.xyxy.cpu().numpy()):
                        light_detections.append((raw_box, 9, float(light_boxes.conf[light_index].cpu().item())))
                traffic_states = {gate_id: detector.detect_state(frame, light_detections)[0]
                                  for gate_id, detector in detectors.items()}
                state_values = set(traffic_states.values())
                traffic_state = next(iter(state_values)) if len(state_values) == 1 else 'MULTI'

                # Xử lý cắt vạch & vi phạm
                annotated_frame, frame_violations = violation_engine.process_tracks(
                    frame=frame,
                    tracks=tracks,
                    traffic_state=traffic_states,
                    frame_idx=frame_idx
                )
                for gate_id, detector in detectors.items():
                    detector.draw_detections(annotated_frame)
                    trace.writerow([frame_idx, round((frame_idx - 1)/fps_in, 3), gate_id,
                                    detector.raw_state, traffic_states[gate_id], len(tracks),
                                    len(violation_engine.violated_ids)])
                if frame_idx % 30 == 0:
                    trace_file.flush()
                if frame_violations:
                    report_gen.export_reports(violation_engine.violations_log)

                # Tính toán FPS
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, now - prev_time))
                prev_time = now

                # Vẽ HUD Dashboard
                line_desc = ' | '.join(f'{key}:{value}' for key, value in traffic_states.items())
                draw_hud(
                    frame=annotated_frame,
                    traffic_state=traffic_state,
                    total_violations=len(violation_engine.violated_ids),
                    violation_ids=[event["track_id"] for event in violation_engine.violations_log],
                    fps=fps,
                    stop_lines_info=line_desc,
                    camera_id=camera_id
                )

                if writer:
                    writer.write(annotated_frame)

            if not headless:
                cv2.imshow("He Thong Phat Hien Vuot Den Do", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 32:
                    paused = not paused
                    logger.info(f"{'Tạm dừng' if paused else 'Tiếp tục'}")

    except KeyboardInterrupt:
        logger.info('Đã dừng; lưu các sự kiện xử lý được.')
    finally:
        run_metadata.update(processed_frames=frame_idx, completed=reached_end)
        (output_dir / 'run_metadata.json').write_text(json.dumps(run_metadata, indent=2), encoding='utf-8')
        cap.release()
        trace_file.close()
        crossings = violation_engine.crossing_tracker.crossings
        (output_dir / 'crossings.json').write_text(json.dumps(crossings, ensure_ascii=False, indent=2), encoding='utf-8')
        if writer:
            writer.release()
            logger.info(f"Video kết quả đã hoàn tất: {Path(output_video).resolve()}")
        if not headless:
            cv2.destroyAllWindows()

        # Xuất các sự kiện cần người dùng kiểm tra.
        report_gen = ReportGenerator(output_dir=output_dir)
        report_gen.export_reports(violation_engine.violations_log)

        logger.info(f"\n[ĐÃ LƯU] {frame_idx} frame; số sự kiện nghi vượt đèn đỏ: {len(violation_engine.violated_ids)}")
        logger.info(f"Báo cáo JSON: {output_dir / 'events.json'}")
        logger.info(f"Báo cáo CSV:  {output_dir / 'events.csv'}")
        logger.info(f"Báo cáo Markdown: {output_dir / 'events.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện phương tiện vượt đèn đỏ tự động định vị vạch dừng")
    parser.add_argument("--video", "--source", dest="video", type=str, default="dataset_videos/red1.mp4", help="Đường dẫn file video hoặc ID camera")
    parser.add_argument("--camera-id", type=str, default=None, help="ID định danh camera (cam_01, cam_red1...)")
    parser.add_argument("--redetect", action="store_true", help="Bỏ qua cache và ép buộc quét lại vạch mới")
    parser.add_argument("--max-lines", type=int, default=None, help="Số lượng vạch dừng tối đa cần phát hiện (1 hoặc 2)")
    parser.add_argument("--output", type=str, default="output_violation.mp4", help="Đường dẫn lưu video kết quả")
    parser.add_argument("--stabilize", action="store_true", help="Bật chống rung lắc camera bằng Homography")
    parser.add_argument("--headless", action="store_true", help="Chạy chế độ ngầm không mở cửa sổ giao diện")
    parser.add_argument("--debug", action="store_true", help="Xuất ảnh debug chi tiết: dòng xe, Road ROI, cụm xe dừng, ứng viên kèm bảng điểm")
    parser.add_argument("--device", type=str, default=None,
                        help="Thiết bị suy luận: auto, cpu hoặc số GPU như 0")
    parser.add_argument('--calibrate', action='store_true', help='Thiết lập lại vạch, đèn và làn xe cho video')
    parser.add_argument('--scene-config', help='Cấu hình góc quay đã xác nhận')
    parser.add_argument('--scene-calibrated', action='store_true', help='Dùng cấu hình vạch và đèn đã hiệu chỉnh (mặc định)')
    parser.add_argument('--experimental-auto', action='store_true', help='Thử tự dò vạch cho video chưa có cấu hình; chỉ để chẩn đoán')
    parser.add_argument('--vehicle-weights', help='Checkpoint xe khác để đối chiếu trên cùng video')
    args = parser.parse_args()

    run_pipeline(
        video_source=args.video,
        camera_id=args.camera_id,
        redetect=args.redetect,
        max_lines=args.max_lines,
        output_video=args.output,
        stabilize=args.stabilize,
        headless=args.headless,
        debug=args.debug,
        device=args.device,
        scene_config=args.scene_config,
        calibrate=args.calibrate,
        experimental_auto=args.experimental_auto and not args.scene_calibrated,
        vehicle_weights=args.vehicle_weights,
    )
