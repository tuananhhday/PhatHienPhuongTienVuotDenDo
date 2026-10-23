import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.stopline_detector import StopLineDetector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("PreviewStopLine")


def preview_and_confirm(
    video_path: str,
    camera_id: str,
    redetect: bool = False,
    max_lines: int = None,
    output_image: str = "preview_stopline.jpg",
    headless: bool = False,
    debug: bool = False
):
    video_p = Path(video_path)
    if not video_p.exists():
        logger.error(f"File video không tồn tại: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_p))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error("Không thể đọc frame từ video.")
        return

    # 1. Chạy detector đề xuất vạch dừng
    detector = StopLineDetector(config_path="config.yaml")
    proposed_lines = detector.detect(
        video_p,
        camera_id=camera_id,
        redetect=redetect,
        max_lines=max_lines,
        debug=debug
    )

    if not proposed_lines:
        logger.warning("[CẢNH BÁO] Không phát hiện thấy vạch dừng đề xuất tự động! Chuyển sang chế độ vẽ tay thủ công.")
        lines = []
    else:
        lines = [cfg["line"] for cfg in proposed_lines]
        for idx, cfg in enumerate(proposed_lines):
            conf = cfg.get("confidence", 0.0)
            if conf < 0.40:
                logger.warning(f"[CẢNH BÁO] Vạch {idx+1} có độ tin cậy thấp ({conf:.2f} < 0.40). Cần kiểm tra kỹ trước khi xác nhận!")

    clone = frame.copy()
    current_clicks = []
    saved_lines = [l.copy() for l in lines]

    def render_canvas():
        nonlocal clone
        clone = frame.copy()
        h, w = frame.shape[:2]

        # Header panel
        cv2.rectangle(clone, (0, 0), (w, 55), (20, 20, 20), -1)
        header_text = f"XAC NHAN VACH DUNG - CAMERA: {camera_id} | Che do: CV"
        cv2.putText(clone, header_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Vẽ các vạch hiện có
        for i, (p1, p2) in enumerate(saved_lines):
            cv2.line(clone, tuple(p1), tuple(p2), (0, 255, 0), 4)
            cv2.circle(clone, tuple(p1), 6, (0, 0, 255), -1)
            cv2.circle(clone, tuple(p2), 6, (0, 0, 255), -1)
            cv2.putText(
                clone, f"VACH {i+1}: {p1} -> {p2}",
                (p1[0], max(30, p1[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
            )

        # Vẽ điểm đang click dở
        if len(current_clicks) == 1:
            cv2.circle(clone, current_clicks[0], 7, (0, 255, 255), -1)

        # Hướng dẫn phím tắt ở đáy màn hình
        cv2.rectangle(clone, (0, h - 40), (w, h), (20, 20, 20), -1)
        guide_text = "[s]: LUU VA XAC NHAN | [r]: RESET VE DE XUAT | [Click 2 diem]: Chinh vach | [q]: THOAT"
        cv2.putText(clone, guide_text, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_clicks.append((x, y))
            if len(current_clicks) == 2:
                saved_lines.clear()
                saved_lines.append([current_clicks[0], current_clicks[1]])
                logger.info(f"Đã cập nhật vạch thủ công: {current_clicks[0]} -> {current_clicks[1]}")
                current_clicks.clear()
            render_canvas()

    render_canvas()
    cv2.imwrite(output_image, clone)
    logger.info(f"Đã xuất ảnh xem trước vạch dừng ra: {output_image}")

    if headless:
        logger.info("[Headless] Đã lưu kết quả vạch dừng đề xuất tự động vào cache.")
        return

    window_name = f"Xem truoc & Xac nhan vach dung - {camera_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 800)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'): # Save
            # Chuyển đổi và lưu vào cache configs/<camera_id>.json
            confirmed = []
            h, w = frame.shape[:2]
            for i, (p1, p2) in enumerate(saved_lines):
                ym = (p1[1] + p2[1]) / 2
                confirmed.append({
                    "id": f"stop_line_{i+1}",
                    "name": f"Vach dung (Xac nhan - Y={int(ym)})",
                    "line": [list(p1), list(p2)],
                    "direction": "TOP_TO_BOTTOM" if ym < h * 0.55 else "BOTTOM_TO_TOP",
                    "confidence": 1.0,
                    "method": "manual_confirmed"
                })
            detector.save_to_cache(camera_id, confirmed, mode="manual_confirmed", video_path=str(video_p))
            logger.info(f"[THÀNH CÔNG] Đã lưu cấu hình vạch xác nhận cho '{camera_id}'!")
            break
        elif key == ord('r'): # Reset
            saved_lines = [l.copy() for l in lines]
            current_clicks.clear()
            render_canvas()
            logger.info("Đã khôi phục lại vạch do AI/CV đề xuất.")
        elif key == ord('q'):
            logger.info("Thoát xem trước mà không lưu thay đổi.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xem trước và xác nhận vạch dừng bán tự động")
    parser.add_argument("--video", type=str, default="dataset_videos/red1.mp4", help="Đường dẫn file video")
    parser.add_argument("--camera-id", type=str, default="cam_01", help="ID định danh camera (cam_01, cam_red1...)")
    parser.add_argument("--redetect", action="store_true", help="Bỏ qua cache và ép phát hiện lại")
    parser.add_argument("--max-lines", type=int, default=None, help="Số lượng vạch dừng tối đa cần phát hiện (ví dụ: 1 hoặc 2)")
    parser.add_argument("--output-img", type=str, default="preview_stopline.jpg", help="Đường dẫn xuất ảnh xem trước")
    parser.add_argument("--headless", action="store_true", help="Chạy không mở cửa sổ giao diện")
    parser.add_argument("--debug", action="store_true", help="Xuất ảnh debug chi tiết: dòng xe, Road ROI, cụm xe dừng, ứng viên kèm bảng điểm")
    args = parser.parse_args()

    preview_and_confirm(
        video_path=args.video,
        camera_id=args.camera_id,
        redetect=args.redetect,
        max_lines=args.max_lines,
        output_image=args.output_img,
        headless=args.headless,
        debug=args.debug
    )
