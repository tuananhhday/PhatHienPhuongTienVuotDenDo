"""Phát hiện cụm đèn giao thông và suy ra màu đang bật."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("TrafficLightDetector")


class TrafficLightDetector:
    """Dùng YOLO tìm đèn, HSV đọc màu và lọc nhiễu theo thời gian."""

    STATE_COLORS = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
        "GREEN": (0, 255, 0),
        "UNKNOWN": (160, 160, 160),
    }

    def __init__(
        self,
        yolo_model=None,
        conf_thresh: float = 0.25,
        hsv_ranges: Optional[Dict] = None,
        max_missing_frames: int = 10,
        state_confirm_frames: int = 3,
        min_color_pixels: int = 3,
        min_color_ratio: float = 0.002,
        color_margin: float = 1.15,
        traffic_roi: Optional[Dict] = None,
        device=None,
        fixed_boxes=None,
        classifier=None,
        movement='straight',
    ):
        self.model = yolo_model
        self.conf_thresh = conf_thresh
        self.current_state = "UNKNOWN"
        self.max_missing_frames = max(1, int(max_missing_frames))
        self.state_confirm_frames = max(1, int(state_confirm_frames))
        self.min_color_pixels = max(1, int(min_color_pixels))
        self.min_color_ratio = max(0.0, float(min_color_ratio))
        self.color_margin = max(1.0, float(color_margin))
        self.traffic_roi = traffic_roi
        self.device = device
        self.fixed_boxes = fixed_boxes
        self.classifier = classifier
        self.movement = movement
        self.raw_state = 'UNKNOWN'
        self.missing_frames = 0
        self.pending_state = "UNKNOWN"
        self.pending_count = 0
        self.detected_boxes: List[Tuple[int, int, int, int]] = []
        self.detected_lights: List[Dict] = []

        self.hsv_ranges = hsv_ranges or {
            "RED": {
                "lower1": np.array([0, 100, 100]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([160, 100, 100]),
                "upper2": np.array([180, 255, 255]),
            },
            "YELLOW": {
                "lower": np.array([15, 100, 100]),
                "upper": np.array([35, 255, 255]),
            },
            "GREEN": {
                "lower": np.array([40, 100, 100]),
                "upper": np.array([90, 255, 255]),
            },
        }

    def _inside_traffic_roi(self, box: Tuple[int, int, int, int], width: int, height: int) -> bool:
        if not self.traffic_roi:
            return True
        x1, y1, x2, y2 = box
        center_x = ((x1 + x2) / 2) / max(1, width)
        center_y = ((y1 + y2) / 2) / max(1, height)
        return (
            self.traffic_roi.get("x_min_ratio", 0.0) <= center_x <=
            self.traffic_roi.get("x_max_ratio", 1.0)
            and self.traffic_roi.get("y_min_ratio", 0.0) <= center_y <=
            self.traffic_roi.get("y_max_ratio", 1.0)
        )

    def _classify_crop(self, crop: np.ndarray) -> Tuple[str, Dict[str, float]]:
        if crop.size == 0:
            return "UNKNOWN", {name: 0.0 for name in ("RED", "YELLOW", "GREEN")}

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.hsv_ranges["RED"]["lower1"], self.hsv_ranges["RED"]["upper1"]),
            cv2.inRange(hsv, self.hsv_ranges["RED"]["lower2"], self.hsv_ranges["RED"]["upper2"]),
        )
        masks = {
            "RED": red_mask,
            "YELLOW": cv2.inRange(
                hsv, self.hsv_ranges["YELLOW"]["lower"], self.hsv_ranges["YELLOW"]["upper"]
            ),
            "GREEN": cv2.inRange(
                hsv, self.hsv_ranges["GREEN"]["lower"], self.hsv_ranges["GREEN"]["upper"]
            ),
        }
        area = float(crop.shape[0] * crop.shape[1])
        pixels = {name: cv2.countNonZero(mask) for name, mask in masks.items()}
        ratios = {name: value / area for name, value in pixels.items()}
        ranked = sorted(pixels, key=pixels.get, reverse=True)
        best, second = ranked[0], ranked[1]
        if pixels[best] < self.min_color_pixels or ratios[best] < self.min_color_ratio:
            return "UNKNOWN", ratios
        if pixels[second] and pixels[best] < pixels[second] * self.color_margin:
            return "UNKNOWN", ratios
        return best, ratios

    def _mark_missing_or_ambiguous(self) -> str:
        self.missing_frames += 1
        self.pending_state = "UNKNOWN"
        self.pending_count = 0
        # Không giữ RED khi mất tín hiệu vì có thể tạo một sự kiện vi phạm sai.
        if self.current_state == "RED" or self.missing_frames >= self.max_missing_frames:
            self.current_state = "UNKNOWN"
        return self.current_state

    def _accept_observation(self, observed_state: str) -> str:
        self.missing_frames = 0
        if observed_state == self.current_state:
            self.pending_state = "UNKNOWN"
            self.pending_count = 0
            return self.current_state

        # Trong lúc chờ đủ số frame xác nhận màu mới, trạng thái là chưa chắc chắn.
        # Điều này đặc biệt tránh giữ RED thêm vài frame sau khi đèn đã đổi màu.
        self.current_state = "UNKNOWN"

        if observed_state == self.pending_state:
            self.pending_count += 1
        else:
            self.pending_state = observed_state
            self.pending_count = 1

        if self.pending_count >= self.state_confirm_frames:
            self.current_state = observed_state
            self.pending_state = "UNKNOWN"
            self.pending_count = 0
        return self.current_state

    def detect_state(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[List] = None,
    ) -> Tuple[str, List[Tuple[int, int, int, int]]]:
        """Trả về trạng thái ổn định và các bbox đèn phát hiện được."""
        candidates = []

        def in_fixed_region(box):
            if self.fixed_boxes is None:
                return True
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                       for rx1, ry1, rx2, ry2 in self.fixed_boxes)

        # In a calibrated scene fixed_boxes is a *search region*, not the
        # final classifier crop. YOLO supplies a tight lamp bbox inside it.
        if yolo_detections is not None:
            threshold = min(self.conf_thresh, 0.10) if self.fixed_boxes is not None else self.conf_thresh
            for box, cls_id, conf in yolo_detections:
                if int(cls_id) == 9 and float(conf) >= threshold and in_fixed_region(box):
                    candidates.append((tuple(map(int, box)), float(conf)))
        elif self.model is not None:
            threshold = min(self.conf_thresh, 0.10) if self.fixed_boxes is not None else self.conf_thresh
            results = self.model.predict(
                frame, conf=threshold, classes=[9], device=self.device, verbose=False
            )
            if results and results[0].boxes is not None:
                result_boxes = results[0].boxes
                for index, raw_box in enumerate(result_boxes.xyxy.cpu().numpy()):
                    box = tuple(map(int, raw_box))
                    confidence = float(result_boxes.conf[index].cpu().item())
                    if in_fixed_region(box):
                        candidates.append((box, confidence))

        # If the detector misses a very small signal, retain the user's box as
        # a fallback. HSV may still read an obvious red/green lamp from it.
        if not candidates and self.fixed_boxes is not None:
            candidates = [(tuple(map(int, box)), 1.0) for box in self.fixed_boxes]

        self.detected_boxes = [box for box, _ in candidates]
        self.detected_lights = []
        height, width = frame.shape[:2]
        observed_states = []

        for box, confidence in candidates:
            if not self._inside_traffic_roi(box, width, height):
                continue
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            clipped_box = (x1, y1, x2, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                state, ratios = "UNKNOWN", {name: 0.0 for name in ("RED", "YELLOW", "GREEN")}
            else:
                state, ratios = self._classify_crop(frame[y1:y2, x1:x2])
            classifier_label = None
            if self.classifier is not None and x2 > x1 and y2 > y1:
                predicted_state, classifier_confidence, classifier_label = self.classifier.classify(
                    frame[y1:y2, x1:x2], self.movement)
                # A low-confidence neural prediction must not erase a clear
                # color reading from the fallback region.
                if predicted_state != 'UNKNOWN':
                    state, confidence = predicted_state, classifier_confidence
            self.detected_lights.append({
                "box": clipped_box,
                "state": state,
                "confidence": confidence,
                "color_ratios": ratios,
                "classifier_label": classifier_label,
            })
            if state != "UNKNOWN":
                observed_states.append(state)

        self.detected_boxes = [light["box"] for light in self.detected_lights]

        # Không tự chọn màu khi các cụm đèn trong ảnh đang hiển thị màu khác nhau.
        unique_states = set(observed_states)
        self.raw_state = next(iter(unique_states)) if len(unique_states) == 1 else 'UNKNOWN'
        if len(unique_states) != 1:
            return self._mark_missing_or_ambiguous(), self.detected_boxes

        return self._accept_observation(observed_states[0]), self.detected_boxes

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ bbox và màu đọc được cho từng cụm đèn trên frame kết quả."""
        if self.traffic_roi:
            height, width = frame.shape[:2]
            x1 = int(width * self.traffic_roi.get("x_min_ratio", 0.0))
            y1 = int(height * self.traffic_roi.get("y_min_ratio", 0.0))
            x2 = int(width * self.traffic_roi.get("x_max_ratio", 1.0))
            y2 = int(height * self.traffic_roi.get("y_max_ratio", 1.0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)
            cv2.putText(
                frame, "VUNG DEN GIAM SAT", (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1,
            )
        for light in self.detected_lights:
            x1, y1, x2, y2 = light["box"]
            state = light["state"]
            color = self.STATE_COLORS[state]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"DEN {state} {light['confidence']:.2f}"
            cv2.putText(
                frame, label, (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
            )
        return frame
