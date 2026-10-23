import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("TrajectoryAnalyzer")


class TrajectoryAnalyzer:
    """
    Module phân tích quỹ đạo và chuyển động phương tiện (Vehicle Motion & Trajectory Analysis):
    1. Ước lượng hướng chuyển động chủ đạo của dòng xe (Dominant Traffic Flow Direction vector).
    2. Tạo Road ROI = Vùng bao các quỹ đạo xe (Convex hull / Dilated contour).
    3. Gom cụm vị trí đáy bbox của các xe đứng yên (vận tốc ~ 0) để tìm cụm xe dừng.
    """

    def __init__(
        self,
        yolo_model: Optional[YOLO] = None,
        model_weights: str = "weights/yolo11x.pt",
        tracker_cfg: str = "bytetrack.yaml",
        track_classes: Optional[List[int]] = None,
        track_names: Optional[List[str]] = None,
        conf_thresh: float = 0.35,
        min_displacement: float = 25.0, # pixel tối thiểu để coi là xe đang di chuyển
        max_stop_speed: float = 3.5     # pixel/frame tối đa để coi là xe đứng yên
    ):
        self.model = yolo_model
        self.model_weights = model_weights
        self.tracker_cfg = tracker_cfg
        self.track_classes = track_classes
        self.track_names = track_names or ["bicycle", "car", "motorcycle", "bus", "truck"]
        self.conf_thresh = conf_thresh
        self.min_displacement = min_displacement
        self.max_stop_speed = max_stop_speed

    def extract_motion_features(
        self,
        video_source: Union[str, Path, cv2.VideoCapture],
        sample_frames: int = 120,
        step: int = 1
    ) -> Dict:
        """
        Chạy YOLO Tracking trên các frame đầu của video để trích xuất:
        - dominant_flow_vector: vector đơn vị hướng dòng xe (ux, uy)
        - road_roi_mask: mặt nạ nhị phân vùng bao quỹ đạo xe
        - road_roi_polygon: đa giác bao quanh vùng xe chạy
        - stopped_points: danh sách tọa độ (x, y) đáy xe dừng
        - trajectories: danh sách đường đi từng xe {track_id: [(x, y), ...]}
        """
        if isinstance(video_source, (str, Path)):
            cap = cv2.VideoCapture(str(video_source))
            should_release = True
        else:
            cap = video_source
            should_release = False

        if not cap.isOpened():
            logger.error(f"Không thể mở video để trích xuất quỹ đạo: {video_source}")
            return self._empty_result((1080, 1080))

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.model is None:
            self.model = YOLO(self.model_weights)
        if self.track_classes is None:
            self.track_classes = [int(index) for index, name in self.model.names.items()
                                  if name in self.track_names]
        if not self.track_classes:
            raise ValueError(f"Checkpoint has none of the selected vehicle classes: {self.model.names}")

        trajectories: Dict[int, List[Tuple[int, int]]] = {} # track_id -> list of (x, y)
        frame_speeds: Dict[int, List[float]] = {}          # track_id -> list of speeds
        stopped_points: List[Tuple[int, int]] = []

        logger.info(f"Đang phân tích quỹ đạo xe trên {sample_frames} khung hình mẫu...")

        frame_count = 0
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

        while frame_count < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % step != 0:
                continue

            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_cfg,
                classes=self.track_classes,
                conf=self.conf_thresh,
                verbose=False
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes_data = results[0].boxes
                for i in range(len(boxes_data)):
                    if boxes_data.id is None or boxes_data.id[i] is None:
                        continue
                    track_id = int(boxes_data.id[i].cpu().numpy())
                    box = boxes_data.xyxy[i].cpu().numpy().astype(int)

                    # Điểm tiếp đất: trung điểm cạnh đáy bbox
                    bottom_center = (int((box[0] + box[2]) / 2), int(box[3]))

                    if track_id not in trajectories:
                        trajectories[track_id] = []
                        frame_speeds[track_id] = []

                    pts = trajectories[track_id]
                    if len(pts) > 0:
                        disp = np.hypot(bottom_center[0] - pts[-1][0], bottom_center[1] - pts[-1][1])
                        frame_speeds[track_id].append(disp)
                    pts.append(bottom_center)

        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        if should_release:
            cap.release()

        # 1. TÍNH HƯỚNG CHUYỂN ĐỘNG CHỦ ĐẠO (DOMINANT TRAFFIC FLOW)
        motion_vectors = []
        motion_weights = []
        all_bottom_points = []

        for tid, pts in trajectories.items():
            all_bottom_points.extend(pts)
            if len(pts) >= 5:
                # Vector di chuyển tổng thể từ điểm đầu đến điểm cuối
                p_start = np.array(pts[0], dtype=np.float32)
                p_end = np.array(pts[-1], dtype=np.float32)
                vec = p_end - p_start
                dist = float(np.linalg.norm(vec))

                # Nếu xe di chuyển quãng đường đáng kể
                if dist >= self.min_displacement:
                    unit_vec = vec / dist
                    motion_vectors.append(unit_vec)
                    motion_weights.append(dist)

                # Kiểm tra xe đứng yên (vận tốc trung bình rất thấp)
                speeds = frame_speeds.get(tid, [])
                if len(speeds) >= 10 and np.mean(speeds[-10:]) < self.max_stop_speed:
                    # Ghi nhận vị trí xe dừng
                    stopped_points.append(pts[-1])

        if len(motion_vectors) > 0:
            motion_arr = np.array(motion_vectors)
            # Trọng số theo quãng đường di chuyển (xe đi xa phản ánh đúng dòng đường chính)
            weights_arr = np.array(motion_weights)
            weighted_dx = float(np.average(motion_arr[:, 0], weights=weights_arr))
            weighted_dy = float(np.average(motion_arr[:, 1], weights=weights_arr))
            norm = np.hypot(weighted_dx, weighted_dy)
            if norm > 1e-4:
                flow_vector = (weighted_dx / norm, weighted_dy / norm)
            else:
                flow_vector = (0.0, 1.0)
        else:
            flow_vector = (0.0, 1.0)

        # 2. TẠO ROAD ROI POLYGON & MASK TỪ QUỸ ĐẠO XE
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_polygon = []

        if len(all_bottom_points) >= 10:
            pts_arr = np.array(all_bottom_points, dtype=np.int32)
            # Tìm Convex Hull bao bọc toàn bộ điểm xe chạy
            hull = cv2.convexHull(pts_arr)
            # Giãn nhẹ (dilate) polygon để bao phủ toàn bộ lòng đường
            cv2.drawContours(roi_mask, [hull], -1, 255, -1)
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
            roi_mask = cv2.dilate(roi_mask, dilate_kernel, iterations=1)

            # Lấy lại contour của ROI sau khi giãn
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                roi_polygon = largest_c.reshape(-1, 2).tolist()
        else:
            # Fallback nếu không có đủ xe
            cv2.rectangle(roi_mask, (0, int(h * 0.45)), (w, int(h * 0.85)), 255, -1)
            roi_polygon = [[0, int(h * 0.45)], [w, int(h * 0.45)], [w, int(h * 0.85)], [0, int(h * 0.85)]]

        # 3. GOM CỤM XE DỪNG (STOPPED CLUSTERS)
        stopped_clusters = []
        if len(stopped_points) > 0:
            # Gom các điểm xe dừng gần nhau (< 60px)
            stop_arr = np.array(stopped_points)
            for pt in stop_arr:
                matched = False
                for cl in stopped_clusters:
                    if np.hypot(cl["center"][0] - pt[0], cl["center"][1] - pt[1]) < 60:
                        cl["points"].append(pt)
                        cl["center"] = np.mean(cl["points"], axis=0).astype(int).tolist()
                        matched = True
                        break
                if not matched:
                    stopped_clusters.append({"center": pt.tolist(), "points": [pt]})

        logger.info(
            f"Trích xuất thành công: Hướng dòng xe={flow_vector}, "
            f"Số quỹ đạo xe={len(trajectories)}, Số điểm xe dừng={len(stopped_points)}"
        )

        return {
            "flow_vector": flow_vector,
            "road_roi_mask": roi_mask,
            "road_roi_polygon": roi_polygon,
            "stopped_points": stopped_points,
            "stopped_clusters": stopped_clusters,
            "trajectories": trajectories,
            "frame_shape": (h, w)
        }

    def _empty_result(self, shape: Tuple[int, int]) -> Dict:
        h, w = shape
        return {
            "flow_vector": (0.0, 1.0),
            "road_roi_mask": np.ones((h, w), dtype=np.uint8) * 255,
            "road_roi_polygon": [[0, int(h*0.45)], [w, int(h*0.45)], [w, int(h*0.85)], [0, int(h*0.85)]],
            "stopped_points": [],
            "stopped_clusters": [],
            "trajectories": {},
            "frame_shape": shape
        }
