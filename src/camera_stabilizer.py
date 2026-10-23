import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("CameraStabilizer")

class CameraStabilizer:
    """
    Module Ổn Định Hóa Rung Lắc Camera (Camera Vibration Stabilization).
    - Tính ma trận Homography giữa frame hiện tại và frame tham chiếu (ORB + RANSAC).
    - Tự động warp tọa độ vạch dừng theo chuyển động camera thực tế.
    - Làm mượt chuyển động theo thời gian bằng bộ lọc EMA (Exponential Moving Average).
    """

    def __init__(
        self,
        enabled: bool = False,
        method: str = "ORB",
        max_features: int = 1000,
        ransac_thresh: float = 3.0,
        ema_alpha: float = 0.25
    ):
        self.enabled = enabled
        self.method = method.upper()
        self.max_features = max_features
        self.ransac_thresh = ransac_thresh
        self.ema_alpha = ema_alpha

        # Khởi tạo detector
        if self.method == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.ref_gray: Optional[np.ndarray] = None
        self.ref_kps = None
        self.ref_descs = None
        self.smoothed_lines: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {}

    def set_reference_frame(self, frame: np.ndarray):
        """Thiết lập khung hình tham chiếu chuẩn (thường là frame đầu tiên)"""
        if not self.enabled:
            return

        h, w = frame.shape[:2]
        self.ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mặt nạ chỉ bắt đặc trưng ở các vật thể cố định (cột đèn, tòa nhà, cây) ở nửa trên
        # tránh bắt phải xe cộ đang chạy ở mặt đường phía dưới
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:int(h * 0.55), :] = 255

        self.ref_kps, self.ref_descs = self.detector.detectAndCompute(self.ref_gray, mask)
        logger.info(f"Đã lưu khung hình tham chiếu chống rung: {len(self.ref_kps)} điểm đặc trưng {self.method}")

    def stabilize_lines(self, lines: List[Dict], current_frame: np.ndarray) -> List[Dict]:
        """
        Warp và làm mượt danh sách vạch dừng theo độ rung của khung hình hiện tại.
        """
        if not self.enabled or self.ref_descs is None or len(lines) == 0:
            return lines

        h, w = current_frame.shape[:2]
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Lấy đặc trưng trên nửa trên khung hình hiện tại
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:int(h * 0.55), :] = 255
        curr_kps, curr_descs = self.detector.detectAndCompute(curr_gray, mask)

        if curr_descs is None or len(curr_kps) < 15:
            return lines

        # Khớp đặc trưng
        matches = self.matcher.match(self.ref_descs, curr_descs)
        if len(matches) < 15:
            return lines

        # Sắp xếp và lấy các match tốt nhất
        matches = sorted(matches, key=lambda m: m.distance)[:150]

        pts_ref = np.float32([self.ref_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_curr = np.float32([curr_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Tính toán Homography matrix H
        H, inliers = cv2.findHomography(pts_ref, pts_curr, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return lines

        # Warp từng vạch dừng
        updated_lines = []
        for line_cfg in lines:
            line_id = line_cfg.get("id", "line_1")
            (x1, y1), (x2, y2) = line_cfg["line"]

            # Vector điểm dạng homogeneous
            pts_line = np.float32([[[x1, y1]], [[x2, y2]]])
            warped_pts = cv2.perspectiveTransform(pts_line, H)

            wx1, wy1 = int(round(warped_pts[0][0][0])), int(round(warped_pts[0][0][1]))
            wx2, wy2 = int(round(warped_pts[1][0][0])), int(round(warped_pts[1][0][1]))

            # Làm mượt chuyển động bằng EMA (Exponential Moving Average)
            if line_id in self.smoothed_lines:
                (sx1, sy1), (sx2, sy2) = self.smoothed_lines[line_id]
                alpha = self.ema_alpha
                fx1 = int(round(alpha * wx1 + (1 - alpha) * sx1))
                fy1 = int(round(alpha * wy1 + (1 - alpha) * sy1))
                fx2 = int(round(alpha * wx2 + (1 - alpha) * sx2))
                fy2 = int(round(alpha * wy2 + (1 - alpha) * sy2))
            else:
                fx1, fy1, fx2, fy2 = wx1, wy1, wx2, wy2

            self.smoothed_lines[line_id] = ((fx1, fy1), (fx2, fy2))

            new_cfg = line_cfg.copy()
            new_cfg["line"] = [(fx1, fy1), (fx2, fy2)]
            updated_lines.append(new_cfg)

        return updated_lines
