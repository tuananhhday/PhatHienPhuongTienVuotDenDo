import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from src.trajectory_analyzer import TrajectoryAnalyzer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("StopLineDetector")


class StopLineDetector:
    """
    Module Tự Động Phát Hiện Vạch Dừng - Kiến trúc: SINH ỨNG VIÊN -> CHẤM ĐIỂM -> CHỌN
    - Sinh ứng viên bằng CLAHE + Ngưỡng thích nghi + HoughLinesP + Gộp đoạn thẳng.
    - Dùng dữ liệu Track xe: Hướng dòng xe, ROI lòng đường, Cụm xe dừng.
    - Chấm điểm đa tiêu chí: Góc vuông góc dòng xe, Số quỹ đạo cắt, Khoảng cách tới xe dừng, Độ dài.
    - Tự động chọn ứng viên điểm cao nhất; cảnh báo nếu điểm thấp hơn ngưỡng.
    """

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config_path = Path(config_path)
        self.cfg = self._load_config()

        # Cache dir
        self.cache_dir = Path(self.cfg.get("camera", {}).get("cache_dir", "configs"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Tham số cấu hình
        classical_cfg = self.cfg.get("stopline_detector", {}).get("classical", {})
        self.sample_frames = classical_cfg.get("sample_frames", 120)
        self.roi_y_min = classical_cfg.get("roi", {}).get("y_min_ratio", 0.45)
        self.roi_y_max = classical_cfg.get("roi", {}).get("y_max_ratio", 0.85)

        # Trọng số chấm điểm
        weights_cfg = classical_cfg.get("scoring_weights", {})
        self.w_angle = weights_cfg.get("angle_perpendicular", 0.35)
        self.w_traj = weights_cfg.get("trajectory_intersection", 0.25)
        self.w_stop = weights_cfg.get("stopped_cluster_distance", 0.20)
        self.w_len = weights_cfg.get("relative_length", 0.15)
        self.w_contrast = weights_cfg.get("paint_contrast", 0.10)
        self.w_crosswalk = weights_cfg.get("crosswalk_relation", 0.05)

        # Ngưỡng chọn
        sel_cfg = classical_cfg.get("selection", {})
        self.min_confidence = sel_cfg.get("min_confidence_thresh", 0.52)
        self.top_k = sel_cfg.get("top_k_candidates", 5)
        self.max_lines = sel_cfg.get("max_lines", 2)
        self.secondary_dist_thresh = sel_cfg.get("secondary_distance_thresh", 48)

        # Trajectory analyzer
        vehicle_cfg = self.cfg.get("models", {}).get("vehicle_detector", {})
        self.traj_analyzer = TrajectoryAnalyzer(
            model_weights=vehicle_cfg.get("weights", "weights/yolo11x.pt"),
            tracker_cfg=vehicle_cfg.get("tracker", "bytetrack.yaml"),
            track_names=vehicle_cfg.get("track_names", ["bicycle", "car", "motorcycle", "bus", "truck"]),
            conf_thresh=vehicle_cfg.get("conf", 0.35)
        )

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_cache_path(self, camera_id: str) -> Path:
        return self.cache_dir / f"{camera_id}.json"

    def load_from_cache(self, camera_id: str) -> Optional[List[Dict]]:
        cache_file = self.get_cache_path(camera_id)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("detector_mode", "").startswith("ai_"):
                    logger.info("Bỏ qua cache vạch dừng từ model cũ: %s", cache_file)
                    return None
                lines = data.get("lines", [])
                logger.info(f"Đã tải thành công {len(lines)} vạch dừng từ cache: {cache_file}")
                return lines
            except Exception as e:
                logger.warning(f"Lỗi khi đọc file cache {cache_file}: {e}")
        return None

    def save_to_cache(self, camera_id: str, lines: List[Dict], mode: str, video_path: str = ""):
        cache_file = self.get_cache_path(camera_id)
        data = {
            "camera_id": camera_id,
            "source_video": str(video_path),
            "detector_mode": mode,
            "total_lines": len(lines),
            "lines": lines
        }
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Đã lưu cấu hình vạch dừng vào cache: {cache_file}")
        except Exception as e:
            logger.error(f"Không thể ghi cache {cache_file}: {e}")

    # =========================================================================
    # ENTRYPOINT
    # =========================================================================
    def detect(
        self,
        video_source: Union[str, Path, cv2.VideoCapture],
        camera_id: Optional[str] = None,
        redetect: bool = False,
        max_lines: Optional[int] = None,
        debug: bool = False
    ) -> List[Dict]:
        source_name = str(video_source)
        if max_lines is not None:
            if max_lines < 1:
                raise ValueError("max_lines must be positive")
            self.max_lines = max_lines
        if camera_id is None:
            camera_id = Path(source_name).stem if isinstance(video_source, (str, Path)) else "cam_default"

        if not redetect:
            cached = self.load_from_cache(camera_id)
            if cached is not None and len(cached) > 0:
                return cached

        logger.info(f"Thực hiện phát hiện vạch dừng: SINH ỨNG VIÊN -> CHẤM ĐIỂM -> CHỌN cho '{camera_id}'...")

        # 1. Trích xuất đặc trưng chuyển động từ quỹ đạo xe (Motion & Trajectory)
        motion_info = self.traj_analyzer.extract_motion_features(
            video_source=video_source,
            sample_frames=min(self.sample_frames, 100)
        )

        # 2. Sinh tất cả ứng viên bằng CLAHE + Adaptive Threshold + Hough
        candidates, bg_frame = self._generate_candidates(video_source, motion_info)

        if not candidates:
            logger.warning("Không tìm thấy ứng viên vạch dừng nào trên mặt đường.")
            return []

        # 3. Chấm điểm từng ứng viên theo đa tiêu chí
        scored_candidates = self._score_candidates(candidates, motion_info, bg_frame)

        # 4. Xuất ảnh debug nếu được yêu cầu
        if debug:
            self._export_debug_visualizations(bg_frame, scored_candidates, motion_info, camera_id)

        # 5. Lựa chọn ứng viên tốt nhất
        selected_lines = self._select_best_candidate(scored_candidates, camera_id, bg_frame)

        if selected_lines:
            self.save_to_cache(camera_id, selected_lines, mode="classical", video_path=source_name)

        return selected_lines

    def _generate_candidates(
        self,
        video_source: Union[str, Path, cv2.VideoCapture],
        motion_info: Dict
    ) -> Tuple[List[Dict], np.ndarray]:
        frames, H, W = self._sample_video_frames(video_source, self.sample_frames)
        if not frames:
            return [], np.zeros((1080, 1080, 3), dtype=np.uint8)

        # Ảnh nền tĩnh bằng median trải đều trên toàn bộ khung hình mẫu
        sample_step = max(1, len(frames) // 25)
        bg_frame = np.median(frames[::sample_step][:25], axis=0).astype(np.uint8)

        # Vùng xét mặt đường ROI
        roi_top = int(H * self.roi_y_min)
        roi_bottom = int(H * self.roi_y_max)
        roi = bg_frame[roi_top:roi_bottom, :]

        # Áp dụng CLAHE trên kênh sáng L của không gian màu LAB (cân bằng sáng thích nghi)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl_l = clahe.apply(l_channel)
        enhanced_roi = cv2.merge((cl_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_roi, cv2.COLOR_LAB2BGR)
        enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

        # Ngưỡng thích nghi cục bộ (Adaptive Threshold)
        adaptive_mask = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -5
        )

        # Lọc màu sơn trắng HSV bổ trợ
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_hsv = cv2.inRange(hsv_roi, np.array([0, 0, 120]), np.array([180, 65, 255]))
        combined_binary = cv2.bitwise_or(adaptive_mask, white_hsv)

        # Canny edge detection
        edges = cv2.Canny(combined_binary, 40, 130)

        # HoughLinesP tìm tất cả đoạn thẳng
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=30, minLineLength=30, maxLineGap=25
        )

        if lines is None:
            return [], bg_frame

        # Thu thập toàn bộ đoạn thẳng và quy đổi tọa độ về toàn khung hình
        raw_segments = []
        for l in lines:
            x1, y1, x2, y2 = (l[0] if len(l.shape) > 2 else l).astype(int)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.hypot(dx, dy)
            if length >= 25:
                raw_segments.append((min(x1, x2), y1 + roi_top, max(x1, x2), y2 + roi_top, length))

        # Gộp các đoạn thẳng gần nhau và đồng hướng (Collinear Merging)
        merged_candidates = self._merge_collinear_segments(raw_segments, H, W, motion_info)
        return merged_candidates, bg_frame

    def _merge_collinear_segments(
        self,
        segments: List[Tuple],
        H: int,
        W: int,
        motion_info: Dict
    ) -> List[Dict]:
        """Gộp các đoạn thẳng gần nhau thành các đường ứng viên duy nhất kèm đặc trưng"""
        # Gom nhóm theo trục Y
        groups = []
        for x1, y1, x2, y2, l in segments:
            ym = (y1 + y2) / 2
            matched = False
            for grp in groups:
                if abs(grp['ym'] - ym) < 28:
                    grp['segments'].append((x1, y1, x2, y2, l))
                    grp['ym'] = np.mean([(it[1] + it[3]) / 2 for it in grp['segments']])
                    matched = True
                    break
            if not matched:
                groups.append({'ym': ym, 'segments': [(x1, y1, x2, y2, l)]})

        candidates = []
        road_mask = motion_info.get("road_roi_mask")

        for idx, grp in enumerate(groups):
            segs = grp['segments']
            min_x = min(it[0] for it in segs)
            max_x = max(it[2] for it in segs)
            span_x = max_x - min_x
            tot_len = sum(it[4] for it in segs)

            if span_x < W * 0.15:
                continue

            # Fit line bằng cv2.fitLine
            pts = []
            for it in segs:
                pts.append([it[0], it[1]])
                pts.append([it[2], it[3]])
            pts_arr = np.array(pts, dtype=np.int32)
            vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            slope = vy / vx if abs(vx) > 1e-4 else 0.0
            slope = max(-0.15, min(0.15, slope))

            mid_x = (min_x + max_x) / 2
            mid_y = int(grp['ym'])
            
            # Mở rộng vạch ra toàn bề rộng mặt đường (0 -> W-1) để nối liền vạch bị xe che khuất
            ext_x1 = 0
            ext_x2 = W - 1
            ext_y1 = int(round(mid_y + slope * (ext_x1 - mid_x)))
            ext_y2 = int(round(mid_y + slope * (ext_x2 - mid_x)))
            ext_y1 = max(0, min(H - 1, ext_y1))
            ext_y2 = max(0, min(H - 1, ext_y2))

            midpoint = (int((ext_x1 + ext_x2) / 2), int((ext_y1 + ext_y2) / 2))

            # Bỏ ứng viên nằm ngoài Road ROI
            if road_mask is not None:
                mx, my = min(W - 1, max(0, midpoint[0])), min(H - 1, max(0, midpoint[1]))
                if road_mask[my, mx] == 0:
                    continue # Bỏ qua vì nằm ngoài vùng xe chạy

            dx = ext_x2 - ext_x1
            dy = ext_y2 - ext_y1
            length = float(np.hypot(dx, dy))
            angle_rad = float(np.arctan2(dy, dx))
            angle_deg = float(np.degrees(angle_rad))

            candidates.append({
                "id": f"cand_{idx+1}",
                "p1": (ext_x1, ext_y1),
                "p2": (ext_x2, ext_y2),
                "orig_p1": (int(min_x), y1 if 'y1' in locals() else ext_y1),
                "orig_p2": (int(max_x), y2 if 'y2' in locals() else ext_y2),
                "midpoint": midpoint,
                "length": length,
                "span_x": span_x,
                "total_paint_len": tot_len,
                "angle_deg": angle_deg,
                "unit_vec": (dx / max(1e-4, length), dy / max(1e-4, length))
            })

        return candidates

    # =========================================================================
    # 2. CHẤM ĐIỂM ĐA TIÊU CHÍ (MULTI-CRITERIA SCORING)
    # =========================================================================
    def _score_candidates(self, candidates: List[Dict], motion_info: Dict, bg_frame: Optional[np.ndarray] = None) -> List[Dict]:
        flow_vec = motion_info.get("flow_vector", (0.0, 1.0)) # (ux, uy)
        trajectories = motion_info.get("trajectories", {})
        stopped_clusters = motion_info.get("stopped_clusters", [])
        h, w = motion_info.get("frame_shape", (1080, 1080))

        # Tính bản đồ tương phản biên cạnh dọc (Sobel Y) trên ảnh nền mặt đường
        sobel_y = None
        if bg_frame is not None:
            gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY) if len(bg_frame.shape) == 3 else bg_frame
            sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

        scored_list = []

        for cand in candidates:
            p1, p2 = cand["p1"], cand["p2"]
            cand_vec = cand["unit_vec"] # (lx, ly)
            midpoint = cand["midpoint"]

            # --- Tiêu chí 1: Góc so với hướng dòng xe (Càng vuông góc càng cao) ---
            # Tích có hướng |lx*uy - ly*ux| = |sin(alpha)|
            # Vuông góc -> sin(alpha) = 1.0; Song song (vạch phân làn) -> sin(alpha) = 0.0
            cross_prod = abs(cand_vec[0] * flow_vec[1] - cand_vec[1] * flow_vec[0])
            s_angle = float(cross_prod)
            # Phạt nặng các vạch gần song song với hướng dòng xe (< 0.40)
            if s_angle < 0.40:
                s_angle *= 0.2

            # --- Tiêu chí 2: Số quỹ đạo xe di chuyển thực tế cắt qua ứng viên ---
            intersecting_count = 0
            moving_tracks_count = 0
            for tid, pts in trajectories.items():
                if len(pts) >= 4:
                    disp = float(np.linalg.norm(np.array(pts[-1], dtype=float) - np.array(pts[0], dtype=float)))
                    if disp >= 35.0: # Chỉ xét xe thực sự di chuyển (loại bỏ xe đứng yên rung lắc)
                        moving_tracks_count += 1
                        if self._check_trajectory_intersects_line(pts, p1, p2):
                            intersecting_count += 1

            total_moving = max(1, moving_tracks_count)
            s_traj = float(min(1.0, intersecting_count / max(1, 0.35 * total_moving)))

            # --- Tiêu chí 3: Khoảng cách tới cụm xe dừng (Ngay trước đầu xe dừng) ---
            s_stop = 0.5 # Giá trị trung lập mặc định nếu không có xe dừng
            if stopped_clusters:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                slope = dy / dx if dx != 0 else 0.0
                best_stop_score = 0.0
                has_front_car = False

                for cl in stopped_clusters:
                    c_pt = cl["center"]
                    # Tọa độ Y của vạch ngay tại vị trí X của xe dừng
                    line_y_at_car = p1[1] + slope * (c_pt[0] - p1[0])
                    vert_dist = float(line_y_at_car - c_pt[1])
                    dist_along_flow = vert_dist * float(flow_vec[1])

                    # Vạch dừng chuẩn nằm ngay phía trước đầu xe dừng 25 - 250px theo chiều tiến
                    if 20 <= dist_along_flow <= 260:
                        score_cl = float(np.exp(-((dist_along_flow - 90) ** 2) / (2 * (55 ** 2))))
                        if score_cl > best_stop_score:
                            best_stop_score = score_cl
                            has_front_car = True

                if has_front_car:
                    s_stop = best_stop_score
                else:
                    s_stop = 0.30

            # --- Tiêu chí 4: Độ dài tương đối so với bề rộng mặt đường ---
            s_len = float(min(1.0, cand["span_x"] / (w * 0.50)))

            # --- Tiêu chí 5: Độ tương phản biên cạnh vạch sơn thật (Loại vệt loang nước) ---
            s_contrast = 0.5
            if sobel_y is not None:
                mask = np.zeros(sobel_y.shape, dtype=np.uint8)
                cv2.line(mask, p1, p2, 255, 6)
                mean_contrast = float(cv2.mean(sobel_y, mask=mask)[0])
                s_contrast = float(min(1.0, mean_contrast / 5.0))

            # Tổng điểm có trọng số
            total_score = (
                self.w_angle * s_angle +
                self.w_traj * s_traj +
                self.w_stop * s_stop +
                self.w_len * s_len +
                self.w_contrast * s_contrast
            )
            total_score = round(float(total_score), 3)

            cand_result = cand.copy()
            cand_result["sub_scores"] = {
                "s_angle": round(s_angle, 3),
                "s_traj": round(s_traj, 3),
                "s_stop": round(s_stop, 3),
                "s_len": round(s_len, 3),
                "s_contrast": round(s_contrast, 3)
            }
            cand_result["intersecting_tracks"] = intersecting_count
            cand_result["total_score"] = total_score
            scored_list.append(cand_result)

        scored_list.sort(key=lambda x: x["total_score"], reverse=True)
        return scored_list

    def _check_trajectory_intersects_line(self, track_pts: List[Tuple[int, int]], p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Kiểm tra đoạn nối quỹ đạo xe có cắt ngang đoạn thẳng p1-p2 không"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        for i in range(len(track_pts) - 1):
            A, B = track_pts[i], track_pts[i + 1]
            if (ccw(A, p1, p2) != ccw(B, p1, p2)) and (ccw(A, B, p1) != ccw(A, B, p2)):
                return True
        return False

    # =========================================================================
    # 3. CHỌN ỨNG VIÊN TỐT NHẤT & CẢNH BÁO
    # =========================================================================
    def _select_best_candidate(self, scored_candidates: List[Dict], camera_id: str, bg_frame: np.ndarray) -> List[Dict]:
        if not scored_candidates:
            return []

        best = scored_candidates[0]
        h, w = bg_frame.shape[:2]

        # In bảng điểm top-5 ứng viên ra terminal
        logger.info(f"\n--- BẢNG ĐIỂM TOP-{min(self.top_k, len(scored_candidates))} ỨNG VIÊN VẠCH DỪNG CHO '{camera_id}' ---")
        logger.info(f"{'ID':<8} | {'Tọa độ (p1 -> p2)':<28} | {'Góc(S_ang)':<10} | {'Quỹ đạo(S_tr)':<13} | {'Xe dừng(S_st)':<13} | {'Độ dài(S_len)':<13} | {'Tương phản(S_cnt)':<15} | {'TỔNG ĐIỂM':<10}")
        logger.info("-" * 128)

        for cand in scored_candidates[:self.top_k]:
            sc = cand["sub_scores"]
            coord_str = f"({cand['p1'][0]},{cand['p1'][1]})->({cand['p2'][0]},{cand['p2'][1]})"
            logger.info(
                f"{cand['id']:<8} | {coord_str:<28} | {sc['s_angle']:<10.2f} | {sc['s_traj']:<13.2f} | "
                f"{sc['s_stop']:<13.2f} | {sc['s_len']:<13.2f} | {sc.get('s_contrast', 0.5):<15.2f} | {cand['total_score']:<10.3f}"
            )

        # Kiểm tra ngưỡng tin cậy
        if best["total_score"] < self.min_confidence:
            logger.warning(
                f"[CẢNH BÁO] Ứng viên điểm cao nhất ({best['id']}) có điểm {best['total_score']} < {self.min_confidence} (ĐỘ TIN CẬY THẤP!). "
                f"Đã xuất ảnh top-5 ứng viên. Vui lòng chạy 'preview_stopline.py' để xác nhận."
            )
            # Xuất ảnh cảnh báo top-5
            warn_img_path = f"debug_warning_top5_{camera_id}.jpg"
            self._draw_top_k_warning_image(bg_frame, scored_candidates[:self.top_k], warn_img_path, camera_id)

        # Chọn danh sách vạch dừng hợp lệ (hỗ trợ tự động 1 hoặc 2 vạch nếu có vạch thứ 2 hợp lệ)
        selected = []
        for cand in scored_candidates:
            if len(selected) >= self.max_lines:
                break

            if cand["total_score"] < self.min_confidence:
                continue

            p1, p2 = cand["p1"], cand["p2"]
            avg_y = int((p1[1] + p2[1]) / 2)

            # Kiểm tra khoảng cách với các vạch đã chọn trước đó
            too_close = False
            for prev in selected:
                prev_y = int((prev["line"][0][1] + prev["line"][1][1]) / 2)
                if abs(avg_y - prev_y) < self.secondary_dist_thresh:
                    too_close = True
                    break

            if too_close:
                continue

            final_line = [(p1[0], avg_y), (p2[0], avg_y)]
            line_idx = len(selected) + 1
            direction = "TOP_TO_BOTTOM" if avg_y < h * 0.55 else "BOTTOM_TO_TOP"

            selected.append({
                "id": f"stop_line_{line_idx}",
                "name": f"Vach {line_idx} (Chon tu dong - Y={avg_y})",
                "line": final_line,
                "direction": direction,
                "confidence": cand["total_score"],
                "method": "candidate_scoring_selection"
            })

        # Fallback nếu không có vạch nào vượt ngưỡng
        if not selected and scored_candidates:
            p1, p2 = best["p1"], best["p2"]
            avg_y = int((p1[1] + p2[1]) / 2)
            direction = "TOP_TO_BOTTOM" if avg_y < h * 0.55 else "BOTTOM_TO_TOP"
            selected.append({
                "id": "stop_line_1",
                "name": f"Vach dung (Chon tu dong - Y={avg_y})",
                "line": [(p1[0], avg_y), (p2[0], avg_y)],
                "direction": direction,
                "confidence": best["total_score"],
                "method": "candidate_scoring_selection"
            })

        logger.info(f"[KẾT QUẢ CHỌN] Đã tự động chọn {len(selected)} vạch dừng:")
        for s in selected:
            logger.info(f"   -> {s['name']}: {s['line']} (Điểm số: {s['confidence']})")
        logger.info("")

        return selected

    # =========================================================================
    # 4. CHẾ ĐỘ DEBUG: XUẤT ẢNH VISUALIZATION
    # =========================================================================
    def _export_debug_visualizations(
        self,
        bg_frame: np.ndarray,
        scored_candidates: List[Dict],
        motion_info: Dict,
        camera_id: str
    ):
        h, w = bg_frame.shape[:2]

        # 1. ẢNH DEBUG 1: Quỹ đạo xe + Vector dòng xe + Road ROI + Cụm xe dừng
        canvas_motion = bg_frame.copy()

        # Vẽ Road ROI Polygon
        roi_poly = motion_info.get("road_roi_polygon", [])
        if len(roi_poly) > 0:
            pts_poly = np.array(roi_poly, dtype=np.int32)
            cv2.polylines(canvas_motion, [pts_poly], True, (0, 255, 0), 2)
            # Overlay mờ
            overlay = canvas_motion.copy()
            cv2.fillPoly(overlay, [pts_poly], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, canvas_motion, 0.85, 0, canvas_motion)

        # Vẽ quỹ đạo xe
        trajectories = motion_info.get("trajectories", {})
        for tid, pts in trajectories.items():
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(canvas_motion, pts[i], pts[i + 1], (255, 255, 0), 2)

        # Vẽ vector dòng xe chủ đạo ở góc trên
        flow = motion_info.get("flow_vector", (0.0, 1.0))
        arrow_start = (int(w * 0.15), int(h * 0.20))
        arrow_end = (int(arrow_start[0] + flow[0] * 80), int(arrow_start[1] + flow[1] * 80))
        cv2.arrowedLine(canvas_motion, arrow_start, arrow_end, (0, 215, 255), 4, tipLength=0.3)
        cv2.putText(canvas_motion, f"Huong dong xe: ({flow[0]:.2f}, {flow[1]:.2f})", (arrow_start[0] - 50, arrow_start[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

        # Vẽ cụm xe dừng
        for cl in motion_info.get("stopped_clusters", []):
            cp = cl["center"]
            cv2.circle(canvas_motion, tuple(cp), 10, (0, 0, 255), -1)
            cv2.putText(canvas_motion, "Xe dung", (cp[0] + 12, cp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out_motion_path = f"debug_motion_roi_{camera_id}.jpg"
        cv2.imwrite(out_motion_path, canvas_motion)
        logger.info(f"[DEBUG] Đã xuất ảnh quỹ đạo & ROI dòng xe: {out_motion_path}")

        # 2. ẢNH DEBUG 2: Tất cả ứng viên tô màu theo điểm số + Bảng điểm
        canvas_cand = bg_frame.copy()
        for idx, cand in enumerate(scored_candidates):
            score = cand["total_score"]
            # Màu theo điểm số: > 0.60 (Xanh lá), 0.40 - 0.60 (Vàng), < 0.40 (Đỏ)
            if idx == 0:
                color = (0, 255, 0) # Top 1: Xanh lá đậm
                thickness = 4
            elif score >= 0.50:
                color = (0, 215, 255) # Vàng cam
                thickness = 3
            else:
                color = (0, 0, 255) # Đỏ
                thickness = 2

            cv2.line(canvas_cand, cand["p1"], cand["p2"], color, thickness)
            cv2.putText(
                canvas_cand, f"{cand['id']}: {score:.2f}",
                (cand["p1"][0], max(25, cand["p1"][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        out_cand_path = f"debug_candidates_{camera_id}.jpg"
        cv2.imwrite(out_cand_path, canvas_cand)
        logger.info(f"[DEBUG] Đã xuất ảnh ứng viên chấm điểm: {out_cand_path}")

    def _draw_top_k_warning_image(self, bg_frame: np.ndarray, top_cands: List[Dict], out_path: str, camera_id: str):
        canvas = bg_frame.copy()
        colors = [(0, 255, 0), (0, 215, 255), (255, 100, 0), (255, 0, 255), (0, 0, 255)]

        for i, cand in enumerate(top_cands):
            c = colors[i % len(colors)]
            cv2.line(canvas, cand["p1"], cand["p2"], c, 3)
            label = f"Top {i+1} ({cand['id']}) - Score: {cand['total_score']:.2f}"
            cv2.putText(canvas, label, (cand["p1"][0], max(30, cand["p1"][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

        cv2.putText(canvas, f"CANH BAO DO TIN CAY THAP (< {self.min_confidence}) - VUI LONG XAC NHAN",
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(out_path, canvas)

    def _sample_video_frames(self, video_source, num_samples: int):
        if isinstance(video_source, (str, Path)):
            cap = cv2.VideoCapture(str(video_source))
            should_release = True
        else:
            cap = video_source
            should_release = False

        if not cap.isOpened():
            return [], 0, 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        indices = np.linspace(0, min(total - 1, 150), min(num_samples, total), dtype=int)
        frames = []
        orig_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_pos)
        if should_release:
            cap.release()

        if not frames:
            return [], 0, 0
        h, w = frames[0].shape[:2]
        return frames, h, w
