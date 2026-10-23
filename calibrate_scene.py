"""Fast fixed-camera setup: two stop-line endpoints and one traffic-light box."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.scene_config import scene_path, video_identity, save_scene

HEADER_HEIGHT = 82


def calibrate(source, output=None):
    cap = cv2.VideoCapture(str(source))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f'Cannot read video: {source}')
    h, w = frame.shape[:2]
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    scale = min(1.0, 1200 / w, 760 / h)
    window = 'Thiet lap nhanh - q: huy'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Frame', window, 0, max(1, total - 1), lambda _: None)
    points, gates = [], []

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4 and HEADER_HEIGHT <= y < HEADER_HEIGHT + int(h * scale):
            points.append([int(np.clip(x / scale, 0, w - 1)),
                           int(np.clip((y - HEADER_HEIGHT) / scale, 0, h - 1))])

    def make_gate():
        a, b, p, q = points
        # Simple calibration has no directional click. Use the same perspective
        # prior as automatic detection; it is correct for ordinary forward views.
        direction = 'BOTTOM_TO_TOP' if (a[1] + b[1]) / 2 >= h * .55 else 'TOP_TO_BOTTOM'
        return {
            'id': f'gate_{len(gates) + 1}', 'name': f'VACH {len(gates) + 1}',
            'line': [a, b], 'direction': direction,
            'signal_boxes': [[min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])]],
            'movement': 'straight', 'anchor': [0.5, 1.0],
            'method': 'quick_scene_calibrated', 'confidence': 1.0,
        }

    cv2.setMouseCallback(window, mouse)
    previous_frame, error = 0, ''
    try:
        while True:
            position = cv2.getTrackbarPos('Frame', window)
            if position != previous_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, candidate = cap.read()
                if ok:
                    frame = candidate
                previous_frame = position

            canvas = frame.copy()
            for gate in gates:
                cv2.line(canvas, tuple(gate['line'][0]), tuple(gate['line'][1]), (0, 255, 0), 3)
                x1, y1, x2, y2 = gate['signal_boxes'][0]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), 2)
            for point in points:
                cv2.circle(canvas, tuple(point), 5, (0, 255, 255), -1)
            if len(points) >= 2:
                cv2.line(canvas, tuple(points[0]), tuple(points[1]), (0, 255, 255), 3)
            if len(points) >= 4:
                cv2.rectangle(canvas, tuple(points[2]), tuple(points[3]), (255, 200, 0), 2)

            display = cv2.resize(canvas, (int(w * scale), int(h * scale)))
            display = np.vstack([np.full((HEADER_HEIGHT, display.shape[1], 3), 20, np.uint8), display])
            if len(points) < 2:
                instruction = '1/2: Click 2 dau VACH DUNG'
            elif len(points) < 4:
                instruction = '2/2: Click 2 goc khung OM SAT CUM DEN giao thong'
            else:
                instruction = 'DA DU 4 DIEM: nhan S de luu, N de them vach'
            status = error or f'Da chon: {len(points)}/4 | u: lui 1 diem | r: lam lai | q: huy'
            for index, text in enumerate([instruction, status, 'Keo thanh Frame de tim khung anh ro.']):
                cv2.putText(display, text, (8, 22 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
            cv2.imshow(window, display)

            key = cv2.waitKey(20) & 0xff
            if 65 <= key <= 90:
                key += 32
            if key == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == ord('r'):
                points.clear(); error = ''
            elif key == ord('u') and points:
                points.pop(); error = ''
            elif key in (ord('n'), ord('s')) and len(points) == 4:
                scene = {'schema_version': 2, 'video_identity': video_identity(source),
                         'source_video': str(Path(source).resolve()), 'frame_size': [w, h],
                         'gates': gates + [make_gate()]}
                if key == ord('n'):
                    gates.append(make_gate()); points.clear(); error = ''
                else:
                    target = Path(output) if output else scene_path(source)
                    save_scene(scene, target, source, frame.shape)
                    print(f'Saved: {target}')
                    return scene
            elif key == ord('s'):
                error = f'Chua luu: can du 4 diem, hien co {len(points)}/4.'
    finally:
        cap.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    calibrate(args.video, args.output)
