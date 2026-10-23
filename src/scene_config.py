"""Video-bound geometry for fixed cameras; no sharing coordinates across scenes."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from src.crossing_tracker import signed_distance


def video_identity(source):
    path = Path(str(source))
    if not path.is_file():
        raise ValueError('Calibration currently requires a local video file')
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open('rb') as stream:
        digest.update(stream.read(1024 * 1024))
        stream.seek(max(0, size - 1024 * 1024))
        digest.update(stream.read(1024 * 1024))
    return digest.hexdigest()


def scene_path(source, directory='configs/scenes'):
    return Path(directory) / f'{Path(source).stem}_{video_identity(source)[:12]}.json'


def validate_scene(scene, source, frame_shape):
    """Validate detailed (v1) and quick 4-click (v2) camera profiles."""
    if scene.get('schema_version') not in (1, 2) or scene.get('video_identity') != video_identity(source):
        raise ValueError('Cau hinh khong thuoc video nay. Chay lai --calibrate cho video moi.')
    h, w = frame_shape[:2]
    if scene.get('frame_size') != [w, h]:
        raise ValueError('Kich thuoc video da thay doi; can xac nhan lai cau hinh.')
    gates = scene.get('gates', [])
    if not gates or len({g.get('id') for g in gates}) != len(gates):
        raise ValueError('Can it nhat mot vung giam sat voi ID rieng.')

    for gate in gates:
        points = np.asarray(gate.get('line'), dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError('Vach can hai diem hop le.')
        approach = gate.get('approach_point')
        if approach is not None and abs(signed_distance(approach, points)) < 3:
            raise ValueError('Diem phia xe di toi qua sat vach.')
        if approach is None and gate.get('direction') not in ('TOP_TO_BOTTOM', 'BOTTOM_TO_TOP'):
            raise ValueError('Cau hinh nhanh can chieu xe hop le.')
        if gate.get('movement', 'straight') not in ('straight', 'left', 'right'):
            raise ValueError('Huong di khong hop le.')
        lane = gate.get('lane_polygon')
        if lane is not None:
            lane = np.asarray(lane, dtype=np.float32)
            if lane.shape != (4, 2) or not np.isfinite(lane).all() or not cv2.isContourConvex(lane) or abs(cv2.contourArea(lane)) < 25:
                raise ValueError('Vung lan xe can bon goc theo thu tu quanh mot da giac loi.')
        anchor = gate.get('anchor', [0.5, 1.0])
        if len(anchor) != 2 or not all(0 <= v <= 1 for v in anchor):
            raise ValueError('Diem dai dien bbox phai nam trong khoang 0..1.')
        check_points = [*points.tolist()]
        if approach is not None:
            check_points.append(approach)
        if lane is not None:
            check_points.extend(lane.tolist())
        for point in check_points:
            if not all(np.isfinite(point)) or not (0 <= point[0] < w and 0 <= point[1] < h):
                raise ValueError('Toa do vuot ngoai anh.')
        if not gate.get('signal_boxes'):
            raise ValueError('Can chon cum den dieu khien.')
        for x1, y1, x2, y2 in gate['signal_boxes']:
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and x2-x1 >= 5 and y2-y1 >= 5):
                raise ValueError('Khung den qua nho hoac nam ngoai anh.')
    return scene

def load_scene(path, source, frame_shape):
    return validate_scene(json.loads(Path(path).read_text(encoding='utf-8')), source, frame_shape)


def save_scene(scene, path, source, frame_shape):
    validate_scene(scene, source, frame_shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding='utf-8')
