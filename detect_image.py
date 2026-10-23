"""Detect road vehicles in one image or every image in a folder."""
import argparse
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VEHICLE_NAMES = {'bicycle', 'car', 'motorcycle', 'bus', 'truck'}


def resolve_device(requested: str):
    value = str(requested or 'auto').lower()
    if value == 'auto':
        return 0 if torch.cuda.is_available() else 'cpu'
    if value == 'cpu':
        return 'cpu'
    if not torch.cuda.is_available():
        raise RuntimeError('Da yeu cau GPU nhung PyTorch khong nhan CUDA. Dung --device cpu hoac cai PyTorch CUDA.')
    return int(value.removeprefix('cuda:'))


def list_images(source: Path):
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'File khong phai anh ho tro: {source}')
        return [source]
    if source.is_dir():
        images = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f'Khong tim thay anh trong: {source}')
        return images
    raise FileNotFoundError(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='images/input', help='Mot file anh hoac thu muc anh')
    parser.add_argument('--output', default='images/output', help='Thu muc luu anh da ve bbox')
    parser.add_argument('--weights', default=None, help='Checkpoint YOLO xe; mac dinh lay tu config.yaml')
    parser.add_argument('--device', default='auto', help='auto, cpu, 0, 1...')
    parser.add_argument('--conf', type=float, default=0.25, help='Nguong confidence')
    parser.add_argument('--imgsz', type=int, default=None, help='Kich thuoc inference; mac dinh tu config.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8')) or {}
    vehicle_cfg = config.get('models', {}).get('vehicle_detector', {})
    weights = args.weights or vehicle_cfg.get('weights', 'weights/coco/yolo11x.pt')
    if not Path(weights).is_file():
        raise FileNotFoundError(f'Khong tim thay trong so: {weights}')
    device = resolve_device(args.device)
    images = list_images(Path(args.source))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    class_ids = [index for index, name in model.names.items() if name in VEHICLE_NAMES]
    if not class_ids:
        raise ValueError('Checkpoint khong co cac class xe COCO: bicycle, car, motorcycle, bus, truck.')
    imgsz = args.imgsz or vehicle_cfg.get('imgsz', 960)
    print(f'Dang dung {weights} tren {device}; xu ly {len(images)} anh.')

    total = 0
    for image_path in images:
        result = model.predict(str(image_path), classes=class_ids, conf=args.conf,
                               imgsz=imgsz, device=device, verbose=False)[0]
        rendered = result.plot()
        target = output / f'{image_path.stem}_detected{image_path.suffix}'
        if not cv2.imwrite(str(target), rendered):
            raise RuntimeError(f'Khong ghi duoc: {target}')
        count = 0 if result.boxes is None else len(result.boxes)
        total += count
        print(f'{image_path.name}: {count} phuong tien -> {target}')
    print(f'Hoan tat: {total} phuong tien trong {len(images)} anh.')


if __name__ == '__main__':
    main()
