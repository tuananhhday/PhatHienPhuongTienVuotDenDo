"""Sample independent camera videos for annotation; never invent training labels."""
import argparse
import json
from pathlib import Path
import cv2
from src.scene_config import video_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--split', choices=['train','val','test'], required=True)
    parser.add_argument('--output', type=Path, default=Path('scene_dataset'))
    parser.add_argument('--interval', type=float, default=1.0, help='Seconds between sampled frames')
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError('Interval must be positive')
    identity = video_identity(args.video)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if identity in manifest and manifest[identity]['split'] != args.split:
        raise ValueError('Một video không được xuất vào nhiều split. Tách theo camera/video, không random frame.')
    folder = args.output/'images'/args.split
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(args.interval*fps))
    files = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                name = f'{identity[:12]}_{frame_index:07d}.jpg'
                if not cv2.imwrite(str(folder/name),frame):
                    raise OSError(f'Cannot save {name}')
                files.append({'file':str((folder/name).relative_to(args.output)),
                              'frame':frame_index,'seconds':frame_index/fps})
            frame_index += 1
    finally:
        cap.release()
    manifest[identity] = {'source':str(Path(args.video).resolve()),'split':args.split,
                          'annotation_status':'unlabeled','frames':files}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    truth_path = args.output/f'{identity[:12]}_events_template.json'
    if not truth_path.exists():
        truth_path.write_text(json.dumps({'video_identity':identity,'reviewed':False,'events':[]},indent=2),encoding='utf-8')
    print(f'Saved {len(files)} unlabeled images to {folder}; annotate before training.')


if __name__ == '__main__':
    main()
