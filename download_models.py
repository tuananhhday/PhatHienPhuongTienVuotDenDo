"""Download pinned Autoware and Ultralytics pretrained models, verify SHA256."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

REVISION = '527c4905afd8f1bfcc03253529a105f7fa5a6d36'
BASE = f'https://huggingface.co/AutowareFoundation/traffic_light_classifier/resolve/{REVISION}/'
FILES = {
    'traffic_light_classifier_mobilenetv2_batch_1.onnx': '455b71b3b20d3a96aa0e49f32714ba50421f668a2f9b9907c30b1346ac8a3703',
    'lamp_labels.txt': '1a5a49eeec5593963eab8d70f48b8a01bfb07e753e9688eb1510ad26e803579d',
    'coco/yolo11x.pt': '7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24',
}


def download():
    root = Path(__file__).resolve().parent / 'weights'
    root.mkdir(exist_ok=True)
    for name, expected in FILES.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target.read_bytes()).hexdigest() == expected:
            print(f'OK: {target}')
            continue
        url = ('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt'
               if name == 'coco/yolo11x.pt' else BASE + name)
        with urlopen(url, timeout=120) as response:
            content = response.read()
        if sha256(content).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
        temporary = target.with_suffix(target.suffix + '.part')
        temporary.write_bytes(content)
        temporary.replace(target)
        print(f'Downloaded: {target}')


if __name__ == '__main__':
    download()
