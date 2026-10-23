"""Download a reproducible vehicle subset of official COCO 2017 for pipeline practice."""
import json
import random
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml

COCO_VEHICLES = {2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}


def download_file(url, destination):
    destination = Path(destination)
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as out:
                import shutil
                shutil.copyfileobj(response, out)
            partial.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def download_coco_vehicles(root, train_count=1000, val_count=200, test_count=200, seed=42):
    """Train from train2017; disjoint validation/test from val2017.

    This is a practice subset, NOT an official COCO test benchmark or a new domain.
    Images containing crowd annotations for selected classes are excluded rather than
    treating crowd regions as unlabelled background.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    settings = {"train": train_count, "val": val_count, "test": test_count, "seed": seed}
    settings_path = root / "subset_settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Subset settings changed. Use a new dataset directory.")
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    archive = root / "annotations_trainval2017.zip"
    # Path-style official S3 endpoint has a valid TLS certificate (the dotted bucket hostname does not).
    download_file("https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
    mapping = {category: index for index, category in enumerate(COCO_VEHICLES)}
    rng = random.Random(seed)
    sources = []
    licenses = []
    with zipfile.ZipFile(archive) as zipped:
        for official, split_counts in (("train2017", {"train": train_count}),
                                       ("val2017", {"val": val_count, "test": test_count})):
            with zipped.open(f"annotations/instances_{official}.json") as stream:
                coco = json.load(stream)
            licenses = coco.get("licenses", [])
            annotations, crowds = defaultdict(list), set()
            for ann in coco["annotations"]:
                if ann["category_id"] in mapping:
                    if ann.get("iscrowd", 0):
                        crowds.add(ann["image_id"])
                    else:
                        annotations[ann["image_id"]].append(ann)
            images = {im["id"]: im for im in coco["images"]}
            remaining = sorted(set(annotations) - crowds)
            rng.shuffle(remaining)
            for split, count in split_counts.items():
                if count < 5 or count > len(remaining):
                    raise ValueError(f"Choose at least 5 and at most {len(remaining)} images for {split}.")
                # Guarantee each class is represented, then fill from the same seeded pool.
                selected = []
                for category in mapping:
                    match = next((i for i in remaining if i not in selected and
                                  any(a["category_id"] == category for a in annotations[i])), None)
                    if match is None:
                        raise ValueError(f"Not enough images for category {category}")
                    selected.append(match)
                selected += [i for i in remaining if i not in selected][:count-len(selected)]
                chosen = set(selected)
                remaining = [i for i in remaining if i not in chosen]
                for number, image_id in enumerate(selected, 1):
                    im = images[image_id]
                    filename = im["file_name"]
                    image_path = root / "images" / split / filename
                    download_file(f"https://s3.amazonaws.com/images.cocodataset.org/{official}/{filename}", image_path)
                    rows = []
                    for ann in annotations[image_id]:
                        x, y, bw, bh = ann["bbox"]
                        x1, y1 = max(0., x), max(0., y)
                        x2, y2 = min(im["width"], x+bw), min(im["height"], y+bh)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        rows.append(f"{mapping[ann['category_id']]} {(x1+x2)/2/im['width']:.7f} "
                                    f"{(y1+y2)/2/im['height']:.7f} {(x2-x1)/im['width']:.7f} {(y2-y1)/im['height']:.7f}")
                    label_path = root / "labels" / split / Path(filename).with_suffix(".txt")
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("\n".join(rows), encoding="utf-8")
                    sources.append({"split": split, "official_split": official, **im})
                    if number % 100 == 0 or number == count:
                        print(f"COCO {split}: {number}/{count}")
            del coco, annotations, images
    (root / "sources_and_licenses.json").write_text(
        json.dumps({"source": "https://cocodataset.org/", "terms": "https://cocodataset.org/#termsofuse",
                    "licenses": licenses,
                    "images": sources}, ensure_ascii=False, indent=2), encoding="utf-8")
    data = {"path": str(root.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": list(COCO_VEHICLES.values())}
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")
    return data_yaml
