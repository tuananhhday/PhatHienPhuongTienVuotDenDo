"""Shared, real-data training and COCO evaluation; also embedded in the Colab notebook."""
import hashlib
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

MODEL_NAMES = ("ssd300_vgg16", "yolov8s", "yolo11x")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_names(config):
    names = config.get("names")
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        if sorted(names) != list(range(len(names))):
            raise ValueError("Class IDs must be consecutive, starting at 0.")
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("data.yaml needs a nonempty list of unique class names.")
    return names


def image_to_label(path):
    parts = list(Path(path).parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Image path must contain an images directory: {path}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(path, count):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing label: {path}. Negative images need an empty .txt file.")
    labels = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(x) for x in line.split()]
            if len(values) != 5:
                raise ValueError("Expected class cx cy width height; segmentation labels are unsupported")
            cls, cx, cy, width, height = values
            if not all(math.isfinite(x) for x in values):
                raise ValueError("Non-finite value")
            if cls != int(cls) or not 0 <= cls < count:
                raise ValueError("Invalid class ID")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("Coordinates must be normalized to [0, 1] with positive box size")
            if min(cx - width / 2, cy - height / 2) < -1e-4 or max(cx + width / 2, cy + height / 2) > 1.0001:
                raise ValueError("Box extends beyond image")
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        labels.append((int(cls), cx, cy, width, height))
    return labels


def prepare_detection_data(yaml_path, output_yaml):
    """Normalize relocated ZIP exports, validate labels and detect exact split leakage.

    Supports directory-based images/train or train/images exports, not image-list files.
    Camera/video group separation must be performed by the dataset author.
    """
    yaml_path = Path(yaml_path).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    names = class_names(config)
    local_root = yaml_path.parent
    configured_root = Path(str(config.get("path", ".")))
    if not configured_root.is_absolute():
        configured_root = local_root / configured_root
    splits, manifest, seen = {}, {}, {}
    for split in ("train", "val", "test"):
        raw = config.get(split)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"data.yaml must define {split} as an image directory.")
        # Prefer directories next to the supplied YAML over obsolete machine-specific paths.
        aliases = ("val", "valid") if split == "val" else (split,)
        candidates = [local_root / raw]
        candidates += [local_root / alias / "images" for alias in aliases]
        candidates += [local_root / "images" / alias for alias in aliases]
        candidates += [configured_root / raw]
        directory = next((p.resolve() for p in candidates if p.is_dir()), None)
        if directory is None:
            raise FileNotFoundError(f"Cannot locate {split} images next to {yaml_path}.")
        images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"Empty split: {directory}")
        counts = [0] * len(names)
        split_records = []
        for img in images:
            with Image.open(img) as im:
                im.verify()
            label = image_to_label(img)
            rows = read_labels(label, len(names))
            for cls, *_ in rows:
                counts[cls] += 1
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError(f"Image leakage between splits: {seen[digest][1]} and {img}")
            seen[digest] = (split, str(img))
            split_records.append({"image": str(img.relative_to(directory)), "sha256": digest,
                                  "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest()})
        if any(n == 0 for n in counts):
            raise ValueError(f"{split} set has classes without boxes: {dict(zip(names, counts))}. "
                             "Every evaluated class must be represented in every split.")
        splits[split] = str(directory)
        manifest[split] = {"images": len(images), "instances": dict(zip(names, counts)), "files": split_records}
        print(split, len(images), "images;", dict(zip(names, counts)))
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump({**splits, "names": names}, allow_unicode=True), encoding="utf-8")
    (output_yaml.parent / "dataset_manifest.json").write_text(
        json.dumps({"names": names, "splits": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_yaml


class TrafficDataset(Dataset):
    """YOLO class IDs 0..C-1; SSD target IDs 1..C (0 is background)."""
    def __init__(self, data_yaml, split="train", augment=False):
        self.config = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        self.names = class_names(self.config)
        self.images = sorted(p for p in Path(self.config[split]).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as im:
            image = TF.to_tensor(im.convert("RGB"))
        _, h, w = image.shape
        rows = read_labels(image_to_label(path), len(self.names))
        boxes, labels = [], []
        for cls, cx, cy, bw, bh in rows:
            boxes.append([max(0, (cx-bw/2)*w), max(0, (cy-bh/2)*h),
                          min(w, (cx+bw/2)*w), min(h, (cy+bh/2)*h)])
            labels.append(cls + 1)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target = {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor(index)}
        return image, target


def collate_detection(batch):
    return tuple(zip(*batch))


def build_ssd(num_classes, pretrained=True):
    from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
    from torchvision.models.detection.ssd import SSDClassificationHead
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT if pretrained else None,
                        weights_backbone=None, score_thresh=0.001, nms_thresh=0.6,
                        detections_per_img=100)
    channels = [layer.in_channels for layer in model.head.classification_head.module_list]
    anchors = model.anchor_generator.num_anchors_per_location()
    # Retain COCO-pretrained feature extractor AND box regressor; replace only class head.
    model.head.classification_head = SSDClassificationHead(channels, anchors, num_classes + 1)
    return model


@torch.inference_mode()
def predict_detector(model, image, device, kind="ssd", imgsz=640, conf=0.001):
    """Return a framework-neutral prediction with 0-based class labels on CPU."""
    if kind == "ssd":
        model.eval()
        raw = model([image.to(device)])[0]
        keep = raw["scores"] >= conf
        return {"boxes": raw["boxes"][keep].cpu(), "scores": raw["scores"][keep].cpu(),
                "labels": raw["labels"][keep].cpu() - 1}
    result = model.predict(TF.to_pil_image(image), imgsz=imgsz, device=device,
                           conf=conf, iou=0.6, max_det=100, half=False, verbose=False)[0]
    return {"boxes": result.boxes.xyxy.cpu(), "scores": result.boxes.conf.cpu(),
            "labels": result.boxes.cls.long().cpu()}


@torch.inference_mode()
def evaluate_detector(model, dataset, device, kind="ssd", imgsz=640):
    """Identical COCO box evaluator for both frameworks, labels mapped to 0..C-1."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    if kind == "ssd":
        model.eval()
    for index in range(len(dataset)):
        image, target = dataset[index]
        pred = predict_detector(model, image, device, kind, imgsz, conf=0.001)
        truth = {"boxes": target["boxes"], "labels": target["labels"] - 1}
        metric.update([pred], [truth])
    raw = metric.compute()
    result = {"mAP50_95": float(raw["map"]), "mAP50": float(raw["map_50"]),
              "mAR100": float(raw["mar_100"]), "images": len(dataset)}
    result["per_class_AP"] = {dataset.names[int(cls)]: float(ap)
                              for cls, ap in zip(raw["classes"].reshape(-1), raw["map_per_class"].reshape(-1))}
    return result


def box_iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros((len(boxes1), len(boxes2)))
    area1 = ((boxes1[:, 2]-boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3]-boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2]-boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3]-boxes2[:, 1]).clamp(min=0))
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (right_bottom-left_top).clamp(min=0).prod(2)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-9)


@torch.inference_mode()
def detection_diagnostics(model, dataset, device, kind="ssd", imgsz=640,
                          conf_threshold=0.25, iou_threshold=0.5):
    """Fixed-threshold P/R/F1, confusion matrix and class PR curves at IoU=0.50."""
    count = len(dataset.names)
    background = count
    confusion = np.zeros((count+1, count+1), dtype=np.int64)  # rows=true, cols=pred
    pr_records = {c: [] for c in range(count)}
    gt_totals = np.zeros(count, dtype=np.int64)

    for index in range(len(dataset)):
        image, target = dataset[index]
        pred = predict_detector(model, image, device, kind, imgsz, conf=0.001)
        gt_boxes, gt_labels = target["boxes"], target["labels"] - 1
        for c in range(count):
            gt_totals[c] += int((gt_labels == c).sum())
            class_gt = gt_boxes[gt_labels == c]
            class_mask = pred["labels"] == c
            class_boxes = pred["boxes"][class_mask]
            class_scores = pred["scores"][class_mask]
            order = class_scores.argsort(descending=True)
            used = set()
            ious = box_iou_matrix(class_boxes[order], class_gt)
            for row, original in enumerate(order.tolist()):
                is_tp = 0
                if len(class_gt):
                    values, columns = ious[row].sort(descending=True)
                    for value, column in zip(values.tolist(), columns.tolist()):
                        if value >= iou_threshold and column not in used:
                            used.add(column)
                            is_tp = 1
                            break
                pr_records[c].append((float(class_scores[original]), is_tp))

        keep = pred["scores"] >= conf_threshold
        p_boxes, p_labels = pred["boxes"][keep], pred["labels"][keep]
        ious = box_iou_matrix(gt_boxes, p_boxes)
        candidates = []
        for g in range(len(gt_boxes)):
            for p in range(len(p_boxes)):
                if float(ious[g, p]) >= iou_threshold:
                    candidates.append((float(ious[g, p]), g, p))
        matched_gt, matched_pred = set(), set()
        for _, g, p in sorted(candidates, reverse=True):
            if g not in matched_gt and p not in matched_pred:
                confusion[int(gt_labels[g]), int(p_labels[p])] += 1
                matched_gt.add(g)
                matched_pred.add(p)
        for g in set(range(len(gt_boxes))) - matched_gt:
            confusion[int(gt_labels[g]), background] += 1
        for p in set(range(len(p_boxes))) - matched_pred:
            confusion[background, int(p_labels[p])] += 1

    per_class, curves = {}, {}
    for c, name in enumerate(dataset.names):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "tp": tp, "fp": fp, "fn": fn, "support": int(gt_totals[c])}
        records = sorted(pr_records[c], key=lambda item: item[0], reverse=True)
        scores = np.array([r[0] for r in records], dtype=float)
        tps = np.array([r[1] for r in records], dtype=float)
        fps = 1-tps
        cumulative_tp, cumulative_fp = np.cumsum(tps), np.cumsum(fps)
        curve_precision = cumulative_tp / np.maximum(cumulative_tp+cumulative_fp, 1e-9)
        curve_recall = cumulative_tp / max(int(gt_totals[c]), 1)
        # Keep curves compact enough for Drive/JSON while preserving their shape.
        sample = np.unique(np.linspace(0, max(len(scores)-1, 0), min(200, len(scores)), dtype=int)) if len(scores) else []
        curves[name] = {"score": scores[sample].tolist() if len(scores) else [],
                        "precision": curve_precision[sample].tolist() if len(scores) else [],
                        "recall": curve_recall[sample].tolist() if len(scores) else []}
    macro = {metric: float(np.mean([values[metric] for values in per_class.values()]))
             for metric in ("precision", "recall", "f1")}
    return {"conf_threshold": conf_threshold, "iou_threshold": iou_threshold,
            "macro": macro, "per_class": per_class, "confusion_matrix": confusion.tolist(),
            "labels": dataset.names + ["background"], "pr_curves": curves}


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def train_ssd(data_yaml, output_dir, epochs=80, batch=8, lr=0.002, workers=2, seed=42, resume=False):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = TrafficDataset(data_yaml, "train", augment=True)
    val = TrafficDataset(data_yaml, "val")
    loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=workers,
                        collate_fn=collate_detection, pin_memory=device == "cuda")
    last, best = output_dir / "last.pth", output_dir / "best.pth"
    if not resume and (last.exists() or best.exists()):
        raise FileExistsError("SSD run exists. Use resume=True or a new RUN_NAME.")
    if resume and not last.exists():
        raise FileNotFoundError(last)
    model = build_ssd(len(train.names), pretrained=not resume).to(device)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(1, int(epochs*.6)), max(2, int(epochs*.85))], gamma=.1)
    start, best_map, history = 0, -1.0, []
    if resume:
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["names"] != train.names or checkpoint["epochs"] != epochs:
            raise ValueError("Resume needs the same class order and epochs budget.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best_map, history = checkpoint["epoch"], checkpoint["best_map"], checkpoint["history"]
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        if torch.cuda.is_available() and checkpoint["cuda_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_state"])
    for epoch in range(start, epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SSD loss at epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        metrics = evaluate_detector(model, val, device)
        score = metrics["mAP50_95"]
        improved = score > best_map
        best_map = max(best_map, score)
        history.append({"epoch": epoch+1, "loss": total/len(loader), **metrics})
        checkpoint = {"architecture": "ssd300_vgg16", "model": model.state_dict(),
                      "names": train.names, "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch+1, "epochs": epochs,
                      "best_map": best_map, "history": history, "random_state": random.getstate(),
                      "numpy_state": np.random.get_state(), "torch_state": torch.get_rng_state(),
                      "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        atomic_save(checkpoint, last)
        if improved:
            atomic_save({"architecture": "ssd300_vgg16", "model": model.state_dict(),
                         "names": train.names, "epoch": epoch+1, "val_metrics": metrics}, best)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"SSD {epoch+1}/{epochs}: loss={total/len(loader):.4f}, val mAP50-95={score:.4f}")
    del model, optimizer
    metadata_path = output_dir / "training_meta.json"
    previous = json.loads(metadata_path.read_text()) if resume and metadata_path.exists() else {}
    metadata = {"model": "ssd300_vgg16", "epochs_completed": epochs,
                "training_seconds": previous.get("training_seconds", 0)+time.perf_counter()-started,
                "peak_vram_mb": max(previous.get("peak_vram_mb", 0),
                                    torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0),
                "batch": batch, "seed": seed}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def train_yolo(model_name, data_yaml, output_root, epochs=80, imgsz=640, batch=4, workers=2,
               seed=42, resume=False):
    from ultralytics import YOLO
    if model_name not in ("yolov8s", "yolo11x"):
        raise ValueError(model_name)
    output_root = Path(output_root)
    run = output_root / model_name
    last = run / "weights" / "last.pt"
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if resume:
        if not last.exists():
            raise FileNotFoundError(last)
        trained = YOLO(str(last))
        trained.train(resume=True, device=0, workers=workers, batch=batch)
    else:
        if run.exists():
            raise FileExistsError(f"{run} exists. Use resume=True or a new RUN_NAME.")
        trained = YOLO(model_name + ".pt")
        trained.train(
            data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
            project=str(output_root), name=model_name, workers=workers, seed=seed,
            deterministic=True, patience=epochs, optimizer="AdamW", lr0=0.001,
            amp=True, plots=True, save=True, save_period=5, cache=False)
    metadata_path = run / "training_meta.json"
    previous = json.loads(metadata_path.read_text()) if resume and metadata_path.exists() else {}
    completed = 0
    results_csv = run / "results.csv"
    if results_csv.exists():
        import pandas as pd
        completed = len(pd.read_csv(results_csv))
    metadata = {"model": model_name, "epochs_completed": completed,
                "training_seconds": previous.get("training_seconds", 0)+time.perf_counter()-started,
                "peak_vram_mb": max(previous.get("peak_vram_mb", 0),
                                    torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0),
                "batch": batch, "seed": seed, "imgsz": imgsz}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del trained
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run / "weights" / "best.pt"


@torch.inference_mode()
def benchmark_detector(model, dataset, device, kind="ssd", imgsz=640, limit=100):
    """Batch=1 wall time: preprocessing + inference + NMS + CPU boxes; excludes file I/O."""
    count = min(limit, len(dataset))
    if kind == "ssd":
        model.eval()

    def predict(image):
        if kind == "ssd":
            result = model([image.to(device)])[0]
            return {k: v.cpu() for k, v in result.items()}
        return model.predict(image, imgsz=imgsz, device=device, conf=0.001,
                             iou=0.6, max_det=100, half=False, verbose=False)[0].boxes.cpu()

    first = TF.to_pil_image(dataset[0][0]) if kind != "ssd" else dataset[0][0]
    for _ in range(5):
        predict(first)
    times = []
    for index in range(count):
        image = dataset[index][0]
        if kind != "ssd":
            image = TF.to_pil_image(image)
        if str(device) != "cpu":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict(image)
        if str(device) != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    return {"latency_ms_mean": float(np.mean(times)), "latency_ms_p95": float(np.percentile(times, 95)),
            "fps": 1000/float(np.mean(times)), "timed_images": count}


def evaluate_run(data_yaml, output_root, imgsz=640):
    """Evaluate only existing best checkpoints, never choose weights on the test split."""
    from ultralytics import YOLO
    output_root = Path(output_root)
    test = TrafficDataset(data_yaml, "test")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in MODEL_NAMES:
        kind = "ssd" if name == "ssd300_vgg16" else "yolo"
        checkpoint_path = output_root/name/("best.pth" if kind == "ssd" else "weights/best.pt")
        if not checkpoint_path.exists():
            print("Skipped (not trained):", name)
            continue
        if kind == "ssd":
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["names"] != test.names:
                raise ValueError("SSD checkpoint classes differ from dataset")
            model = build_ssd(len(test.names), pretrained=False).to(device)
            model.load_state_dict(checkpoint["model"])
        else:
            model = YOLO(str(checkpoint_path))
            if [model.names[i] for i in range(len(model.names))] != test.names:
                raise ValueError("YOLO checkpoint classes differ from dataset")
        metrics = evaluate_detector(model, test, device, kind, imgsz)
        diagnostics = detection_diagnostics(model, test, device, kind, imgsz)
        speed = benchmark_detector(model, test, device, kind, imgsz)
        parameters = sum(parameter.numel() for parameter in
                         (model.parameters() if kind == "ssd" else model.model.parameters()))
        meta_path = output_root/name/"training_meta.json"
        training_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        rows.append({"model": name, "input_size": 300 if kind == "ssd" else imgsz,
                     "checkpoint": str(checkpoint_path), "parameters_m": parameters/1e6,
                     "checkpoint_mb": checkpoint_path.stat().st_size/2**20,
                     "training_minutes": training_meta.get("training_seconds", 0)/60,
                     "peak_vram_mb": training_meta.get("peak_vram_mb", 0),
                     **metrics, **speed, "diagnostics": diagnostics})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not rows:
        raise FileNotFoundError("No best checkpoint found. Train at least one model first.")
    report = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "protocol": "COCO bbox AP; conf=0.001, NMS IoU=0.6, max_det=100; batch=1 FP32; native input sizes",
              "results": rows}
    (output_root/"test_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import pandas as pd
    table = pd.DataFrame([{k: v for k, v in row.items()
                           if k not in ("per_class_AP", "diagnostics")} |
                          {"precision": row["diagnostics"]["macro"]["precision"],
                           "recall": row["diagnostics"]["macro"]["recall"],
                           "f1": row["diagnostics"]["macro"]["f1"]} for row in rows])
    table.to_csv(output_root/"test_benchmark.csv", index=False)
    return table


def plot_dataset_statistics(manifest, output_path):
    import matplotlib.pyplot as plt
    splits = list(manifest["splits"])
    names = manifest["names"]
    images = [manifest["splits"][split]["images"] for split in splits]
    instances = np.array([[manifest["splits"][split]["instances"][name] for name in names]
                          for split in splits])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(splits, images, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_title("Images by split")
    axes[0].set_ylabel("Images")
    bottom = np.zeros(len(splits))
    for index, name in enumerate(names):
        axes[1].bar(splits, instances[:, index], bottom=bottom, label=name)
        bottom += instances[:, index]
    axes[1].set_title("Annotated instances by class")
    axes[1].set_ylabel("Instances")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def _load_best_model(name, output_root, names, device):
    from ultralytics import YOLO
    kind = "ssd" if name == "ssd300_vgg16" else "yolo"
    path = Path(output_root)/name/("best.pth" if kind == "ssd" else "weights/best.pt")
    if kind == "ssd":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_ssd(len(names), pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        model = YOLO(str(path))
    return model, kind, path


def _draw_boxes(ax, image, boxes, labels, names, scores=None, title=""):
    import matplotlib.patches as patches
    ax.imshow(image.permute(1, 2, 0).clamp(0, 1))
    colors = plt_colormap(len(names))
    for index, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [float(x) for x in box]
        label = int(label)
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                       linewidth=1.5, edgecolor=colors[label]))
        text = names[label]
        if scores is not None:
            text += f" {float(scores[index]):.2f}"
        ax.text(x1, max(0, y1-2), text, fontsize=7, color="white",
                bbox={"facecolor": colors[label], "alpha": .8, "pad": 1, "edgecolor": "none"})
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plt_colormap(count):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    return [cmap(i/max(count-1, 1)) for i in range(count)]


def create_paper_artifacts(data_yaml, output_root, imgsz=640, qualitative_images=3):
    """Create figures, tables and a draft results section from completed checkpoints."""
    import matplotlib.pyplot as plt
    import pandas as pd
    output_root = Path(output_root)
    artifact_dir = output_root / "paper_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((output_root/"test_benchmark.json").read_text(encoding="utf-8"))
    rows = report["results"]
    table = pd.read_csv(output_root/"test_benchmark.csv")
    manifest = json.loads((Path(data_yaml).parent/"dataset_manifest.json").read_text(encoding="utf-8"))
    plot_dataset_statistics(manifest, artifact_dir/"dataset_distribution.png")
    plt.close("all")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = table["model"].tolist()
    axes[0].bar(labels, table["mAP50_95"], color="#4c78a8")
    axes[0].set_ylim(0, 1); axes[0].set_title("Test mAP@0.50:0.95")
    axes[1].bar(labels, table["fps"], color="#54a24b"); axes[1].set_title("Detector FPS")
    axes[2].bar(labels, table["parameters_m"], color="#e45756"); axes[2].set_title("Parameters (million)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(artifact_dir/"accuracy_speed_complexity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    per_class = pd.DataFrame({row["model"]: row["per_class_AP"] for row in rows})
    ax = per_class.plot.bar(figsize=(11, 5), rot=20)
    ax.set_ylabel("AP@0.50:0.95"); ax.set_ylim(0, 1); ax.set_title("Per-class test AP")
    ax.figure.tight_layout(); ax.figure.savefig(artifact_dir/"per_class_ap.png", dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    detail_rows = []
    for row in rows:
        for class_name, values in row["diagnostics"]["per_class"].items():
            detail_rows.append({"model": row["model"], "class": class_name,
                                "AP50_95": row["per_class_AP"].get(class_name), **values})
    pd.DataFrame(detail_rows).to_csv(artifact_dir/"per_class_metrics.csv", index=False)

    for row in rows:
        diagnostics = row["diagnostics"]
        matrix = np.array(diagnostics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="Blues")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(diagnostics["labels"])), diagnostics["labels"], rotation=35, ha="right")
        ax.set_yticks(range(len(diagnostics["labels"])), diagnostics["labels"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{row['model']} confusion matrix (IoU≥0.5, conf≥0.25)")
        fig.colorbar(image, ax=ax); fig.tight_layout()
        fig.savefig(artifact_dir/f"confusion_{row['model']}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, curve in diagnostics["pr_curves"].items():
            ax.plot(curve["recall"], curve["precision"], label=name)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
               title=f"{row['model']} PR curves at IoU=0.50")
        ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(artifact_dir/f"pr_{row['model']}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for name in MODEL_NAMES:
        if name == "ssd300_vgg16":
            history_path = output_root/name/"history.json"
            if history_path.exists():
                frame = pd.DataFrame(json.loads(history_path.read_text()))
                axes[0].plot(frame["epoch"], frame["loss"], label=name)
                axes[1].plot(frame["epoch"], frame["mAP50_95"], label=name)
        else:
            csv_path = output_root/name/"results.csv"
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
                frame.columns = [column.strip() for column in frame.columns]
                epoch = frame.get("epoch", pd.Series(range(1, len(frame)+1)))
                loss_cols = [column for column in frame if column.startswith("train/") and column.endswith("loss")]
                map_cols = [column for column in frame if "mAP50-95" in column]
                if loss_cols:
                    axes[0].plot(epoch, frame[loss_cols].sum(axis=1), label=name)
                if map_cols:
                    axes[1].plot(epoch, frame[map_cols[0]], label=name)
    axes[0].set_title("Training loss (scale differs by architecture)"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation mAP@0.50:0.95"); axes[1].set(xlabel="Epoch", ylabel="mAP")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(artifact_dir/"training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    test = TrafficDataset(data_yaml, "test")
    indices = np.linspace(0, len(test)-1, min(qualitative_images, len(test)), dtype=int)
    images_targets = [test[int(index)] for index in indices]
    predictions = {name: [] for name in MODEL_NAMES}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for name in MODEL_NAMES:
        model, kind, _ = _load_best_model(name, output_root, test.names, device)
        for image, _ in images_targets:
            predictions[name].append(predict_detector(model, image, device, kind, imgsz, conf=.25))
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    fig, axes = plt.subplots(len(images_targets), 4, figsize=(18, 4.5*len(images_targets)), squeeze=False)
    for row_index, (image, target) in enumerate(images_targets):
        _draw_boxes(axes[row_index, 0], image, target["boxes"], target["labels"]-1,
                    test.names, title="Ground truth")
        for column, name in enumerate(MODEL_NAMES, 1):
            pred = predictions[name][row_index]
            _draw_boxes(axes[row_index, column], image, pred["boxes"], pred["labels"],
                        test.names, pred["scores"], title=name)
    fig.tight_layout(); fig.savefig(artifact_dir/"qualitative_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    best = table.loc[table["mAP50_95"].idxmax()]
    fastest = table.loc[table["fps"].idxmax()]
    dataset_lines = [f"| {split} | {manifest['splits'][split]['images']} | "
                     f"{sum(manifest['splits'][split]['instances'].values())} |"
                     for split in manifest["splits"]]
    result_lines = [f"| {row.model} | {row.mAP50_95:.3f} | {row.mAP50:.3f} | "
                    f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | "
                    f"{row.fps:.1f} | {row.parameters_m:.1f} | {row.checkpoint_mb:.1f} | "
                    f"{row.training_minutes:.1f} | {row.peak_vram_mb:.0f} |"
                    for _, row in table.iterrows()]
    markdown = f"""# Kết quả thực nghiệm

## Thiết lập thực nghiệm

Cả ba mô hình sử dụng cùng cách chia train/validation/test và cùng thứ tự lớp. SSD300-VGG16 dùng
đầu vào 300×300; YOLOv8s và YOLO11x dùng {imgsz}×{imgsz}. Checkpoint tốt nhất chỉ được chọn theo
mAP trên validation. Độ chính xác cuối cùng được đo một lần trên test bằng COCO box AP.
Precision, Recall, F1 và confusion matrix dùng confidence 0,25 và IoU 0,50. Tốc độ detector được đo
với batch=1, FP32 trên **{report['device']}**, sau warm-up, không tính đọc file và tracking.

| Tập | Số ảnh | Số đối tượng |
|---|---:|---:|
{chr(10).join(dataset_lines)}

## So sánh định lượng

| Model | mAP50-95 | mAP50 | Precision | Recall | F1 | FPS | Params (M) | Weights (MB) | Train (min) | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(result_lines)}

Mô hình có mAP50-95 cao nhất trên test là **{best['model']}** ({best['mAP50_95']:.3f}).
Mô hình có tốc độ detector cao nhất là **{fastest['model']}** ({fastest['fps']:.1f} FPS).
Kết luận này chỉ áp dụng cho dataset, độ phân giải và phần cứng của thí nghiệm. Chưa thể suy rộng
sang toàn bộ hệ thống vượt đèn đỏ nếu chưa đánh giá ở mức sự kiện.

## Hình và bảng đã tạo

- `dataset_distribution.png`: phân bố ảnh và lớp.
- `training_curves.png`: loss và validation mAP theo epoch.
- `accuracy_speed_complexity.png`: độ chính xác, tốc độ và số tham số.
- `per_class_ap.png`: AP của từng lớp phương tiện.
- `pr_*.png` và `confusion_*.png`: PR curve và confusion matrix của từng model.
- `qualitative_comparison.png`: ground truth và dự đoán của ba model.

## Hạn chế

Thí nghiệm dùng một seed nên chưa có khoảng tin cậy. Tập con COCO phù hợp để kiểm tra quy trình,
nhưng nghiên cứu về giao thông Việt Nam cần thay bằng dữ liệu camera thực tế và chia theo video/camera.
Các chỉ số bounding box chưa đo độ ổn định ID tracking, nhận diện trạng thái đèn, định vị vạch dừng
hoặc Precision/Recall của sự kiện vi phạm hoàn chỉnh.
"""
    (artifact_dir/"EXPERIMENTAL_RESULTS.md").write_text(markdown, encoding="utf-8")
    per_class.to_csv(artifact_dir/"per_class_ap.csv")
    return artifact_dir, table
