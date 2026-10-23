import json
import zipfile
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from training.traffic_training import TrafficDataset, build_ssd, prepare_detection_data, read_labels


def dataset(tmp_path):
    for index, split in enumerate(("train", "val", "test")):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (100, 80), (40*index, 100, 20)).save(tmp_path / "images" / split / "one.png")
        (tmp_path / "labels" / split / "one.txt").write_text("0 0.5 0.5 0.4 0.5")
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"path": "/obsolete/export", "train": "../train/images",
                                  "val": "../valid/images", "test": "../test/images", "names": ["car"]}))
    return path


def test_relocated_export_and_ssd_label_mapping(tmp_path):
    path = prepare_detection_data(dataset(tmp_path), tmp_path / "normalized.yaml")
    image, target = TrafficDataset(path)[0]
    assert image.shape == (3, 80, 100)
    assert target["labels"].tolist() == [1]
    torch.testing.assert_close(target["boxes"], torch.tensor([[30., 20., 70., 60.]]))


def test_exact_split_leakage_is_rejected(tmp_path):
    source = dataset(tmp_path)
    (tmp_path / "images/test/one.png").write_bytes((tmp_path / "images/train/one.png").read_bytes())
    with pytest.raises(ValueError, match="leakage"):
        prepare_detection_data(source, tmp_path / "normalized.yaml")


@pytest.mark.parametrize("text", ["0 .5 .5 .2 .2 .4 .4", "0 nan .5 .2 .2", "1 .5 .5 .2 .2",
                                       "0 .5 .5 0 .2", "0 .99 .5 .5 .2"])
def test_invalid_labels_are_rejected(tmp_path, text):
    label = tmp_path / "bad.txt"
    label.write_text(text)
    with pytest.raises(ValueError):
        read_labels(label, 1)


def test_negative_and_missing_labels(tmp_path):
    label = tmp_path / "negative.txt"
    label.write_text("")
    assert read_labels(label, 1) == []
    with pytest.raises(FileNotFoundError):
        read_labels(tmp_path / "missing.txt", 1)


def test_ssd_custom_classes_train_and_infer_without_download():
    torch.set_num_threads(2)
    model = build_ssd(2, pretrained=False)
    image = torch.rand(3, 100, 140)
    model.train()
    losses = model([image], [{"boxes": torch.tensor([[10., 10., 80., 70.]]),
                              "labels": torch.tensor([2])}])
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.classification_head.module_list[0].weight.grad is not None
    model.eval()
    with torch.no_grad():
        prediction = model([image])[0]
    assert prediction["boxes"].shape[1] == 4
    assert set(prediction["labels"].tolist()) <= {1, 2}


def test_common_evaluator_perfect_and_wrong_predictions(tmp_path):
    pytest.importorskip("torchmetrics")
    from training.traffic_training import detection_diagnostics, evaluate_detector
    normalized = prepare_detection_data(dataset(tmp_path), tmp_path / "normalized.yaml")
    samples = TrafficDataset(normalized, "test")

    class Detector:
        def eval(self):
            return self

        def __call__(self, images):
            return [{"boxes": torch.tensor([[30., 20., 70., 60.]]), "labels": torch.tensor([1]),
                     "scores": torch.tensor([.99])}]

    metrics = evaluate_detector(Detector(), samples, "cpu")
    assert metrics["mAP50_95"] == pytest.approx(1)
    assert metrics["per_class_AP"]["car"] == pytest.approx(1)
    diagnostics = detection_diagnostics(Detector(), samples, "cpu", conf_threshold=.25)
    assert diagnostics["macro"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert diagnostics["confusion_matrix"] == [[1, 0], [0, 0]]
    assert diagnostics["pr_curves"]["car"]["recall"][-1] == pytest.approx(1)

    class WrongDetector(Detector):
        def __call__(self, images):
            return [{"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([1]),
                     "scores": torch.tensor([.99])}]

    assert evaluate_detector(WrongDetector(), samples, "cpu")["mAP50_95"] == pytest.approx(0)


def test_ssd_one_epoch_checkpoint_and_resume(tmp_path, monkeypatch):
    pytest.importorskip("torchmetrics")
    from training import traffic_training as training
    torch.set_num_threads(2)
    normalized = prepare_detection_data(dataset(tmp_path), tmp_path / "normalized.yaml")
    original_builder = training.build_ssd
    monkeypatch.setattr(training, "build_ssd", lambda num_classes, pretrained=True:
                        original_builder(num_classes, pretrained=False))
    run = tmp_path / "run"
    best = training.train_ssd(normalized, run, epochs=1, batch=1, workers=0)
    assert best.is_file()
    last = torch.load(run / "last.pth", weights_only=False, map_location="cpu")
    assert last["epoch"] == 1 and last["names"] == ["car"]
    assert last["optimizer"]["state"]
    assert len(last["history"]) == 1
    metadata = json.loads((run / "training_meta.json").read_text())
    assert metadata["epochs_completed"] == 1 and metadata["training_seconds"] > 0
    # A completed SSD run resumes without retraining or changing history.
    training.train_ssd(normalized, run, epochs=1, batch=1, workers=0, resume=True)
    assert len(json.loads((run / "history.json").read_text())) == 1
    with pytest.raises(FileExistsError):
        training.train_ssd(normalized, run, epochs=1, batch=1, workers=0)


def test_coco_subset_uses_disjoint_sources_and_maps_classes(tmp_path, monkeypatch):
    from training import coco_subset
    images, annotations = [], []
    for index in range(20):
        images.append({"id": index, "file_name": f"{index}.png", "width": 100, "height": 80, "license": 1})
        for category in coco_subset.COCO_VEHICLES:
            annotations.append({"image_id": index, "category_id": category, "bbox": [20, 20, 40, 30]})
    archive = tmp_path / "annotations_trainval2017.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for official in ("train2017", "val2017"):
            zipped.writestr(f"annotations/instances_{official}.json", json.dumps(
                {"images": images, "annotations": annotations, "licenses": [{"id": 1, "name": "test"}]}))

    def fake_download(url, destination):
        path = Path(destination)
        if path.suffix == ".zip":
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80)).save(path)

    monkeypatch.setattr(coco_subset, "download_file", fake_download)
    result = coco_subset.download_coco_vehicles(tmp_path, 5, 5, 5)
    config = yaml.safe_load(result.read_text())
    assert config["names"] == ["bicycle", "car", "motorcycle", "bus", "truck"]
    val = {p.name for p in (tmp_path / "images/val").iterdir()}
    test = {p.name for p in (tmp_path / "images/test").iterdir()}
    assert not val & test
    sources = json.loads((tmp_path / "sources_and_licenses.json").read_text())
    assert sources["licenses"][0]["id"] == 1


def test_paper_artifacts_are_generated_from_measured_results(tmp_path, monkeypatch):
    try:
        import pandas as pd
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"Local application policy blocked a plotting dependency: {exc}")
    from training import traffic_training as training
    normalized = prepare_detection_data(dataset(tmp_path), tmp_path / "normalized.yaml")

    class Detector:
        def eval(self): return self
        def __call__(self, images):
            return [{"boxes": torch.tensor([[30., 20., 70., 60.]]), "labels": torch.tensor([1]),
                     "scores": torch.tensor([.99])}]

    samples = TrafficDataset(normalized, "test")
    diagnostics = training.detection_diagnostics(Detector(), samples, "cpu")
    rows = []
    for index, name in enumerate(training.MODEL_NAMES):
        rows.append({"model": name, "input_size": 300 if index == 0 else 640,
                     "checkpoint": "fixture", "parameters_m": 10+index,
                     "checkpoint_mb": 20+index, "training_minutes": 1+index,
                     "peak_vram_mb": 100+index, "mAP50_95": .5+index*.1, "mAP50": .7,
                     "mAR100": .6, "images": 1, "per_class_AP": {"car": .5+index*.1},
                     "latency_ms_mean": 10, "latency_ms_p95": 11, "fps": 100-index*10,
                     "timed_images": 1, "diagnostics": diagnostics})
        run = tmp_path/name
        run.mkdir()
        if name == "ssd300_vgg16":
            (run/"history.json").write_text(json.dumps(
                [{"epoch": 1, "loss": 1., "mAP50_95": .5}]))
        else:
            (run/"results.csv").write_text("epoch,train/box_loss,metrics/mAP50-95(B)\n1,1.0,0.5\n")
    report = {"device": "test-cpu", "results": rows}
    (tmp_path/"test_benchmark.json").write_text(json.dumps(report))
    pd.DataFrame([{k: v for k, v in row.items() if k not in ("per_class_AP", "diagnostics")} |
                  {"precision": 1., "recall": 1., "f1": 1.} for row in rows]).to_csv(
                      tmp_path/"test_benchmark.csv", index=False)
    monkeypatch.setattr(training, "_load_best_model",
                        lambda name, output_root, names, device: (Detector(), "ssd", Path("fixture")))
    artifact_dir, table = training.create_paper_artifacts(normalized, tmp_path, qualitative_images=1)
    assert len(table) == 3
    expected = ["EXPERIMENTAL_RESULTS.md", "dataset_distribution.png", "training_curves.png",
                "accuracy_speed_complexity.png", "per_class_ap.png", "per_class_metrics.csv",
                "qualitative_comparison.png"]
    assert all((artifact_dir/name).is_file() for name in expected)
    text = (artifact_dir/"EXPERIMENTAL_RESULTS.md").read_text(encoding="utf-8")
    assert "yolo11x" in text and "0.700" in text and "test-cpu" in text


