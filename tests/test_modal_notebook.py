"""Validate the portable notebook and its persistence/completion guards without GPU."""
import json
from pathlib import Path

import nbformat
import pytest
from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return nbformat.read(ROOT / "notebooks/train-traffic-3-models-modal-h100.ipynb", as_version=4)


def test_modal_cells_compile_and_embed_current_sources():
    nb = notebook()
    nbformat.validate(nb)
    shell = InteractiveShell.instance()
    sources = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    for index, source in enumerate(sources):
        compile(shell.input_transformer_manager.transform_cell(source), f"modal-cell-{index}", "exec")
    for filename in ("traffic_training.py", "coco_subset.py"):
        assert (ROOT / "training" / filename).read_text(encoding="utf-8") in sources
    assert "google.colab" not in "\n".join(sources)
    assert "/content/" not in "\n".join(sources)


def test_completion_requires_full_epochs_and_checkpoint(tmp_path):
    source = notebook().cells[8].source
    function = source[source.index("def completed("):]
    scope = {"OUTPUT_ROOT": tmp_path, "EPOCHS": 3, "json": json}
    exec(function, scope)
    completed = scope["completed"]
    for model, best in [("ssd300_vgg16", "best.pth"), ("yolo11x", "weights/best.pt")]:
        run = tmp_path / model
        (run / best).parent.mkdir(parents=True)
        assert not completed(model)
        meta = run / "training_meta.json"
        meta.write_text(json.dumps({"epochs_completed": 3}))
        assert not completed(model)
        (run / best).write_bytes(b"test")
        assert completed(model)
        meta.write_text(json.dumps({"epochs_completed": 2}))
        assert not completed(model)


def test_pilot_skips_test_and_paper():
    nb = notebook()
    # Deliberately provide no evaluator/model globals: pilot must never touch test.
    for index in (18, 20):
        exec(nb.cells[index].source, {"PILOT": True})


def test_config_requires_existing_volume_and_separates_runs(tmp_path, monkeypatch):
    mount = tmp_path / "attached-volume"
    source = notebook().cells[3].source.replace(
        'Path("/mnt/traffic-training")', f"Path({str(mount)!r})")
    with pytest.raises(FileNotFoundError, match="Volume"):
        exec(source, {})
    assert not mount.exists()
    mount.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TORCH_HOME", "test-placeholder")
    pilot, full = {}, {}
    exec(source.replace("PILOT = False", "PILOT = True"), pilot)
    exec(source, full)
    assert pilot["EPOCHS"] == 3 and full["EPOCHS"] == 80
    assert pilot["OUTPUT_ROOT"] != full["OUTPUT_ROOT"]
    assert pilot["DATA_ROOT"] == full["DATA_ROOT"]
    assert pilot["OUTPUT_ROOT"].is_relative_to(mount)


def test_export_isolated_to_current_run(tmp_path):
    import zipfile
    output = tmp_path / "runs/experiment_full"
    output.mkdir(parents=True)
    (output / "test_benchmark.csv").write_text("measured")
    pilot = tmp_path / "runs/experiment_pilot"
    pilot.mkdir()
    (pilot / "pilot.txt").write_text("pilot")
    scope = {"VOLUME_PATH": tmp_path, "OUTPUT_ROOT": output, "PILOT": False}
    for _ in range(2):
        exec(notebook().cells[22].source, scope)
        with zipfile.ZipFile(scope["archive_path"]) as archive:
            assert archive.namelist() == ["test_benchmark.csv"]
