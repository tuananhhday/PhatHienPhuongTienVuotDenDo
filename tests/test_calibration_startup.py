"""Calibration must remain usable when Windows blocks the inference runtime."""
import builtins
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import main
import calibrate_scene


ROOT = Path(__file__).resolve().parents[1]


def block_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise OSError("[WinError 4551] An Application Control policy has blocked this file")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_help_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", "import sys, runpy; "
         "sys.modules['torch'] = None; "
         "sys.argv = ['main.py', '--help']; "
         "runpy.run_path('main.py', run_name='__main__')"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "--calibrate" in result.stdout


@pytest.mark.parametrize("saved", [False, True])
def test_calibration_runs_before_blocked_inference(monkeypatch, tmp_path, saved):
    block_torch(monkeypatch)
    calls = []

    class Capture:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((100, 100, 3), dtype=np.uint8)

        def set(self, *args):
            pass

        def release(self):
            self.released = True

    cap = Capture()
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda source: cap)

    def setup(source, target):
        calls.append(source)
        if saved:
            return {"gates": [{"id": "gate_1"}]}
        return None

    monkeypatch.setattr(calibrate_scene, "calibrate", setup)
    arguments = dict(video_source="red4.mp4", calibrate=True,
                     scene_config=str(tmp_path / "scene.json"))
    if saved:
        with pytest.raises(main.InferenceDependencyError, match="Windows App Control"):
            main.run_pipeline(**arguments)
    else:
        main.run_pipeline(**arguments)
    assert calls == ["red4.mp4"]
    assert cap.released


def test_unrelated_library_error_is_preserved(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch":
            raise OSError("a different DLL problem")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(OSError, match="different DLL"):
        main.resolve_inference_device()
