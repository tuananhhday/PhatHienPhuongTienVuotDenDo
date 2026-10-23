"""Check the crossing rule and the selected local detector checkpoint."""
import json
from pathlib import Path

import numpy as np
import yaml
import cv2
import pytest

from src.violation_engine import ViolationEngine
from src.stopline_detector import StopLineDetector
from src.trajectory_analyzer import TrajectoryAnalyzer
from src.report_generator import ReportGenerator
from src.traffic_light_detector import TrafficLightDetector


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_uses_pretrained_baseline_and_keeps_finetuned_checkpoint():
    vehicle = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["models"]["vehicle_detector"]
    assert vehicle["weights"] == "weights/coco/yolo11x.pt"
    assert (ROOT / 'weights/yolo11x.pt').is_file()
    assert (ROOT / vehicle["weights"]).is_file()
    assert vehicle["track_names"] == ["bicycle", "car", "motorcycle", "bus", "truck"]
    light = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["models"]["traffic_light"]
    assert light["weights"] == "weights/yolov8s.pt"
    assert (ROOT / light["weights"]).is_file()


def test_trajectory_uses_names_from_custom_checkpoint():
    class Model:
        names = {0: "bicycle", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}

    analyzer = TrajectoryAnalyzer(yolo_model=Model())
    analyzer.track_classes = [i for i, name in analyzer.model.names.items()
                              if name in analyzer.track_names]
    assert analyzer.track_classes == [0, 1, 2, 3, 4]


def test_old_segmentation_cache_is_ignored(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"camera": {"cache_dir": str(tmp_path / "cache")}}))
    detector = StopLineDetector(config)
    detector.get_cache_path("camera").write_text(json.dumps({
        "detector_mode": "ai_maskrcnn", "lines": [{"id": "old"}]}))
    assert detector.load_from_cache("camera") is None


def test_red_crossing_requires_correct_direction_and_three_frames(tmp_path, monkeypatch):
    line = {"id": "line", "name": "line", "line": [(0, 50), (99, 50)],
            "direction": "TOP_TO_BOTTOM"}
    engine = ViolationEngine([line], tmp_path, min_violation_frames=3)
    monkeypatch.setattr(engine, "_record_violation", lambda **kwargs: {"frame_index": kwargs["frame_idx"]})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def step(index, y, state="RED"):
        track = {"id": 1, "box": [40, y-10, 60, y], "class_name": "O to", "conf": .9}
        return engine.process_tracks(frame, [track], state, index)[1]

    assert not step(1, 40)
    assert not step(2, 51)
    assert not step(3, 53)
    assert step(4, 55) == [{"frame_index": 4}]
    assert not step(5, 58)


def test_reverse_crossing_and_green_are_not_violations(tmp_path):
    line = {"id": "line", "name": "line", "line": [(0, 50), (99, 50)],
            "direction": "TOP_TO_BOTTOM"}
    engine = ViolationEngine([line], tmp_path, min_violation_frames=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for index, (y, state) in enumerate([(60, "RED"), (40, "RED"), (60, "GREEN"),
                                        (40, "GREEN")], 1):
        track = {"id": 1, "box": [40, y-10, 60, y], "class_name": "O to", "conf": .9}
        assert not engine.process_tracks(frame, [track], state, index)[1]


def test_report_contains_candidate_events_without_fines(tmp_path):
    reporter = ReportGenerator(tmp_path)
    reporter.export_reports([])
    assert json.loads((tmp_path / "events.json").read_text(encoding="utf-8")) == []
    event = {"event_id": "EV-1", "track_id": 1, "vehicle_type": "car", "confidence": .9,
             "frame_index": 5, "timestamp": "2026-10-15 12:00:00",
             "stop_line_violated": "line", "scene_image": "scene.jpg",
             "vehicle_crop_image": "vehicle.jpg"}
    reporter.export_reports([event])
    assert json.loads((tmp_path / "events.json").read_text(encoding="utf-8")) == [event]
    assert "fine" not in (tmp_path / "events.csv").read_text(encoding="utf-8-sig").lower()
    assert "cần kiểm tra" in (tmp_path / "events.md").read_text(encoding="utf-8")


def test_light_becomes_unknown_when_no_longer_visible():
    detector = TrafficLightDetector(max_missing_frames=2)
    detector.current_state = "RED"
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert detector.detect_state(frame, yolo_detections=[])[0] == "UNKNOWN"
    assert detector.detect_state(frame, yolo_detections=[])[0] == "UNKNOWN"


def test_light_color_is_automatic_and_requires_stable_frames():
    detector = TrafficLightDetector(state_confirm_frames=2, max_missing_frames=2)
    detection = [([5, 5, 25, 35], 9, 0.9)]

    red = np.zeros((40, 30, 3), dtype=np.uint8)
    red[5:35, 5:25] = (0, 0, 255)
    assert detector.detect_state(red, detection)[0] == "UNKNOWN"
    assert detector.detect_state(red, detection)[0] == "RED"
    assert detector.detected_lights[0]["state"] == "RED"

    green = np.zeros_like(red)
    green[5:35, 5:25] = (0, 255, 0)
    assert detector.detect_state(green, detection)[0] == "UNKNOWN"
    assert detector.detect_state(green, detection)[0] == "GREEN"

    annotated = detector.draw_detections(green.copy())
    assert not np.array_equal(annotated, green)


def test_conflicting_traffic_lights_do_not_immediately_turn_red():
    detector = TrafficLightDetector(state_confirm_frames=1, max_missing_frames=2)
    detector.current_state = "GREEN"
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[5:35, 5:25] = (0, 0, 255)
    frame[5:35, 35:55] = (0, 255, 0)
    detections = [([5, 5, 25, 35], 9, 0.9), ([35, 5, 55, 35], 9, 0.9)]

    assert detector.detect_state(frame, detections)[0] == "GREEN"
    assert detector.detect_state(frame, detections)[0] == "UNKNOWN"


def test_traffic_light_roi_selects_signal_for_monitored_lane():
    detector = TrafficLightDetector(
        state_confirm_frames=1,
        traffic_roi={"x_min_ratio": 0.5, "x_max_ratio": 1.0,
                     "y_min_ratio": 0.0, "y_max_ratio": 1.0},
    )
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[5:35, 5:25] = (0, 255, 0)
    frame[5:35, 35:55] = (0, 0, 255)
    detections = [([5, 5, 25, 35], 9, 0.9), ([35, 5, 55, 35], 9, 0.9)]

    state, boxes = detector.detect_state(frame, detections)
    assert state == "RED"
    assert boxes == [(35, 5, 55, 35)]


def test_violation_records_id_and_increments_only_once(tmp_path):
    line = {"id": "line", "name": "line", "line": [(0, 50), (99, 50)],
            "direction": "TOP_TO_BOTTOM"}
    engine = ViolationEngine([line], tmp_path, min_violation_frames=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def step(index, y):
        track = {"id": 42, "box": [40, y-10, 60, y], "class_name": "O to", "conf": .9}
        return engine.process_tracks(frame, [track], "RED", index)[1]

    assert not step(1, 40)
    event = step(2, 55)[0]
    assert event["track_id"] == 42
    assert event["violation_number"] == 1
    assert len(engine.violated_ids) == 1
    assert not step(3, 60)
    assert len(engine.violations_log) == 1


def test_three_frame_video_runs_with_real_models(tmp_path, monkeypatch):
    import main

    source = cv2.VideoCapture(str(ROOT / "dataset_videos/red1.mp4"))
    ok, first = source.read()
    if not ok:
        pytest.skip("Sample video unavailable")
    height, width = first.shape[:2]
    sample = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(sample), cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height))
    writer.write(first)
    for _ in range(2):
        ok, frame = source.read()
        if ok:
            writer.write(frame)
    writer.release()
    source.release()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["general"]["output_dir"] = str(tmp_path / "events")
    from src.scene_config import save_scene, video_identity
    profile = tmp_path / 'scene.json'
    scene = {'schema_version': 1, 'video_identity': video_identity(sample),
             'frame_size': [width, height], 'gates': [{
                 'id': 'gate_1', 'name': 'LAN 1', 'line': [[0, height//2], [width-1, height//2]],
                 'approach_point': [width//2, height-1], 'movement': 'straight',
                 'signal_boxes': [[int(width*.936),int(height*.097),int(width*.956),int(height*.151)]],
                 'lane_polygon': [[0,0],[width-1,0],[width-1,height-1],[0,height-1]],
             }]}
    save_scene(scene, profile, sample, first.shape)
    monkeypatch.setattr(main, "load_yaml_config", lambda *args: cfg)
    monkeypatch.chdir(ROOT)
    output_video = tmp_path / "output.mp4"
    main.run_pipeline(str(sample), camera_id="cam_01", headless=True, output_video=str(output_video),
                      scene_config=str(profile))
    assert output_video.is_file() and output_video.stat().st_size > 0
    reports = list((tmp_path / "events").rglob("events.json"))
    assert len(reports) == 1
    assert isinstance(json.loads(reports[0].read_text(encoding="utf-8")), list)
    analyzer = TrajectoryAnalyzer(model_weights="weights/yolo11x.pt")
    motion = analyzer.extract_motion_features(str(sample), sample_frames=3)
    assert analyzer.track_classes == [0, 1, 2, 3, 4]
    assert motion["frame_shape"] == (height, width)
