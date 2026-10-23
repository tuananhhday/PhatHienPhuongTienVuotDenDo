import json
import numpy as np
import pytest
from src.crossing_tracker import CrossingTracker
from src.violation_engine import ViolationEngine
from src.scene_config import video_identity, validate_scene


GATE = {'id': 'one', 'name': 'one', 'line': [(0,50),(100,50)], 'approach_point': (50,20)}


def test_numpy_points_and_light_loss_after_crossing():
    tracker = CrossingTracker(min_frames=3)
    assert tracker.update(7, np.array([40,40]), GATE, 'RED', 1) is None
    assert tracker.update(7, np.array([40,52]), GATE, 'RED', 2) is None
    assert tracker.update(7, np.array([40,55]), GATE, 'UNKNOWN', 3) is None
    event = tracker.update(7, np.array([40,58]), GATE, 'GREEN', 4)
    assert event['crossing_frame'] == 2
    assert event['light_at_crossing'] == 'RED'
    assert event['confirmed_frame'] == 4


def test_short_detection_gap_and_no_permanent_ignore_after_green():
    tracker = CrossingTracker(min_frames=1, max_gap=5)
    for frame, y, light in [(1,40,'GREEN'),(2,60,'GREEN'),(3,40,'RED')]:
        assert tracker.update(1,(40,y),GATE,light,frame) is None
    assert tracker.update(1,(40,60),GATE,'RED',6)['track_id'] == 1


def test_long_track_gap_is_not_interpolated():
    tracker = CrossingTracker(min_frames=1, max_gap=3)
    tracker.update(1,(40,40),GATE,'RED',1)
    assert tracker.update(1,(40,60),GATE,'RED',20) is None


def test_vertical_line_and_reversed_endpoints():
    for endpoints in [[(50,0),(50,100)],[(50,100),(50,0)]]:
        gate = dict(GATE, line=endpoints, approach_point=(20,50))
        tracker = CrossingTracker(min_frames=1)
        tracker.update(1,(40,40),gate,'RED',1)
        assert tracker.update(1,(60,40),gate,'RED',2) is not None


def test_crossing_line_extension_is_ignored():
    tracker = CrossingTracker(min_frames=1)
    tracker.update(1,(150,40),GATE,'RED',1)
    assert tracker.update(1,(150,60),GATE,'RED',2) is None


def test_unknown_crossing_is_reviewable_and_never_promoted_after_red():
    tracker = CrossingTracker(min_frames=1)
    tracker.update(1,(40,40),GATE,'UNKNOWN',1)
    assert tracker.update(1,(40,60),GATE,'UNKNOWN',2) is None
    assert tracker.update(1,(40,65),GATE,'RED',3) is None
    assert tracker.crossings[0]['status'] == 'review_unknown_light'


def test_multi_gate_colors_and_no_cross_gate_poisoning(tmp_path):
    gate2 = dict(GATE, id='two', line=[(0,70),(100,70)])
    engine = ViolationEngine([GATE,gate2],tmp_path,min_violation_frames=1)
    frame = np.zeros((100,100,3),np.uint8)
    for index, y in enumerate([40,60,80],1):
        _, events = engine.process_tracks(frame,[{'id':1,'box':[30,y-10,50,y]}],
                                          {'one':'GREEN','two':'RED'},index)
    assert len(events) == 1
    assert events[0]['gate_id'] == 'two'


def test_profiles_are_bound_to_video_content(tmp_path):
    a, b = tmp_path/'a.mp4', tmp_path/'b.mp4'
    a.write_bytes(b'first-video'); b.write_bytes(b'other-video')
    gate = dict(GATE, signal_boxes=[[5,5,15,30]], movement='straight',
                lane_polygon=[[0,0],[99,0],[99,99],[0,99]], line=[[0,50],[99,50]])
    scene = {'schema_version':1,'video_identity':video_identity(a),'frame_size':[100,100],'gates':[gate]}
    assert validate_scene(scene,a,(100,100,3)) is scene
    with pytest.raises(ValueError, match='video'):
        validate_scene(scene,b,(100,100,3))
