from evaluate_events import evaluate


def event(time, gate='one'):
    return {'gate_id':gate, 'video_time_seconds':time}


def test_duplicates_count_as_false_positives_and_wrong_gate_does_not_match():
    result = evaluate([event(1),event(1.1),event(3,'other')],[event(1),event(3)])
    assert (result['tp'],result['fp'],result['fn']) == (1,2,1)
    assert result['recall'] == .5


def test_matching_does_not_greedily_lose_a_valid_match():
    result = evaluate([event(1.1),event(.7)],[event(1),event(1.5)], tolerance=.5)
    assert result['tp'] == 2


def test_empty_ground_truth_does_not_invent_perfect_accuracy():
    result = evaluate([],[])
    assert result['precision'] is None and result['recall'] is None
