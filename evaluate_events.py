"""Match one video's red-light events to reviewed ground truth by gate and time."""
import argparse
import json
import math
from pathlib import Path


def evaluate(predicted, expected, tolerance=0.5):
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError('Tolerance must be finite and nonnegative')
    for event in [*predicted, *expected]:
        if not event.get('gate_id') or not math.isfinite(event['video_time_seconds']) or event['video_time_seconds'] < 0:
            raise ValueError('Every event needs a gate_id and nonnegative video_time_seconds')
    # Maximum-cardinality bipartite matching: one prediction can explain one event.
    neighbours = []
    for pred in predicted:
        candidates = [index for index, truth in enumerate(expected)
                      if pred['gate_id'] == truth['gate_id'] and
                      abs(pred['video_time_seconds'] - truth['video_time_seconds']) <= tolerance]
        candidates.sort(key=lambda index: abs(pred['video_time_seconds'] - expected[index]['video_time_seconds']))
        neighbours.append(candidates)
    matches = {}

    def augment(pred_index, seen):
        for truth_index in neighbours[pred_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            if truth_index not in matches or augment(matches[truth_index], seen):
                matches[truth_index] = pred_index
                return True
        return False

    for index in range(len(predicted)):
        augment(index, set())
    tp = len(matches)
    fp, fn = len(predicted)-tp, len(expected)-tp
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    return {'tp':tp, 'fp':fp, 'fn':fn, 'precision':precision, 'recall':recall,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'tolerance_seconds':tolerance,
            'false_positive_events':[p for i,p in enumerate(predicted) if i not in matches.values()],
            'missed_events':[e for i,e in enumerate(expected) if i not in matches]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='violations/run_... directory')
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=0.5)
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text(encoding='utf-8'))
    scene = json.loads((args.run/'scene.json').read_text(encoding='utf-8'))
    metadata = json.loads((args.run/'run_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('completed') is not True:
        raise ValueError('Run chưa xử lý hết video; không so với ground truth của toàn bộ video')
    if truth.get('reviewed') is not True:
        raise ValueError('Ground truth must be manually reviewed, including every missed event')
    if truth.get('video_identity') != scene['video_identity']:
        raise ValueError('Ground truth and predictions belong to different videos')
    predicted = json.loads((args.run/'events.json').read_text(encoding='utf-8'))
    result = evaluate(predicted, truth['events'], args.tolerance)
    (args.run/'event_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
