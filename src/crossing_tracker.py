"""Directional, finite-segment crossings with evidence captured at crossing time."""
from dataclasses import dataclass
import math


def signed_distance(point, line):
    (ax, ay), (bx, by) = line
    length = math.hypot(bx - ax, by - ay)
    if length < 1:
        raise ValueError("Stop line must have two distinct endpoints")
    return ((point[0] - ax) * (by - ay) - (point[1] - ay) * (bx - ax)) / length


def after_distance(point, gate):
    """Positive on departure side, independent of endpoint order and line orientation."""
    line = gate["line"]
    if "approach_point" in gate:
        approach = signed_distance(gate["approach_point"], line)
        if abs(approach) < 1:
            raise ValueError("Approach point must be away from the stop line")
        return signed_distance(point, line) * (-1 if approach > 0 else 1)
    # Legacy profiles; new calibrated gates always store an approach point.
    orientation = 1 if line[1][0] >= line[0][0] else -1
    direction = 1 if gate.get("direction") == "BOTTOM_TO_TOP" else -1
    return signed_distance(point, line) * orientation * direction


def intersects_segment(a, b, line):
    """Intersect the actual movement segment with the finite stop-line segment."""
    da, db = signed_distance(a, line), signed_distance(b, line)
    if da * db > 0 or abs(da - db) < 1e-9:
        return False
    t = da / (da - db)
    p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    (ax, ay), (bx, by) = line
    projection = ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / (
        (bx - ax) ** 2 + (by - ay) ** 2)
    return 0 <= projection <= 1


@dataclass
class TrackState:
    point: tuple
    frame: int
    light: str
    before_point: tuple | None = None
    pending: dict | None = None
    confirmations: int = 0


class CrossingTracker:
    def __init__(self, min_frames=3, margin=2.0, max_gap=5):
        self.min_frames = max(1, int(min_frames))
        self.margin = max(0.0, float(margin))
        self.max_gap = max(1, int(max_gap))
        self.tracks = {}
        self.crossings = []

    def update(self, track_id, point, gate, light, frame):
        key = (int(track_id), gate["id"])
        state = self.tracks.get(key)
        distance = after_distance(point, gate)
        if state is None or not 0 < frame - state.frame <= self.max_gap:
            self.tracks[key] = TrackState(point, frame, light,
                                         point if distance < -self.margin else None)
            return None

        result = None
        if distance < -self.margin:
            if state.pending is not None:
                state.pending['status'] = 'rejected_reversed_before_confirmation'
            state.before_point = point
            state.pending = None
            state.confirmations = 0
        elif state.pending is not None:
            if distance > 0:
                state.confirmations += 1
            else:
                state.pending['status'] = 'rejected_returned_to_line'
                state.pending = None
                state.confirmations = 0
        elif distance > 0 and state.before_point is not None:
            if intersects_segment(state.before_point, point, gate["line"]):
                # Across a missing-detection gap, require red at both observations.
                known_red = light == "RED" and (frame - state.frame == 1 or state.light == "RED")
                status = "red_candidate" if known_red else (
                    "non_red" if light in ("GREEN", "YELLOW") else "review_unknown_light")
                if light == "RED" and not known_red:
                    status = "review_detection_gap"
                crossing = {"track_id": int(track_id), "gate_id": gate["id"],
                            "crossing_frame": frame, "light_at_crossing": light,
                            "status": status}
                self.crossings.append(crossing)
                if known_red:
                    state.pending = crossing
                    state.confirmations = 1
            state.before_point = None

        if state.pending is not None and state.confirmations >= self.min_frames and distance >= self.margin:
            state.pending['status'] = 'confirmed_red_crossing'
            result = dict(state.pending, confirmed_frame=frame)
            state.pending = None
            state.confirmations = 0
        state.point, state.frame, state.light = point, frame, light
        return result

    def prune(self, frame):
        self.tracks = {key: value for key, value in self.tracks.items()
                       if frame - value.frame <= self.max_gap}
