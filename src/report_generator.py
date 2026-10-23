"""Export candidate red-light events for manual review."""

import csv
import json
from pathlib import Path


FIELDS = ("violation_number", "event_id", "track_id", "vehicle_type", "confidence", "frame_index",
          "timestamp", "stop_line_violated", "scene_image", "vehicle_crop_image",
          "crossing_frame", "confirmed_frame", "video_time_seconds", "light_at_crossing", "gate_id", "status")


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_reports(self, events):
        (self.output_dir / "events.json").write_text(
            json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

        with (self.output_dir / "events.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(events)

        lines = ["# Sự kiện nghi vượt đèn đỏ", "",
                 f"Số sự kiện cần kiểm tra: {len(events)}", "",
                 "| STT | Mã sự kiện | Track ID | Xe | Frame | Thời gian | Vạch |",
                 "|---:|---|---:|---|---:|---|---|"]
        for index, event in enumerate(events, 1):
            lines.append(f"| {event.get('violation_number', index)} | {event['event_id']} | "
                         f"{event['track_id']} | {event['vehicle_type']} | "
                         f"{event['frame_index']} | {event['timestamp']} | "
                         f"{event['stop_line_violated']} |")
        lines.extend(["", "Các sự kiện do thuật toán đề xuất; cần xem ảnh và video để xác nhận."])
        (self.output_dir / "events.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
