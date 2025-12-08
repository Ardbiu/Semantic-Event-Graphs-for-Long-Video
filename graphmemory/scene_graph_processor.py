import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class InteractionState:
    a_id: int
    b_id: int
    a_label: str
    b_label: str
    last_seen_ts: float
    missing_frames: int = 0
    start_ts: float = 0.0
    start_frame: int = 0


class SceneGraphProcessor:
    """
    Tracks pairwise proximity interactions and only emits state changes.
    """

    def __init__(self, distance_threshold: float = 100.0, end_buffer: int = 5, focus_class: str = "person"):
        self.distance_threshold = distance_threshold
        self.end_buffer = end_buffer
        self.focus_class = focus_class.lower() if focus_class else None
        self.active: Dict[Tuple[int, int], InteractionState] = {}
        self.event_log: List[dict] = []

    @staticmethod
    def _centroid(bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _format_ts(seconds: Optional[float]) -> str:
        if seconds is None:
            return "00:00"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _order_pair(self, det_a, det_b):
        # Prefer person first for readability.
        if det_a["label"].lower() == "person":
            return det_a, det_b
        if det_b["label"].lower() == "person":
            return det_b, det_a
        return det_a, det_b

    def _append_event(self, timestamp: float, frame: int, event_type: str, subj_label: str, subj_id: int, obj_label: str, obj_id: int):
        self.event_log.append(
            {
                "timestamp": float(timestamp),
                "frame": int(frame),
                "type": event_type,
                "subject": f"{subj_label}-{subj_id}",
                "object": f"{obj_label}-{obj_id}",
            }
        )

    def update(self, detections: List[dict], timestamp: Optional[float], frame_index: int):
        """
        detections: list of {"id": int, "label": str, "bbox": (x1,y1,x2,y2)}
        Returns a list of textual events representing state changes.
        """
        events = []
        present_pairs = set()

        # Build centroids.
        processed = []
        for det in detections:
            if det.get("id") is None:
                continue
            cx, cy = self._centroid(det["bbox"])
            processed.append({**det, "centroid": (cx, cy)})

        # Detect interactions in this frame.
        for i in range(len(processed)):
            for j in range(i + 1, len(processed)):
                det_a, det_b = processed[i], processed[j]
                if self.focus_class:
                    if det_a["label"].lower() != self.focus_class and det_b["label"].lower() != self.focus_class:
                        continue
                dist = math.hypot(det_a["centroid"][0] - det_b["centroid"][0], det_a["centroid"][1] - det_b["centroid"][1])
                if dist > self.distance_threshold:
                    continue

                # Order for consistent keys.
                ordered_a, ordered_b = self._order_pair(det_a, det_b)
                key = (ordered_a["id"], ordered_b["id"])
                present_pairs.add(key)

                if key not in self.active:
                    start_ts = timestamp if timestamp is not None else 0.0
                    self.active[key] = InteractionState(
                        a_id=ordered_a["id"],
                        b_id=ordered_b["id"],
                        a_label=ordered_a["label"],
                        b_label=ordered_b["label"],
                        last_seen_ts=start_ts,
                        missing_frames=0,
                        start_ts=start_ts,
                        start_frame=frame_index,
                    )
                    ts = self._format_ts(timestamp or 0.0)
                    self._append_event(start_ts, frame_index, "START", ordered_a["label"], ordered_a["id"], ordered_b["label"], ordered_b["id"])
                    events.append(f"[TIMESTAMP {ts}] START: {ordered_a['label']}-{ordered_a['id']} interacting with {ordered_b['label']}-{ordered_b['id']}.")
                else:
                    state = self.active[key]
                    state.last_seen_ts = timestamp if timestamp is not None else state.last_seen_ts
                    state.missing_frames = 0

        # Handle interactions that may have ended.
        for key, state in list(self.active.items()):
            if key not in present_pairs:
                state.missing_frames += 1
                if state.missing_frames > self.end_buffer:
                    ts = self._format_ts(timestamp or state.last_seen_ts)
                    end_ts = timestamp if timestamp is not None else state.last_seen_ts
                    self._append_event(end_ts, frame_index, "END", state.a_label, state.a_id, state.b_label, state.b_id)
                    events.append(f"[TIMESTAMP {ts}] END: {state.a_label}-{state.a_id} disengaged from {state.b_label}-{state.b_id}.")
                    del self.active[key]

        return events

    def save_events_to_json(self, filename: Union[str, Path]):
        path = Path(filename)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.event_log, f, indent=2)

    def generate_timeline_plot(self, filename: Union[str, Path]):
        """
        Create a simple Gantt-style chart of interactions using START/END events.
        """
        # Lazy import to avoid heavy matplotlib initialization during main loop.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Build segments from event log.
        open_interactions: Dict[Tuple[str, str], dict] = {}
        segments: List[Tuple[str, float, float]] = []
        for evt in self.event_log:
            key = (evt["subject"], evt["object"])
            if evt["type"] == "START":
                open_interactions[key] = {"start": evt["timestamp"]}
            elif evt["type"] == "END" and key in open_interactions:
                start_time = open_interactions[key]["start"]
                end_time = evt["timestamp"]
                if end_time > start_time:
                    segments.append((f"{key[0]} & {key[1]}", start_time, end_time - start_time))
                del open_interactions[key]

        if not segments:
            # Nothing to plot; create an empty figure.
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No interactions recorded", ha="center", va="center")
            ax.axis("off")
            fig.savefig(filename, bbox_inches="tight")
            plt.close(fig)
            return

        fig, ax = plt.subplots(figsize=(8, 4 + len(segments) * 0.2))
        y_ticks = []
        y_labels = []
        for idx, (label, start, duration) in enumerate(segments):
            ax.barh(idx, duration, left=start, height=0.4, align="center")
            y_ticks.append(idx)
            y_labels.append(label)

        ax.set_xlabel("Time (s)")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_title("Interaction Timeline")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)
