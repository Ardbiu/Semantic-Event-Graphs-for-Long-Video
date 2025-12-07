"""
Utilities for building a lightweight graph memory from YOLO-style detections.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BBox = Tuple[float, float, float, float]


@dataclass
class Track:
    """Simple trajectory container for a single object."""

    id: int
    cls: str
    frames: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    bboxes: List[BBox] = field(default_factory=list)

    def add_observation(self, frame_index: int, time_s: float, bbox: BBox) -> None:
        """Append an observation for this track."""
        self.frames.append(frame_index)
        self.times.append(time_s)
        self.bboxes.append(bbox)


@dataclass
class Event:
    """Represents a simple temporal event derived from tracks."""

    id: int
    time_s: float
    frame_index: int
    description: str
    object_ids: List[int] = field(default_factory=list)


@dataclass
class GraphMemory:
    """Tracks and events captured for a video."""

    tracks: Dict[int, Track] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert GraphMemory to a JSON-serializable dict."""
        tracks_dict = {
            str(track_id): {
                "id": track.id,
                "cls": track.cls,
                "frames": track.frames,
                "times": track.times,
                "bboxes": [list(bbox) for bbox in track.bboxes],
            }
            for track_id, track in self.tracks.items()
        }
        events_list = [
            {
                "id": event.id,
                "time_s": event.time_s,
                "frame_index": event.frame_index,
                "description": event.description,
                "object_ids": event.object_ids,
            }
            for event in self.events
        ]
        return {"tracks": tracks_dict, "events": events_list}

    @classmethod
    def from_dict(cls, data: dict) -> "GraphMemory":
        """Reconstruct a GraphMemory object from a dictionary."""
        tracks_data = data.get("tracks", {})
        tracks: Dict[int, Track] = {}
        for key, track_info in tracks_data.items():
            track = Track(
                id=int(track_info["id"]),
                cls=track_info.get("cls", ""),
                frames=list(track_info.get("frames", [])),
                times=list(track_info.get("times", [])),
                bboxes=[tuple(bbox) for bbox in track_info.get("bboxes", [])],
            )
            tracks[int(key)] = track

        events = [
            Event(
                id=int(event_info.get("id", idx)),
                time_s=float(event_info.get("time_s", 0.0)),
                frame_index=int(event_info.get("frame_index", 0)),
                description=event_info.get("description", ""),
                object_ids=list(event_info.get("object_ids", [])),
            )
            for idx, event_info in enumerate(data.get("events", []))
        ]
        return cls(tracks=tracks, events=events)


def compute_iou(b1: BBox, b2: BBox) -> float:
    """Compute intersection over union for two bounding boxes."""
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_b1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area_b2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    union = area_b1 + area_b2 - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


class SimpleTracker:
    """Greedy IoU-based tracker that links detections across frames."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_track_id = 0
        self.tracks: Dict[int, Track] = {}
        self.active_tracks: Dict[int, int] = {}  # track_id -> last seen frame index
        self.missed_counts: Dict[int, int] = {}  # track_id -> missed frames

    def update(self, frame_index: int, time_s: float, detections: List[dict]) -> None:
        """
        Update tracker with detections from a single frame.

        Args:
            frame_index: Index of the current frame.
            time_s: Timestamp in seconds.
            detections: Iterable of detection dicts containing class, bbox, score.
        """
        detections = detections or []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        candidate_track_ids = [
            track_id
            for track_id, last_seen in self.active_tracks.items()
            if frame_index - last_seen <= self.max_missed
        ]
        pairs: List[Tuple[float, int, int]] = []

        for track_id in candidate_track_ids:
            track = self.tracks[track_id]
            if not track.bboxes:
                continue
            track_bbox = track.bboxes[-1]
            for det_idx, det in enumerate(detections):
                det_cls = det.get("class")
                bbox = det.get("bbox")
                if det_cls != track.cls or bbox is None:
                    continue
                iou = compute_iou(track_bbox, bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, track_id, det_idx))

        # Greedy matching on descending IoU.
        for _, track_id, det_idx in sorted(pairs, reverse=True):
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            detection = detections[det_idx]
            bbox = detection.get("bbox")
            if bbox is None:
                continue
            self.tracks[track_id].add_observation(frame_index, time_s, bbox)
            self.active_tracks[track_id] = frame_index
            self.missed_counts[track_id] = 0
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)

        # Create new tracks for unmatched detections.
        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detections:
                continue
            bbox = detection.get("bbox")
            det_cls = detection.get("class")
            if bbox is None or det_cls is None:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            track = Track(id=track_id, cls=det_cls)
            track.add_observation(frame_index, time_s, bbox)
            self.tracks[track_id] = track
            self.active_tracks[track_id] = frame_index
            self.missed_counts[track_id] = 0

        # Update missed counters for unmatched tracks.
        for track_id in list(self.active_tracks.keys()):
            if track_id in matched_tracks:
                continue
            self.missed_counts[track_id] = self.missed_counts.get(track_id, 0) + 1
            if self.missed_counts[track_id] > self.max_missed:
                # Remove from active tracking but keep the data.
                self.active_tracks.pop(track_id, None)

    def finalize(self) -> Dict[int, Track]:
        """Return all tracks recorded by the tracker."""
        return self.tracks


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    """Return the (x, y) center of a bounding box."""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def track_pair_min_dist_over_time(track_a: Track, track_b: Track) -> List[Tuple[int, float, float]]:
    """
    Compute distances for frames where both tracks exist.

    Returns:
        List of tuples (frame_index, time_s, distance) sorted by frame index.
    """
    a_points = {
        frame: (time, bbox)
        for frame, time, bbox in zip(track_a.frames, track_a.times, track_a.bboxes)
    }
    b_points = {
        frame: (time, bbox)
        for frame, time, bbox in zip(track_b.frames, track_b.times, track_b.bboxes)
    }
    common_frames = sorted(set(a_points.keys()) & set(b_points.keys()))
    distances: List[Tuple[int, float, float]] = []
    for frame in common_frames:
        time_a, bbox_a = a_points[frame]
        time_b, bbox_b = b_points[frame]
        cx_a, cy_a = bbox_center(bbox_a)
        cx_b, cy_b = bbox_center(bbox_b)
        dist = math.dist((cx_a, cy_a), (cx_b, cy_b))
        distances.append((frame, max(time_a, time_b), dist))
    return distances


def extract_events_from_tracks(tracks: Dict[int, Track]) -> List[Event]:
    """
    Generate simple appearance/disappearance and interaction events from tracks.
    """
    events: List[Event] = []
    next_event_id = 0
    track_list = [track for track in tracks.values() if track.frames]
    track_list.sort(key=lambda t: t.id)

    for track in track_list:
        # Appearance
        events.append(
            Event(
                id=next_event_id,
                time_s=track.times[0],
                frame_index=track.frames[0],
                description=f"{track.cls} {track.id} appears",
                object_ids=[track.id],
            )
        )
        next_event_id += 1
        # Disappearance
        events.append(
            Event(
                id=next_event_id,
                time_s=track.times[-1],
                frame_index=track.frames[-1],
                description=f"{track.cls} {track.id} disappears",
                object_ids=[track.id],
            )
        )
        next_event_id += 1

    # Interaction events
    dist_threshold = 0.2
    for i, track_a in enumerate(track_list):
        for track_b in track_list[i + 1 :]:
            if track_a.cls != "person" and track_b.cls != "person":
                continue
            distances = track_pair_min_dist_over_time(track_a, track_b)
            if len(distances) < 2:
                continue
            prev_state = "near" if distances[0][2] <= dist_threshold else "far"
            for frame_index, time_s, dist in distances[1:]:
                state = "near" if dist <= dist_threshold else "far"
                if prev_state == "far" and state == "near":
                    description = f"{track_a.cls} {track_a.id} approaches {track_b.cls} {track_b.id}"
                    events.append(
                        Event(
                            id=next_event_id,
                            time_s=time_s,
                            frame_index=frame_index,
                            description=description,
                            object_ids=[track_a.id, track_b.id],
                        )
                    )
                    next_event_id += 1
                elif prev_state == "near" and state == "far":
                    description = f"{track_a.cls} {track_a.id} moves away from {track_b.cls} {track_b.id}"
                    events.append(
                        Event(
                            id=next_event_id,
                            time_s=time_s,
                            frame_index=frame_index,
                            description=description,
                            object_ids=[track_a.id, track_b.id],
                        )
                    )
                    next_event_id += 1
                prev_state = state
    return events


def _normalize_bbox(raw_bbox: Optional[object], detection: dict) -> Optional[BBox]:
    """Normalize various bbox formats into (x1, y1, x2, y2)."""
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, dict):
        if {"x1", "y1", "x2", "y2"} <= raw_bbox.keys():
            return (
                float(raw_bbox["x1"]),
                float(raw_bbox["y1"]),
                float(raw_bbox["x2"]),
                float(raw_bbox["y2"]),
            )
        if {"left", "top", "right", "bottom"} <= raw_bbox.keys():
            return (
                float(raw_bbox["left"]),
                float(raw_bbox["top"]),
                float(raw_bbox["right"]),
                float(raw_bbox["bottom"]),
            )
        if {"x_center", "y_center", "width", "height"} <= raw_bbox.keys():
            cx = float(raw_bbox["x_center"])
            cy = float(raw_bbox["y_center"])
            w = float(raw_bbox["width"])
            h = float(raw_bbox["height"])
            return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
        if {"cx", "cy", "w", "h"} <= raw_bbox.keys():
            cx = float(raw_bbox["cx"])
            cy = float(raw_bbox["cy"])
            w = float(raw_bbox["w"])
            h = float(raw_bbox["h"])
            return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
    elif isinstance(raw_bbox, (list, tuple)):
        if len(raw_bbox) == 4:
            return tuple(float(v) for v in raw_bbox)  # type: ignore[return-value]
    # Handle YOLO (x, y, w, h) stored separately.
    if {"x", "y", "w", "h"} <= detection.keys():
        x = float(detection["x"])
        y = float(detection["y"])
        w = float(detection["w"])
        h = float(detection["h"])
        return x, y, x + w, y + h
    return None


def _iter_frames_from_data(data: Iterable[dict]) -> Iterable[Tuple[int, float, List[dict]]]:
    """Yield normalized frame info from arbitrary detection data."""
    for frame_entry in data:
        frame_index = frame_entry.get("frame_index") or frame_entry.get("frame")
        if frame_index is None:
            continue
        time_s = frame_entry.get("time_s")
        if time_s is None:
            time_s = frame_entry.get("timestamp") or frame_entry.get("time") or 0.0
        detections = frame_entry.get("detections")
        if detections is None:
            detections = frame_entry.get("objects", [])
        normalized_detections: List[dict] = []
        for det in detections:
            det_cls = det.get("class") or det.get("label")
            if det_cls is None:
                continue
            bbox = _normalize_bbox(det.get("bbox"), det)
            if bbox is None:
                continue
            score = det.get("score", det.get("confidence", det.get("probability", 0.0)))
            normalized_detections.append(
                {
                    "class": det_cls,
                    "bbox": bbox,
                    "score": float(score) if score is not None else 0.0,
                }
            )
        yield int(frame_index), float(time_s), normalized_detections


def build_tracks_from_json(json_path: str) -> Dict[int, Track]:
    """
    Load detections from json_path and run the SimpleTracker.
    """
    json_data = json.loads(Path(json_path).read_text())
    tracker = SimpleTracker()
    for frame_index, time_s, detections in _iter_frames_from_data(json_data):
        tracker.update(frame_index, time_s, detections)
    return tracker.finalize()


def build_graph_memory_from_detections(json_path: str, output_path: str) -> Tuple[GraphMemory, int]:
    """
    Build the GraphMemory from detections and persist it to output_path.

    Returns:
        The GraphMemory instance and the number of frames processed.
    """
    json_data = json.loads(Path(json_path).read_text())
    tracker = SimpleTracker()
    num_frames = 0
    for frame_index, time_s, detections in _iter_frames_from_data(json_data):
        tracker.update(frame_index, time_s, detections)
        num_frames += 1

    tracks = tracker.finalize()
    events = extract_events_from_tracks(tracks)
    graph_memory = GraphMemory(tracks=tracks, events=events)
    Path(output_path).write_text(json.dumps(graph_memory.to_dict(), indent=2))
    return graph_memory, num_frames


__all__ = [
    "BBox",
    "Track",
    "Event",
    "GraphMemory",
    "SimpleTracker",
    "compute_iou",
    "bbox_center",
    "track_pair_min_dist_over_time",
    "extract_events_from_tracks",
    "build_tracks_from_json",
    "build_graph_memory_from_detections",
]
