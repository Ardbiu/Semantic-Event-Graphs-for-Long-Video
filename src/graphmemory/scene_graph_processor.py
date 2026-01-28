import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Import the new scorer (assuming it's in the same package or accessible)
try:
    from src.graphmemory.interaction_scorer import InteractionScorer
except ImportError:
    # Fallback for direct script execution without package structure
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.graphmemory.interaction_scorer import InteractionScorer

@dataclass
class InteractionState:
    a_id: int
    b_id: int
    a_label: str
    b_label: str
    last_seen_ts: float
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)
    missing_frames: int = 0
    start_ts: float = 0.0
    start_frame: int = 0

class SceneGraphProcessor:
    """
    Tracks pairwise interactions using Semantic-Gated Interaction Events.
    Uses InteractionScorer to compute S(i,j,t).
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        
        self.config = config
        self.focus_class = config.get('focus_class', 'person')  # Optional
        
        # Thresholds
        thresholds = config.get('interaction', {}).get('thresholds', {})
        self.tau_start = thresholds.get('tau_start', 0.6)
        self.tau_end = thresholds.get('tau_end', 0.4)
        self.end_buffer = thresholds.get('end_buffer_frames', 5)
        self.min_duration = thresholds.get('min_duration_frames', 5)
        
        # Scorer
        scorer_config = config.get('interaction', {})
        self.scorer = InteractionScorer(scorer_config)
        
        self.active: Dict[Tuple[int, int], InteractionState] = {}
        self.event_log: List[dict] = []
        
        # History for motion computation {id: detection_dict}
        self.history: Dict[int, dict] = {}
        self.current_frame_detections: Dict[int, dict] = {}

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

    def _append_event(self, timestamp: float, frame: int, event_type: str, 
                     subj_label: str, subj_id: int, obj_label: str, obj_id: int,
                     confidence: float = 1.0, evidence: dict = None):
        self.event_log.append(
            {
                "timestamp": float(timestamp),
                "frame": int(frame),
                "type": event_type,
                "subject": f"{subj_label}-{subj_id}",
                "object": f"{obj_label}-{obj_id}",
                "confidence": confidence,
                "evidence": evidence or {}
            }
        )

    def update(self, detections: List[dict], timestamp: Optional[float], frame_index: int):
        """
        detections: list of {"id": int, "label": str, "bbox": (x1,y1,x2,y2)}
        Returns a list of textual events representing state changes.
        """
        events = []
        present_pairs = set()
        
        # 1. Update History & Current Map
        self.current_frame_detections = {d['id']: d for d in detections if d.get('id') is not None}
        
        # 2. Main Loop Over Pairs
        processed = list(self.current_frame_detections.values())
        
        for i in range(len(processed)):
            for j in range(i + 1, len(processed)):
                det_a, det_b = processed[i], processed[j]
                
                # Filter by focus class if set
                if self.focus_class:
                    if det_a["label"].lower() != self.focus_class and det_b["label"].lower() != self.focus_class:
                        continue
                
                # Retrieve previous states for motion
                prev_a = self.history.get(det_a['id'])
                prev_b = self.history.get(det_b['id'])
                
                # Compute Score
                score, evidence = self.scorer.compute_score(det_a, det_b, prev_a, prev_b)
                
                # Order for consistent keys
                ordered_a, ordered_b = self._order_pair(det_a, det_b)
                key = (ordered_a["id"], ordered_b["id"])
                
                # Logic: S > tau_start -> Interaction Active
                # Logic: S < tau_end -> Interaction Weak (start counter)
                
                if score >= self.tau_start:
                    present_pairs.add(key)
                    
                    if key not in self.active:
                        # NEW START
                        start_ts = timestamp if timestamp is not None else 0.0
                        self.active[key] = InteractionState(
                            a_id=ordered_a["id"],
                            b_id=ordered_b["id"],
                            a_label=ordered_a["label"],
                            b_label=ordered_b["label"],
                            last_seen_ts=start_ts,
                            confidence=score,
                            evidence=evidence,
                            missing_frames=0,
                            start_ts=start_ts,
                            start_frame=frame_index,
                        )
                        ts = self._format_ts(start_ts)
                        
                        # Log START
                        self._append_event(start_ts, frame_index, "START", 
                                         ordered_a["label"], ordered_a["id"], 
                                         ordered_b["label"], ordered_b["id"],
                                         confidence=score, evidence=evidence)
                        
                        events.append(f"[TIMESTAMP {ts}] START: {ordered_a['label']}-{ordered_a['id']} interacting with {ordered_b['label']}-{ordered_b['id']} (Score: {score:.2f})")
                    else:
                        # CONTINUE
                        state = self.active[key]
                        state.last_seen_ts = timestamp if timestamp is not None else state.last_seen_ts
                        state.missing_frames = 0
                        state.confidence = (state.confidence * 0.9) + (score * 0.1) # Moving average
                        state.evidence = evidence
                        
                elif score >= self.tau_end:
                     # In Hysteresis Zone (between tau_end and tau_start)
                     # If already active, keep it active (add to present_pairs)
                     # If not active, do nothing (needs tau_start to trigger)
                     if key in self.active:
                         present_pairs.add(key)
                         state = self.active[key]
                         state.last_seen_ts = timestamp if timestamp is not None else state.last_seen_ts
                         state.missing_frames = 0
                         # Update confidence but maybe lower?
                         state.confidence = (state.confidence * 0.9) + (score * 0.1)
                
                # If score < tau_end, we do NOT add to present_pairs, triggering "missing" logic below.

        # 3. Handle Ended Interactions
        for key, state in list(self.active.items()):
            if key not in present_pairs:
                state.missing_frames += 1
                if state.missing_frames > self.end_buffer:
                    # Check min duration
                    duration_frames = frame_index - state.start_frame
                    if duration_frames >= self.min_duration:
                        ts = self._format_ts(timestamp or state.last_seen_ts)
                        end_ts = timestamp if timestamp is not None else state.last_seen_ts
                        # Log END
                        self._append_event(end_ts, frame_index, "END", 
                                         state.a_label, state.a_id, 
                                         state.b_label, state.b_id,
                                         confidence=state.confidence)
                        events.append(f"[TIMESTAMP {ts}] END: {state.a_label}-{state.a_id} disengaged from {state.b_label}-{state.b_id}.")
                    else:
                        # Prune short events (don't log END, maybe remove START? 
                        # For now, just don't log END, but we already logged START. 
                        # Ideally we buffer STARTs, but that's complex. 
                        # Let's just log END and let graph pruning handle short stuff later if needed.)
                        # Or, better: Post-processing removes short events.
                         ts = self._format_ts(timestamp or state.last_seen_ts)
                         end_ts = timestamp if timestamp is not None else state.last_seen_ts
                         self._append_event(end_ts, frame_index, "END", 
                                          state.a_label, state.a_id, 
                                          state.b_label, state.b_id,
                                          confidence=state.confidence)

                    del self.active[key]

        # 4. Update History
        self.history = self.current_frame_detections.copy()
        
        return events

    def save_events_to_json(self, filename: Union[str, Path]):
        path = Path(filename)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.event_log, f, indent=2)

    def generate_timeline_plot(self, filename: Union[str, Path]):
        # Lazy import
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
        ax.set_title("Interaction Timeline (Semantic Gated)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)

