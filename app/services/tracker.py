"""
PersonTracker — IoU-based multi-person tracker with stable track IDs.
Tracks persons across frames without DeepSORT dependency (MVP friendly).
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """A tracked person with history."""
    track_id: str
    bbox: List[int]           # [x1, y1, x2, y2]
    confidence: float
    action: str = "unknown"
    age: int = 0              # frames since last seen
    history: List[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    camera_id: str = ""

    # Behavior state machine
    action_history: List[str] = field(default_factory=list)
    consecutive_action_count: int = 0
    last_confirmed_action: str = "unknown"

    def update(self, bbox: List[int], confidence: float, action: str):
        self.bbox = bbox
        self.confidence = confidence
        self.age = 0

        # Smooth action — require 3 consecutive frames to confirm
        self.action_history.append(action)
        if len(self.action_history) > 5:
            self.action_history.pop(0)

        # Most frequent action in window = confirmed
        if self.action_history:
            from collections import Counter
            most_common = Counter(self.action_history).most_common(1)[0]
            if most_common[1] >= 2:  # appeared at least twice in last 5 frames
                self.last_confirmed_action = most_common[0]

        self.action = self.last_confirmed_action

    @property
    def center(self) -> Tuple[float, float]:
        x = (self.bbox[0] + self.bbox[2]) / 2
        y = (self.bbox[1] + self.bbox[3]) / 2
        return x, y

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


class PersonTracker:
    """
    Simple IoU-based multi-person tracker.
    Assigns stable track IDs across frames using bbox overlap.
    No GPU required — runs in ~1ms.
    """

    def __init__(
        self,
        camera_id: str = "unknown",
        iou_threshold: float = 0.3,
        max_age: int = 10,        # frames before track is removed
        id_prefix: str = "track",
    ):
        self.camera_id = camera_id
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.id_prefix = id_prefix

        self._tracks: Dict[str, Track] = {}
        self._next_id: int = 1
        self._frame_count: int = 0

    def _new_track_id(self) -> str:
        tid = f"{self.id_prefix}-{self.camera_id}-{self._next_id:04d}"
        self._next_id += 1
        return tid

    def update(
        self,
        detections: List[Dict],  # [{"bbox": [...], "confidence": ..., "action": ...}]
    ) -> List[Track]:
        """
        Match detections to existing tracks or create new ones.
        Returns list of active tracks with updated info.
        """
        self._frame_count += 1

        # Age all existing tracks
        for track in self._tracks.values():
            track.age += 1

        if not detections:
            self._cleanup()
            return []

        matched_track_ids = set()
        matched_det_indices = set()

        # Build IoU matrix
        track_ids = list(self._tracks.keys())
        det_bboxes = [d["bbox"] for d in detections]
        track_bboxes = [self._tracks[tid].bbox for tid in track_ids]

        if track_ids and det_bboxes:
            iou_matrix = np.zeros((len(track_ids), len(det_bboxes)))
            for i, tbbox in enumerate(track_bboxes):
                for j, dbbox in enumerate(det_bboxes):
                    iou_matrix[i, j] = compute_iou(tbbox, dbbox)

            # Greedy matching — highest IoU first
            flat_indices = np.argsort(iou_matrix.ravel())[::-1]
            for idx in flat_indices:
                i, j = divmod(idx, len(det_bboxes))
                if i in matched_track_ids or j in matched_det_indices:
                    continue
                if iou_matrix[i, j] < self.iou_threshold:
                    break

                tid = track_ids[i]
                det = detections[j]
                self._tracks[tid].update(
                    bbox=det["bbox"],
                    confidence=det.get("confidence", 1.0),
                    action=det.get("action", "unknown"),
                )
                matched_track_ids.add(i)
                matched_det_indices.add(j)

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                new_id = self._new_track_id()
                self._tracks[new_id] = Track(
                    track_id=new_id,
                    bbox=det["bbox"],
                    confidence=det.get("confidence", 1.0),
                    action=det.get("action", "unknown"),
                    camera_id=self.camera_id,
                )

        self._cleanup()
        return [t for t in self._tracks.values() if t.age == 0]

    def _cleanup(self):
        """Remove stale tracks."""
        stale = [tid for tid, t in self._tracks.items() if t.age > self.max_age]
        for tid in stale:
            del self._tracks[tid]

    def get_all_tracks(self) -> List[Track]:
        return list(self._tracks.values())

    def get_track(self, track_id: str) -> Optional[Track]:
        return self._tracks.get(track_id)
