"""
BehaviorAnalyzer — MediaPipe Pose-based behavior classification.

Actions detected:
- standing   : person is upright, not moving much
- walking    : legs alternating, moderate speed
- running    : legs alternating, fast / high knee
- sitting    : hips low, knees bent
- lying      : body nearly horizontal
- unknown    : pose not visible / low confidence

Violation mapping:
- lying     -> LYING violation (possible medical emergency)
- running   -> RUNNING violation (no running in lobby/corridor)
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_WRIST = 15
RIGHT_WRIST = 16


# Violation config — which actions trigger alerts
VIOLATION_ACTIONS = {
    "lying": {
        "type": "lying",
        "severity": "high",
        "description": "Person detected lying on ground",
    },
    "running": {
        "type": "running",
        "severity": "medium",
        "description": "Person running in restricted area",
    },
}


class BehaviorAnalyzer:
    """
    Analyzes human pose within a person bounding box to classify behavior.
    Uses MediaPipe Pose — runs on CPU, ~15-30ms per person ROI.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,   # Process each ROI independently
            model_complexity=0,        # Lite model for speed
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        logger.info("BehaviorAnalyzer initialized with MediaPipe Pose Lite")

    def analyze(self, frame: np.ndarray, bbox: List[int]) -> Dict:
        """
        Analyze behavior for a single person ROI.

        Args:
            frame: Full frame (BGR)
            bbox:  [x1, y1, x2, y2] bounding box

        Returns:
            {
                "action": str,          # walking/running/standing/sitting/lying/unknown
                "confidence": float,    # pose detection confidence 0-1
                "violation": bool,
                "violation_type": str | None,
                "violation_severity": str | None,
                "keypoints": dict | None,
            }
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return self._unknown_result()

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return self._unknown_result()

        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_roi)

        if not results.pose_landmarks:
            return self._unknown_result()

        landmarks = results.pose_landmarks.landmark
        action = self._classify_action(landmarks)

        # Check violation
        violation_info = VIOLATION_ACTIONS.get(action)
        violation = violation_info is not None

        return {
            "action": action,
            "confidence": float(landmarks[NOSE].visibility),
            "violation": violation,
            "violation_type": violation_info["type"] if violation else None,
            "violation_severity": violation_info["severity"] if violation else None,
            "violation_description": violation_info["description"] if violation else None,
            "keypoints": self._extract_keypoints(landmarks),
        }

    def _classify_action(self, landmarks) -> str:
        """Classify action from pose landmarks (normalized 0-1 coordinates)."""
        try:
            nose = landmarks[NOSE]
            l_hip = landmarks[LEFT_HIP]
            r_hip = landmarks[RIGHT_HIP]
            l_knee = landmarks[LEFT_KNEE]
            r_knee = landmarks[RIGHT_KNEE]
            l_ankle = landmarks[LEFT_ANKLE]
            r_ankle = landmarks[RIGHT_ANKLE]
            l_shoulder = landmarks[LEFT_SHOULDER]
            r_shoulder = landmarks[RIGHT_SHOULDER]

            # Skip if core landmarks invisible
            if l_hip.visibility < 0.3 or r_hip.visibility < 0.3:
                return "unknown"

            hip_y = (l_hip.y + r_hip.y) / 2
            knee_y = (l_knee.y + r_knee.y) / 2
            ankle_y = (l_ankle.y + r_ankle.y) / 2
            shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            nose_y = nose.y

            # --- LYING: body nearly horizontal ---
            # nose and hips close in Y, both low on frame
            if abs(nose_y - hip_y) < 0.18:
                return "lying"

            # --- SITTING: hips higher than knees, knees near ankles ---
            if hip_y > 0.55 and (knee_y - ankle_y) < 0.18:
                return "sitting"

            # --- RUNNING: knees high, feet alternating far apart ---
            ankle_diff = abs(l_ankle.y - r_ankle.y)
            if ankle_diff > 0.18 and shoulder_y < 0.45:
                return "running"

            # --- WALKING: feet alternating, moderate ---
            if ankle_diff > 0.10:
                return "walking"

            # --- STANDING: upright, hips above midpoint ---
            if hip_y < 0.65:
                return "standing"

            return "unknown"

        except (IndexError, AttributeError):
            return "unknown"

    def _extract_keypoints(self, landmarks) -> Dict:
        """Extract key pose points for downstream use."""
        key_indices = {
            "nose": NOSE,
            "left_shoulder": LEFT_SHOULDER,
            "right_shoulder": RIGHT_SHOULDER,
            "left_hip": LEFT_HIP,
            "right_hip": RIGHT_HIP,
            "left_knee": LEFT_KNEE,
            "right_knee": RIGHT_KNEE,
            "left_ankle": LEFT_ANKLE,
            "right_ankle": RIGHT_ANKLE,
        }
        return {
            name: {
                "x": round(landmarks[idx].x, 3),
                "y": round(landmarks[idx].y, 3),
                "visibility": round(landmarks[idx].visibility, 3),
            }
            for name, idx in key_indices.items()
        }

    def _unknown_result(self) -> Dict:
        return {
            "action": "unknown",
            "confidence": 0.0,
            "violation": False,
            "violation_type": None,
            "violation_severity": None,
            "violation_description": None,
            "keypoints": None,
        }

    def release(self):
        self.pose.close()


# Per-camera singleton instances
_analyzers: Dict[str, BehaviorAnalyzer] = {}


def get_behavior_analyzer(camera_id: str = "default") -> BehaviorAnalyzer:
    """Get or create a BehaviorAnalyzer per camera."""
    global _analyzers
    if camera_id not in _analyzers:
        _analyzers[camera_id] = BehaviorAnalyzer()
    return _analyzers[camera_id]
