"""
Detection Service — MVP with Behavior Analysis & Person Tracking.
Pipeline: YOLO person detect → Tracker (stable IDs) → Behavior Analyze → Face Detect
"""
import cv2
import numpy as np
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

# PyTorch 2.6+ compatibility fix
import torch
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
import mediapipe as mp

from app.services.tracker import PersonTracker
from app.services.behavior_analyzer import BehaviorAnalyzer, VIOLATION_ACTIONS

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Full detection pipeline with behavior analysis and tracking.
    Single instance per process, per-camera trackers.
    """

    def __init__(self, model_path: str = "models/yolov8n.pt"):
        logger.info(f"Loading YOLO model: {model_path}")
        self.yolo = YOLO(model_path)

        logger.info("Initializing MediaPipe Face Detection")
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            min_detection_confidence=0.5
        )

        # BehaviorAnalyzer uses MediaPipe Pose
        self.behavior_analyzer = BehaviorAnalyzer(min_detection_confidence=0.4)

        # Per-camera trackers
        self._trackers: Dict[str, PersonTracker] = {}

        logger.info("==> DetectionService ready (YOLO + Pose + Face + Tracker)")

    def _get_tracker(self, camera_id: str) -> PersonTracker:
        if camera_id not in self._trackers:
            self._trackers[camera_id] = PersonTracker(
                camera_id=camera_id,
                iou_threshold=0.3,
                max_age=15,
            )
        return self._trackers[camera_id]

    def _detect_persons_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLOv8 — persons only (class 0)."""
        results = self.yolo(frame, classes=[0], verbose=False)
        detections = []
        for result in results:
            for idx, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "person_id": idx,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                })
        return detections

    def _detect_face_in_roi(self, frame: np.ndarray, bbox: List[int]) -> bool:
        """Quick face presence check in a person ROI."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = self.face_detector.process(rgb_roi)
        return result.detections is not None

    def process_frame(self, frame: np.ndarray, camera_id: str = "default") -> Dict:
        """
        Full pipeline for a single frame.

        Returns:
            {
                camera_id: str,
                timestamp: str,
                total_persons: int,
                detections: [
                    {
                        track_id: str,
                        person_id: int,
                        bbox: [x1,y1,x2,y2],
                        confidence: float,
                        action: str,
                        face_detected: bool,
                        behavior_confidence: float,
                        violation_detected: bool,
                        violation_type: str | None,
                        violation_severity: str | None,
                        violation_description: str | None,
                        timestamp: str,
                    }
                ],
                violations: [...]   # subset of detections with violation_detected=True
            }
        """
        tracker = self._get_tracker(camera_id)
        now = datetime.now().isoformat()

        # 1. YOLO person detection
        raw_detections = self._detect_persons_yolo(frame)

        # 2. Behavior analysis per person (before tracking so action is available)
        for det in raw_detections:
            behavior = self.behavior_analyzer.analyze(frame, det["bbox"])
            det["action"] = behavior["action"]
            det["behavior_confidence"] = behavior["confidence"]
            det["violation"] = behavior["violation"]
            det["violation_type"] = behavior["violation_type"]
            det["violation_severity"] = behavior["violation_severity"]
            det["violation_description"] = behavior["violation_description"]

        # 3. Update tracker — get stable track IDs
        active_tracks = tracker.update(raw_detections)

        # 4. Face detection per tracked person
        output_detections = []
        violations = []

        for track in active_tracks:
            face_detected = self._detect_face_in_roi(frame, track.bbox)

            # Re-read violation from tracker action (smoothed)
            violation_info = VIOLATION_ACTIONS.get(track.action)
            has_violation = violation_info is not None

            det_record = {
                "track_id": track.track_id,
                "person_id": int(track.track_id.split("-")[-1]) if track.track_id else 0,
                "bbox": track.bbox,
                "confidence": round(track.confidence, 3),
                "action": track.action,
                "face_detected": face_detected,
                "behavior_confidence": round(track.confidence, 3),
                "violation_detected": has_violation,
                "violation_type": violation_info["type"] if has_violation else None,
                "violation_severity": violation_info["severity"] if has_violation else None,
                "violation_description": violation_info["description"] if has_violation else None,
                "timestamp": now,
            }

            output_detections.append(det_record)
            if has_violation:
                violations.append(det_record)

        return {
            "camera_id": camera_id,
            "timestamp": now,
            "total_persons": len(output_detections),
            "detections": output_detections,
            "violations": violations,
        }

    def process_image(self, image: np.ndarray) -> Dict:
        """Backward-compatible wrapper (used by existing HTTP endpoint)."""
        result = self.process_frame(image, camera_id="http-upload")
        return result

    def detect_from_bytes(self, image_bytes: bytes, camera_id: str = "unknown") -> Dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return self.process_frame(img, camera_id=camera_id)

    async def detect_from_bytes_async(
        self, image_bytes: bytes, camera_id: str = "unknown"
    ) -> Dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.detect_from_bytes, image_bytes, camera_id
        )

    async def detect_batch(
        self, frames: List[Tuple[bytes, str]]
    ) -> List[Dict]:
        tasks = [
            self.detect_from_bytes_async(image_bytes, camera_id)
            for image_bytes, camera_id in frames
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                camera_id = frames[i][1]
                logger.error(f"Batch detection error for {camera_id}: {result}")
                output.append({
                    "detections": [],
                    "violations": [],
                    "total_persons": 0,
                    "camera_id": camera_id,
                    "error": str(result),
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                output.append(result)
        return output

    def release(self):
        self.face_detector.close()
        self.behavior_analyzer.release()
        logger.info("DetectionService released")


# ─── Singleton ────────────────────────────────────────────────────────────────
_service_instance: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    global _service_instance
    if _service_instance is None:
        from app.config import settings
        _service_instance = DetectionService(model_path=settings.YOLO_MODEL_PATH)
    return _service_instance
