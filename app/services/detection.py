"""
Simplified Detection Service for MVP
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

# PyTorch 2.6+ fix
import torch
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

from ultralytics import YOLO
import mediapipe as mp


class SimpleDetectionService:
    """Simplified detection service for MVP"""
    
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        print(f"Loading YOLO model from {model_path}...")
        self.yolo_model = YOLO(model_path)
        
        print("Initializing MediaPipe Face Detection...")
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.5
        )
        
        print("✅ Detection service ready!")
    
    def detect_persons(self, image: np.ndarray) -> List[Dict]:
        """Detect persons using YOLO"""
        results = self.yolo_model(image, classes=[0], verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf > 0.5:
                    detections.append({
                        'person_id': idx,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'timestamp': datetime.now().isoformat()
                    })
        
        return detections
    
    def detect_faces_in_roi(self, image: np.ndarray, bbox: List[int]) -> bool:
        """Check if face exists in bounding box"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_roi)
        
        return results.detections is not None
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Main processing function
        Returns: {detections, total, timestamp}
        """
        # Detect persons
        detections = self.detect_persons(image)
        
        # Add face detection to each person
        for detection in detections:
            detection['face_detected'] = self.detect_faces_in_roi(
                image, detection['bbox']
            )
        
        return {
            'detections': detections,
            'total_persons': len(detections),
            'timestamp': datetime.now().isoformat()
        }
    
    def release(self):
        """Cleanup resources"""
        self.face_detection.close()
        print("Detection service released")


# Global instance
_detection_service = None

def get_detection_service() -> SimpleDetectionService:
    """Get or create detection service singleton"""
    global _detection_service
    if _detection_service is None:
        _detection_service = SimpleDetectionService()
    return _detection_service
