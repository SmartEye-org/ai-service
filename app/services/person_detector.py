"""
Person Detection Service - Module riêng cho phát hiện người
Sử dụng YOLOv8 model
"""
import cv2
import numpy as np
from typing import List, Dict
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


class PersonDetector:
    def __init__(self, model_path: str = "models/yolov8n.pt", confidence_threshold: float = 0.5):
        print(f"[PersonDetector] Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"[PersonDetector] ✅ Model loaded! (threshold={confidence_threshold})")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        # Run YOLO detection, chỉ lấy class 0 (person)
        results = self.model(image, classes=[0], verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Lọc theo confidence threshold
                if conf > self.confidence_threshold:
                    detections.append({
                        'person_id': idx,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'timestamp': datetime.now().isoformat()
                    })
        
        print(f"[PersonDetector] Detected {len(detections)} persons")
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        img_draw = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            person_id = det['person_id']
            
            # Vẽ bounding box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ label
            label = f"Person {person_id}: {conf:.2f}"
            cv2.putText(img_draw, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_draw


# Global singleton instance
_person_detector_instance = None

def get_person_detector(model_path: str = "models/yolov8n.pt", 
                       confidence_threshold: float = 0.5) -> PersonDetector:
    global _person_detector_instance
    if _person_detector_instance is None:
        _person_detector_instance = PersonDetector(model_path, confidence_threshold)
    return _person_detector_instance
