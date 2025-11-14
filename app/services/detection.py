"""
Simplified Detection Service for MVP with Async & Batch Support
"""
import cv2
import numpy as np
import asyncio
from typing import List, Dict, Tuple
from datetime import datetime
import logging

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

logger = logging.getLogger(__name__)


class SimpleDetectionService:
    """Simplified detection service for MVP with async support"""
    
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        print(f"Loading YOLO model from {model_path}...")
        self.yolo_model = YOLO(model_path)
        
        print("Initializing MediaPipe Face Detection...")
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.5
        )
        
        print("==> Detection service ready!")
    
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
    
    def detect_from_bytes(self, image_bytes: bytes, camera_id: str = "unknown") -> Dict:
        """
        Process detection from raw bytes
        
        Args:
            image_bytes: Image data in bytes
            camera_id: Camera identifier
        
        Returns:
            Detection results
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Process
        results = self.process_image(img)
        results['camera_id'] = camera_id
        
        return results
    
    async def detect_from_bytes_async(
        self, 
        image_bytes: bytes, 
        camera_id: str = "unknown"
    ) -> Dict:
        """
        Async version of detect_from_bytes for better performance
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.detect_from_bytes, 
            image_bytes, 
            camera_id
        )
    
    async def detect_batch(
        self, 
        frames: List[Tuple[bytes, str]]
    ) -> List[Dict]:
        """
        Process multiple frames in batch for better performance
        
        Args:
            frames: List of (image_bytes, camera_id) tuples
        
        Returns:
            List of detection results
        """
        tasks = [
            self.detect_from_bytes_async(image_bytes, camera_id)
            for image_bytes, camera_id in frames
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and return successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                camera_id = frames[i][1]
                logger.error(f"Batch detection failed for camera {camera_id}: {result}")
                # Return empty result for failed detection
                successful_results.append({
                    "detections": [],
                    "total_persons": 0,
                    "camera_id": camera_id,
                    "error": str(result),
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                successful_results.append(result)
        
        return successful_results
    
    def detect_from_frame_skip(
        self,
        image_bytes: bytes,
        camera_id: str,
        frame_number: int,
        skip_rate: int = 2
    ) -> Dict:
        """
        Process frame with skipping logic
        
        Args:
            image_bytes: Image data
            camera_id: Camera identifier
            frame_number: Current frame number
            skip_rate: Process every Nth frame
        
        Returns:
            Detection result or cached result
        """
        # Only process every Nth frame
        if frame_number % skip_rate != 0:
            return {
                "detections": [],
                "total_persons": 0,
                "camera_id": camera_id,
                "skipped": True,
                "frame_number": frame_number,
                "timestamp": datetime.now().isoformat(),
            }
        
        return self.detect_from_bytes(image_bytes, camera_id)
    
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
