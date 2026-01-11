"""
Face Recognition Service - InsightFace (ArcFace)
Độ chính xác cao, production-ready
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from insightface.app import FaceAnalysis
import os


class FaceDetector:
    """
    Phát hiện và nhận diện khuôn mặt sử dụng InsightFace (ArcFace)
    Database: embeddings_data/face_db.npz (face embeddings)
    """
    
    def __init__(self, db_path: str = "embeddings_data/face_db.npz", confidence_threshold: float = 0.5):
        print(f"[FaceDetector] Initializing InsightFace ArcFace model...")
        
        try:
            # Load InsightFace model
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
            
            # Load face database
            if os.path.exists(db_path):
                data = np.load(db_path, allow_pickle=True)
                self.embeddings_db = data["embeddings"]
                self.names_db = data["names"]
                self.embeddings_db = self._l2_normalize(self.embeddings_db)
                
                print(f"[FaceDetector] Loaded {len(self.names_db)} embeddings")
                print(f"[FaceDetector] Known: {list(set(self.names_db))}")
            else:
                print(f"[FaceDetector] WARNING: Database not found: {db_path}")
                print(f"[FaceDetector] Run: python scripts/build_face_db.py")
                self.embeddings_db = np.array([])
                self.names_db = np.array([])
            
            self.confidence_threshold = confidence_threshold
            print(f"[FaceDetector] Threshold: {confidence_threshold}")
            print(f"[FaceDetector] Ready!")
            
        except Exception as e:
            print(f"[FaceDetector] Failed: {e}")
            raise
    
    def _l2_normalize(self, x, axis=-1, eps=1e-10):
        return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), eps))
    
    def detect(self, image: np.ndarray, roi: Optional[List[int]] = None) -> List[Dict]:
        """
        Phát hiện và nhận diện khuôn mặt - InsightFace ArcFace
        
        Args:
            image: Input image (BGR format)
            roi: Region of Interest [x1, y1, x2, y2] (optional)
            
        Returns:
            List of detections with face_id, name, bbox, confidence (similarity)
        """
        # Crop image if ROI provided
        if roi:
            x1, y1, x2, y2 = roi
            img_crop = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            img_crop = image
            offset_x, offset_y = 0, 0
        
        if img_crop.size == 0:
            return []
        
        # InsightFace expects BGR (same as OpenCV)
        # Detect faces
        faces = self.app.get(img_crop)
        
        detections = []
        for idx, face in enumerate(faces):
            # Get bounding box (x1, y1, x2, y2)
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Convert to absolute coordinates
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
            
            # Recognition: Compare embedding with database
            if len(self.embeddings_db) > 0:
                emb = self._l2_normalize(face.embedding.reshape(1, -1))
                sims = self.embeddings_db @ emb.T  # Cosine similarity
                sims = sims.flatten()
                
                # CRITICAL: Use argmax (high similarity = match)
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                
                # Check threshold
                if best_score >= self.confidence_threshold:
                    person_name = str(self.names_db[best_idx])
                    confidence = best_score
                else:
                    person_name = "Unknown"
                    confidence = 0.0
            else:
                person_name = "Unknown"
                confidence = 0.0
            
            detections.append({
                'face_id': idx,
                'name': person_name,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'landmarks': [],
                'timestamp': datetime.now().isoformat()
            })
        
        if detections:
            print(f"[FaceDetector] Detected {len(detections)} faces:")
            for det in detections:
                print(f"  - {det['name']} (similarity: {det['confidence']:.3f})")
        else:
            print(f"[FaceDetector] No faces detected")
            
        return detections
    
    def check_face_in_roi(self, image: np.ndarray, roi: List[int]) -> bool:
        """
        Kiểm tra xem có khuôn mặt trong ROI không
        """
        detections = self.detect(image, roi)
        return len(detections) > 0
    
    def get_known_identities(self) -> List[str]:
        """
        Lấy danh sách tên người đã train trong database
        """
        return list(set(self.names_db.tolist()))
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Vẽ bounding box và tên người lên ảnh
        """
        img_result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name = det['name']
            confidence = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green if known, Red if unknown
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_result, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_result, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return img_result
    
    def release(self):
        """Cleanup resources"""
        print("[FaceDetector] Released")


# Global singleton instance
_face_detector_instance = None

def get_face_detector(db_path: str = "embeddings_data/face_db.npz", confidence_threshold: float = 0.5) -> FaceDetector:
    """
    Get singleton instance của FaceDetector (InsightFace)
    """
    global _face_detector_instance
    if _face_detector_instance is None:
        _face_detector_instance = FaceDetector(db_path, confidence_threshold)
    return _face_detector_instance
