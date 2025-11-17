"""
Test face recognition với InsightFace
Nhận diện khuôn mặt từ ảnh mới
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

DB_PATH = "embeddings_data/face_db.npz"


def l2_normalize(x, axis=-1, eps=1e-10):
    """Normalize vector to unit length"""
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), eps))


def load_db(path=DB_PATH):
    """Load face database"""
    print(f"Loading face database from {path}...")
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]   # shape (N, 512)
    names = data["names"]             # shape (N,)
    # Đảm bảo đã normalize
    embeddings = l2_normalize(embeddings)
    print(f"Loaded {len(names)} embeddings")
    print(f"   Unique persons: {list(set(names))}")
    return embeddings, names


def recognize_image(image_path, threshold=0.5):
    """
    Nhận diện khuôn mặt trong ảnh
    
    Args:
        image_path: Đường dẫn ảnh
        threshold: Ngưỡng similarity (0.4-0.6 recommended)
    """
    print(f"\nRecognizing faces in: {image_path}")
    print(f"   Threshold: {threshold}")
    
    # 1. Load database
    embeddings_db, names_db = load_db(DB_PATH)
    
    # 2. Khởi tạo model ArcFace + detector
    print("\nInitializing InsightFace model...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))   # -1: CPU, 0: GPU
    
    # 3. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return
    
    print(f"Image loaded: {img.shape}")
    
    # 4. Detect faces
    print("\n🔎 Detecting faces...")
    faces = app.get(img)
    
    if len(faces) == 0:
        print("No face detected in image.")
        return
    
    print(f"Detected {len(faces)} face(s)")
    
    # 5. Nhận diện từng face
    print("\nRecognition results:")
    for i, face in enumerate(faces):
        # Extract embedding
        emb = l2_normalize(face.embedding)
        
        # Tính cosine similarity với tất cả người trong DB
        # Vì tất cả vector đều đã normalize → cos_sim = dot product
        sims = embeddings_db @ emb
        best_idx = int(np.argmax(sims))  # ARGMAX - similarity cao nhất
        best_score = float(sims[best_idx])
        best_name = names_db[best_idx]
        
        # Get bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        if best_score >= threshold:
            print(f"\n   Face {i}:")
            print(f"     Name: {best_name}")
            print(f"     Similarity: {best_score:.3f}")
            print(f"     BBox: ({x1}, {y1}, {x2}, {y2})")
            
            # Draw on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{best_name}: {best_score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print(f"\n   Face {i}:")
            print(f"     UNKNOWN")
            print(f"     Best match: {best_name} (similarity: {best_score:.3f})")
            print(f"     BBox: ({x1}, {y1}, {x2}, {y2})")
            
            # Draw on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "UNKNOWN", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Save result
    output_path = "recognition_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"\n💾 Result saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test face recognition')
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    recognize_image(args.image, threshold=args.threshold)
