"""
Build face database từ ảnh training
Sử dụng InsightFace (ArcFace) để extract embeddings

Cấu trúc thư mục:
face_data/
  ├── Nguyen_the_vinh/
  │   ├── 001.jpg
  │   ├── 002.jpg
  │   └── ...
  ├── Lo_van_bang/
  │   ├── 001.jpg
  │   └── ...
"""
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path

DATA_DIR = "face_data"         # Thư mục chứa ảnh từng người
DB_PATH = "embeddings_data/face_db.npz"  # File lưu database


def l2_normalize(x, axis=-1, eps=1e-10):
    """Normalize vector to unit length"""
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), eps))


def build_face_db():
    """Build face database from images"""
    
    print(" Initializing InsightFace ArcFace model...")
    # ctx_id: -1 = CPU, 0 = GPU
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print(" Model loaded!")
    
    if not os.path.exists(DATA_DIR):
        print(f" Data directory not found: {DATA_DIR}")
        print("\n Please create data structure:")
        print("   face_data/")
        print("     ├── PersonName1/")
        print("     │   ├── image1.jpg")
        print("     │   └── image2.jpg")
        print("     └── PersonName2/")
        print("         └── image1.jpg")
        return
    
    all_embeddings = []
    all_names = []
    
    person_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    
    if not person_folders:
        print(f" No person folders found in {DATA_DIR}")
        return
    
    print(f"\n👥 Found {len(person_folders)} persons:")
    for person_name in person_folders:
        print(f"   - {person_name}")
    
    print("\n Processing images...")
    
    total_images = 0
    total_faces = 0
    
    for person_name in person_folders:
        person_dir = os.path.join(DATA_DIR, person_name)
        print(f"\n📂 Processing: {person_name}")
        
        person_faces = 0
        for file in os.listdir(person_dir):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            total_images += 1
            img_path = os.path.join(person_dir, file)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"     Cannot read image: {file}")
                continue
            
            # Detect faces
            faces = app.get(img)
            
            if len(faces) == 0:
                print(f"     No face detected in {file}")
                continue
            
            if len(faces) > 1:
                print(f"    Multiple faces in {file}, using first one")
            
            # Get embedding (lấy face đầu tiên)
            emb = faces[0].embedding
            emb = l2_normalize(emb)
            
            all_embeddings.append(emb)
            all_names.append(person_name)
            person_faces += 1
            total_faces += 1
            
            print(f"    {file} → embedding shape: {emb.shape}")
        
        print(f"   Total: {person_faces} faces encoded")
    
    if len(all_embeddings) == 0:
        print("\n No faces found. Check your data folder.")
        return
    
    # Save to numpy file
    embeddings = np.vstack(all_embeddings).astype("float32")
    names = np.array(all_names)
    
    # Create output directory
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(DB_PATH, embeddings=embeddings, names=names)
    
    print(f"\n Saved face database to {DB_PATH}")
    print(f"\n Summary:")
    print(f"   Total images processed: {total_images}")
    print(f"   Total embeddings: {len(names)}")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Unique persons: {len(set(names))}")
    print(f"\n Embeddings per person:")
    for person in set(names):
        count = list(names).count(person)
        print(f"   - {person}: {count} embeddings")


if __name__ == "__main__":
    build_face_db()
