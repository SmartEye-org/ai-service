"""
Simple test script for API
"""
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("✅ Health check passed\n")

def test_detection(image_path: str):
    """Test detection endpoint"""
    print(f"Testing detection with image: {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'camera_id': 'camera-test'}
        response = requests.post(
            f"{API_URL}/api/v1/detect",
            files=files,
            data=data
        )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Detected: {result['total_persons']} persons")
        print(f"NGSI-LD entities: {len(result.get('ngsi_ld_entities', []))}")
        print("✅ Detection test passed\n")
    else:
        print(f"❌ Detection failed: {response.text}\n")

if __name__ == "__main__":
    test_health()
    
    if len(sys.argv) > 1:
        test_detection(sys.argv[1])
    else:
        print("Usage: python scripts/test_api.py <image_path>")
