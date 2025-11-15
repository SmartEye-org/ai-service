"""
gRPC Client Test Script
Test các chức năng của Detection Service qua gRPC
"""
import grpc
import cv2
import sys
from pathlib import Path
import os

# Add proto generated files to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# Import generated proto files
try:
    from proto import detection_service_pb2
    from proto import detection_service_pb2_grpc
except ImportError:
    print("⚠️  Proto files chưa được generate!")
    print("Run: python -m grpc_tools.protoc -I./proto --python_out=./proto --grpc_python_out=./proto ./proto/detection_service.proto")
    sys.exit(1)


def test_health_check(stub):
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("🔍 Testing HealthCheck...")
    print("="*60)
    
    request = detection_service_pb2.HealthCheckRequest(service="detection")
    
    try:
        response = stub.HealthCheck(request)
        print(f"✅ Health Check Response:")
        print(f"   Healthy: {response.healthy}")
        print(f"   Version: {response.version}")
        print(f"   Timestamp: {response.timestamp}")
        print(f"   Message: {response.message}")
        return True
    except grpc.RpcError as e:
        print(f"❌ Health Check Failed:")
        print(f"   Code: {e.code()}")
        print(f"   Details: {e.details()}")
        return False


def test_detect_person(stub, image_path: str, camera_id: str = "camera-test"):
    """Test person detection"""
    print("\n" + "="*60)
    print(f"🔍 Testing DetectPerson with image: {image_path}")
    print("="*60)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Read and encode image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return False
    
    # Encode to JPEG
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        print("❌ Failed to encode image")
        return False
    
    
    image_bytes = encoded_image.tobytes()
    
    # Create request
    image_msg = detection_service_pb2.Image(
        image_data=image_bytes,
        camera_id=camera_id,
        timestamp=""
    )
    request = detection_service_pb2.DetectPersonRequest(image=image_msg)
    
    # Call gRPC
    try:
        response = stub.DetectPerson(request)
        
        if response.success:
            print(f"✅ Person Detection Success:")
            print(f"   Total Persons: {response.total_persons}")
            print(f"   Timestamp: {response.timestamp}")
            print(f"   Message: {response.message}")
            
            if response.detections:
                print(f"\n   Detections:")
                for det in response.detections:
                    bbox = det.bbox
                    print(f"   - Person {det.person_id}:")
                    print(f"     BBox: ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})")
                    print(f"     Confidence: {det.confidence:.3f}")
                    print(f"     Face Detected: {det.face_detected}")
            
            return True
        else:
            print(f"❌ Person Detection Failed:")
            print(f"   Message: {response.message}")
            return False
            
    except grpc.RpcError as e:
        print(f"❌ gRPC Error:")
        print(f"   Code: {e.code()}")
        print(f"   Details: {e.details()}")
        return False


def test_detect_face(stub, image_path: str):
    """Test face detection"""
    print("\n" + "="*60)
    print(f"🔍 Testing DetectFace with image: {image_path}")
    print("="*60)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Read and encode image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return False
    
    # Encode to JPEG
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        print("❌ Failed to encode image")
        return False
    
    image_bytes = encoded_image.tobytes()
    
    # Create request
    image_msg = detection_service_pb2.Image(
        image_data=image_bytes,
        camera_id="camera-test",
        timestamp=""
    )
    request = detection_service_pb2.DetectFaceRequest(image=image_msg)
    
    # Call gRPC
    try:
        response = stub.DetectFace(request)
        
        if response.success:
            print(f"✅ Face Detection Success:")
            print(f"   Total Faces: {response.total_faces}")
            print(f"   Timestamp: {response.timestamp}")
            print(f"   Message: {response.message}")
            
            if response.detections:
                print(f"\n   Detections:")
                for det in response.detections:
                    bbox = det.bbox
                    print(f"   - Face {det.face_id}:")
                    print(f"     BBox: ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})")
                    print(f"     Confidence: {det.confidence:.3f}")
            
            return True
        else:
            print(f"❌ Face Detection Failed:")
            print(f"   Message: {response.message}")
            return False
            
    except grpc.RpcError as e:
        print(f"❌ gRPC Error:")
        print(f"   Code: {e.code()}")
        print(f"   Details: {e.details()}")
        return False


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test gRPC Detection Service')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--camera-id', default='camera-test', help='Camera ID')
    parser.add_argument('--test', choices=['all', 'health', 'person', 'face'], 
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    # Connect to gRPC server
    server_address = f'{args.host}:{args.port}'
    print(f"🔌 Connecting to gRPC server: {server_address}")
    
    try:
        channel = grpc.insecure_channel(server_address)
        stub = detection_service_pb2_grpc.DetectionServiceStub(channel)
        
        # Run tests
        results = []
        
        if args.test in ['all', 'health']:
            results.append(('HealthCheck', test_health_check(stub)))
        
        if args.test in ['all', 'person']:
            if args.image:
                results.append(('DetectPerson', test_detect_person(stub, args.image, args.camera_id)))
            else:
                print("\n⚠️  Skip DetectPerson: No image provided (use --image)")
        
        if args.test in ['all', 'face']:
            if args.image:
                results.append(('DetectFace', test_detect_face(stub, args.image)))
            else:
                print("\n⚠️  Skip DetectFace: No image provided (use --image)")
        
        # Summary
        print("\n" + "="*60)
        print("📊 Test Summary")
        print("="*60)
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {test_name}")
        
        # Return exit code
        all_pass = all(success for _, success in results)
        sys.exit(0 if all_pass else 1)
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        sys.exit(1)
    finally:
        channel.close()


if __name__ == '__main__':
    main()
