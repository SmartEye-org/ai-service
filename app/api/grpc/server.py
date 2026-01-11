"""
gRPC Server Implementation
Triển khai các service được định nghĩa trong proto/detection_service.proto
"""
import grpc
from concurrent import futures
import cv2
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Import generated proto files (sẽ được generate từ .proto)
try:
    from proto import detection_service_pb2
    from proto import detection_service_pb2_grpc
except ImportError as e:
    print(f"⚠️  Proto files chưa được generate! Error: {e}")
    print("Run: python -m grpc_tools.protoc -I./proto --python_out=./proto --grpc_python_out=./proto ./proto/detection_service.proto")
    sys.exit(1)

# Import detection services
from app.services.person_detector import get_person_detector
from app.services.face_detector import get_face_detector
from app.services.behavior_analyzer import get_behavior_analyzer


class DetectionServicer(detection_service_pb2_grpc.DetectionServiceServicer):
    """
    Implementation của DetectionService gRPC
    """
    
    def __init__(self):
        """Initialize all detection services"""
        print("🚀 Initializing gRPC Detection Service...")
        
        # Load detection modules
        self.person_detector = get_person_detector()
        self.face_detector = get_face_detector()
        self.behavior_analyzer = get_behavior_analyzer()
        
        print("✅ gRPC Detection Service ready!")
    
    def DetectPerson(self, request, context):
        """
        Phát hiện người trong ảnh
        IMPLEMENTED - Phase 1
        """
        try:
            # Decode image from bytes
            image_bytes = request.image.image_data
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Invalid image data')
                return detection_service_pb2.DetectPersonResponse(
                    success=False,
                    message="Invalid image data"
                )
            
            # Detect persons
            detections = self.person_detector.detect(image)
            
            # Check face for each person
            for detection in detections:
                bbox = detection['bbox']
                detection['face_detected'] = self.face_detector.check_face_in_roi(
                    image, bbox
                )
            
            # Build response
            response_detections = []
            for det in detections:
                bbox_msg = detection_service_pb2.BoundingBox(
                    x1=det['bbox'][0],
                    y1=det['bbox'][1],
                    x2=det['bbox'][2],
                    y2=det['bbox'][3]
                )
                
                person_det = detection_service_pb2.PersonDetection(
                    person_id=det['person_id'],
                    bbox=bbox_msg,
                    confidence=det['confidence'],
                    face_detected=det['face_detected'],
                    timestamp=det['timestamp']
                )
                response_detections.append(person_det)
            
            return detection_service_pb2.DetectPersonResponse(
                detections=response_detections,
                total_persons=len(detections),
                timestamp=datetime.now().isoformat(),
                success=True,
                message=f"Detected {len(detections)} persons"
            )
            
        except Exception as e:
            print(f"❌ Error in DetectPerson: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return detection_service_pb2.DetectPersonResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    def DetectFace(self, request, context):
        """
        Phát hiện khuôn mặt
        FUTURE IMPLEMENTATION - Phase 2
        """
        try:
            # Decode image
            image_bytes = request.image.image_data
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Invalid image data')
                return detection_service_pb2.DetectFaceResponse(
                    success=False,
                    message="Invalid image data"
                )
            
            # Extract ROI if provided
            roi = None
            if request.HasField('roi'):
                roi = [
                    request.roi.x1,
                    request.roi.y1,
                    request.roi.x2,
                    request.roi.y2
                ]
            
            # Detect faces
            face_detections = self.face_detector.detect(image, roi)
            
            # Build response
            response_detections = []
            for det in face_detections:
                bbox_msg = detection_service_pb2.BoundingBox(
                    x1=det['bbox'][0],
                    y1=det['bbox'][1],
                    x2=det['bbox'][2],
                    y2=det['bbox'][3]
                )
                
                face_det = detection_service_pb2.FaceDetection(
                    face_id=det['face_id'],
                    bbox=bbox_msg,
                    confidence=det['confidence'],
                    landmarks=det['landmarks'],
                    timestamp=det['timestamp']
                )
                response_detections.append(face_det)
            
            return detection_service_pb2.DetectFaceResponse(
                detections=response_detections,
                total_faces=len(face_detections),
                timestamp=datetime.now().isoformat(),
                success=True,
                message=f"Detected {len(face_detections)} faces"
            )
            
        except Exception as e:
            print(f"❌ Error in DetectFace: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return detection_service_pb2.DetectFaceResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    def RecognizeFace(self, request, context):
        """
        Nhận diện khuôn mặt
        FUTURE IMPLEMENTATION - Phase 3
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Face recognition not implemented yet')
        return detection_service_pb2.RecognizeFaceResponse(
            success=False,
            message="Face recognition not implemented yet"
        )
    
    def AnalyzeBehavior(self, request, context):
        """
        Phân tích hành vi
        FUTURE IMPLEMENTATION - Phase 3
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Behavior analysis not implemented yet')
        return detection_service_pb2.AnalyzeBehaviorResponse(
            success=False,
            message="Behavior analysis not implemented yet"
        )
    
    def FullAnalysis(self, request, context):
        """
        Phân tích toàn diện (person + face + behavior)
        FUTURE IMPLEMENTATION - Phase 3
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Full analysis not implemented yet')
        return detection_service_pb2.FullAnalysisResponse(
            success=False,
            message="Full analysis not implemented yet"
        )
    
    def HealthCheck(self, request, context):
        """
        Health check endpoint
        IMPLEMENTED
        """
        return detection_service_pb2.HealthCheckResponse(
            healthy=True,
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            message="Service is healthy"
        )


def serve(port: int = 8000, max_workers: int = 10):
    """
    Start gRPC server
    
    Args:
        port: Port to listen on
        max_workers: Max thread pool workers
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    # Add servicer
    detection_service_pb2_grpc.add_DetectionServiceServicer_to_server(
        DetectionServicer(), server
    )
    
    # Bind port
    server.add_insecure_port(f'[::]:{port}')
    
    print(f"🚀 gRPC Server starting on port {port}...")
    server.start()
    print(f"✅ gRPC Server listening on [::]:{port}")
    print(f"   Max workers: {max_workers}")
    print(f"   Services available:")
    print(f"   - DetectPerson (IMPLEMENTED)")
    print(f"   - DetectFace (IMPLEMENTED)")
    print(f"   - RecognizeFace (FUTURE)")
    print(f"   - AnalyzeBehavior (FUTURE)")
    print(f"   - FullAnalysis (FUTURE)")
    print(f"   - HealthCheck (IMPLEMENTED)")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gRPC server...")
        server.stop(0)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='gRPC Detection Service')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--workers', type=int, default=10, help='Max thread pool workers')
    
    args = parser.parse_args()
    
    serve(port=args.port, max_workers=args.workers)
