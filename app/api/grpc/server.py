"""
gRPC Server Implementation — Phase 2 complete.
Implements: DetectPerson (with tracking+behavior), FullAnalysis, HealthCheck.
"""
import grpc
from concurrent import futures
import cv2
import numpy as np
from datetime import datetime
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Project root on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

try:
    from proto import detection_service_pb2
    from proto import detection_service_pb2_grpc
except ImportError as e:
    print(f"⚠️  Proto files not generated! Error: {e}")
    print("Run: python -m grpc_tools.protoc -I./proto --python_out=./proto --grpc_python_out=./proto ./proto/detection_service.proto")
    sys.exit(1)

from app.services.detection import get_detection_service
from app.services.face_detector import get_face_detector


def _bbox_msg(bbox: list) -> "detection_service_pb2.BoundingBox":
    """Convert [x1,y1,x2,y2] list to BoundingBox proto message."""
    return detection_service_pb2.BoundingBox(
        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]
    )


def _person_detection_msg(det: dict) -> "detection_service_pb2.PersonDetection":
    """Convert detection dict to PersonDetection proto message."""
    return detection_service_pb2.PersonDetection(
        person_id=det.get("person_id", 0),
        bbox=_bbox_msg(det["bbox"]),
        confidence=float(det.get("confidence", 0.0)),
        face_detected=bool(det.get("face_detected", False)),
        timestamp=det.get("timestamp", datetime.now().isoformat()),
        track_id=det.get("track_id", ""),
        action=det.get("action", "unknown"),
        violation_detected=bool(det.get("violation_detected", False)),
        violation_type=det.get("violation_type") or "",
        violation_severity=det.get("violation_severity") or "",
        violation_description=det.get("violation_description") or "",
    )


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode bytes to BGR numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image data")
    return img


class DetectionServicer(detection_service_pb2_grpc.DetectionServiceServicer):
    """gRPC DetectionService — Phase 2 implementation."""

    def __init__(self):
        logger.info("Initializing gRPC DetectionServicer...")
        self.detection_service = get_detection_service()
        self.face_detector = get_face_detector()
        logger.info("✅ gRPC DetectionServicer ready")

    # ─── DetectPerson ──────────────────────────────────────────────────────────

    def DetectPerson(self, request, context):
        """
        Full person detection pipeline:
        YOLO → Tracker → Behavior → Face check
        """
        try:
            img = _decode_image(request.image.image_data)
            camera_id = request.image.camera_id or "unknown"

            result = self.detection_service.process_frame(img, camera_id=camera_id)

            detections_pb = [_person_detection_msg(d) for d in result["detections"]]
            violations_pb = [_person_detection_msg(d) for d in result["violations"]]

            return detection_service_pb2.DetectPersonResponse(
                detections=detections_pb,
                violations=violations_pb,
                total_persons=result["total_persons"],
                timestamp=result["timestamp"],
                success=True,
                message=f"Detected {result['total_persons']} persons, {len(result['violations'])} violations",
            )

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return detection_service_pb2.DetectPersonResponse(success=False, message=str(e))

        except Exception as e:
            logger.error(f"DetectPerson error: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return detection_service_pb2.DetectPersonResponse(success=False, message=str(e))

    # ─── DetectFace ────────────────────────────────────────────────────────────

    def DetectFace(self, request, context):
        try:
            img = _decode_image(request.image.image_data)

            roi = None
            if request.HasField("roi"):
                roi = [request.roi.x1, request.roi.y1, request.roi.x2, request.roi.y2]

            face_dets = self.face_detector.detect(img, roi)

            response_dets = [
                detection_service_pb2.FaceDetection(
                    face_id=d["face_id"],
                    bbox=_bbox_msg(d["bbox"]),
                    confidence=float(d["confidence"]),
                    landmarks=d.get("landmarks", []),
                    timestamp=d.get("timestamp", datetime.now().isoformat()),
                )
                for d in face_dets
            ]

            return detection_service_pb2.DetectFaceResponse(
                detections=response_dets,
                total_faces=len(face_dets),
                timestamp=datetime.now().isoformat(),
                success=True,
                message=f"Detected {len(face_dets)} faces",
            )

        except Exception as e:
            logger.error(f"DetectFace error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return detection_service_pb2.DetectFaceResponse(success=False, message=str(e))

    # ─── RecognizeFace (placeholder for Phase 3 ArcFace) ──────────────────────

    def RecognizeFace(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("ArcFace recognition not implemented yet (Phase 3)")
        return detection_service_pb2.RecognizeFaceResponse(
            success=False,
            message="ArcFace recognition coming in Phase 3",
        )

    # ─── AnalyzeBehavior ───────────────────────────────────────────────────────

    def AnalyzeBehavior(self, request, context):
        """
        Analyze behavior for pre-detected persons.
        Accepts PersonDetection list + frame image.
        """
        try:
            img = _decode_image(request.image.image_data)
            camera_id = request.camera_id or "unknown"

            behaviors = []
            for person in request.persons:
                bbox = [person.bbox.x1, person.bbox.y1, person.bbox.x2, person.bbox.y2]
                result = self.detection_service.behavior_analyzer.analyze(img, bbox)

                b_type = _action_to_behavior_type(result["action"])
                behaviors.append(
                    detection_service_pb2.BehaviorDetection(
                        track_id=person.track_id,
                        behavior=b_type,
                        behavior_label=result["action"],
                        confidence=float(result["confidence"]),
                        description=result.get("violation_description") or result["action"],
                        timestamp=datetime.now().isoformat(),
                        is_violation=result["violation"],
                        violation_severity=result.get("violation_severity") or "",
                    )
                )

            return detection_service_pb2.AnalyzeBehaviorResponse(
                behaviors=behaviors,
                success=True,
                message=f"Analyzed {len(behaviors)} persons",
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"AnalyzeBehavior error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return detection_service_pb2.AnalyzeBehaviorResponse(success=False, message=str(e))

    # ─── FullAnalysis ──────────────────────────────────────────────────────────

    def FullAnalysis(self, request, context):
        """
        One-call full pipeline: YOLO + Tracker + Behavior + Face.
        This is the recommended endpoint for streaming use.
        """
        try:
            img = _decode_image(request.image.image_data)
            camera_id = request.camera_id or request.image.camera_id or "unknown"

            result = self.detection_service.process_frame(img, camera_id=camera_id)

            detections_pb = [_person_detection_msg(d) for d in result["detections"]]
            violations_pb = [_person_detection_msg(d) for d in result["violations"]]

            return detection_service_pb2.FullAnalysisResponse(
                detections=detections_pb,
                violations=violations_pb,
                total_persons=result["total_persons"],
                success=True,
                message=(
                    f"Full analysis: {result['total_persons']} persons, "
                    f"{len(result['violations'])} violations"
                ),
                timestamp=result["timestamp"],
                camera_id=camera_id,
            )

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return detection_service_pb2.FullAnalysisResponse(success=False, message=str(e))

        except Exception as e:
            logger.error(f"FullAnalysis error: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return detection_service_pb2.FullAnalysisResponse(success=False, message=str(e))

    # ─── HealthCheck ───────────────────────────────────────────────────────────

    def HealthCheck(self, request, context):
        return detection_service_pb2.HealthCheckResponse(
            healthy=True,
            version="2.0.0",
            timestamp=datetime.now().isoformat(),
            message="DetectionService healthy — Behavior analysis active",
        )


def _action_to_behavior_type(action: str) -> int:
    """Map action string to BehaviorType enum value."""
    mapping = {
        "standing": 1,  # STANDING
        "sitting":  2,  # SITTING
        "walking":  3,  # WALKING
        "running":  4,  # RUNNING
        "lying":    5,  # LYING
    }
    return mapping.get(action, 0)  # 0 = UNKNOWN


def serve(port: int = 50051, max_workers: int = 4):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    detection_service_pb2_grpc.add_DetectionServiceServicer_to_server(
        DetectionServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"🚀 gRPC server starting on [::]:{port}")
    server.start()
    logger.info(
        f"✅ gRPC server ready | DetectPerson ✓ | FullAnalysis ✓ | AnalyzeBehavior ✓"
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(0)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.workers)
