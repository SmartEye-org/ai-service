#!/bin/bash
# Regenerate gRPC Python bindings from proto file
# Run this after modifying proto/detection_service.proto

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Regenerating proto bindings..."

python -m grpc_tools.protoc \
    -I ./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    ./proto/detection_service.proto

echo "✅ Proto bindings regenerated successfully!"
echo ""
echo "Generated files:"
ls -la proto/*_pb2*.py
