"""
Script để compile proto files
Generate Python code từ .proto definition
"""
import subprocess
import sys
from pathlib import Path


def compile_proto():
    """Compile proto files using grpc_tools"""
    
    proto_dir = Path("proto")
    proto_file = proto_dir / "detection_service.proto"
    
    if not proto_file.exists():
        print(f"❌ Proto file not found: {proto_file}")
        sys.exit(1)
    
    print(f"🔨 Compiling proto file: {proto_file}")
    print(f"   Output directory: {proto_dir}")
    
    # Compile command
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        str(proto_file)
    ]
    
    print(f"\n📝 Command: {' '.join(cmd)}\n")
    
    # Run compilation
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Proto compilation successful!")
        print(f"\nGenerated files:")
        print(f"   - {proto_dir}/detection_service_pb2.py")
        print(f"   - {proto_dir}/detection_service_pb2_grpc.py")
        
        # Create __init__.py if not exists
        init_file = proto_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"   - {init_file} (created)")
        
        return True
    else:
        print("❌ Proto compilation failed!")
        if result.stdout:
            print(f"\nStdout:\n{result.stdout}")
        if result.stderr:
            print(f"\nStderr:\n{result.stderr}")
        return False


if __name__ == "__main__":
    success = compile_proto()
    sys.exit(0 if success else 1)
