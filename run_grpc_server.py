"""
Wrapper script to run gRPC server
Usage: python run_grpc_server.py [--port PORT] [--workers WORKERS]
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run gRPC server
from app.api.grpc.server import serve

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='gRPC Detection Service')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on (default: 8000)')
    parser.add_argument('--workers', type=int, default=10, help='Max thread pool workers (default: 10)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting gRPC Detection Service...")
    print(f"   Port: {args.port}")
    print(f"   Workers: {args.workers}")
    print()
    
    serve(port=args.port, max_workers=args.workers)
