"""
Test WebSocket Connection and Stream Start
"""
import asyncio
import websockets
import json
import requests
from datetime import datetime

async def test_stream_start():
    """Test stream start with WebSocket connection"""
    
    camera_id = "camera-01"
    ws_uri = f"ws://127.0.0.1:8000/ws/detect/{camera_id}"
    api_url = f"http://127.0.0.1:8000/stream/start?camera_id={camera_id}"
    
    print("=" * 60)
    print("🧪 Testing Stream Start with WebSocket")
    print("=" * 60)
    
    # Step 1: Check status before connection
    print(f"\n📊 Step 1: Check initial status...")
    response = requests.get("http://127.0.0.1:8000/stream/status")
    print(f"Status: {response.json()}")
    
    # Step 2: Connect WebSocket
    print(f"\n🔌 Step 2: Connecting WebSocket to {ws_uri}...")
    
    try:
        async with websockets.connect(ws_uri) as websocket:
            print(f"✅ WebSocket connected!")
            
            # Step 3: Call start stream API
            print(f"\n🚀 Step 3: Calling start stream API...")
            response = requests.post(api_url)
            result = response.json()
            
            print(f"\n📨 Response:")
            print(json.dumps(result, indent=2))
            
            if result['status'] == 'started':
                print("\n✅ SUCCESS! Stream started with active connection")
            else:
                print("\n❌ FAILED! Still no clients")
            
            # Step 4: Check overall status
            print(f"\n📊 Step 4: Check overall status...")
            response = requests.get("http://127.0.0.1:8000/stream/status")
            status = response.json()
            print(json.dumps(status, indent=2))
            
            # Keep connection alive for a moment
            print(f"\n⏳ Keeping connection alive for 3 seconds...")
            await asyncio.sleep(3)
            
            print(f"\n✅ Test completed!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure AI service is running:")
        print("   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")

if __name__ == "__main__":
    print(f"⏰ Test started at {datetime.now().strftime('%H:%M:%S')}")
    asyncio.run(test_stream_start())
