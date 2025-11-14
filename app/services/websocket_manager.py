"""
WebSocket Connection Manager
Manages WebSocket connections for real-time detection streaming
"""
import asyncio
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manage WebSocket connections for real-time detection streaming"""
    
    def __init__(self):
        # Map of camera_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of WebSocket -> camera_id for cleanup
        self.connection_camera_map: Dict[WebSocket, str] = {}
        # Connection statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'errors': 0,
        }
    
    async def connect(self, websocket: WebSocket, camera_id: str):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = set()
        
        self.active_connections[camera_id].add(websocket)
        self.connection_camera_map[websocket] = camera_id
        
        self.stats['total_connections'] += 1
        self.stats['active_connections'] += 1
        
        logger.info(f"WebSocket connected for camera {camera_id}. "
                   f"Active connections: {self.stats['active_connections']}")
    
    def disconnect(self, websocket: WebSocket):
        """Unregister a WebSocket connection"""
        if websocket in self.connection_camera_map:
            camera_id = self.connection_camera_map[websocket]
            
            if camera_id in self.active_connections:
                self.active_connections[camera_id].discard(websocket)
                
                # Clean up empty camera sets
                if not self.active_connections[camera_id]:
                    del self.active_connections[camera_id]
            
            del self.connection_camera_map[websocket]
            self.stats['active_connections'] -= 1
            
            logger.info(f"WebSocket disconnected for camera {camera_id}. "
                       f"Active connections: {self.stats['active_connections']}")
    
    async def send_to_camera(self, camera_id: str, message: dict):
        """Send message to all connections subscribed to a camera"""
        if camera_id not in self.active_connections:
            return
        
        disconnected = set()
        
        for connection in self.active_connections[camera_id]:
            try:
                await connection.send_json(message)
                self.stats['messages_sent'] += 1
            except Exception as e:
                logger.error(f"Error sending message to camera {camera_id}: {e}")
                disconnected.add(connection)
                self.stats['errors'] += 1
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all active connections"""
        for camera_id in list(self.active_connections.keys()):
            await self.send_to_camera(camera_id, message)
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        return {
            **self.stats,
            'cameras_monitored': len(self.active_connections),
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    def get_camera_connections(self, camera_id: str) -> int:
        """Get number of connections for a specific camera"""
        return len(self.active_connections.get(camera_id, set()))

# Global instance
manager = ConnectionManager()
