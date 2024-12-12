from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict
import json

router = APIRouter()
active_connections: Dict[str, WebSocket] = {}

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
    except WebSocketDisconnect:
        del active_connections[client_id]

async def broadcast_progress(batch_id: str, progress: dict):
    """Broadcast progress updates to connected clients"""
    message = json.dumps({
        "type": "progress_update",
        "batch_id": batch_id,
        "data": progress
    })
    for connection in active_connections.values():
        await connection.send_text(message) 