import os
import sys

try:
    import audioop_lts as audioop
    sys.modules['audioop'] = audioop
    sys.modules['pyaudioop'] = audioop
except ImportError:
    pass

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# --- Pointing HuggingFace Cache to Google Drive ---
cache_dir = "/content/drive/MyDrive/EchoNetCache/huggingface"
os.makedirs(cache_dir, exist_ok=True)

os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.services.processing_service import TranscriptionSession

app = FastAPI(
    title="EchoNet Backend Gateway",
    description="FastAPI live streaming pipeline for custom high-speed Speech-to-Text inference"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "model_loaded": True
    }

@app.websocket("/api/stream-transcribe")
async def stream_transcribe(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected for streaming transcription")
    
    session = TranscriptionSession()
    
    try:
        import json
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                data = message["bytes"]
                # Directly pipe raw Float32 PCM to our session processor
                result = await session.process_chunk(data)
                if result:
                    await websocket.send_json(result)
                    
            elif "text" in message:
                text_data = message["text"]
                try:
                    command = json.loads(text_data)
                    if command.get("type") == "stop":
                        print("[WS] Received stop signal. Finalizing remaining audio...")
                        
                        final_result = await session.flush()
                        if final_result:
                            await websocket.send_json(final_result)
                        
                        await websocket.send_json({"type": "closed"})
                        break
                except Exception as e:
                    print(f"[WS] Failed to parse control message: {e}")
                continue
            else:
                break
                            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        final_result = await session.flush()
        if final_result:
            try:
                await websocket.send_json(final_result)
            except Exception:
                pass
    except Exception as ws_err:
        print(f"WebSocket error: {ws_err}")
        try:
            await websocket.send_json({"type": "error", "message": str(ws_err)})
        except Exception:
            pass
