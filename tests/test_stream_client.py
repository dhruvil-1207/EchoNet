import asyncio
import websockets
import json
from pydub import AudioSegment
import io
import os

async def test_stream():
    url = "ws://localhost:8000/api/stream-transcribe"
    print(f"Connecting to WebSocket at {url}...")
    
    try:
        async with websockets.connect(url) as ws:
            print("Connected successfully!")
            
            # Load a local WAV file from workspace root
            wav_path = "input_audio.wav"
            if not os.path.exists(wav_path):
                # Fallback to enhanced_audio
                wav_path = "enhanced_audio.wav"
                
            if not os.path.exists(wav_path):
                print("Error: No test audio WAV files found in root. Make sure input_audio.wav exists.")
                return
                
            print(f"Loading test audio from {wav_path}...")
            audio = AudioSegment.from_wav(wav_path)
            
            # Export to WebM in-memory buffer
            webm_io = io.BytesIO()
            audio.export(webm_io, format="webm")
            webm_bytes = webm_io.getvalue()
            
            print(f"Total WebM audio size: {len(webm_bytes)} bytes. Duration: {len(audio)/1000.0} seconds.")
            
            # Stream in chunks simulating periodic 500ms intervals
            chunk_count = 5
            chunk_size = len(webm_bytes) // chunk_count
            if chunk_size == 0:
                chunk_size = 1024
                
            async def receive_messages():
                try:
                    async for message in ws:
                        data = json.loads(message)
                        print(f"[RECV] Type: {data.get('type')}, Text: '{data.get('text')}'")
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed by server.")
                except Exception as e:
                    print(f"Error receiving messages: {e}")
            
            # Start background task to listen for transcription outputs
            recv_task = asyncio.create_task(receive_messages())
            
            # Send the audio chunks sequentially
            for i in range(0, len(webm_bytes), chunk_size):
                chunk = webm_bytes[i:i+chunk_size]
                print(f"[SEND] Sending chunk of size {len(chunk)} bytes...")
                await ws.send(chunk)
                await asyncio.sleep(0.5)  # Simulate 500ms real-time delay
                
            # Wait for final transcription processing
            print("Finished sending. Waiting 3 seconds for final transcription responses...")
            await asyncio.sleep(3)
            
            # Close connection
            await ws.close()
            recv_task.cancel()
            
    except Exception as e:
        print(f"Connection/Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream())
