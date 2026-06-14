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

import scipy.io.wavfile as wavfile
import numpy as np

from audio_pipeline.vad.chunk_extractor import extract_speech_chunks

USE_D_DRIVE = os.path.exists("D:\\")

if USE_D_DRIVE:
    os.environ["HF_HOME"] = "D:\\EchoNetCache\\huggingface"
    os.environ["HF_HUB_CACHE"] = "D:\\EchoNetCache\\huggingface"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

from audio_pipeline.pipeline import enhance_audio
from audio_pipeline.vad.vad_detector import detect_speech_segments
from audio_pipeline.denoising.auto_noise_reducer import reduce_noise_auto
from backend.services.transcription_service import transcribe_audio_chunk

# Import the optimized session handler
from backend.services.processing_service import TranscriptionSession

app = FastAPI(
    title="EchoNet Backend Gateway",
    description="FastAPI ingestion pipeline for custom high-speed Speech-to-Text inference"
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


def get_temp_path(filename):
    if USE_D_DRIVE:
        temp_dir = "D:\\EchoNetTemp"
        os.makedirs(temp_dir, exist_ok=True)
        return os.path.join(temp_dir, filename)

    return filename


@app.post("/api/transcribe")
async def transcribe_speech(file: UploadFile = File(...)):

    if not file.filename.endswith(
        (".webm", ".wav", ".mp3")
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format profile."
        )

    temp_file_path = get_temp_path(
        f"temp_{file.filename}"
    )

    try:

        input_file_node = get_temp_path(
            "input_audio.wav"
        )

        output_file_node = get_temp_path(
            "enhanced_audio.wav"
        )

        try:
            with open(
                temp_file_path,
                "wb"
            ) as buffer:
                buffer.write(
                    await file.read()
                )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Phase 1 Error (Save Temp): {str(e)}"
            )

        try:
            my_audio_track = AudioSegment.from_file(
                temp_file_path,
                format="webm"
            )

            # Exporting to 16000 Hz mono PCM
            my_audio_track.export(
                input_file_node,
                format="wav",
                parameters=[
                    "-ac", "1",
                    "-ar", "16000"
                ]
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Phase 2 Error (pydub conversion): {str(e)}"
            )

        try:
            speech_segments = enhance_audio(
                input_path=input_file_node,
                output_path=output_file_node,
                generate_visualizations=True
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Phase 3 Error (audio enhancement): {str(e)}"
            )

        try:

            sample_rate, enhanced_audio = wavfile.read(
                output_file_node
            )

            print("\nMAIN.PY AUDIO INFO")
            print("dtype:", enhanced_audio.dtype)
            print("shape:", enhanced_audio.shape)
            print("min:", enhanced_audio.min())
            print("max:", enhanced_audio.max())

            print("\nSpeech Segments:")
            print(speech_segments)

            chunks = [enhanced_audio]

            print("\nChunk Count:")
            print(len(chunks))

            if len(chunks) == 0:
                raise Exception(
                    "No speech segments detected."
                )

            full_transcript = []

            for i, chunk in enumerate(chunks, start=1):

                print(f"\nTranscribing Chunk {i}...")

                # Chunk is already natively at 16000 Hz here
                chunk = chunk.astype(np.float32) / 32768.0

                print("\nWHISPER INPUT INFO")
                print("dtype:", chunk.dtype)
                print("shape:", chunk.shape)
                print("min:", np.min(chunk))
                print("max:", np.max(chunk))
                print("mean abs:", np.mean(np.abs(chunk)))

                wavfile.write(
                    "debug_chunk.wav",
                    16000,
                    (chunk * 32767).astype(np.int16)
                )

                chunk_text = transcribe_audio_chunk(chunk)

                print(f"Chunk {i} Transcript: '{chunk_text}'")

                if chunk_text:
                    full_transcript.append(chunk_text)

            transcribed_text = " ".join(full_transcript)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in whisper transcription: {e}")

            raise HTTPException(
                status_code=500,
                detail=f"Phase 4 Error (transcription inference): {str(e)}"
            )

        print(
            f"Transcription Complete: '{transcribed_text}'"
        )

        return {
            "transcript": transcribed_text
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected Internal Failure: {str(e)}"
        )

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


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
