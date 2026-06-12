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
    import io
    await websocket.accept()
    print("WebSocket client connected for streaming transcription")
    
    accumulated_bytes = bytearray()
    last_finalized_sample_index = 0
    finalized_transcripts = []
    
    try:
        import json
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                data = message["bytes"]
                accumulated_bytes.extend(data)
                print(f"[WS] Received chunk of {len(data)} bytes. Accumulated: {len(accumulated_bytes)} bytes.")
            elif "text" in message:
                text_data = message["text"]
                try:
                    command = json.loads(text_data)
                    if command.get("type") == "stop":
                        print("[WS] Received stop signal. Finalizing remaining audio...")
                        try:
                            # Try to decode the final accumulated buffer
                            audio_segment = AudioSegment.from_file(io.BytesIO(accumulated_bytes), format="webm")
                            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                            if audio_segment.sample_width == 2:
                                samples /= 32768.0
                            elif audio_segment.sample_width == 4:
                                samples /= 2147483648.0
                            else:
                                samples /= 128.0
                                
                            total_samples = len(samples)
                            if total_samples > last_finalized_sample_index:
                                segment_audio = samples[last_finalized_sample_index:]
                                try:
                                    denoised_audio = reduce_noise_auto(segment_audio, 16000, use_rnnoise=True)
                                except Exception as e:
                                    print(f"RNNoise failed on final chunk: {e}. Using raw.")
                                    denoised_audio = segment_audio
                                    
                                text = transcribe_audio_chunk(denoised_audio)
                                if text:
                                    finalized_transcripts.append(text)
                                    await websocket.send_json({
                                        "type": "final",
                                        "text": text,
                                        "segment_index": len(finalized_transcripts) - 1
                                    })
                        except Exception as e:
                            print(f"[WS] Final decode failed: {e}")
                        
                        await websocket.send_json({"type": "closed"})
                        break
                except Exception as e:
                    print(f"[WS] Failed to parse control message: {e}")
                continue
            else:
                break
                
            try:
                # Decode accumulated audio using pydub
                audio_segment = AudioSegment.from_file(io.BytesIO(accumulated_bytes), format="webm")
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                if audio_segment.sample_width == 2:
                    samples /= 32768.0
                elif audio_segment.sample_width == 4:
                    samples /= 2147483648.0
                else:
                    samples /= 128.0
            except Exception as e:
                # Log the exception for visibility
                print(f"[WS] Decoding failed: {e}")
                continue
                
            total_samples = len(samples)
            speech_segments = detect_speech_segments(samples, 16000)
            
            for idx, seg in enumerate(speech_segments):
                start_s = seg["start"]
                end_s = seg["end"]
                
                if end_s <= last_finalized_sample_index:
                    continue
                    
                is_last_segment = (idx == len(speech_segments) - 1)
                silence_passed = (total_samples - end_s) >= 12000  # 750ms silence threshold
                
                if not is_last_segment or silence_passed:
                    segment_audio = samples[start_s:end_s]
                    
                    try:
                        denoised_audio = reduce_noise_auto(segment_audio, 16000, use_rnnoise=True)
                    except Exception as e:
                        print(f"Denoising failed: {e}. Using raw segment.")
                        denoised_audio = segment_audio
                        
                    text = transcribe_audio_chunk(denoised_audio)
                    if text:
                        finalized_transcripts.append(text)
                        await websocket.send_json({
                            "type": "final",
                            "text": text,
                            "segment_index": len(finalized_transcripts) - 1
                        })
                    
                    last_finalized_sample_index = end_s
            
            # Handle partial transcription of active segment
            if len(speech_segments) > 0:
                last_seg = speech_segments[-1]
                if last_seg["start"] >= last_finalized_sample_index:
                    active_start = last_seg["start"]
                    active_audio = samples[active_start:]
                    
                    if len(active_audio) >= 8000:  # At least 500ms of active audio
                        partial_text = transcribe_audio_chunk(active_audio)
                        if partial_text:
                            await websocket.send_json({
                                "type": "partial",
                                "text": partial_text
                            })
                            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as ws_err:
        print(f"WebSocket error: {ws_err}")
        try:
            await websocket.send_json({"type": "error", "message": str(ws_err)})
        except Exception:
            pass
