import os
import sys
import scipy.io.wavfile as wavfile
import numpy as np

from audio_pipeline.vad.chunk_extractor import extract_speech_chunks

USE_D_DRIVE = os.path.exists("D:\\")

if USE_D_DRIVE:
    os.environ["HF_HOME"] = "D:\\EchoNetCache\\huggingface"
    os.environ["HF_HUB_CACHE"] = "D:\\EchoNetCache\\huggingface"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from faster_whisper import WhisperModel

from audio_pipeline.pipeline import enhance_audio

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

print("Loading Faster-Whisper Medium Model locally...")

download_root = (
    "D:\\EchoNetCache\\whisper"
    if USE_D_DRIVE
    else None
)

if download_root:
    os.makedirs(download_root, exist_ok=True)

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    download_root=download_root
)

print("Model loaded successfully and sitting in RAM!")


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
                generate_visualizations=False
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

            chunks = extract_speech_chunks(
                enhanced_audio,
                speech_segments
            )

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

                segments, info = model.transcribe(
                    chunk,
                    beam_size=5,
                    temperature=0.0
                )

                chunk_text = "".join(
                    segment.text
                    for segment in segments
                ).strip()

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
