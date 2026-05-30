import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from audio_processor import apply_high_pass_filter
import time

app = FastAPI(
    title="EchoNet Backend Gateway",
    description="FastAPI ingestion pipeline for custom PyTorch Speech-to-Text inference"
)

# Configure CORS so your React frontend (running on Vite) can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace this with your specific live frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. System Diagnostic Health Check ---
@app.get("/api/health")
async def health_check():
    """
    Returns the operational status of the server gateway.
    """
    return {"status": "online", "model_loaded": False}

# --- 2. Audio Ingestion & Transcription Endpoint ---
@app.post("/api/transcribe")
async def transcribe_speech(file: UploadFile = File(...)):
    """
    Receives binary audio blobs from the React frontend, runs format validation,
    and forwards data to the PyTorch acoustic sequence model.
    """
    # Validate file format before touching memory allocations
    if not file.filename.endswith(('.webm', '.wav', '.mp3')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported audio format profile. Please provide webm, wav, or mp3 data."
        )

    # Establish a temporary file path to store the incoming binary audio stream
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Hardcoded static filenames so they overwrite every single request
        input_file_node = "input_audio.wav"
        output_file_node = "enhanced_audio.wav"
        
        # Write the network chunk stream to your local disk temporarily
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # 1. Open the file and capture the returned object instance
        my_audio_track = AudioSegment.from_file(temp_file_path, format="webm")

        # 2. Export using the static input node string
        my_audio_track.export(
            input_file_node, 
            format="wav", 
            parameters=["-ac", "1", "-ar", "16000"]
        ) 
        
        # 3. Process the file mathematically
        apply_high_pass_filter(input_path=input_file_node, output_path=output_file_node)

        mock_model_output = "EchoNet backend server is active! Audio pipeline processing complete."
        return {"transcript": mock_model_output}

    except Exception as e:
        # Catch unexpected I/O or processing failures gracefully
        raise HTTPException(status_code=500, detail=f"Internal Server Processing Failure: {str(e)}")
        
    finally:
        # Clean up the system disk memory immediately after execution completes
        pass