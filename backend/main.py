import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
        # Write the network chunk stream to your local disk temporarily
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # ----------------===================================================----------------
        # PLACEHOLDER: This is where we will route the file to our PyTorch MFCC feature
        # extractor and custom GRU neural layers in the next step!
        # ----------------===================================================----------------
        mock_model_output = "EchoNet backend server is active! PyTorch layer integration is next."
        
        return {"transcript": mock_model_output}

    except Exception as e:
        # Catch unexpected I/O or processing failures gracefully
        raise HTTPException(status_code=500, detail=f"Internal Server Processing Failure: {str(e)}")
        
    finally:
        # Clean up the system disk memory immediately after execution completes
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)