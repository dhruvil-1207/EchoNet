import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from audio_processor import apply_high_pass_filter

# Import the high-speed engine
from faster_whisper import WhisperModel

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

# Load the model into memory once when the server starts
# We use device="cpu" to run locally without needing a giant GPU, 
# and compute_type="int8" to compress the memory footprint without losing accuracy.
print("Loading Faster-Whisper Medium Model locally...")
model = WhisperModel("medium", device="cpu", compute_type="int8")
print("Model loaded successfully and sitting in RAM!")


@app.get("/api/health")
async def health_check():
    # health check to show the model is active!
    return {"status": "online", "model_loaded": True}


@app.post("/api/transcribe")
async def transcribe_speech(file: UploadFile = File(...)):
    if not file.filename.endswith(('.webm', '.wav', '.mp3')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported audio format profile."
        )

    temp_file_path = f"temp_{file.filename}"
    
    try:
        input_file_node = "input_audio.wav"
        output_file_node = "enhanced_audio.wav"
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        my_audio_track = AudioSegment.from_file(temp_file_path, format="webm")
        my_audio_track.export(
            input_file_node, 
            format="wav", 
            parameters=["-ac", "1", "-ar", "16000"]
        ) 
        
        apply_high_pass_filter(input_path=input_file_node, output_path=output_file_node)

        # vad_filter=True acts as an extra layer to skip any silent gaps
        segments, info = model.transcribe(
            output_file_node, 
            vad_filter=True,
            beam_size=5,          # Tracks the top 5 sentence paths for better context
            temperature=0.0,      # Forces strict literal matching, reducing hallucination
            initial_prompt="EchoNet, FastAPI, React, Git, terminal, audio pipeline" # Prime its memory
        )
        
        # Combine the generator chunks into a single string text sentence
        transcribed_text = "".join([segment.text for segment in segments]).strip()

        print(f"Transcription Complete: '{transcribed_text}'")
        
        # Return the actual, live transcribed words back to React!
        return {"transcript": transcribed_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Processing Failure: {str(e)}")
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)