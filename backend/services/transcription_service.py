import os
import numpy as np
from faster_whisper import WhisperModel

# Detect system configuration
# Cache Whisper exactly like we did in main.py
download_root = "/content/drive/MyDrive/EchoNetCache/whisper"
os.makedirs(download_root, exist_ok=True)
print("Loading Faster-Whisper Medium Model locally...")
model = WhisperModel(
    "turbo", 
    device="cuda",
    compute_type="float16",
    download_root=download_root
)
print("Model loaded successfully and sitting in VRAM!")


def transcribe_audio_chunk(chunk: np.ndarray, beam_size: int = 5, initial_prompt: str = None) -> str:
    """
    Runs Faster-Whisper inference on a normalized mono float32 numpy array chunk.
    
    Args:
        chunk: NumPy array of shape (N,) with float32 values in range [-1.0, 1.0].
        beam_size: Controls accuracy vs speed. (1 for fast partials, 5 for finals).
        initial_prompt: Context from the previous sentence to prevent hallucinations.
        
    Returns:
        String containing the transcribed text.
    """
    try:
        segments, info = model.transcribe(
            chunk,
            beam_size=beam_size,
            temperature=0.0,
            initial_prompt=initial_prompt
        )
        chunk_text = "".join(segment.text for segment in segments).strip()
        return chunk_text
    except Exception as e:
        print(f"Error during transcription inference: {e}")
        return ""
