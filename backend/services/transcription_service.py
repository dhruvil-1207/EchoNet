import os
import numpy as np
from faster_whisper import WhisperModel

# Detect system configuration
USE_D_DRIVE = os.path.exists("D:\\")
download_root = "D:\\EchoNetCache\\whisper" if USE_D_DRIVE else None

if download_root:
    os.makedirs(download_root, exist_ok=True)

print("Loading Faster-Whisper Medium Model locally...")
model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    download_root=download_root
)
print("Model loaded successfully and sitting in RAM!")


def transcribe_audio_chunk(chunk: np.ndarray) -> str:
    """
    Runs Faster-Whisper inference on a normalized mono float32 numpy array chunk.
    
    Args:
        chunk: NumPy array of shape (N,) with float32 values in range [-1.0, 1.0].
        
    Returns:
        String containing the transcribed text.
    """
    try:
        segments, info = model.transcribe(
            chunk,
            beam_size=5,
            temperature=0.0
        )
        chunk_text = "".join(segment.text for segment in segments).strip()
        return chunk_text
    except Exception as e:
        print(f"Error during transcription inference: {e}")
        return ""
