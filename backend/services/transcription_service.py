import os
import numpy as np
from faster_whisper import WhisperModel

# Google Drive model cache
DOWNLOAD_ROOT = "/content/drive/MyDrive/EchoNetCache/whisper"

os.makedirs(DOWNLOAD_ROOT, exist_ok=True)

print("Loading Faster-Whisper Medium Model locally...")

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16",
    download_root=DOWNLOAD_ROOT
)

print("Model loaded successfully and sitting in RAM!")


def transcribe_audio_chunk(
    chunk: np.ndarray,
    beam_size: int = 5
) -> str:
    """
    Runs Faster-Whisper inference on a normalized mono float32 numpy array chunk.

    Args:
        chunk:
            NumPy array of shape (N,)
            with float32 values in range [-1.0, 1.0]

        beam_size:
            Whisper beam size.
            Use 1 for partial/live transcripts.
            Use 5 for finalized transcripts.

    Returns:
        Transcribed text string.
    """

    try:
        segments, info = model.transcribe(
            chunk,
            beam_size=beam_size,
            temperature=0.0
        )

        chunk_text = "".join(
            segment.text
            for segment in segments
        ).strip()

        return chunk_text

    except Exception as e:
        print(
            f"Error during transcription inference: {e}",
            flush=True
        )
        return ""