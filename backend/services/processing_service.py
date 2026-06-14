import numpy as np
import scipy.signal as signal
from silero_vad import load_silero_vad, VADIterator
import asyncio
import torch

from audio_pipeline.preprocessing.high_pass_filter import apply_high_pass_filter
from audio_pipeline.denoising.auto_noise_reducer import reduce_noise_auto
from backend.services.transcription_service import transcribe_audio_chunk

# Detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Silero VAD on device: {device}")

# Load and move VAD to GPU
_vad_model = load_silero_vad()
if hasattr(_vad_model, 'to'):
    _vad_model = _vad_model.to(device)

_SAMPLE_RATE = 16000
_BUFFER_MAX_SAMPLES   = _SAMPLE_RATE * 30
_FINALIZE_SILENCE_MS  = 700
_PARTIAL_EVERY_S      = 1.0
_MIN_CHUNK_S          = 0.5
_VAD_WINDOW_SAMPLES   = 512

def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data ** 2)))

class TranscriptionSession:
    def __init__(self):
        self.vad = VADIterator(
            _vad_model,
            threshold=0.5,
            sampling_rate=_SAMPLE_RATE,
            min_silence_duration_ms=_FINALIZE_SILENCE_MS,
            speech_pad_ms=100,
        )
        self._buffer: list[np.ndarray] = []
        self._samples_since_partial = 0
        self._finalized: list[str] = []
        
        self._vad_buffer = np.array([], dtype=np.float32)

    def _denoise(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return data
        
        # Use custom pipeline denoiser instead of RNNoise
        denoised = reduce_noise_auto(data, _SAMPLE_RATE, use_rnnoise=False)
        return denoised.astype(np.float32)

    async def process_chunk(self, raw_bytes: bytes) -> dict | None:
        if not raw_bytes:
            return None

        # Bulletproof: Drop trailing fragmented bytes if it's not a multiple of 4
        remainder = len(raw_bytes) % 4
        if remainder != 0:
            raw_bytes = raw_bytes[:-remainder]
            
        if not raw_bytes:
            return None

        chunk = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        
        self._vad_buffer = np.concatenate((self._vad_buffer, chunk))
        should_finalize = False
        
        while len(self._vad_buffer) >= _VAD_WINDOW_SAMPLES:
            window = self._vad_buffer[:_VAD_WINDOW_SAMPLES]
            self._vad_buffer = self._vad_buffer[_VAD_WINDOW_SAMPLES:]
            vad_result = self.vad(window)
            
            if vad_result is not None and "end" in vad_result:
                should_finalize = True

        chunk = apply_high_pass_filter(chunk, _SAMPLE_RATE)
        chunk = self._denoise(chunk)
        
        if _rms(chunk) < 0.005 and not should_finalize:
            total = sum(len(c) for c in self._buffer)
            if total > _BUFFER_MAX_SAMPLES:
                return await self._finalize()
            return None

        self._buffer.append(chunk)
        self._samples_since_partial += len(chunk)

        if should_finalize:
            return await self._finalize()

        if self._samples_since_partial >= _SAMPLE_RATE * _PARTIAL_EVERY_S:
            self._samples_since_partial = 0
            return await self._partial()

        return None

    async def flush(self) -> dict | None:
        if self._buffer:
            return await self._finalize()
        return None

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._buffer) if self._buffer else np.array([], dtype=np.float32)

    async def _partial(self) -> dict | None:
        audio = self._audio()
        if len(audio) < _SAMPLE_RATE * _MIN_CHUNK_S:
            return None
        
        context = " ".join(self._finalized)[-200:] if self._finalized else None
        
        text = await asyncio.to_thread(transcribe_audio_chunk, audio, beam_size=1, initial_prompt=context)
        
        if not text:
            return None
        return {"type": "partial", "text": text}

    async def _finalize(self) -> dict | None:
        audio = self._audio()
        self._buffer = []
        self._samples_since_partial = 0

        if len(audio) < _SAMPLE_RATE * _MIN_CHUNK_S:
            return None

        context = " ".join(self._finalized)[-200:] if self._finalized else None
        
        text = await asyncio.to_thread(transcribe_audio_chunk, audio, beam_size=5, initial_prompt=context)
        
        if not text:
            return None

        self._finalized.append(text)
        return {
            "type": "final",
            "text": text,
            "full_transcript": " ".join(self._finalized),
        }
