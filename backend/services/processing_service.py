"""
Processing Service Module
-------------------------
Acts as the central router and brain of the EchoNet pipeline. It receives raw bytes 
from WebSockets, coordinates denoising, voice activity detection, and transcription.
"""

import numpy as np
import asyncio
import datetime
import time

# Core AI Pipeline Imports
from audio_pipeline.preprocessing.high_pass_filter import apply_high_pass_filter
from audio_pipeline.vad.vad_detector import StreamingVAD
from audio_pipeline.denoising.deepfilter_reducer import StreamingDeepFilter
from backend.services.transcription_service import transcribe_audio_chunk

# External Helper Managers (Zero-Latency Integration)
from audio_pipeline.visualization.exporter import DiagnosticExporter
from audio_pipeline.recording.session_recorder import SessionRecorder

# --- GLOBAL TOGGLE ---
ENABLE_VISUALIZATIONS = True

# --- SYSTEM CONSTANTS ---
_SAMPLE_RATE = 16000 # hz (Required frequency for Whisper and VAD neural networks)
_BUFFER_MAX_SAMPLES   = _SAMPLE_RATE * 30 # 30 seconds max buffer limit to prevent RAM crashes
_FINALIZE_SILENCE_MS  = 700 # ms (If silence lasts longer than this, the sentence is officially over)
_PARTIAL_EVERY_S      = 1.0 # Trigger a live UI text update every 1 second of speech
_MIN_CHUNK_S          = 0.5 # Minimum audio length required to prevent Whisper hallucinations
_VAD_WINDOW_SAMPLES   = 512 # Exactly 32ms. Silero VAD strictly requires a rigid 512-sample tensor.

def _rms(data: np.ndarray) -> float:
    """
    Calculates the Root Mean Square (mathematical loudness) of an audio array.
    Used as an ultra-lightweight gatekeeper before hitting the heavier neural networks.
    """
    return float(np.sqrt(np.mean(data ** 2)))

class TranscriptionSession:
    """
    Maintains the state, audio buffers, and memory for a single active WebSocket connection.
    A new instance is spawned every time a user connects to the server.
    """
    def __init__(self):
        # 1. Initialize Neural Networks
        self.vad = StreamingVAD(
            sample_rate=_SAMPLE_RATE,
            min_silence_duration_ms=_FINALIZE_SILENCE_MS,
        )
        self.denoiser = StreamingDeepFilter()
        
        # 2. Initialize Background Managers

        # Force India Standard Time (UTC + 5:30)
        ist_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exporter = DiagnosticExporter(sample_rate=_SAMPLE_RATE, session_timestamp=self.session_timestamp)
        self.recorder = SessionRecorder(sample_rate=_SAMPLE_RATE, session_timestamp = self.session_timestamp)
        
        # 3. Initialize Audio Buffers & State
        self._buffer: list[np.ndarray] = []              # The main sentence being built for Whisper
        self._raw_sentence_buffer: list[np.ndarray] = [] # The dirty mic input for the current sentence
        
        self._session_raw_history: list[np.ndarray] = [] # The entire 1-hour uncut raw mic recording
        self._session_clean_history: list[np.ndarray] = [] # The entire 1-hour cut & cleaned audio recording

        self._samples_since_partial = 0
        self._finalized: list[str] = []
        self._vad_buffer = np.array([], dtype=np.float32)

    async def process_chunk(self, raw_bytes: bytes) -> dict | None:
        """
        The main processing loop. Receives arbitrary-sized internet byte packets, aligns them,
        denoises them, runs VAD windowing, and decides whether to trigger a transcription.
        """
        if not raw_bytes: return None

        # Bulletproof byte alignment: drop incomplete Float32 bytes from network stutters
        remainder = len(raw_bytes) % 4
        if remainder != 0:
            raw_bytes = raw_bytes[:-remainder]
            
        if not raw_bytes: return None

        chunk = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        
        # Save a copy of the messy audio for our graphs and .wav files
        self._raw_sentence_buffer.append(chunk)
        self._session_raw_history.append(chunk)
        
        # Denoise the chunk before VAD so the VAD isn't tricked by fan noise
        chunk = self.denoiser.process(chunk)
        
        # Add the clean audio to the VAD holding tank
        self._vad_buffer = np.concatenate((self._vad_buffer, chunk))
        should_finalize = False
        
        # Slicing Engine: VAD strictly requires 512 samples. Loop and slice until the tank is empty.
        while len(self._vad_buffer) >= _VAD_WINDOW_SAMPLES:
            window = self._vad_buffer[:_VAD_WINDOW_SAMPLES]
            self._vad_buffer = self._vad_buffer[_VAD_WINDOW_SAMPLES:]
            
            vad_result = self.vad(window)
            if vad_result is not None and "end" in vad_result:
                should_finalize = True

        chunk = apply_high_pass_filter(chunk, _SAMPLE_RATE)
        
        # If the volume is under 0.5% (dead silence) and the sentence isn't ending, throw it away
        if _rms(chunk) < 0.005 and not should_finalize:
            total = sum(len(c) for c in self._buffer)
            # Hard limit: Force a finalization if someone screams for 30 seconds straight
            if total > _BUFFER_MAX_SAMPLES:
                return await self._finalize()
            return None

        self._buffer.append(chunk)
        self._samples_since_partial += len(chunk)

        # Trigger logic
        if should_finalize:
            return await self._finalize()

        if self._samples_since_partial >= _SAMPLE_RATE * _PARTIAL_EVERY_S:
            self._samples_since_partial = 0
            return await self._partial()

        return None

    async def flush(self) -> dict | None:
        """
        Called when the WebSocket disconnects or the user hits 'Stop'.
        Forces the final transcription and dumps the entire session history to .wav files.
        """
        # Save the full 1-hour session to the hard drive in a background thread
        asyncio.create_task(asyncio.to_thread(
            self.recorder.save_session, 
            self._session_raw_history, 
            self._session_clean_history
        ))

        # Transcribe any leftover audio
        if self._buffer:
            return await self._finalize()
        return None

    def _audio(self) -> np.ndarray:
        """ Helper: Glues the clean audio buffer list into a single flat array """
        return np.concatenate(self._buffer) if self._buffer else np.array([], dtype=np.float32)

    def _raw_audio(self) -> np.ndarray:
        """ Helper: Glues the raw audio buffer list into a single flat array """
        return np.concatenate(self._raw_sentence_buffer) if self._raw_sentence_buffer else np.array([], dtype=np.float32)

    async def _partial(self) -> dict | None:
        """
        Triggers every 1 second. Uses beam_size=1 for instantaneous, "live-typing" speed.
        Does NOT erase the buffer.
        """
        audio = self._audio()
        if len(audio) < _SAMPLE_RATE * _MIN_CHUNK_S:
            return None
        
        context = " ".join(self._finalized)[-200:] if self._finalized else None
        
        # --- BENCHMARK STOPWATCH ---
        start_time = time.time()
        text = await asyncio.to_thread(transcribe_audio_chunk, audio, beam_size=1, initial_prompt=context)
        latency_ms = (time.time() - start_time) * 1000
        print(f"⚡ [Benchmark] Partial Latency: {latency_ms:.1f} ms")
        # ---------------------------

        if not text: return None
        return {"type": "partial", "text": text}


    async def _finalize(self) -> dict | None:
        """
        Triggers when the VAD detects 700ms of silence. Uses beam_size=5 for perfect accuracy.
        Erases the buffer and AI memory for the next sentence, and launches visualization graphs.
        """
        audio = self._audio()
        raw_audio = self._raw_audio()
        
        # Save this perfect sentence to our permanent session recording log
        self._session_clean_history.append(audio)
        
        # 1. Erase all buffers and stopwatch instantly for the next sentence
        self._buffer = []
        self._raw_sentence_buffer = []
        self._samples_since_partial = 0
        
        # 2. Erase the short-term memory of the AI networks to prevent hallucination looping
        self.vad.reset_states()
        self.denoiser.reset_state()

        if len(audio) < _SAMPLE_RATE * _MIN_CHUNK_S:
            return None

        # 3. Highly-accurate transcription
        context = " ".join(self._finalized)[-200:] if self._finalized else None
        
        # --- BENCHMARK STOPWATCH ---
        start_time = time.time()
        text = await asyncio.to_thread(transcribe_audio_chunk, audio, beam_size=5, initial_prompt=context)
        latency_ms = (time.time() - start_time) * 1000
        
        audio_duration_sec = len(audio) / _SAMPLE_RATE
        rtf = (latency_ms / 1000) / audio_duration_sec if audio_duration_sec > 0 else 0
        
        print(f"🔥 [Benchmark] Final Latency: {latency_ms:.1f} ms | Audio Length: {audio_duration_sec:.1f}s | RTF: {rtf:.2f}x")
        # ---------------------------

        if not text: return None

        self._finalized.append(text)

        
        # 4. Generate the Diagnostic PNG Graphs in the background without causing lag
        if ENABLE_VISUALIZATIONS:
            asyncio.create_task(asyncio.to_thread(
                self.exporter.generate_graphs, 
                raw_audio, 
                audio, 
                text
            ))

        return {
            "type": "final",
            "text": text,
            "full_transcript": " ".join(self._finalized),
        }
