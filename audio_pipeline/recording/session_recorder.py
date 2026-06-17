import os
import datetime
import numpy as np
from scipy.io import wavfile

class SessionRecorder:
    """
    Handles background saving of complete audio sessions to disk.
    Ensures that streaming performance is never blocked by slow hard-drive I/O operations.
    """
    def __init__(self, session_timestamp: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.session_timestamp = session_timestamp
        self.recordings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)

    def save_session(self, raw_history: list[np.ndarray], clean_history: list[np.ndarray]):
        """
        Takes the complete memory history of a session and saves them as .wav files.
        This function is strictly designed to be called within an asyncio.to_thread background worker.
        """
        if not raw_history and not clean_history:
            return

        # Generate a unified timestamp for this specific session
        timestamp = self.session_timestamp
        
        # 1. Save the complete Raw Microphone Audio
        if raw_history:
            raw_audio = np.concatenate(raw_history)
            raw_path = os.path.join(self.recordings_dir, f"{timestamp}_session_raw.wav")
            wavfile.write(raw_path, self.sample_rate, raw_audio)
            print(f"[SessionRecorder] Saved raw audio: {raw_path}")

        # 2. Save the perfectly cleaned, VAD-filtered Audio
        if clean_history:
            clean_audio = np.concatenate(clean_history)
            clean_path = os.path.join(self.recordings_dir, f"{timestamp}_session_clean_vad.wav")
            wavfile.write(clean_path, self.sample_rate, clean_audio)
            print(f"[SessionRecorder] Saved clean audio: {clean_path}")
