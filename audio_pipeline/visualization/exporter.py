import os
import re
import datetime
import numpy as np

# Import your existing visualization tools
from audio_pipeline.visualization.waveform import save_waveform
from audio_pipeline.visualization.spectrogram import save_spectrogram
from audio_pipeline.visualization.vad_visualizer import save_vad_visualization

class DiagnosticExporter:
    def __init__(self, session_timestamp: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.session_timestamp = session_timestamp
        
        # Base reports directory
        self.reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")

        # Dynamically create the session-specific subfolders!
        self.waveforms_dir = os.path.join(self.reports_dir, "waveforms", self.session_timestamp)
        self.spectrograms_dir = os.path.join(self.reports_dir, "spectrograms", self.session_timestamp)
        self.vad_dir = os.path.join(self.reports_dir, "vad", self.session_timestamp)
        
        os.makedirs(self.waveforms_dir, exist_ok=True)
        os.makedirs(self.spectrograms_dir, exist_ok=True)
        os.makedirs(self.vad_dir, exist_ok=True)

    def generate_graphs(self, raw_audio: np.ndarray, clean_audio: np.ndarray, text: str):
        if len(raw_audio) == 0 or len(clean_audio) == 0:
            return

        # Generate a unique timestamp for this specific sentence
        # Force the individual sentence graphs to be in IST too!
        ist_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)
        sentence_time = ist_now.strftime("%H%M%S")
        safe_text = re.sub(r'[^a-zA-Z0-9]', '_', text[:20].strip())
        
        # 1. Map paths into their specific folders
        raw_wave_path = os.path.join(self.waveforms_dir, f"{sentence_time}_{safe_text}_raw.png")
        clean_wave_path = os.path.join(self.waveforms_dir, f"{sentence_time}_{safe_text}_clean.png")
        
        raw_spec_path = os.path.join(self.spectrograms_dir, f"{sentence_time}_{safe_text}_raw.png")
        clean_spec_path = os.path.join(self.spectrograms_dir, f"{sentence_time}_{safe_text}_clean.png")
        
        vad_path = os.path.join(self.vad_dir, f"{sentence_time}_{safe_text}_vad.png")
        
        # 2. Command the graphing tools
        save_waveform(raw_audio, self.sample_rate, raw_wave_path, title=f"Raw Mic: {text[:30]}")
        save_waveform(clean_audio, self.sample_rate, clean_wave_path, title=f"Denoised: {text[:30]}")
        
        save_spectrogram(raw_audio, self.sample_rate, raw_spec_path, title=f"Raw Spectrogram: {text[:30]}")
        save_spectrogram(clean_audio, self.sample_rate, clean_spec_path, title=f"Clean Spectrogram: {text[:30]}")
        
        speech_segments = [{"start": 0, "end": len(clean_audio)}]
        save_vad_visualization(clean_audio, self.sample_rate, speech_segments, vad_path, title=f"VAD Detection: {text[:30]}")
        
        print(f"[DiagnosticExporter] Saved 5 visual reports for session {self.session_timestamp}!")
