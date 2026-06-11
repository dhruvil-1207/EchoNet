import numpy as np
import scipy.io.wavfile as wavfile

from audio_pipeline.preprocessing.mono_converter import convert_to_mono
from audio_pipeline.preprocessing.normalizer import normalize_audio
from audio_pipeline.preprocessing.high_pass_filter import apply_high_pass_filter
from audio_pipeline.denoising.auto_noise_reducer import reduce_noise_auto
from audio_pipeline.visualization.waveform import save_waveform
from audio_pipeline.visualization.spectrogram import save_spectrogram

from audio_pipeline.vad.vad_detector import detect_speech_segments
from audio_pipeline.visualization.vad_visualizer import save_vad_visualization

from backend.utils.profiler import profile

@profile
def enhance_audio(input_path, output_path, generate_visualizations=False):

    sample_rate, data = wavfile.read(input_path)

    if generate_visualizations:
        save_waveform(
            data,
            sample_rate,
            "reports/waveforms/01_original.png",
            "Original Audio"
        )

        save_spectrogram(
            data,
            sample_rate,
            "reports/spectrograms/01_original.png",
            "Original Audio"
        )

    data = convert_to_mono(data)

    if generate_visualizations:
        save_waveform(
            data,
            sample_rate,
            "reports/waveforms/02_mono.png",
            "Mono Audio"
        )

        save_spectrogram(
            data,
            sample_rate,
            "reports/spectrograms/02_mono.png",
            "Mono Audio"
        )

    data = data.astype(np.float32)

    data = normalize_audio(data)

    if generate_visualizations:
        save_waveform(
            data,
            sample_rate,
            "reports/waveforms/03_normalized.png",
            "Normalized Audio"
        )

        save_spectrogram(
            data,
            sample_rate,
            "reports/spectrograms/03_normalized.png",
            "Normalized Audio"
        )

    print("Starting high-pass filter...")
    data = apply_high_pass_filter(data, sample_rate)
    print("High-pass filter completed")

    if generate_visualizations:
        save_waveform(
            data,
            sample_rate,
            "reports/waveforms/04_highpass.png",
            "High Pass Filter Output"
        )

        save_spectrogram(
            data,
            sample_rate,
            "reports/spectrograms/04_highpass.png",
            "High Pass Filter Output"
        )

    print("INPUT mean abs =", np.mean(np.abs(data)))
    print("Starting noise reduction...")
    data = reduce_noise_auto(data, sample_rate, use_rnnoise=False)
    print("Noise reduction completed")


    if generate_visualizations:
        save_waveform(
            data,
            sample_rate,
            "reports/waveforms/05_denoised.png",
            "Denoised Audio"
        )

        save_spectrogram(
            data,
            sample_rate,
            "reports/spectrograms/05_denoised.png",
            "Denoised Audio"
        )

    print("\nVAD INPUT INFO")
    print("dtype:", data.dtype)
    print("shape:", data.shape)
    print("min:", np.min(data))
    print("max:", np.max(data))
    print("mean abs:", np.mean(np.abs(data)))

    # Pass the normalized float32 directly to Silero VAD without scaling it down
    vad_audio = data.astype(np.float32)

    speech_segments = detect_speech_segments(
        vad_audio,
        sample_rate
    )

    print("Speech Segments:")
    print(speech_segments)

    if generate_visualizations:
        save_vad_visualization(
            data,
            sample_rate,
            speech_segments,
            "reports/vad/vad_detection.png",
            "VAD Detection"
        )

    print(type(data))
    print(data.dtype)
    print("\nBEFORE FINAL CONVERSION")
    print("dtype:", data.dtype)
    print("min:", np.min(data))
    print("max:", np.max(data))
    print("mean abs:", np.mean(np.abs(data)))
    if data.dtype != np.int16:

        data = np.clip(
            data,
            -1.0,
            1.0
        )

        data = (
            data * 32767
        ).astype(np.int16)
      
    print("\nAFTER FINAL CONVERSION")
    print("dtype:", data.dtype)
    print("min:", np.min(data))
    print("max:", np.max(data))
    print("mean abs:", np.mean(np.abs(data)))

    print("Before WAV Save:")
    print(
        data.dtype,
        np.min(data),
        np.max(data)
    )
    wavfile.write(
        output_path,
        sample_rate,
        data
    )

    print(f"Enhanced audio saved to: {output_path}")

    return speech_segments
