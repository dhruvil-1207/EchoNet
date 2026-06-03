import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import noisereduce as nr


def apply_high_pass_filter(input_path, output_path):
    # Read audio
    sample_rate, data = wavfile.read(input_path)

    # Convert stereo to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Convert to float32
    data = data.astype(np.float32)

    # Normalize audio
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    # High-pass filter (remove rumble below ~80Hz)
    cutoff = 80.0
    nyquist = 0.5 * sample_rate

    b, a = signal.butter(
        N=4,
        Wn=cutoff / nyquist,
        btype="high"
    )

    filtered_data = signal.filtfilt(b, a, data)

    # Noise reduction
    denoised_data = nr.reduce_noise(
        y=filtered_data,
        sr=sample_rate,
        prop_decrease=0.75,
        n_std_thresh_stationary=1.5,
        time_mask_smooth_ms=200
    )

    # Prevent clipping
    denoised_data = np.clip(denoised_data, -1.0, 1.0)

    # Convert back to int16 WAV
    final_audio = (denoised_data * 32767).astype(np.int16)

    # Save
    wavfile.write(
        output_path,
        sample_rate,
        final_audio
    )

    print(f"Enhanced audio saved to: {output_path}")