import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import noisereduce as nr

def apply_high_pass_filter(input_path, output_path):
    # Read the raw array numbers from disk
    sample_rate, data = wavfile.read(input_path)

    # Force a 1D vector matrix if it's stereo
    if len(data.shape) > 1:
        data = data[:, 0]

    b, a = signal.butter(N=4, Wn=(80.0 / (0.5 * sample_rate)), btype="high")

    # Run the filter over the data array numbers
    filtered_data = signal.filtfilt(b, a, data) 
    
    denoised_data = nr.reduce_noise(
        y=filtered_data, 
        sr=sample_rate,
        prop_decrease=0.85,            # Leaves 15% noise floor so voice transitions sound natural
        n_std_thresh_stationary=1.5,   # Raises threshold to protect softer speech volumes
        time_mask_smooth_ms=200         # 200ms smoothing window to prevent aggressive clamping
    )

    # Cast the numbers safely back to 16-bit integers
    final_array = np.clip(denoised_data, -32768, 32767).astype(np.int16)


    # Flush the array to your enhanced file
    wavfile.write(output_path, sample_rate, final_array)
