from pyrnnoise import RNNoise
import numpy as np
import scipy.signal as signal
import traceback

rnnoise = RNNoise(sample_rate=48000)

def reduce_noise_rnnoise(data, sample_rate, strength=0.75):
    print(f"Input Samples (at {sample_rate}Hz): {len(data)}")
    print(f"Input dtype: {data.dtype}")

    try:
        data_float = data.astype(np.float32)

        if sample_rate != 48000:
            data_48k = signal.resample_poly(data_float, 48000, sample_rate)
        else:
            data_48k = data_float

        denoised_frames = []

        for prob, frame in rnnoise.denoise_chunk(data_48k, partial=True):
            denoised_frames.append(frame)

        output_48k = np.concatenate(denoised_frames, axis=0).reshape(-1)
        output_normalized_48k = output_48k.astype(np.float32) / 32768.0

        if sample_rate != 48000:
            denoised = signal.resample_poly(output_normalized_48k, sample_rate, 48000)
        else:
            denoised = output_normalized_48k

        min_len = min(len(data_float), len(denoised))
        original = data_float[:min_len]
        denoised = denoised[:min_len]

        # 0.0 = original audio, 1.0 = full RNNoise
        output = (1.0 - strength) * original + strength * denoised

        return np.clip(output, -1.0, 1.0).astype(np.float32)

    except Exception as e:
        print("\nRNNOISE CRASH:")
        traceback.print_exc()
        raise e