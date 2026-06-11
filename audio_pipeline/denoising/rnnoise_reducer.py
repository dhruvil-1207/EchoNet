from pyrnnoise import RNNoise
import numpy as np
import scipy.signal as signal
import traceback

rnnoise = RNNoise(sample_rate=48000)

def reduce_noise_rnnoise(data, sample_rate):
    print(f"Input Samples (at {sample_rate}Hz): {len(data)}")
    print(f"Input dtype: {data.dtype}")

    try:
        # 1. Upsample 16kHz to 48kHz (Factor of 3)
        data_float = data.astype(np.float32)
        data_48k = signal.resample_poly(data_float, 3, 1)

        denoised_frames = []

        # 2. Process with RNNoise at 48kHz
        for prob, frame in rnnoise.denoise_chunk(data_48k, partial=True):
            denoised_frames.append(frame)

        print(f"Frames Produced (at 48kHz): {len(denoised_frames)}")

        output_48k = np.concatenate(
            denoised_frames,
            axis=0
        ).reshape(-1)

        # 3. Convert int16 frames to normalized float32 range [-1.0, 1.0]
        output_normalized_48k = output_48k.astype(np.float32) / 32768.0

        # 4. Downsample back to 16kHz (Factor of 3)
        output_16k = signal.resample_poly(output_normalized_48k, 1, 3)

        print("\nRNNOISE OUTPUT (NORMALIZED 16kHz)")
        print("dtype:", output_16k.dtype)
        print("min:", np.min(output_16k))
        print("max:", np.max(output_16k))
        print("mean abs:", np.mean(np.abs(output_16k)))

        return output_16k

    except Exception as e:
        print("\nRNNOISE CRASH:")
        traceback.print_exc()
        raise e
