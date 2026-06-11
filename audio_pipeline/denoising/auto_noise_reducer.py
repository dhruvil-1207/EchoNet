import numpy as np

from audio_pipeline.denoising.noise_reducer import (
    reduce_noise_spectral
)

from audio_pipeline.denoising.rnnoise_reducer import (
    reduce_noise_rnnoise
)


def estimate_snr(data):

    noise_sample = data[:8000]  # first 0.5 sec at 16kHz

    noise_power = np.mean(noise_sample ** 2)
    signal_power = np.mean(data ** 2)

    if noise_power == 0:
        return 100

    snr = 10 * np.log10(
        signal_power / noise_power
    )

    return snr


def reduce_noise_auto(
    data,
    sample_rate,
    use_rnnoise=False
):

    if use_rnnoise:

        print("Using RNNoise")

        return reduce_noise_rnnoise(
            data,
            sample_rate
        )

    print("Using Spectral Denoising")

    return reduce_noise_spectral(
        data,
        sample_rate
    )