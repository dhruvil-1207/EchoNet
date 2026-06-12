import numpy as np

from audio_pipeline.denoising.noise_reducer import (
    reduce_noise_spectral
)

from audio_pipeline.denoising.rnnoise_reducer import (
    reduce_noise_rnnoise
)


def reduce_noise_auto(
    data,
    sample_rate,
    use_rnnoise=False,
    rnnoise_strength=0.75
):
    if use_rnnoise:
        print("Using RNNoise")
        return reduce_noise_rnnoise(
            data,
            sample_rate,
            strength=rnnoise_strength
        )

    print("Using Spectral Denoising")
    return reduce_noise_spectral(
        data,
        sample_rate
    )