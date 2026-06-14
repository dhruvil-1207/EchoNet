import numpy as np

from audio_pipeline.denoising.noise_reducer import (
    reduce_noise_spectral
)

# from audio_pipeline.denoising.rnnoise_reducer import (
#     reduce_noise_rnnoise
# )


def reduce_noise_auto(
    data,
    sample_rate,
    use_rnnoise=False
):

    return reduce_noise_spectral(
        data,
        sample_rate
    )