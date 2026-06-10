import noisereduce as nr

NOISE_REDUCTION_STRENGTH = 0.75
# Remove 75% of detected noise

def reduce_noise(data, sample_rate):
    data = nr.reduce_noise(
        y=data,
        sr=sample_rate,
        prop_decrease=NOISE_REDUCTION_STRENGTH,
        n_std_thresh_stationary=1.5,
        time_mask_smooth_ms=200
    )

    return data