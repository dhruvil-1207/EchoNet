import scipy.signal as signal
from backend.utils.profiler import profile


HIGH_PASS_CUTOFF = 80

@profile
def apply_high_pass_filter(data, sample_rate):
    nyquist = 0.5 * sample_rate

    b, a = signal.butter(
        N=4,
        Wn=HIGH_PASS_CUTOFF / nyquist,
        btype="high"
    )

    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data