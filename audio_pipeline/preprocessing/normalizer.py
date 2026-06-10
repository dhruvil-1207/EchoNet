import numpy as np

def normalize_audio(data):
    max_val = np.max(np.abs(data))

    if max_val > 0:
        data = data / max_val

    return data