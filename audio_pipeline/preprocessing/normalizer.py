import numpy as np

def normalize_audio(data):
    data = data.astype(np.float32) / 32767.0

    return data