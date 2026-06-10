import numpy as np

def convert_to_mono(data):
    if len(data.shape) > 1:
        return np.mean(data, axis=1)

    return data