from silero_vad import load_silero_vad
from silero_vad import get_speech_timestamps
import numpy as np

model = load_silero_vad()


def detect_speech_segments(data, sample_rate):

    data = np.ascontiguousarray(
        data,
        dtype=np.float32
    )

    speech_segments = get_speech_timestamps(
        data,
        model,
        sampling_rate=sample_rate
    )

    return speech_segments