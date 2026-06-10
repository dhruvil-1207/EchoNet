from silero_vad import load_silero_vad
from silero_vad import get_speech_timestamps


model = load_silero_vad()


def detect_speech_segments(data, sample_rate):

    speech_segments = get_speech_timestamps(
        data,
        model,
        sampling_rate=sample_rate
    )

    return speech_segments