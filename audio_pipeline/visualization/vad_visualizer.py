import matplotlib.pyplot as plt
import numpy as np


def save_vad_visualization(
    data,
    sample_rate,
    speech_segments,
    output_path,
    title="VAD Detection"
):

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    time = np.arange(len(data)) / sample_rate

    plt.figure(figsize=(12, 4))

    plt.plot(
        time,
        data,
        linewidth=0.8
    )

    for segment in speech_segments:

        start_time = (
            segment["start"] / sample_rate
        )

        end_time = (
            segment["end"] / sample_rate
        )

        plt.axvspan(
            start_time,
            end_time,
            alpha=0.3
        )

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(
        f"VAD visualization saved to: {output_path}"
    )