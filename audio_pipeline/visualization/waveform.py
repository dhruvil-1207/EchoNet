import matplotlib.pyplot as plt
import numpy as np
import os


def save_waveform( data, sample_rate, output_path, title="Audio Waveform"):
    time = np.arange(len(data)) / sample_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time, data)

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"Waveform saved to: {output_path}")