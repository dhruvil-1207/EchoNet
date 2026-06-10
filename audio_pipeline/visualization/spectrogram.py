import matplotlib.pyplot as plt


def save_spectrogram(data,sample_rate,output_path,title="Audio Spectrogram"):
    plt.figure(figsize=(12, 4))

    plt.specgram( data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="viridis")

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Spectrogram saved to: {output_path}")