def extract_speech_chunks(
    data,
    speech_segments
):
    chunks = []

    for segment in speech_segments:

        start_sample = segment["start"]
        end_sample = segment["end"]

        chunk = data[
            start_sample:end_sample
        ]

        chunks.append(chunk)

    return chunks