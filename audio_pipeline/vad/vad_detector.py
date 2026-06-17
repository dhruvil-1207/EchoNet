from silero_vad import load_silero_vad, VADIterator
import torch

# Force CPU strictly to prevent PyTorch state crashes. It is incredibly fast.
device = torch.device('cpu')
model = load_silero_vad()
if hasattr(model, 'to'):
    model = model.to(device)

class StreamingVAD:
    """ Used by preocessing_service.py for live WebSocket streaming """
    def __init__(self, sample_rate=16000, threshold=0.5, min_silence_duration_ms=700, speech_pad_ms=100):
        #speech_pad_ms is the extra ms we add to the speech to make sure we capture the whole word (for example, when we speak cats, 's' is very silent so we might miss it)
        self.iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )
        
    def __call__(self, window):
        return self.iterator(window)

    def reset_states(self):
        self.iterator.reset_states()
