import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import sys
import types

# --- Compatibility Patch for DeepFilterNet vs TorchAudio 2.2+ ---
if not hasattr(torchaudio, 'backend'):
    # Create a completely empty dummy class to satisfy the broken import
    class DummyAudioMetaData:
        pass
    sys.modules['torchaudio.backend'] = types.ModuleType('torchaudio.backend')
    sys.modules['torchaudio.backend.common'] = types.ModuleType('torchaudio.backend.common')
    sys.modules['torchaudio.backend.common'].AudioMetaData = DummyAudioMetaData
# ----------------------------------------------------------------

from df.enhance import init_df, enhance

class StreamingDeepFilter:
    def __init__(self):
        # Initialize DeepFilterNet model and its streaming memory state
        self.model, self.df_state, _ = init_df()
        
        # Fast PyTorch resampling modules
        self.resample_up = T.Resample(orig_freq=16000, new_freq=48000)
        self.resample_down = T.Resample(orig_freq=48000, new_freq=16000)
        
    def process(self, chunk_16k_np: np.ndarray) -> np.ndarray:
        """
        Takes raw 16kHz audio, denoises it through DeepFilterNet, 
        and returns clean 16kHz audio.
        """
        # Convert numpy array to PyTorch tensor shape (1, N)
        tensor_16k = torch.from_numpy(chunk_16k_np).unsqueeze(0).to(torch.float32)
        
        # 1. Upsample to 48kHz for DeepFilterNet
        tensor_48k = self.resample_up(tensor_16k)
        
        # 2. Scrub noise (This automatically updates df_state for continuous streaming)
        enhanced_48k = enhance(self.model, self.df_state, tensor_48k, atten_lim_db=100)
        
        # 3. Downsample back to 16kHz for VAD and Whisper
        enhanced_16k = self.resample_down(enhanced_48k)
        
        # Return as flat 1D numpy array
        return enhanced_16k.squeeze(0).numpy()

    def reset_state(self):
        """ Reset the noise memory profile when a sentence ends """
        _, self.df_state, _ = init_df()
