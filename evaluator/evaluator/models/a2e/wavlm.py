"""WavLM audio embedding model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from ..base import AudioEmbeddingModel
from ..registry import register_audio_embedding_model


@register_audio_embedding_model(
    'wavlm',
    default_name='microsoft/wavlm-base',
    description='WavLM self-supervised audio representation model from Microsoft'
)
class WavLMAudioModel(AudioEmbeddingModel):

    @dataclass
    class Params:
        size: str = "base"
        pooling: str = "mean"
        SIZES: ClassVar[Dict[str, str]] = {
            "base": "microsoft/wavlm-base",
            "base-plus": "microsoft/wavlm-base-plus",
            "large": "microsoft/wavlm-large",
        }
    """
    Audio embedding model using WavLM from Microsoft.
    
    WavLM is a self-supervised speech representation learning model that
    achieves state-of-the-art performance on various speech tasks.
    
    Available models:
        - microsoft/wavlm-base: Base model (768 hidden dim)
        - microsoft/wavlm-base-plus: Base+ model with improved training
        - microsoft/wavlm-large: Large model (1024 hidden dim)
    """
    
    def __init__(self,
                 model_name: str = "microsoft/wavlm-base",
                 pooling: str = "mean"):
        """
        Initialize WavLM audio model.
        
        Args:
            model_name: HuggingFace model name (e.g., "microsoft/wavlm-base")
            pooling: Pooling strategy for hidden states ("mean" or "cls")
        """
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor
        
        self.model_name = model_name
        self.pooling = pooling
        self.device = torch.device("cpu")
        
        # Initialize WavLM model
        self.model = WavLMModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # WavLM uses Wav2Vec2FeatureExtractor for preprocessing
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Get hidden dimension from model config
        self.hidden_dim = self.model.config.hidden_size
    
    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.model.to(device)
        return self
    
    def preprocess_audio(self, audio_list: List[torch.Tensor],
                        sampling_rates: List[int]):
        """
        Preprocess audio using WavLM feature extractor.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: Corresponding sampling rates
            
        Returns:
            Tuple of (input_values, attention_mask)
        """
        processed_audio = []
        
        for idx in range(len(audio_list)):
            audio = audio_list[idx]
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)
            
            sr = sampling_rates[idx]
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            
            # Convert to numpy and ensure 1D
            audio_np = audio.squeeze().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)
            
            processed_audio.append(audio_np)
        
        # Extract features
        inputs = self.feature_extractor(
            processed_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        return inputs.input_values, inputs.attention_mask
    
    def encode_from_features(self, features: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Encode preprocessed features into embeddings.
        
        Args:
            features: Input values from feature extractor
            attention_mask: Optional attention mask
            
        Returns:
            Array of embeddings with shape (batch_size, hidden_dim)
        """
        features = features.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            
            # Forward pass through WavLM
            outputs = self.model(
                input_values=features,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            
            # Pool hidden states
            if self.pooling == "mean":
                if attention_mask is not None:
                    # Create mask for hidden states (may have different seq length)
                    hidden_seq_len = hidden_states.size(1)
                    # WavLM downsamples like Wav2Vec2 (conv layers)
                    # Compute output sequence length
                    output_lengths = self._get_feat_extract_output_lengths(
                        attention_mask.sum(dim=-1)
                    )
                    # Create mask for pooling
                    pooling_mask = torch.zeros(
                        hidden_states.size(0), hidden_seq_len,
                        device=self.device, dtype=torch.bool
                    )
                    for i, length in enumerate(output_lengths):
                        pooling_mask[i, :length] = True
                    
                    # Masked mean pooling
                    hidden_states_masked = hidden_states * pooling_mask.unsqueeze(-1)
                    embeddings = hidden_states_masked.sum(dim=1) / pooling_mask.sum(dim=1, keepdim=True)
                else:
                    embeddings = hidden_states.mean(dim=1)
            else:  # cls - use first token
                embeddings = hidden_states[:, 0, :]
            
            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        torch.cuda.empty_cache()
        return embeddings.cpu().numpy()
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute output lengths after feature extraction (conv layers).
        
        WavLM uses the same conv architecture as Wav2Vec2.
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1
        
        # WavLM has conv layers with kernel_sizes and strides from config
        # Default: [(512,10,5), (512,3,2), (512,3,2), (512,3,2), (512,3,2), (512,2,2), (512,2,2)]
        conv_kernel = getattr(self.model.config, 'conv_kernel', [10, 3, 3, 3, 3, 2, 2])
        conv_stride = getattr(self.model.config, 'conv_stride', [5, 2, 2, 2, 2, 2, 2])
        
        for kernel_size, stride in zip(conv_kernel, conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        
        return input_lengths.to(torch.long)
    
    def encode_audio(self, audio_list: List[torch.Tensor],
                    sampling_rates: List[int],
                    show_progress: bool = False) -> np.ndarray:
        """
        Encode audio directly into embeddings.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: Corresponding sampling rates
            show_progress: Whether to show progress bar (not used)
            
        Returns:
            Array of embeddings with shape (N, hidden_dim)
        """
        features, attention_mask = self.preprocess_audio(audio_list, sampling_rates)
        return self.encode_from_features(features, attention_mask)
    
    def name(self) -> str:
        """Return model name."""
        return f"WavLMAudioModel - model:{self.model_name} - pooling:{self.pooling} - dim:{self.hidden_dim}"
