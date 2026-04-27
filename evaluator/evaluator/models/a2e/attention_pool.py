"""Attention pooling audio embedding model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from ..base import AudioEmbeddingModel
from ..registry import register_audio_embedding_model


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence embeddings."""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(AttentionPooling, self).__init__()
        # These are defined in original APM but not used - kept for checkpoint compatibility
        self.w1 = nn.Linear(input_dim, output_dim)
        self.w2 = nn.Linear(output_dim, 1)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1),
        )

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, input_dim)
        scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch_size, input_dim)
        return pooled


class ProjectionHead(nn.Module):
    """Projection head to map pooled features to embedding space."""
    
    def __init__(self, input_dim, emb_dim, dropout=0.1):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.proj(x)


@register_audio_embedding_model('attention_pool', default_name='openai/whisper-large', description='Attention pooling audio embedding model')
class AttentionPoolAudioModel(AudioEmbeddingModel):
    """
    Audio embedding model using attention pooling architecture.
    Compatible with pre-trained APM models from attention_pool_model/.
    """

    @dataclass
    class Params:
        size: str = "large"
        emb_dim: int = 2048
        dropout: float = 0.1
        SIZES: ClassVar[Dict[str, str]] = {
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
        }

    def __init__(self,
                 audio_encoder_name: str = "openai/whisper-large",
                 emb_dim: int = 2048,
                 model_path: Optional[str] = None,
                 dropout: float = 0.1):
        """
        Initialize attention pool audio model.
        
        Args:
            audio_encoder_name: Name of the audio encoder (e.g., "openai/whisper-large")
            emb_dim: Embedding dimension (should match text embedder)
            model_path: Path to pre-trained APM model weights (.pt file)
            dropout: Dropout rate for attention pooling and projection
        """
        from transformers import WhisperModel, WhisperFeatureExtractor
        
        self.audio_encoder_name = audio_encoder_name
        self.emb_dim = emb_dim
        self.model_path = model_path
        self.device = torch.device("cpu")
        
        # Initialize audio encoder
        whisper_model = WhisperModel.from_pretrained(audio_encoder_name)
        self.audio_encoder = whisper_model.get_encoder()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        
        self.hidden_dim = 1280  # Whisper encoder hidden dim
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(audio_encoder_name)
        
        # Initialize attention pooling and projection
        self.attention_pool = AttentionPooling(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=dropout
        )
        self.projection_head = ProjectionHead(
            input_dim=self.hidden_dim,
            emb_dim=emb_dim,
            dropout=dropout
        )
        
        # Load pre-trained weights if provided
        if model_path:
            self._load_weights(model_path)
    
    def _load_weights(self, model_path: str):
        """Load pre-trained APM weights."""
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Extract weights for attention pooling and projection head
        # APM checkpoint has keys: audio_enc.*, attn_pool.*, proj.*
        attn_pool_state = {}
        proj_head_state = {}
        
        for key, value in state_dict.items():
            if key.startswith('attn_pool.'):
                new_key = key.replace('attn_pool.', '')
                attn_pool_state[new_key] = value
            elif key.startswith('proj.proj.'):
                # Strip 'proj.' prefix to get 'proj.0.weight' etc.
                new_key = key.replace('proj.', '', 1)
                proj_head_state[new_key] = value
        
        # Load weights
        if attn_pool_state:
            self.attention_pool.load_state_dict(attn_pool_state, strict=True)
        if proj_head_state:
            self.projection_head.load_state_dict(proj_head_state, strict=True)
    
    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.audio_encoder.to(device)
        self.attention_pool.to(device)
        self.projection_head.to(device)
        return self
    
    def preprocess_audio(self, audio_list: List[torch.Tensor], 
                        sampling_rates: List[int]):
        """Preprocess audio using Whisper feature extractor."""
        processed_audio = []
        
        for idx in range(len(audio_list)):
            if not isinstance(audio_list[idx], torch.Tensor):
                audio_list[idx] = torch.tensor(audio_list[idx])
            
            sr = sampling_rates[idx]
            audio_sample = audio_list[idx]
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio_list[idx] = torchaudio.functional.resample(audio_sample, sr, 16000)
                sampling_rates[idx] = 16000
            
            processed_audio.append(audio_list[idx].squeeze().numpy())
        
        # Extract features
        inputs = self.feature_extractor(
            processed_audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        return inputs.input_features, inputs.attention_mask
    
    def encode_from_features(self, features: torch.Tensor, 
                           attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Encode preprocessed features into embeddings."""
        features = features.to(self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        
        with torch.no_grad():
            self.audio_encoder.eval()
            self.attention_pool.eval()
            self.projection_head.eval()
            
            # Encode audio
            encoder_outputs = self.audio_encoder(
                input_features=features,
                attention_mask=attention_mask
            )
            enc_hidden = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            
            # Adjust mask for downsampled sequence length
            if attention_mask is not None:
                attention_mask = attention_mask[:, ::2]
            
            # Apply attention pooling
            pooled = self.attention_pool(enc_hidden, attention_mask)  # (batch, hidden_dim)
            
            # Project to embedding space
            embeddings = self.projection_head(pooled)  # (batch, emb_dim)
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        torch.cuda.empty_cache()
        return embeddings.cpu().numpy()
    
    def encode_audio(self, audio_list: List[torch.Tensor], 
                    sampling_rates: List[int], 
                    show_progress: bool = False) -> np.ndarray:
        """Encode audio directly into embeddings."""
        features, attention_mask = self.preprocess_audio(audio_list, sampling_rates)
        return self.encode_from_features(features, attention_mask)
    
    def name(self) -> str:
        """Return model name."""
        name = f"AttentionPoolAudioModel - encoder:{self.audio_encoder_name} - emb_dim:{self.emb_dim}"
        if self.model_path:
            name += f" - weights:{self.model_path}"
        return name
