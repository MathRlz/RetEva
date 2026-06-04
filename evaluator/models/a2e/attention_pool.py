"""Attention pooling audio embedding model implementation.

The attention-pool + projection architecture is encoder-agnostic; the encoder
specifics (feature extractor, frozen encoder, hidden dim, and the post-encoder mask
downsampling) live behind a small ``_EncoderBackend`` so the same model works on a
Whisper encoder (``attention_pool``) or a SeamlessM4T-v2 encoder
(``attention_pool_m4t``). The backend is chosen from the encoder name.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple
import math
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


def _resample_to_16k(audio_list: List[torch.Tensor], sampling_rates: List[int]) -> List[np.ndarray]:
    """Resample each clip to 16 kHz and return mono numpy arrays."""
    processed = []
    for idx in range(len(audio_list)):
        audio = audio_list[idx]
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        sr = sampling_rates[idx]
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            sampling_rates[idx] = 16000
        processed.append(audio.squeeze().numpy())
    return processed


# ----------------------------------------------------------------------------
# Encoder backends — the only encoder-specific pieces.
# ----------------------------------------------------------------------------
class _EncoderBackend(ABC):
    """Feature extraction + frozen encoder + mask downsampling for one encoder."""

    hidden_dim: int
    encoder: nn.Module
    processor: object

    @abstractmethod
    def preprocess(self, audio_list, sampling_rates) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_features, attention_mask) for the encoder."""

    @abstractmethod
    def run_encoder(self, encoder, features, mask) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the (model-owned) encoder and return (hidden_states, pooled_mask).

        ``pooled_mask`` is the attention mask aligned to the encoder *output*
        sequence length (each encoder downsamples differently).
        """

    def to(self, device):
        self.encoder.to(device)

    def eval(self):
        self.encoder.eval()

    def parameters(self):
        return self.encoder.parameters()


class _WhisperBackend(_EncoderBackend):
    """Whisper encoder backend (hidden_dim 1280, fixed 2x mask downsample)."""

    def __init__(self, audio_encoder_name: str):
        from transformers import WhisperModel, WhisperFeatureExtractor

        self.hidden_dim = 1280
        self.encoder = WhisperModel.from_pretrained(audio_encoder_name).get_encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.processor = WhisperFeatureExtractor.from_pretrained(audio_encoder_name)

    def preprocess(self, audio_list, sampling_rates):
        processed = _resample_to_16k(audio_list, sampling_rates)
        inputs = self.processor(
            processed, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        return inputs.input_features, inputs.attention_mask

    def run_encoder(self, encoder, features, mask):
        outputs = encoder(input_features=features, attention_mask=mask)
        # Whisper's conv stack downsamples the frame axis by 2.
        pooled_mask = mask[:, ::2] if mask is not None else None
        return outputs.last_hidden_state, pooled_mask


class _M4TBackend(_EncoderBackend):
    """SeamlessM4T-v2 speech encoder backend (hidden_dim 1024, length-ratio mask)."""

    def __init__(self, audio_encoder_name: str):
        from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor

        self.hidden_dim = 1024
        self.processor = AutoProcessor.from_pretrained(audio_encoder_name)
        self.encoder = SeamlessM4Tv2ForSpeechToText.from_pretrained(audio_encoder_name).get_encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def preprocess(self, audio_list, sampling_rates):
        processed = _resample_to_16k(audio_list, sampling_rates)
        inputs = self.processor(
            audio=processed, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        return inputs.input_features, inputs.attention_mask

    def run_encoder(self, encoder, features, mask):
        outputs = encoder(features, attention_mask=mask)
        hidden = outputs.last_hidden_state
        if mask is None:
            return hidden, None
        # M4T downsamples by a model-specific factor: rescale the mask by the
        # output/input sequence-length ratio (verbatim from the M4T encoder).
        seq_out = hidden.shape[1]
        seq_in = mask.shape[1]
        lengths = mask.to(torch.long).sum(dim=1)
        out_lengths = torch.ceil(lengths.float() * seq_out / seq_in).to(torch.long)
        out_lengths = torch.clamp(out_lengths, min=1)
        positions = torch.arange(seq_out, device=mask.device).unsqueeze(0)
        pooled_mask = (positions < out_lengths.unsqueeze(1)).to(mask.dtype)
        return hidden, pooled_mask


def _select_backend(audio_encoder_name: str) -> _EncoderBackend:
    """Pick the encoder backend from the encoder name."""
    name = (audio_encoder_name or "").lower()
    if "seamless" in name or "m4t" in name:
        return _M4TBackend(audio_encoder_name)
    return _WhisperBackend(audio_encoder_name)


@register_audio_embedding_model('attention_pool', default_name='openai/whisper-large', description='Attention pooling audio embedding model (Whisper encoder)')
class AttentionPoolAudioModel(AudioEmbeddingModel):
    """
    Audio embedding model using attention pooling architecture.
    Compatible with pre-trained APM models from attention_pool_model/.

    The audio encoder is selected from ``audio_encoder_name``: a Whisper name uses
    the Whisper backend, a SeamlessM4T name uses the M4T backend (or use the
    dedicated ``attention_pool_m4t`` model type).
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
            audio_encoder_name: Name of the audio encoder (Whisper or SeamlessM4T).
            emb_dim: Embedding dimension (should match text embedder).
            model_path: Path to pre-trained APM model weights (.pt file).
            dropout: Dropout rate for attention pooling and projection.
        """
        self.audio_encoder_name = audio_encoder_name
        self.emb_dim = emb_dim
        self.model_path = model_path
        self.device = torch.device("cpu")

        # Encoder-specific backend (feature extractor + frozen encoder + mask logic).
        self.backend = _select_backend(audio_encoder_name)
        self.hidden_dim = self.backend.hidden_dim
        # Back-compat aliases: tests / external code reference these directly.
        self.audio_encoder = self.backend.encoder
        self.feature_extractor = self.backend.processor

        # Attention pooling and projection operate at the encoder's hidden dim.
        self.attention_pool = AttentionPooling(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.projection_head = ProjectionHead(
            input_dim=self.hidden_dim,
            emb_dim=emb_dim,
            dropout=dropout,
        )

        if model_path:
            self._load_weights(model_path)

    def _load_weights(self, model_path: str):
        """Load pre-trained APM weights (attn_pool.* and proj.proj.*)."""
        state_dict = torch.load(model_path, map_location=self.device)

        # APM checkpoint has keys: audio_enc.*, attn_pool.*, proj.* — we load only
        # the trained pooling + projection (the encoder is frozen / from_pretrained).
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

        if attn_pool_state:
            self.attention_pool.load_state_dict(attn_pool_state, strict=True)
        if proj_head_state:
            self.projection_head.load_state_dict(proj_head_state, strict=True)

    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.backend.to(device)
        self.audio_encoder = self.backend.encoder
        self.attention_pool.to(device)
        self.projection_head.to(device)
        return self

    def preprocess_audio(self, audio_list: List[torch.Tensor],
                         sampling_rates: List[int]):
        """Preprocess audio with the backend's feature extractor."""
        return self.backend.preprocess(audio_list, sampling_rates)

    def encode_from_features(self, features: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Encode preprocessed features into embeddings."""
        features = features.to(self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

        with torch.no_grad():
            self.backend.eval()
            self.attention_pool.eval()
            self.projection_head.eval()

            # Run the (model-owned) encoder; the backend returns the hidden states
            # and the mask aligned to the encoder's output length.
            enc_hidden, pooled_mask = self.backend.run_encoder(
                self.audio_encoder, features, attention_mask
            )

            pooled = self.attention_pool(enc_hidden, pooled_mask)  # (batch, hidden_dim)
            embeddings = self.projection_head(pooled)              # (batch, emb_dim)
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


@register_audio_embedding_model('attention_pool_m4t', default_name='facebook/seamless-m4t-v2-large', description='Attention pooling audio embedding model (SeamlessM4T-v2 encoder)')
class M4TAttentionPoolAudioModel(AttentionPoolAudioModel):
    """Attention-pool audio model on a SeamlessM4T-v2 speech encoder.

    Identical architecture to :class:`AttentionPoolAudioModel`; this subclass only
    carries its own registry metadata (default name + size table) so name/size
    resolution doesn't fall back to the Whisper sizes. The M4T backend is selected
    automatically from the seamless encoder name.
    """

    @dataclass
    class Params:
        size: str = "v2-large"
        emb_dim: int = 2048
        dropout: float = 0.1
        SIZES: ClassVar[Dict[str, str]] = {
            "v2-large": "facebook/seamless-m4t-v2-large",
        }

    def __init__(self,
                 audio_encoder_name: str = "facebook/seamless-m4t-v2-large",
                 emb_dim: int = 2048,
                 model_path: Optional[str] = None,
                 dropout: float = 0.1):
        super().__init__(audio_encoder_name, emb_dim, model_path, dropout)
