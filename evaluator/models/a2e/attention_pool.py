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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from ..base import AudioEmbeddingModel
from ..registry import register_audio_embedding_model
from ...logging_config import get_logger

logger = get_logger(__name__)


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


class MeanPooling(nn.Module):
    """Masked mean over the sequence axis (no learnable params)."""

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, input_dim)
        if mask is not None:
            m = mask.unsqueeze(-1).to(x.dtype)
            return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        return x.mean(dim=1)


class MeanPoolingWithWhitening(nn.Module):
    """Masked mean, then whiten with precomputed stats (buffers loaded from the checkpoint)."""

    def __init__(self, m: torch.Tensor, W: torch.Tensor):
        super().__init__()
        self.register_buffer("m", m)
        self.register_buffer("W", W)

    def forward(self, x, mask=None):
        from .postprocessing import whiten_batch

        pooled = MeanPooling().forward(x, mask)
        return whiten_batch(pooled, self.m, self.W)


class MeanPoolingWithAbtt(nn.Module):
    """Masked mean, then All-But-The-Top with precomputed stats (buffers from the checkpoint)."""

    def __init__(self, mu: torch.Tensor, pc1: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("pc1", pc1)

    def forward(self, x, mask=None):
        from .postprocessing import abtt_batch

        pooled = MeanPooling().forward(x, mask)
        return abtt_batch(pooled, self.mu, self.pc1)


# Pooling strategies selectable via the ``pooling`` model param (CHOICES, surfaced in the
# builder). All pool to ``hidden_dim``; the projection head then maps to ``emb_dim``.
POOLING_CHOICES = ["attention", "mean", "mean_whiten", "mean_abtt"]


def _make_pooling(pooling: str, hidden_dim: int, dropout: float) -> nn.Module:
    """Build the selected pooling module. The whiten/ABTT stats are placeholder buffers of the
    right shape so a checkpoint's ``attn_pool.*`` buffers overwrite them via ``load_state_dict``."""
    if pooling == "attention":
        return AttentionPooling(input_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout)
    if pooling == "mean":
        return MeanPooling()
    if pooling == "mean_whiten":
        return MeanPoolingWithWhitening(
            m=torch.zeros(hidden_dim), W=torch.eye(hidden_dim)
        )
    if pooling == "mean_abtt":
        return MeanPoolingWithAbtt(
            mu=torch.zeros(hidden_dim), pc1=torch.zeros(hidden_dim)
        )
    raise ValueError(f"Unknown pooling '{pooling}'. Options: {POOLING_CHOICES}")


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
        pooling: str = "attention"
        SIZES: ClassVar[Dict[str, str]] = {
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
        }
        CHOICES: ClassVar[Dict[str, List[str]]] = {"pooling": POOLING_CHOICES}

    def __init__(self,
                 audio_encoder_name: str = "openai/whisper-large",
                 emb_dim: int = 2048,
                 model_path: Optional[str] = None,
                 dropout: float = 0.1,
                 pooling: str = "attention"):
        """
        Initialize attention pool audio model.

        Args:
            audio_encoder_name: Name of the audio encoder (Whisper or SeamlessM4T).
            emb_dim: Embedding dimension (should match text embedder).
            model_path: Path to pre-trained APM model weights (.pt file).
            dropout: Dropout rate for attention pooling and projection.
            pooling: Sequence-pooling strategy — one of ``POOLING_CHOICES`` (attention /
                mean / mean_whiten / mean_abtt). The whiten/ABTT stats load from the checkpoint.
        """
        self.audio_encoder_name = audio_encoder_name
        self.emb_dim = emb_dim
        self.model_path = model_path
        self.pooling_kind = pooling
        self.device = torch.device("cpu")

        # Encoder-specific backend (feature extractor + frozen encoder + mask logic).
        self.backend = _select_backend(audio_encoder_name)
        self.hidden_dim = self.backend.hidden_dim
        # Back-compat aliases: tests / external code reference these directly.
        self.audio_encoder = self.backend.encoder
        self.feature_extractor = self.backend.processor

        # Pooling (selected strategy) and projection operate at the encoder's hidden dim.
        self.pooling = _make_pooling(pooling, self.hidden_dim, dropout)
        # Back-compat alias: external code/tests reference ``attention_pool`` as the pooling slot.
        self.attention_pool = self.pooling
        self.projection_head = ProjectionHead(
            input_dim=self.hidden_dim,
            emb_dim=emb_dim,
            dropout=dropout,
        )

        if model_path:
            self._load_weights(model_path)

    #: Checkpoint key prefixes (the trained ``AudioEmbedder`` state_dict from ``apm_new``):
    #: ``audio_enc.encoder.*`` (the frozen encoder), ``attn_pool.*`` (the selected pooling), and
    #: ``proj.proj.*`` (the projection head — ``AudioEmbedder.proj`` is a ``ProjectionHead`` whose
    #: own ``self.proj`` is the Sequential, hence the doubled prefix).
    _ENCODER_PREFIX = 'audio_enc.encoder.'
    _POOLING_PREFIX = 'attn_pool.'
    _PROJ_PREFIX = 'proj.proj.'

    def _load_weights(self, model_path: str):
        """Load a pre-trained APM checkpoint into the encoder + pooling + projection.

        The checkpoint is a full ``AudioEmbedder`` state_dict. We load:
        ``audio_enc.encoder.*`` → the encoder (so the encoder matches what training used, not
        just the HF name), ``attn_pool.*`` → the selected pooling (attention params, or the
        whiten ``m``/``W`` / ABTT ``mu``/``pc1`` buffers; ``mean`` has none), and ``proj.proj.*``
        → the projection head. A key mismatch is a **loud error** (no silent random init); an
        encoder size mismatch (wrong Whisper variant) is a loud error too."""
        state_dict = torch.load(model_path, map_location=self.device)

        enc_state, pooling_state, proj_head_state, unexpected = {}, {}, {}, []
        for key, value in state_dict.items():
            if key.startswith(self._ENCODER_PREFIX):
                enc_state[key[len(self._ENCODER_PREFIX):]] = value
            elif key.startswith(self._POOLING_PREFIX):
                pooling_state[key[len(self._POOLING_PREFIX):]] = value
            elif key.startswith(self._PROJ_PREFIX):
                proj_head_state[key[len('proj.'):]] = value  # strip one 'proj.' → 'proj.0.weight'
            else:
                unexpected.append(key)

        self._load_encoder_weights(enc_state, model_path)
        self._load_module_weights(
            self.pooling, pooling_state, f"pooling[{self.pooling_kind}]", model_path,
            state_dict, reregister_buffers=True,
        )
        self._load_module_weights(
            self.projection_head, proj_head_state, "projection_head", model_path, state_dict,
        )
        if unexpected:
            logger.warning(
                "APM checkpoint %s: %d unexpected key(s) ignored (e.g. %s)",
                model_path, len(unexpected), unexpected[:3],
            )

    def _load_encoder_weights(self, enc_state: dict, model_path: str) -> None:
        """Load ``audio_enc.encoder.*`` into ``self.backend.encoder`` so the encoder is exactly
        the one the checkpoint was trained against (not just whatever the HF name resolves to).

        A shape mismatch (wrong Whisper/M4T size) raises from ``load_state_dict``; a low match
        fraction (a structurally different encoder) raises here. No encoder weights in the
        checkpoint → keep today's HF-pretrained encoder."""
        if not enc_state:
            logger.info(
                "APM checkpoint %s: no encoder weights — using HF-pretrained encoder '%s'",
                model_path, self.audio_encoder_name,
            )
            return
        target = set(self.backend.encoder.state_dict().keys())
        # strict=False so version-skew keys don't hard-fail; a shape mismatch still raises here.
        result = self.backend.encoder.load_state_dict(enc_state, strict=False)
        matched = len(target) - len(result.missing_keys)
        if not target or matched < 0.9 * len(target):
            raise ValueError(
                f"APM checkpoint {model_path}: encoder weights match only {matched}/{len(target)} "
                f"of the '{self.audio_encoder_name}' encoder parameters — the checkpoint was "
                f"trained against a different encoder. Set audio_embedding.model_name to the "
                f"encoder used in training. Missing e.g.: {sorted(result.missing_keys)[:3]}"
            )
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "APM encoder load (%s): %d missing, %d unexpected key(s) — possible transformers "
                "version skew", model_path, len(result.missing_keys), len(result.unexpected_keys),
            )
        logger.info(
            "APM checkpoint %s: loaded %d encoder param(s) from the checkpoint (matches training)",
            model_path, matched,
        )

    def _load_module_weights(self, module, incoming: dict, label: str, model_path: str,
                             full_state: dict, *, reregister_buffers: bool = False) -> None:
        """Strict-load ``incoming`` into ``module`` after asserting it covers every expected key.

        A missing expected key would otherwise leave that submodule at random init (the old
        ``if incoming:`` guard did this silently). Raise a clear error instead, naming the missing
        keys + the checkpoint's actual top-level prefixes."""
        expected = set(module.state_dict().keys())
        if not expected:  # e.g. plain MeanPooling — nothing to load
            return
        missing = expected - set(incoming)
        if missing:
            prefixes = sorted({k.split('.')[0] for k in full_state})
            raise ValueError(
                f"APM checkpoint {model_path}: {label} is missing weights for "
                f"{sorted(missing)[:6]} ({len(missing)} key(s)) — it would be left at random "
                f"init. The checkpoint's top-level prefixes are {prefixes}; expected the "
                f"'{label}' weights under '{self._POOLING_PREFIX}'/'{self._PROJ_PREFIX}'. "
                f"Check the checkpoint matches pooling='{self.pooling_kind}'."
            )
        if reregister_buffers:
            # Stat buffers (whiten m/W, ABTT mu/pc1) can carry trailing singleton dims from the
            # training pipeline (mu (1,H), pc1 (H,1)) vs. our flat (H,) placeholders. Re-register
            # to the checkpoint's shape so the strict load succeeds; the transforms normalize the
            # shape at apply time.
            for name, tensor in incoming.items():
                if name in module._buffers:
                    module.register_buffer(name, torch.empty_like(tensor))
        module.load_state_dict(incoming, strict=True)
        logger.info("APM checkpoint %s: loaded %s (%d key(s))", model_path, label, len(incoming))

    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.backend.to(device)
        self.audio_encoder = self.backend.encoder
        self.pooling.to(device)
        self.attention_pool = self.pooling  # keep the alias in sync
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
            self.pooling.eval()
            self.projection_head.eval()

            # Run the (model-owned) encoder; the backend returns the hidden states
            # and the mask aligned to the encoder's output length.
            enc_hidden, pooled_mask = self.backend.run_encoder(
                self.audio_encoder, features, attention_mask
            )

            pooled = self.pooling(enc_hidden, pooled_mask)  # (batch, hidden_dim)
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
        name = (f"AttentionPoolAudioModel - encoder:{self.audio_encoder_name}"
                f" - emb_dim:{self.emb_dim} - pooling:{self.pooling_kind}")
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
        pooling: str = "attention"
        SIZES: ClassVar[Dict[str, str]] = {
            "v2-large": "facebook/seamless-m4t-v2-large",
        }
        CHOICES: ClassVar[Dict[str, List[str]]] = {"pooling": POOLING_CHOICES}

    def __init__(self,
                 audio_encoder_name: str = "facebook/seamless-m4t-v2-large",
                 emb_dim: int = 2048,
                 model_path: Optional[str] = None,
                 dropout: float = 0.1,
                 pooling: str = "attention"):
        super().__init__(audio_encoder_name, emb_dim, model_path, dropout, pooling)
