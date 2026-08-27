"""EAM alignment-pool audio embedding model (``side/eam``'s ``AlignmentModel`` architecture).

Wraps a frozen SeamlessM4T-v2 encoder (reusing ``attention_pool.py``'s backend) with the
self-attention → attention-pool → projection head from ``side/eam/alignment_model``. The
module shapes and attribute names (``audio_sa``, ``pool``, ``proj``, ``cross_attn``,
``text_sa``, ``audio_adapter``, ``text_adapter``) mirror ``side/eam`` exactly so a trained
``AlignmentModel`` state_dict loads with no key remapping. ``cross_attn``/``text_sa``/the
adapters are training-only in ``side/eam`` (``encode_audio``'s cross-attention branch is
gated on ``self.training``, always False at inference) — reimplemented here only so a
full checkpoint has somewhere to load, and loaded best-effort since they never affect
``encode_audio``'s output.

``embedding_dim``/``input_dim`` are read from the checkpoint itself (``proj.net.3.weight``
and ``audio_sa.attn.in_proj_weight`` shapes) rather than taken as config, so there is
nothing a user can set inconsistently with the actual trained weights.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base import AudioEmbeddingModel
from ..registry import register_audio_embedding_model
from ...logging_config import get_logger
from .attention_pool import _select_backend, _EncoderBackend

logger = get_logger(__name__)


class _AttentionPool(nn.Module):
    """Single learned softmax-weighted sum over time. Unlike this repo's own
    ``AttentionPooling``/``MeanPooling``, ``side/eam``'s ``AttentionPool`` never accepts a
    mask — reproduced faithfully here, so a padded batch's pad frames get pooled in too,
    matching what the checkpoint was trained on. Not a bug: a deliberate difference from
    this file's mask-aware siblings."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        weights = F.softmax(self.score(x).squeeze(-1), dim=-1)  # (batch, seq_len)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, dim)


class _SharedProjection(nn.Module):
    """Linear -> GELU -> Linear -> LayerNorm, matching ``side/eam``'s ``SharedProjection``."""

    def __init__(self, dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SelfAttentionBlock(nn.Module):
    """Self-attention + residual + LayerNorm, matching ``side/eam``'s ``SelfAttentionBlock``."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attended)


class _CrossAttention(nn.Module):
    """Cross-attention + residual + LayerNorm, matching ``side/eam``'s ``CrossAttention``.

    Training-only in ``side/eam`` (gated on ``self.training``) — never exercised by
    ``encode_audio`` at inference. Reimplemented purely so its checkpoint weights have
    somewhere to load."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, context, context, need_weights=False)
        return self.norm(query + attended)


# Checkpoint-critical vs. training-only key prefixes (see module docstring).
_STRICT_PREFIXES = ("audio_sa.", "pool.", "proj.")
_BEST_EFFORT_PREFIXES = ("cross_attn.", "text_sa.", "audio_adapter.", "text_adapter.")

_PROJ_LAYERNORM_KEY = "proj.net.3.weight"  # shape (embedding_dim,)
_AUDIO_SA_INPROJ_KEY = "audio_sa.attn.in_proj_weight"  # shape (3*input_dim, input_dim)

_DEFAULT_NUM_HEADS = 8
_DEFAULT_DROPOUT = 0.1


@register_audio_embedding_model(
    'eam_alignment',
    default_name='facebook/seamless-m4t-v2-large',
    description='Audio-text alignment pooling model (side/eam AlignmentModel architecture)',
    display_name='EAM alignment pool',
)
class EamAlignmentAudioModel(AudioEmbeddingModel):
    """Audio embedding model using ``side/eam``'s AlignmentModel architecture: a frozen
    speech encoder -> self-attention -> attention-pool -> shared projection, L2-normalized.

    ``embedding_dim``/``input_dim`` are inferred from ``model_path``'s checkpoint (there is
    no untrained mode — a checkpoint is required, unlike ``attention_pool`` which can run
    on a freshly initialized head)."""

    @dataclass
    class Params:
        size: str = "v2-large"
        model_path: Optional[str] = None
        SIZES: ClassVar[Dict[str, str]] = {
            "v2-large": "facebook/seamless-m4t-v2-large",
        }
        DESCRIPTIONS: ClassVar[Dict[str, str]] = {
            "model_path": "Path to a trained side/eam AlignmentModel checkpoint (.pt file, required).",
        }

    def __init__(self,
                 audio_encoder_name: str = "facebook/seamless-m4t-v2-large",
                 model_path: Optional[str] = None):
        if not model_path:
            raise ValueError(
                "eam_alignment requires model_path: a trained side/eam AlignmentModel "
                "checkpoint (embedding_dim/input_dim are read from its weights)."
            )
        self.audio_encoder_name = audio_encoder_name
        self.model_path = model_path
        self.device = torch.device("cpu")

        self.backend: _EncoderBackend = _select_backend(audio_encoder_name)
        self.hidden_dim = self.backend.hidden_dim
        self.audio_encoder = self.backend.encoder
        self.feature_extractor = self.backend.processor

        state_dict = self._read_checkpoint(model_path)
        input_dim, embedding_dim = self._infer_dims(state_dict, model_path)
        if input_dim != self.hidden_dim:
            raise ValueError(
                f"eam_alignment checkpoint {model_path}: trained input_dim={input_dim} does "
                f"not match encoder '{audio_encoder_name}' hidden_dim={self.hidden_dim}. "
                f"Set audio_embedding.model_name to the encoder used in training."
            )
        self.embedding_dim = embedding_dim

        self.audio_sa = _SelfAttentionBlock(input_dim, _DEFAULT_NUM_HEADS, _DEFAULT_DROPOUT)
        self.pool = _AttentionPool(input_dim)
        self.proj = _SharedProjection(input_dim, embedding_dim)
        # Training-only submodules — present only so the checkpoint has somewhere to load.
        self.text_sa = _SelfAttentionBlock(input_dim, _DEFAULT_NUM_HEADS, _DEFAULT_DROPOUT)
        self.cross_attn = _CrossAttention(input_dim, _DEFAULT_NUM_HEADS, _DEFAULT_DROPOUT)

        self._load_weights(state_dict, model_path)

    @staticmethod
    def _read_checkpoint(model_path: str) -> dict:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return state

    @staticmethod
    def _infer_dims(state_dict: dict, model_path: str) -> (int, int):
        try:
            input_dim = state_dict[_AUDIO_SA_INPROJ_KEY].shape[1]
            embedding_dim = state_dict[_PROJ_LAYERNORM_KEY].shape[0]
        except KeyError as exc:
            raise ValueError(
                f"eam_alignment checkpoint {model_path} is missing {exc}: not a side/eam "
                f"AlignmentModel state_dict (expected keys under 'audio_sa.'/'pool.'/'proj.')."
            ) from exc
        return int(input_dim), int(embedding_dim)

    def _load_weights(self, state_dict: dict, model_path: str) -> None:
        strict_state, best_effort_state, unexpected = {}, {}, []
        for key, value in state_dict.items():
            if key.startswith(_STRICT_PREFIXES):
                strict_state[key] = value
            elif key.startswith(_BEST_EFFORT_PREFIXES):
                best_effort_state[key] = value
            else:
                unexpected.append(key)

        modules = {"audio_sa": self.audio_sa, "pool": self.pool, "proj": self.proj}
        expected = {
            f"{name}.{k}" for name, mod in modules.items() for k in mod.state_dict()
        }
        missing = expected - set(strict_state)
        if missing:
            raise ValueError(
                f"eam_alignment checkpoint {model_path}: missing weights for "
                f"{sorted(missing)[:6]} ({len(missing)} key(s)) — audio_sa/pool/proj would be "
                f"left at random init."
            )
        for name, mod in modules.items():
            prefix = f"{name}."
            sub_state = {k[len(prefix):]: v for k, v in strict_state.items() if k.startswith(prefix)}
            mod.load_state_dict(sub_state, strict=True)

        best_effort_modules = {
            "text_sa": self.text_sa, "cross_attn": self.cross_attn,
        }
        for name, mod in best_effort_modules.items():
            prefix = f"{name}."
            sub_state = {k[len(prefix):]: v for k, v in best_effort_state.items() if k.startswith(prefix)}
            if sub_state:
                result = mod.load_state_dict(sub_state, strict=False)
                if result.missing_keys or result.unexpected_keys:
                    logger.warning(
                        "eam_alignment checkpoint %s: %s loaded with %d missing, %d unexpected "
                        "key(s) (training-only submodule, inert at inference)",
                        model_path, name, len(result.missing_keys), len(result.unexpected_keys),
                    )
        if unexpected:
            logger.warning(
                "eam_alignment checkpoint %s: %d unexpected key(s) ignored (e.g. %s)",
                model_path, len(unexpected), unexpected[:3],
            )
        logger.info(
            "eam_alignment checkpoint %s: loaded audio_sa/pool/proj (input_dim=%d, "
            "embedding_dim=%d)", model_path, self.hidden_dim, self.embedding_dim,
        )

    def to(self, device: torch.device):
        self.device = device
        self.backend.to(device)
        self.audio_encoder = self.backend.encoder
        self.audio_sa.to(device)
        self.pool.to(device)
        self.proj.to(device)
        self.text_sa.to(device)
        self.cross_attn.to(device)
        return self

    def preprocess_audio(self, audio_list: List[torch.Tensor], sampling_rates: List[int]):
        return self.backend.preprocess(audio_list, sampling_rates)

    def encode_from_features(self, features: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        features = features.to(self.device)
        # attention_mask is intentionally unused: side/eam's AttentionPool never masks
        # (see _AttentionPool docstring) and audio_sa likewise runs unmasked in side/eam.
        with torch.no_grad():
            self.backend.eval()
            self.audio_sa.eval()
            self.pool.eval()
            self.proj.eval()

            enc_hidden, _ = self.backend.run_encoder(self.audio_encoder, features, attention_mask)
            attended = self.audio_sa(enc_hidden)
            pooled = self.pool(attended)
            embeddings = self.proj(pooled)
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        torch.cuda.empty_cache()
        return embeddings.cpu().numpy()

    def encode_audio(self, audio_list: List[torch.Tensor], sampling_rates: List[int],
                     show_progress: bool = False) -> np.ndarray:
        features, attention_mask = self.preprocess_audio(audio_list, sampling_rates)
        return self.encode_from_features(features, attention_mask)

    def name(self) -> str:
        return (f"EamAlignmentAudioModel - encoder:{self.audio_encoder_name}"
                f" - embedding_dim:{self.embedding_dim} - weights:{self.model_path}")
