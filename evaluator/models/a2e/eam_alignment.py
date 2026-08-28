"""EAM alignment-pool model (``side/eam``'s ``AlignmentModel`` architecture).

``side/eam``'s ``AlignmentModel`` is a genuinely joint audio-text model: ``encode_audio``
and ``encode_text`` both end in the SAME ``pool``/``proj`` modules, so audio and text land
in one shared embedding space by construction. This wraps it as one class implementing
BOTH ``AudioEmbeddingModel`` and ``TextEmbeddingModel`` (mirrors ``clap_style.py``'s
dual-registration pattern) so it can be used as the audio_embedding node, the
text_embedding node, or both at once for cross-modal retrieval.

Only ``audio_sa``/``text_sa``/``pool``/``proj`` are built and loaded — ``side/eam``'s
``cross_attn`` and the optional ``audio_adapter``/``text_adapter`` are training-only
machinery (``encode_audio``/``forward``'s cross-attention branch is gated on
``self.training``, and nothing in this codebase ever constructs a non-``None`` adapter —
grepped ``side/eam`` for every ``AlignmentModel(...)`` call site). At inference there is
nothing to compute with them, so they're not reimplemented at all; any such keys in a
checkpoint are simply ignored.

When an audio_embedding node and a text_embedding node both name ``eam_alignment`` with
the SAME ``model_path``, they get two separate Python wrapper instances (one per family's
registry/service-provider cache — those caches don't cross-reference each other) but a
single shared ``_EamAlignmentCore`` (module-level cache keyed on
``(model_path, audio_encoder_name)``): one loaded encoder pair + one set of
audio_sa/text_sa/pool/proj weights, not two.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base import AudioEmbeddingModel, TextEmbeddingModel
from ..registry import register_audio_embedding_model, register_text_embedding_model
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
    """Linear -> GELU -> Linear -> LayerNorm, matching ``side/eam``'s ``SharedProjection``.
    The SAME instance projects both audio and text — that's what makes the space joint."""

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
    """Self-attention + residual + LayerNorm, matching ``side/eam``'s ``SelfAttentionBlock``.
    Real inference-time modules for both modalities: ``audio_sa`` runs in ``encode_audio``,
    ``text_sa`` runs in ``encode_text`` — neither is training-only."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attended)


class _M4TTextBackend:
    """SeamlessM4T-v2 TEXT encoder backend — the text counterpart to
    ``attention_pool.py``'s ``_M4TBackend`` (speech). ``side/eam`` builds this via
    ``SeamlessM4Tv2ForTextToSpeech(...).get_encoder()`` (see
    ``side/eam/encoder/sonar_space/SonarSpaceEncoder.py`` — a misleading name, it wraps
    M4T's own text encoder, not SONAR)."""

    def __init__(self, encoder_name: str):
        from transformers import SeamlessM4Tv2ForTextToSpeech, AutoProcessor

        self.hidden_dim = 1024
        self.processor = AutoProcessor.from_pretrained(encoder_name)
        self.encoder = SeamlessM4Tv2ForTextToSpeech.from_pretrained(encoder_name).get_encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def preprocess(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        return inputs.input_ids, inputs.attention_mask

    def run_encoder(self, encoder, input_ids, attention_mask):
        return encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def to(self, device):
        self.encoder.to(device)

    def eval(self):
        self.encoder.eval()

    def parameters(self):
        return self.encoder.parameters()


_PROJ_LAYERNORM_KEY = "proj.net.3.weight"  # shape (embedding_dim,)
_AUDIO_SA_INPROJ_KEY = "audio_sa.attn.in_proj_weight"  # shape (3*input_dim, input_dim)
_TEXT_SA_INPROJ_KEY = "text_sa.attn.in_proj_weight"  # shape (3*input_dim, input_dim)

_STRICT_PREFIXES = ("audio_sa.", "text_sa.", "pool.", "proj.")
_DEFAULT_NUM_HEADS = 8
_DEFAULT_DROPOUT = 0.1


class _EamAlignmentCore:
    """The actual weights: two frozen M4T encoders (speech + text) and one set of
    audio_sa/text_sa/pool/proj heads, loaded once from ``model_path``. Shared (via
    ``_get_or_build_core``) between an audio_embedding wrapper and a text_embedding
    wrapper pointed at the same checkpoint, so the same nn.Module instances back both."""

    def __init__(self, encoder_name: str, model_path: str, num_heads: int = _DEFAULT_NUM_HEADS):
        self.encoder_name = encoder_name
        self.model_path = model_path
        self.device = torch.device("cpu")

        self._device_set = False
        self.audio_backend: _EncoderBackend = _select_backend(encoder_name)
        self.text_backend = _M4TTextBackend(encoder_name)
        if self.text_backend.hidden_dim != self.audio_backend.hidden_dim:
            raise ValueError(
                f"eam_alignment: audio backend hidden_dim={self.audio_backend.hidden_dim} != "
                f"text backend hidden_dim={self.text_backend.hidden_dim} for '{encoder_name}' "
                f"— audio_sa/text_sa/pool/proj require both modalities at the same width."
            )
        self.hidden_dim = self.audio_backend.hidden_dim

        state_dict = self._read_checkpoint(model_path)
        input_dim, embedding_dim = self._infer_dims(state_dict, model_path)
        if input_dim != self.hidden_dim:
            raise ValueError(
                f"eam_alignment checkpoint {model_path}: trained input_dim={input_dim} does "
                f"not match encoder '{encoder_name}' hidden_dim={self.hidden_dim}. Set "
                f"audio_embedding.model_name/text_embedding.model_name to the encoder used "
                f"in training."
            )
        self.embedding_dim = embedding_dim

        self.audio_sa = _SelfAttentionBlock(input_dim, num_heads, _DEFAULT_DROPOUT)
        self.text_sa = _SelfAttentionBlock(input_dim, num_heads, _DEFAULT_DROPOUT)
        self.pool = _AttentionPool(input_dim)
        self.proj = _SharedProjection(input_dim, embedding_dim)
        self._load_weights(state_dict, model_path)

    @staticmethod
    def _read_checkpoint(model_path: str) -> dict:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return state

    @staticmethod
    def _infer_dims(state_dict: dict, model_path: str) -> Tuple[int, int]:
        try:
            input_dim = state_dict[_AUDIO_SA_INPROJ_KEY].shape[1]
            embedding_dim = state_dict[_PROJ_LAYERNORM_KEY].shape[0]
        except KeyError as exc:
            raise ValueError(
                f"eam_alignment checkpoint {model_path} is missing {exc}: not a side/eam "
                f"AlignmentModel state_dict (expected keys under 'audio_sa.'/'pool.'/'proj.')."
            ) from exc
        text_sa_shape = state_dict.get(_TEXT_SA_INPROJ_KEY)
        if text_sa_shape is not None and text_sa_shape.shape[1] != input_dim:
            raise ValueError(
                f"eam_alignment checkpoint {model_path}: text_sa input_dim="
                f"{text_sa_shape.shape[1]} != audio_sa input_dim={input_dim} — inconsistent "
                f"checkpoint."
            )
        return int(input_dim), int(embedding_dim)

    def _load_weights(self, state_dict: dict, model_path: str) -> None:
        modules = {
            "audio_sa": self.audio_sa, "text_sa": self.text_sa,
            "pool": self.pool, "proj": self.proj,
        }
        by_prefix: Dict[str, dict] = {name: {} for name in modules}
        unexpected = []
        for key, value in state_dict.items():
            if key.startswith(_STRICT_PREFIXES):
                name, rest = key.split(".", 1)
                by_prefix[name][rest] = value
            else:
                unexpected.append(key)

        expected = {f"{name}.{k}" for name, mod in modules.items() for k in mod.state_dict()}
        got = {f"{name}.{k}" for name, sub in by_prefix.items() for k in sub}
        missing = expected - got
        if missing:
            raise ValueError(
                f"eam_alignment checkpoint {model_path}: missing weights for "
                f"{sorted(missing)[:6]} ({len(missing)} key(s)) — audio_sa/text_sa/pool/proj "
                f"would be left at random init."
            )
        for name, mod in modules.items():
            mod.load_state_dict(by_prefix[name], strict=True)
        if unexpected:
            logger.info(
                "eam_alignment checkpoint %s: %d non-inference key(s) ignored (e.g. %s) — "
                "cross_attn/adapters are training-only in side/eam",
                model_path, len(unexpected), unexpected[:3],
            )
        logger.info(
            "eam_alignment checkpoint %s: loaded audio_sa/text_sa/pool/proj (input_dim=%d, "
            "embedding_dim=%d)", model_path, self.hidden_dim, self.embedding_dim,
        )

    def to(self, device: torch.device):
        if self._device_set and self.device != device:
            logger.warning(
                "eam_alignment core for %s is shared across audio/text embedding roles and "
                "is moving from %s to %s — both roles must use the same device.",
                self.model_path, self.device, device,
            )
        self._device_set = True
        self.device = device
        self.audio_backend.to(device)
        self.text_backend.to(device)
        self.audio_sa.to(device)
        self.text_sa.to(device)
        self.pool.to(device)
        self.proj.to(device)
        return self

    def encode_audio(self, features: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = features.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            self.audio_backend.eval()
            self.audio_sa.eval()
            self.pool.eval()
            self.proj.eval()
            # attention_mask intentionally unused: side/eam's AttentionPool/audio_sa never mask.
            enc_hidden, _ = self.audio_backend.run_encoder(
                self.audio_backend.encoder, features, attention_mask
            )
            attended = self.audio_sa(enc_hidden)
            pooled = self.pool(attended)
            embeddings = F.normalize(self.proj(pooled), p=2, dim=-1)
        torch.cuda.empty_cache()
        return embeddings

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            self.text_backend.eval()
            self.text_sa.eval()
            self.pool.eval()
            self.proj.eval()
            input_ids, attention_mask = self.text_backend.preprocess(texts)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            hidden = self.text_backend.run_encoder(self.text_backend.encoder, input_ids, attention_mask)
            attended = self.text_sa(hidden)
            pooled = self.pool(attended)
            embeddings = F.normalize(self.proj(pooled), p=2, dim=-1)
        torch.cuda.empty_cache()
        return embeddings


_CORE_CACHE: Dict[Tuple[str, str], _EamAlignmentCore] = {}


def _get_or_build_core(encoder_name: str, model_path: str,
                       num_heads: int = _DEFAULT_NUM_HEADS) -> _EamAlignmentCore:
    """Get-or-build the shared core for ``(model_path, encoder_name)``. A second wrapper
    (whichever family — audio or text — builds second) reuses the already-loaded weights
    instead of loading the checkpoint + encoders again."""
    key = (model_path, encoder_name)
    core = _CORE_CACHE.get(key)
    if core is None:
        core = _EamAlignmentCore(encoder_name, model_path, num_heads=num_heads)
        _CORE_CACHE[key] = core
    elif num_heads != _DEFAULT_NUM_HEADS:
        logger.warning(
            "eam_alignment: core for %s already built with a different num_heads config; "
            "the requested num_heads=%d is ignored (the two roles share one instance).",
            model_path, num_heads,
        )
    return core


@register_audio_embedding_model(
    'eam_alignment',
    default_name='facebook/seamless-m4t-v2-large',
    description='Audio-text alignment pooling model (side/eam AlignmentModel architecture, audio side)',
    display_name='EAM alignment pool',
)
@register_text_embedding_model(
    'eam_alignment',
    default_name='facebook/seamless-m4t-v2-large',
    description='Audio-text alignment pooling model (side/eam AlignmentModel architecture, text side); '
                'give the same model_path as the eam_alignment audio_embedding node to share one loaded model',
)
class EamAlignmentModel(AudioEmbeddingModel, TextEmbeddingModel):
    """``side/eam``'s AlignmentModel: a frozen speech/text encoder pair -> self-attention ->
    a SHARED attention-pool + projection head, L2-normalized. Implements both
    ``AudioEmbeddingModel`` (``encode_audio``) and ``TextEmbeddingModel`` (``encode``) so it
    can be used as either node kind, or both for cross-modal retrieval into one space.

    ``embedding_dim``/``input_dim`` are inferred from ``model_path``'s checkpoint — there is
    no untrained mode; a checkpoint is required."""

    @dataclass
    class Params:
        size: str = "v2-large"
        model_path: Optional[str] = None
        config_path: Optional[str] = None
        num_heads: int = _DEFAULT_NUM_HEADS
        SIZES: ClassVar[Dict[str, str]] = {
            "v2-large": "facebook/seamless-m4t-v2-large",
        }
        DESCRIPTIONS: ClassVar[Dict[str, str]] = {
            "model_path": "Path to a trained side/eam AlignmentModel checkpoint (.pt file, "
                          "required). Use the SAME path on the eam_alignment audio_embedding "
                          "and text_embedding nodes to share one loaded model.",
            "config_path": "Path to the training run's model.json (side/eam's "
                           "AlignmentModelConfig, saved next to the checkpoint under "
                           "runs/exp_*/config/). When set, num_heads comes from it and "
                           "input_dim/embedding_dim are cross-checked against the "
                           "checkpoint — the recommended way to guarantee the wrapper "
                           "matches the training-time architecture.",
            "num_heads": "Attention head count the checkpoint was trained with (not "
                         "recoverable from the checkpoint's tensor shapes — a wrong value "
                         "loads without error but computes silently-wrong attention). "
                         "Ignored when config_path is set (model.json wins).",
        }

    def __init__(self,
                 audio_encoder_name: str = "facebook/seamless-m4t-v2-large",
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 num_heads: int = _DEFAULT_NUM_HEADS):
        if not model_path:
            raise ValueError(
                "eam_alignment requires model_path: a trained side/eam AlignmentModel "
                "checkpoint (embedding_dim/input_dim are read from its weights)."
            )
        expected_dims: Optional[Tuple[int, int]] = None
        if config_path:
            train_cfg = self._read_model_json(config_path)
            cfg_heads = train_cfg.get("num_heads")
            if cfg_heads is not None:
                if num_heads != _DEFAULT_NUM_HEADS and int(cfg_heads) != int(num_heads):
                    logger.warning(
                        "eam_alignment: num_heads=%s from config param is overridden by "
                        "%s's num_heads=%s (model.json is the training truth).",
                        num_heads, config_path, cfg_heads,
                    )
                num_heads = int(cfg_heads)
            if "input_dim" in train_cfg and "embedding_dim" in train_cfg:
                expected_dims = (int(train_cfg["input_dim"]), int(train_cfg["embedding_dim"]))
        self.encoder_name = audio_encoder_name
        self.model_path = model_path
        self.core = _get_or_build_core(audio_encoder_name, model_path, num_heads=num_heads)
        if expected_dims is not None:
            got = (self.core.hidden_dim, self.core.embedding_dim)
            if got != expected_dims:
                raise ValueError(
                    f"eam_alignment: {config_path} says (input_dim, embedding_dim)="
                    f"{expected_dims} but checkpoint {model_path} has {got} — the model.json "
                    f"does not belong to this checkpoint."
                )

    @staticmethod
    def _read_model_json(config_path: str) -> dict:
        import json

        try:
            with open(config_path) as fh:
                cfg = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"eam_alignment: could not read config_path {config_path!r} as side/eam's "
                f"model.json (AlignmentModelConfig): {exc}"
            ) from exc
        if not isinstance(cfg, dict):
            raise ValueError(
                f"eam_alignment: {config_path} is not a JSON object (AlignmentModelConfig)."
            )
        return cfg

    def to(self, device: torch.device):
        self.core.to(device)
        return self

    # --- AudioEmbeddingModel ---
    def preprocess_audio(self, audio_list: List[torch.Tensor], sampling_rates: List[int]):
        return self.core.audio_backend.preprocess(audio_list, sampling_rates)

    def encode_from_features(self, features: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        return self.core.encode_audio(features, attention_mask).cpu().numpy()

    def encode_audio(self, audio_list: List[torch.Tensor], sampling_rates: List[int],
                     show_progress: bool = False) -> np.ndarray:
        features, attention_mask = self.preprocess_audio(audio_list, sampling_rates)
        return self.encode_from_features(features, attention_mask)

    # --- TextEmbeddingModel ---
    def encode(self, texts: List[str], show_progress: bool = False,
              desc: str = "Embedding") -> np.ndarray:
        return self.core.encode_text(texts).cpu().numpy()

    def name(self) -> str:
        return (f"EamAlignmentModel - encoder:{self.encoder_name}"
                f" - embedding_dim:{self.core.embedding_dim} - weights:{self.model_path}")
