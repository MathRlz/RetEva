"""Model configuration."""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """
    Configuration for model selection and device assignment.

    One scalar per role (asr / text_emb / audio_emb). In the node-centric graph these are the
    **default** model for each role — used by a graph node that doesn't carry its own ``model`` —
    not a hard one-per-role limit: a node can override per-instance (the executor builds it via
    ``_node_pipeline``), so a graph may run two distinct same-role models.

    Each model component accepts **either** a ``size`` shorthand (resolved via
    the registry's Params.SIZES mapping) **or** an explicit ``model_name``.
    ``model_name`` always wins when both are given.

    Examples:
        >>> ModelConfig(asr_model_type="whisper", asr_size="large-v3")
        >>> ModelConfig(asr_model_type="whisper", asr_model_name="openai/whisper-large-v3")
    """
    # --- ASR ---
    asr_model_type: Optional[str] = "wav2vec2"
    asr_size: Optional[str] = None
    asr_model_name: Optional[str] = None
    asr_adapter_path: Optional[str] = None
    asr_device: str = "cuda:0"
    asr_params: Dict[str, object] = field(default_factory=dict)
    asr_quantization: Optional[str] = None

    # --- Text embedding ---
    text_emb_model_type: Optional[str] = "labse"
    text_emb_size: Optional[str] = None
    text_emb_model_name: Optional[str] = None
    text_emb_adapter_path: Optional[str] = None
    text_emb_device: str = "cuda:1"
    # Override the embedding-space id (else derived from model type/name). Declare the
    # SAME id on a co-trained audio embedder so cross-modal dense retrieval validates.
    text_emb_embedding_space: Optional[str] = None
    text_emb_params: Dict[str, object] = field(default_factory=dict)
    text_emb_quantization: Optional[str] = None

    # --- Audio embedding ---
    audio_emb_model_type: Optional[str] = None
    audio_emb_size: Optional[str] = None
    audio_emb_model_name: Optional[str] = None
    audio_emb_adapter_path: Optional[str] = None
    audio_emb_model_path: Optional[str] = None
    audio_emb_dim: int = 2048
    audio_emb_dropout: float = 0.1
    audio_emb_device: str = "cuda:0"
    # Override the embedding-space id — an APM trained to project audio into a text
    # embedder's space declares that text space here (not derived from the encoder name).
    audio_emb_embedding_space: Optional[str] = None
    audio_emb_params: Dict[str, object] = field(default_factory=dict)
    audio_emb_quantization: Optional[str] = None

    # Global quantization (e.g. "int8" / "4bit"); a per-family `*_quantization` wins over it.
    # Opt-in: applied only by models whose constructor accepts a `quantization` kwarg (else a
    # clear warning + full precision). Default None = today's behaviour.
    quantization: Optional[str] = None

    # NB: there is no ``pipeline_mode`` — the graph is the spec. A config carries an explicit
    # ``graph: {nodes}``; the run label derives from the node kinds (``label_from_graph``).
    # (``graph_override['template']`` survives only for legacy flat-dict back-compat.)

    def quantization_for(self, family: str) -> Optional[str]:
        """Resolve the quantization strategy for ``family`` ('asr'/'text_emb'/'audio_emb'):
        the per-family override if set, else the global default."""
        return getattr(self, f"{family}_quantization", None) or self.quantization

    def auto_configure_devices(self) -> None:
        """Auto-configure device assignments based on hardware availability.

        Picks from the CUDA devices torch reports. With 2+ GPUs, text embedding goes on
        the second to spread load; otherwise everything shares the first, or CPU when
        there are none. Hide a device you must not use (an unsupported iGPU on a ROCm
        box) with ``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``.
        """
        from .base import get_available_gpu_count

        devs = list(range(get_available_gpu_count()))

        if not devs:
            # No GPUs, use CPU for all
            self.asr_device = "cpu"
            self.text_emb_device = "cpu"
            self.audio_emb_device = "cpu"
            return

        primary = f"cuda:{devs[0]}"
        secondary = f"cuda:{devs[1]}" if len(devs) > 1 else primary
        # ASR + audio on the primary GPU; text embedding spread to the second when present.
        self.asr_device = primary
        self.audio_emb_device = primary
        self.text_emb_device = secondary
