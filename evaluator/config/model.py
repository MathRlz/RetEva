"""Model configuration."""
from dataclasses import dataclass, field
from typing import Dict, Optional
from .base import get_available_gpu_count


@dataclass
class ModelConfig:
    """Configuration for model selection and device assignment.

    One scalar per role (asr/text_emb/audio_emb): the graph-node DEFAULT model for that
    role, not a hard limit — a node can override per-instance, so a graph may run two
    distinct same-role models. Each component accepts either a ``size`` shorthand or an
    explicit ``model_name``; ``model_name`` wins when both are given.
    """
    asr_model_type: Optional[str] = "wav2vec2"
    asr_size: Optional[str] = None
    asr_model_name: Optional[str] = None
    asr_adapter_path: Optional[str] = None
    asr_device: str = "cuda:0"
    asr_params: Dict[str, object] = field(default_factory=dict)
    asr_quantization: Optional[str] = None

    text_emb_model_type: Optional[str] = "labse"
    text_emb_size: Optional[str] = None
    text_emb_model_name: Optional[str] = None
    text_emb_adapter_path: Optional[str] = None
    # Path to a trained checkpoint (e.g. eam_alignment) — mirrors audio_emb_model_path.
    text_emb_model_path: Optional[str] = None
    text_emb_device: str = "cuda:1"
    # Override the embedding-space id (else derived from model type/name). Declare the
    # SAME id on a co-trained audio embedder so cross-modal dense retrieval validates.
    text_emb_embedding_space: Optional[str] = None
    text_emb_params: Dict[str, object] = field(default_factory=dict)
    text_emb_quantization: Optional[str] = None

    audio_emb_model_type: Optional[str] = None
    audio_emb_size: Optional[str] = None
    audio_emb_encoder_type: Optional[str] = None   # "whisper" | "m4t"
    audio_emb_encoder_size: Optional[str] = None   # e.g. "large", "v2-large"
    audio_emb_model_name: Optional[str] = None
    audio_emb_adapter_path: Optional[str] = None
    audio_emb_model_path: Optional[str] = None
    audio_emb_dim: int = 2048
    audio_emb_device: str = "cuda:0"
    # Override the embedding-space id — an APM trained to project audio into a text
    # embedder's space declares that text space here (not derived from the encoder name).
    audio_emb_embedding_space: Optional[str] = None
    audio_emb_params: Dict[str, object] = field(default_factory=dict)
    audio_emb_quantization: Optional[str] = None

    # Global quantization (e.g. "int8"/"4bit"); a per-family *_quantization wins over it.
    # Opt-in: applied only by models whose constructor accepts a `quantization` kwarg
    # (else full precision).
    quantization: Optional[str] = None

    # NB: there is no ``pipeline_mode`` — the graph is the spec. A config carries an explicit
    # ``graph: {nodes}`` and the run label derives from the node kinds (``graph_label``).

    def quantization_for(self, family: str) -> Optional[str]:
        """Resolve the quantization strategy for ``family`` ('asr'/'text_emb'/'audio_emb'):
        the per-family override if set, else the global default."""
        return getattr(self, f"{family}_quantization", None) or self.quantization

    def auto_configure_devices(self) -> None:
        """Auto-configure device assignments based on hardware availability.

        Picks from the CUDA devices torch reports: with 2+ GPUs, text embedding goes on
        the second to spread load; otherwise everything shares the first, or CPU when
        there are none. Hide an unusable device (e.g. an iGPU) via
        ``CUDA_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES``.
        """

        devs = list(range(get_available_gpu_count()))

        if not devs:
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
