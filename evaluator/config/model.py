"""Model configuration."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from ..config.types import PipelineMode, to_enum


@dataclass
class ModelConfig:
    """
    Configuration for model selection and device assignment.

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

    # --- Text embedding ---
    text_emb_model_type: Optional[str] = "labse"
    text_emb_size: Optional[str] = None
    text_emb_model_name: Optional[str] = None
    text_emb_adapter_path: Optional[str] = None
    text_emb_device: str = "cuda:1"
    text_emb_params: Dict[str, object] = field(default_factory=dict)

    # --- Audio embedding ---
    audio_emb_model_type: Optional[str] = None
    audio_emb_size: Optional[str] = None
    audio_emb_model_name: Optional[str] = None
    audio_emb_adapter_path: Optional[str] = None
    audio_emb_model_path: Optional[str] = None
    audio_emb_dim: int = 2048
    audio_emb_dropout: float = 0.1
    audio_emb_device: str = "cuda:0"
    audio_emb_params: Dict[str, object] = field(default_factory=dict)

    pipeline_mode: Union[str, PipelineMode] = "asr_text_retrieval"
    
    def __post_init__(self):
        """Normalize pipeline_mode to enum."""
        if isinstance(self.pipeline_mode, str):
            self.pipeline_mode = to_enum(self.pipeline_mode, PipelineMode)
    
    def auto_configure_devices(self) -> None:
        """Auto-configure device assignments based on hardware availability.

        Picks from the *compute-usable* CUDA indices (unsupported archs / iGPUs are
        filtered out), so indices are not assumed contiguous. With 2+ usable GPUs,
        text embedding goes on the second to spread load; otherwise everything shares
        the first usable GPU, or CPU when none are usable.
        """
        from ..devices.capability import usable_gpu_indices

        devs = usable_gpu_indices()

        if not devs:
            # No usable GPUs, use CPU for all
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
