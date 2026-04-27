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
        
        Sets asr_device, text_emb_device, and audio_emb_device to available devices.
        Avoids assigning cuda:1 if only 1 GPU is available.
        """
        import evaluator.config as config_module
        gpu_count = config_module.get_available_gpu_count()
        
        if gpu_count == 0:
            # No GPUs available, use CPU for all
            self.asr_device = "cpu"
            self.text_emb_device = "cpu"
            self.audio_emb_device = "cpu"
        elif gpu_count == 1:
            # Only one GPU, put everything on cuda:0
            self.asr_device = "cuda:0"
            self.text_emb_device = "cuda:0"
            self.audio_emb_device = "cuda:0"
        else:
            # Multiple GPUs: distribute models (ASR on 0, text emb on 1, audio on 0)
            self.asr_device = "cuda:0"
            self.text_emb_device = "cuda:1"
            self.audio_emb_device = "cuda:0"
