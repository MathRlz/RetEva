"""Base class for HuggingFace-based ASR models."""
from typing import List, Optional
import torch
import torchaudio
from ..base import ASRModel


class HuggingFaceASRModel(ASRModel):
    """
    Base class for HuggingFace Transformer-based ASR models.
    
    Provides common functionality for ASR models using HuggingFace transformers,
    including:
    - Device management
    - PEFT/LoRA adapter loading
    - Audio preprocessing (resampling, tensor conversion)
    - Batch processing structure
    
    Subclasses must implement:
    - _create_processor(): Initialize the model-specific processor
    - _create_model(): Initialize the model-specific architecture
    - _extract_features(): Extract features using the processor
    - _generate_transcriptions(): Generate transcriptions from features
    """
    
    def __init__(self, model_name: str, adapter_path: Optional[str] = None):
        """
        Initialize the HuggingFace ASR model.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'openai/whisper-small')
            adapter_path: Optional path to PEFT/LoRA adapter weights
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        
        # Initialize processor and model (to be implemented by subclasses)
        self.processor = self._create_processor()
        self.model = self._create_model()
        
        # Load adapter if provided
        if adapter_path:
            self._load_adapter(adapter_path)
        
        # Default to CPU
        self.device = torch.device("cpu")
    
    def _create_processor(self):
        """Create and return the model-specific processor. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_processor()")
    
    def _create_model(self):
        """Create and return the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_model()")
    
    def _load_adapter(self, adapter_path: str):
        """
        Load PEFT/LoRA adapter weights and merge with base model.
        
        Args:
            adapter_path: Path to adapter weights
            
        Raises:
            ImportError: If peft library is not installed
        """
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        except ImportError:
            raise ImportError(
                "peft library required for adapter loading. Install with: pip install peft"
            )
    
    def to(self, device: torch.device):
        """
        Move model to specified device.
        
        Args:
            device: Target device (cpu, cuda, mps, etc.)
            
        Returns:
            Self for chaining
        """
        self.model.to(device)
        self.device = device
        return self
    
    def preprocess(self, audio_list: List, sampling_rates: List[int]):
        """
        Preprocess audio for model input.
        
        Converts audio to tensors, resamples to 16kHz if needed, and extracts features.
        
        Args:
            audio_list: List of audio arrays or tensors
            sampling_rates: Corresponding sampling rates
            
        Returns:
            Tuple of (features, attention_mask) ready for model input
        """
        processed_audio = []
        
        # Convert to tensors and resample if needed
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
        
        # Extract features using model-specific processor
        return self._extract_features(processed_audio)
    
    def _extract_features(self, processed_audio: List):
        """
        Extract features from preprocessed audio using the processor.
        Must be implemented by subclasses.
        
        Args:
            processed_audio: List of preprocessed audio arrays (16kHz, numpy)
            
        Returns:
            Tuple of (features, attention_mask)
        """
        raise NotImplementedError("Subclasses must implement _extract_features()")
    
    def transcribe_from_features(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """
        Generate transcriptions from preprocessed features.
        
        Args:
            features: Preprocessed audio features
            attention_mask: Optional attention mask
            language: Optional language code for multilingual models
            
        Returns:
            List of transcription strings
        """
        with torch.no_grad():
            features = features.to(self.device)
            attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
            
            # Generate transcriptions using model-specific method
            transcripts = self._generate_transcriptions(features, attention_mask, language)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        return transcripts
    
    def _generate_transcriptions(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        language: Optional[str]
    ) -> List[str]:
        """
        Generate transcriptions from features. Must be implemented by subclasses.
        
        Args:
            features: Features on device
            attention_mask: Attention mask on device
            language: Optional language code
            
        Returns:
            List of transcription strings
        """
        raise NotImplementedError("Subclasses must implement _generate_transcriptions()")
    
    def transcribe(
        self,
        audio: List[torch.Tensor],
        sampling_rates: List[int],
        language: Optional[str] = None
    ) -> List[str]:
        """
        Transcribe raw audio into text.
        
        Args:
            audio: List of audio tensors
            sampling_rates: Corresponding sampling rates
            language: Optional language code for multilingual models
            
        Returns:
            List of transcription strings
        """
        features, attention_mask = self.preprocess(audio, sampling_rates)
        return self.transcribe_from_features(features, attention_mask, language)
    
    def name(self) -> str:
        """
        Return human-readable model name.
        
        Returns:
            Model name string including adapter info if present
        """
        class_name = self.__class__.__name__
        name = f"{class_name} - {self.model_name}"
        if self.adapter_path:
            name += f" + adapter({self.adapter_path})"
        return name
