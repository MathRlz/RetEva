"""Multimodal CLAP audio-text embedding model."""
from dataclasses import dataclass, field
from typing import List, Optional, NamedTuple
import pickle
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, WhisperModel
from ..base import AudioEmbeddingModel, TextEmbeddingModel
from ..registry import register_audio_embedding_model


# ============================================================================
# Custom Unpickler for Loading Checkpoints
# ============================================================================

class RemapUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps clap_model references to local classes."""
    
    def find_class(self, module, name):
        # Remap clap_model.config classes to local classes
        if module == 'clap_model.config':
            # Return the class from the current module's globals
            return globals()[name]
        # Remap clap_model.model classes to local classes
        elif module == 'clap_model.model':
            return globals()[name]
        # Remap clap_model.encoders classes to local classes
        elif module == 'clap_model.encoders':
            return globals()[name]
        return super().find_class(module, name)


def load_checkpoint_with_remap(path, map_location=None):
    """Load checkpoint with module remapping using sys.modules trick."""
    import sys
    from types import ModuleType
    
    # Create fake modules to satisfy the unpickler
    fake_config = ModuleType('clap_model.config')
    fake_model = ModuleType('clap_model.model')
    fake_encoders = ModuleType('clap_model.encoders')
    fake_clap_model = ModuleType('clap_model')
    
    # Populate with our local classes
    for name in ['TextEncoderConfig', 'AudioEncoderConfig', 'ProjectionConfig', 'CLAPConfig']:
        if name in globals():
            setattr(fake_config, name, globals()[name])
    
    for name in ['CLAP', 'CLAPOutput', 'ProjectionHead', 'CrossModalAttention']:
        if name in globals():
            setattr(fake_model, name, globals()[name])
    
    for name in ['TextEncoder', 'AudioEncoder']:
        if name in globals():
            setattr(fake_encoders, name, globals()[name])
    
    # Temporarily inject into sys.modules
    sys.modules['clap_model'] = fake_clap_model
    sys.modules['clap_model.config'] = fake_config
    sys.modules['clap_model.model'] = fake_model
    sys.modules['clap_model.encoders'] = fake_encoders
    
    try:
        # Now load with torch.load (handles persistent IDs properly)
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    finally:
        # Clean up
        sys.modules.pop('clap_model', None)
        sys.modules.pop('clap_model.config', None)
        sys.modules.pop('clap_model.model', None)
        sys.modules.pop('clap_model.encoders', None)
    
    return checkpoint


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class TextEncoderConfig:
    """Configuration for text encoder."""
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    max_length: int = 512


@dataclass
class AudioEncoderConfig:
    """Configuration for audio encoder."""
    model_name: str = "openai/whisper-base"
    sample_rate: int = 16000
    max_duration: float = 30.0


@dataclass
class ProjectionConfig:
    """Configuration for projection heads."""
    input_dim_text: int = 768
    input_dim_audio: int = 512
    projection_dim: int = 512
    hidden_dim: Optional[int] = None  # None = linear projection
    dropout: float = 0.1


@dataclass
class CLAPConfig:
    """Full CLAP model configuration."""
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    audio_encoder: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    
    # Contrastive loss settings
    temperature: float = 0.07
    learnable_temperature: bool = True
    
    # Loss weights
    cross_attention_loss_weight: float = 0.4
    contrastive_loss_weight: float = 0.4
    alignment_loss_weight: float = 0.1
    embedding_alignment_loss_weight: float = 0.1


# ============================================================================
# Encoder Classes
# ============================================================================

class AudioEncoder(nn.Module):
    """Audio encoder using Whisper's encoder."""
    
    def __init__(self, config: Optional[AudioEncoderConfig] = None):
        super().__init__()
        self.config = config or AudioEncoderConfig()
        
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.model_name
        )
        
        whisper = WhisperModel.from_pretrained(self.config.model_name)
        self.model = whisper.encoder
        
        self.sample_rate = self.config.sample_rate
        self.max_duration = self.config.max_duration
        
    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self.model.config.d_model
    
    def forward(
        self,
        input_features: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode audio features to sequence representations.
        
        Args:
            input_features: Mel spectrogram features (batch, n_mels, time)
            
        Returns:
            Audio sequence features (batch, time, hidden_dim)
        """
        outputs = self.model(input_features, attention_mask=attn_mask)
        return outputs.last_hidden_state
    
    def freeze(self):
        """Freeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = True


class TextEncoder(nn.Module):
    """Text encoder using pretrained transformer models."""
    
    def __init__(self, config: Optional[TextEncoderConfig] = None):
        super().__init__()
        self.config = config or TextEncoderConfig()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        
    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self.model.config.hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """
        Encode tokenized text to sequence features.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Tuple of (features, attention_mask)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return outputs.last_hidden_state, attention_mask
    
    def freeze(self):
        """Freeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = True


# ============================================================================
# CLAP Model Components
# ============================================================================

class CLAPOutput(NamedTuple):
    """Output from CLAP model."""
    text_embeds: torch.Tensor
    audio_embeds: torch.Tensor
    text_contrastive_embeds: torch.Tensor
    audio_contrastive_embeds: torch.Tensor
    text_seq: torch.Tensor
    audio_seq: torch.Tensor
    cross_attention_t2a: torch.Tensor
    cross_attention_a2t: torch.Tensor
    logits_per_text: torch.Tensor
    logits_per_audio: torch.Tensor
    contrastive_logits_per_text: torch.Tensor
    contrastive_logits_per_audio: torch.Tensor
    loss: Optional[torch.Tensor]


class ProjectionHead(nn.Module):
    """Projection head to map encoder outputs to shared space."""
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, projection_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, projection_dim),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fine-grained alignment."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            query: (batch, seq_q, dim)
            key: (batch, seq_k, dim)
            value: (batch, seq_k, dim)
            key_mask: (batch, seq_k) - True for valid positions
            
        Returns:
            output: (batch, seq_q, dim)
            attention_weights: (batch, seq_q, seq_k)
        """
        batch_size, seq_q, dim = query.shape
        seq_k = key.shape[1]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if key_mask is not None:
            mask = key_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_q, dim)
        output = self.out_proj(output)
        
        # Average attention weights across heads
        attn_weights_avg = attn_weights.mean(dim=1)
        
        return output, attn_weights_avg


class CLAP(nn.Module):
    """CLAP model with cross-attention for fine-grained audio-text alignment."""
    
    def __init__(self, config: Optional[CLAPConfig] = None):
        super().__init__()
        self.config = config or CLAPConfig()
        
        # Encoders
        self.text_encoder = TextEncoder(self.config.text_encoder)
        self.audio_encoder = AudioEncoder(self.config.audio_encoder)
        
        # Get actual encoder dimensions
        text_hidden_dim = self.text_encoder.output_dim
        audio_hidden_dim = self.audio_encoder.output_dim
        
        proj_config = self.config.projection
        proj_dim = proj_config.projection_dim
        
        # Sequence projection heads
        self.text_seq_projection = ProjectionHead(
            input_dim=text_hidden_dim,
            projection_dim=proj_dim,
            hidden_dim=proj_config.hidden_dim,
            dropout=proj_config.dropout,
        )
        
        self.audio_seq_projection = ProjectionHead(
            input_dim=audio_hidden_dim,
            projection_dim=proj_dim,
            hidden_dim=proj_config.hidden_dim,
            dropout=proj_config.dropout,
        )
        
        # Contrastive heads for independent encoding
        self.text_contrastive_head = ProjectionHead(
            input_dim=text_hidden_dim,
            projection_dim=proj_dim,
            hidden_dim=proj_config.hidden_dim,
            dropout=proj_config.dropout,
        )
        
        self.audio_contrastive_head = ProjectionHead(
            input_dim=audio_hidden_dim,
            projection_dim=proj_dim,
            hidden_dim=proj_config.hidden_dim,
            dropout=proj_config.dropout,
        )
        
        # Cross-modal attention
        self.text_to_audio_attention = CrossModalAttention(
            dim=proj_dim,
            num_heads=8,
            dropout=proj_config.dropout,
        )
        
        self.audio_to_text_attention = CrossModalAttention(
            dim=proj_dim,
            num_heads=8,
            dropout=proj_config.dropout,
        )
        
        # Learnable pooling queries
        self.text_pool_query = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.audio_pool_query = nn.Parameter(torch.randn(1, 1, proj_dim))
        
        # Layer normalization
        self.text_cross_norm = nn.LayerNorm(proj_dim)
        self.audio_cross_norm = nn.LayerNorm(proj_dim)
        
        # Pooling attention
        self.text_pool_attention = CrossModalAttention(
            dim=proj_dim,
            num_heads=8,
            dropout=proj_config.dropout,
        )
        
        self.audio_pool_attention = CrossModalAttention(
            dim=proj_dim,
            num_heads=8,
            dropout=proj_config.dropout,
        )
        
        # Temperature parameter
        if self.config.learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.config.temperature))
            )
        else:
            self.register_buffer(
                'log_temperature',
                torch.log(torch.tensor(self.config.temperature))
            )
    
    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()
    
    @property
    def projection_dim(self) -> int:
        return self.config.projection.projection_dim
    
    def encode_text_contrastive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text using contrastive head (independent encoding for retrieval)."""
        hidden_states, mask = self.text_encoder(input_ids, attention_mask)
        text_features = self.text_contrastive_head(hidden_states)
        
        # Mean pooling over valid tokens
        mask_expanded = mask.unsqueeze(-1).float()
        sum_embeddings = (text_features * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        text_embed = sum_embeddings / sum_mask
        
        # Normalize
        text_embed = F.normalize(text_embed, p=2, dim=-1)
        return text_embed
    
    def encode_audio_contrastive(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio using contrastive head (independent encoding for retrieval)."""
        hidden_states = self.audio_encoder(input_features, attention_mask)
        audio_features = self.audio_contrastive_head(hidden_states)
        
        # Mean pooling over frames
        audio_embed = audio_features.mean(dim=1)
        
        # Normalize
        audio_embed = F.normalize(audio_embed, p=2, dim=-1)
        return audio_embed
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to single embedding (for inference/retrieval)."""
        return self.encode_text_contrastive(input_ids, attention_mask)
    
    def encode_audio(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio to single embedding (for inference/retrieval)."""
        return self.encode_audio_contrastive(input_features, attention_mask)
    
    def freeze_encoders(self):
        """Freeze both encoder backbones."""
        self.text_encoder.freeze()
        self.audio_encoder.freeze()
    
    def unfreeze_encoders(self):
        """Unfreeze both encoder backbones."""
        self.text_encoder.unfreeze()
        self.audio_encoder.unfreeze()


# ============================================================================
# Multimodal Wrapper for Evaluation Pipeline
# ============================================================================


@register_audio_embedding_model('clap_style', description='CLAP-style multimodal audio-text embedding model')
class MultimodalClapStyleModel(AudioEmbeddingModel, TextEmbeddingModel):

    @dataclass
    class Params:
        model_path: str = ""
    """
    Multimodal CLAP model that can encode both audio and text.
    Implements both AudioEmbeddingModel and TextEmbeddingModel interfaces.
    Uses the contrastive heads for retrieval (independent encoding).
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda:0"):
        """
        Initialize CLAP-style audio model.
        
        Args:
            model_path: Path to trained CLAP model checkpoint (.pt file)
            device: Device to load model on
        """
        self.model_path = model_path
        self.device = torch.device(device)
        
        # Load checkpoint with custom unpickler to remap clap_model references
        checkpoint = load_checkpoint_with_remap(model_path, map_location=self.device)
        
        # Extract config if saved with checkpoint, otherwise use default
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = CLAPConfig()
        
        # Initialize model
        self.model = CLAP(config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize audio preprocessor
        audio_encoder_name = config.audio_encoder.model_name
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(audio_encoder_name)
        self.sample_rate = config.audio_encoder.sample_rate
        
        # Initialize text tokenizer
        text_encoder_name = config.text_encoder.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.max_length = config.text_encoder.max_length
        
        self.emb_dim = config.projection.projection_dim
    
    def preprocess_audio(self, audio_list: List[torch.Tensor], 
                        sampling_rates: List[int]):
        """Preprocess audio using Whisper feature extractor."""
        import torchaudio
        
        processed_audio = []
        
        for idx in range(len(audio_list)):
            if not isinstance(audio_list[idx], torch.Tensor):
                audio_list[idx] = torch.tensor(audio_list[idx])
            
            sr = sampling_rates[idx]
            audio_sample = audio_list[idx]
            
            # Resample to 16kHz if needed
            if sr != self.sample_rate:
                audio_list[idx] = torchaudio.functional.resample(
                    audio_sample, sr, self.sample_rate
                )
            
            processed_audio.append(audio_list[idx].squeeze().numpy())
        
        # Extract features
        inputs = self.feature_extractor(
            processed_audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        
        return inputs.input_features
    
    def encode_from_features(self, features: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Encode preprocessed audio features into embeddings using contrastive head.
        
        Args:
            features: Preprocessed audio features
            attention_mask: Optional attention mask (unused for CLAP)
            
        Returns:
            embeddings: (batch, emb_dim) numpy array
        """
        features = features.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_audio(features)
        
        return embeddings.cpu().numpy()
    
    def encode_audio(self, audio_list: List[torch.Tensor], 
                    sampling_rates: List[int], 
                    show_progress: bool = False) -> np.ndarray:
        """
        Encode audio directly into embeddings using contrastive head.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: List of sampling rates
            show_progress: Whether to show progress (unused)
            
        Returns:
            embeddings: (batch, emb_dim) numpy array
        """
        # Preprocess audio
        input_features = self.preprocess_audio(audio_list, sampling_rates)
        input_features = input_features.to(self.device)
        
        with torch.no_grad():
            # Use contrastive head for independent encoding (for retrieval)
            embeddings = self.model.encode_audio(input_features)
        
        return embeddings.cpu().numpy()
    
    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode texts (TextEmbeddingModel interface - delegates to encode_text).
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress (unused)
            
        Returns:
            embeddings: (batch, emb_dim) numpy array
        """
        return self.encode_text(texts)
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using CLAP's internal text encoder (for proper evaluation).
        
        Args:
            texts: List of text strings
            
        Returns:
            embeddings: (batch, emb_dim) numpy array
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # Use contrastive head for independent encoding (matches audio encoding)
            embeddings = self.model.encode_text(input_ids, attention_mask)
        
        return embeddings.cpu().numpy()
    
    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def name(self) -> str:
        """Return model name."""
        return f"ClapStyleAudioModel - path:{self.model_path} - emb_dim:{self.emb_dim}"
