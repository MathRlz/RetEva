"""Shared base for HuggingFace self-supervised speech encoders (HuBERT, WavLM)."""
from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np
from ..base import AudioEmbeddingModel
from ...utils.audio import to_mono_16k


class HFSpeechEncoderEmbedding(AudioEmbeddingModel):

    HF_MODEL_CLASS: str
    DEFAULT_MODEL_NAME: str

    def __init__(self,
                 model_name: Optional[str] = None,
                 pooling: str = "mean"):
        import transformers
        from transformers import Wav2Vec2FeatureExtractor

        model_cls = getattr(transformers, self.HF_MODEL_CLASS)
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.pooling = pooling
        self.device = torch.device("cpu")

        self.model = model_cls.from_pretrained(self.model_name)
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)

        self.hidden_dim = self.model.config.hidden_size

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
        return self

    def preprocess_audio(self, audio_list: List[torch.Tensor],
                        sampling_rates: List[int]):
        processed_audio = [
            to_mono_16k(audio, sr, downmix="mean", as_numpy=True)
            for audio, sr in zip(audio_list, sampling_rates)
        ]

        inputs = self.feature_extractor(
            processed_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )

        return inputs.input_values, inputs.attention_mask

    def encode_from_features(self, features: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        features = features.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            self.model.eval()

            outputs = self.model(
                input_values=features,
                attention_mask=attention_mask,
                output_hidden_states=False
            )

            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            if self.pooling == "mean":
                if attention_mask is not None:
                    hidden_seq_len = hidden_states.size(1)
                    output_lengths = self._get_feat_extract_output_lengths(
                        attention_mask.sum(dim=-1)
                    )
                    # mask out the conv-downsampled padding before mean pooling
                    pooling_mask = torch.zeros(
                        hidden_states.size(0), hidden_seq_len,
                        device=self.device, dtype=torch.bool
                    )
                    for i, length in enumerate(output_lengths):
                        pooling_mask[i, :length] = True

                    hidden_states_masked = hidden_states * pooling_mask.unsqueeze(-1)
                    embeddings = hidden_states_masked.sum(dim=1) / pooling_mask.sum(dim=1, keepdim=True)
                else:
                    embeddings = hidden_states.mean(dim=1)
            else:  # cls - use first token
                embeddings = hidden_states[:, 0, :]

            embeddings = F.normalize(embeddings, p=2, dim=-1)

        torch.cuda.empty_cache()  # release the encoder's activation memory between batches
        return embeddings.cpu().numpy()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1

        conv_kernel = getattr(self.model.config, 'conv_kernel', [10, 3, 3, 3, 3, 2, 2])
        conv_stride = getattr(self.model.config, 'conv_stride', [5, 2, 2, 2, 2, 2, 2])

        for kernel_size, stride in zip(conv_kernel, conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths.to(torch.long)

    def encode_audio(self, audio_list: List[torch.Tensor],
                    sampling_rates: List[int],
                    show_progress: bool = False) -> np.ndarray:
        features, attention_mask = self.preprocess_audio(audio_list, sampling_rates)
        return self.encode_from_features(features, attention_mask)

    def name(self) -> str:
        return f"{type(self).__name__} - model:{self.model_name} - pooling:{self.pooling} - dim:{self.hidden_dim}"
