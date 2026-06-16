"""ADMED Voice — Polish medical spoken-query dataset (audio transcription)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import AudioTranscriptionDataset, register_eval_dataset

if TYPE_CHECKING:
    from ...config.data import DataConfig
    from ..core import QueryDataset


@register_eval_dataset(
    id="admed_voice",
    description="ADMED Voice: Polish medical spoken-query dataset",
)
class AdmedVoiceDataset(AudioTranscriptionDataset):
    """Audio transcription; pure fit for the type defaults (no overrides)."""

    domain = "medical"

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        from ..core import load_admed_voice_corpus, load_audio_file
        from ..runtime import AudioSamplesQueryDataset
        from ..loaders.base import AudioSample

        test_size = getattr(data, "test_size", 0.0)
        train_df, _ = load_admed_voice_corpus(test_size=test_size)

        samples = []
        for idx, row in train_df.reset_index(drop=True).iterrows():
            waveform, sr = load_audio_file(row["file_path"])
            samples.append(AudioSample(
                audio_array=waveform.squeeze().numpy(),
                sampling_rate=sr,
                transcription=str(row["phrase"]),
                sample_id=str(row.get("filename", str(idx))),
                language="pl",
                metadata={"speaker_id": str(row["speaker_id"])} if "speaker_id" in row else {},
            ))
        return AudioSamplesQueryDataset(
            samples,
            trace_limit=getattr(data, "trace_limit", 0),
        )
