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
    """Audio transcription; corpus is speaker-split-derived (no native validation split)."""

    domain = "medical"
    splits = ("train", "test")
    default_split = "both"  # the full corpus unless a train/test holdout is requested

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        import pandas as pd
        from ..core import load_admed_voice_corpus, load_audio_file
        from ..descriptor import resolve_split
        from ..runtime import AudioSamplesQueryDataset
        from ..loaders.base import AudioSample

        split = (resolve_split(data, cls.default_split) or "both").lower()
        if split == "validation":
            raise ValueError(
                "admed_voice has no 'validation' split (train / test / both only)"
            )
        base_path = getattr(data, "data_path", None)
        # train/test are a speaker-disjoint holdout; need a non-zero ratio to form them.
        test_size = getattr(data, "test_size", 0.0)
        if split in ("train", "test") and not test_size:
            test_size = 0.3
        train_df, test_df = load_admed_voice_corpus(
            test_size=test_size, base_path=base_path
        )
        if split == "train":
            df = train_df
        elif split == "test":
            df = test_df
        else:  # both — the full corpus
            df = pd.concat([train_df, test_df]) if len(test_df) else train_df

        samples = []
        for idx, row in df.reset_index(drop=True).iterrows():
            waveform, sr = load_audio_file(row["file_path"])
            # admed filenames repeat across category/speaker dirs (the unique key is the full
            # path) — bare-filename sample_ids collide, and duplicate question_ids make
            # dataset_source drop ALL ground truth (→ empty WER/CER + IR metrics). Build a
            # unique id from the path components.
            uid_parts = [
                str(row[c])
                for c in ("source", "cat_code", "rec_place", "speaker_id", "filename")
                if c in row and row[c] is not None
            ]
            sample_id = "_".join(uid_parts) if uid_parts else str(idx)
            samples.append(AudioSample(
                audio_array=waveform.squeeze().numpy(),
                sampling_rate=sr,
                transcription=str(row["phrase"]),
                sample_id=sample_id,
                language="pl",
                metadata={"speaker_id": str(row["speaker_id"])} if "speaker_id" in row else {},
            ))
        return AudioSamplesQueryDataset(
            samples,
            trace_limit=getattr(data, "trace_limit", 0),
        )
