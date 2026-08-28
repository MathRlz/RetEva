"""C1/F20: previously-silent except-and-skip paths now narrow the catch AND log, so a
shrinking dataset / skipped check is visible instead of disappearing.
"""

import logging

import numpy as np

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.validation import _validate_dataset_compat
from evaluator.datasets.loaders.local import LocalAudioDatasetLoader


def test_local_loader_skips_and_logs_unreadable_audio(caplog, tmp_path):
    loader = LocalAudioDatasetLoader(audio_dir=str(tmp_path))
    good, bad = tmp_path / "good.wav", tmp_path / "bad.wav"

    def fake_load(path):
        if path == bad:
            raise OSError("corrupt header")
        return np.zeros(16, dtype="float32"), 16000

    loader._load_audio = fake_load  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING):
        samples = loader._load_from_paired_files([good, bad])

    assert [s.sample_id for s in samples] == ["good"]  # bad one dropped
    assert any("bad.wav" in r.message for r in caplog.records)  # and logged


def test_dataset_compat_check_does_not_blow_up_on_unresolvable(caplog):
    cfg = EvaluationConfig()
    cfg.data.dataset_name = "totally-unknown-dataset-xyz"
    cfg.data.dataset_source = None
    cfg.data.dataset_type = None
    warnings: list = []
    with caplog.at_level(logging.DEBUG):
        _validate_dataset_compat(cfg, warnings)  # must not raise
    # unresolvable → no compat warning appended, and the skip is logged at debug
    assert any("compat check skipped" in r.message for r in caplog.records)


def test_dataset_compat_check_no_longer_flags_pipeline_mode():
    """The "compatible pipeline modes" allowlist was a second, independently-drifting
    classification on top of the graph's own nodes/edges — it went stale (pubmed_qa's
    hand-maintained override omitted audio_emb_retrieval even though
    pubmed_qa_rag_fulltext_apm.yaml runs it correctly via TTS gap-fill) and produced false
    "Results may be incorrect" warnings on working graphs. The check is gone; only the
    audio-generation-support check remains."""
    cfg = EvaluationConfig()
    cfg.data.dataset_name = "pubmed_qa"
    cfg.data.dataset_type = "pubmed_qa"
    cfg.graph_override = {"template": "audio_emb_retrieval"}
    warnings: list = []
    _validate_dataset_compat(cfg, warnings)
    assert not warnings
