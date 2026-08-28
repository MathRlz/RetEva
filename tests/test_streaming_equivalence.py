"""Roadmap 3a: a windowed run reproduces the whole-run report (mock pipelines, no models).

The parity proof for the windowed query-side driver: the same config / dataset / deterministic
mock pipelines, run whole vs windowed, must yield an identical report — including the per-item
metrics and the bootstrap CIs (which resample the accumulated per-item arrays in dataset order).
"""

import numpy as np
import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.datasets.loaders.base import AudioSample
from evaluator.datasets.runtime import AudioSamplesQueryDataset
from evaluator.evaluation.executor.run import run_graph
from evaluator.evaluation.executor.state import EvaluationContext
from evaluator.models.retrieval.strategy import RetrievalStrategyConfig
from evaluator.pipeline.retrieval_pipeline import RetrievalPipeline
from evaluator.storage.cache import CacheManager
from evaluator.storage.vector_store import InMemoryVectorStore


def _vec(text):
    """A deterministic, process-independent 8-d embedding of a text (so whole == windowed)."""
    v = np.zeros(8, dtype="float32")
    for i, ch in enumerate(str(text)):
        v[i % 8] += float(ord(ch) % 17)
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


class _Named:
    def __init__(self, n):
        self._n = n

    def name(self):
        return self._n


class _MockASR:
    model = _Named("mock-asr")

    def process_dataset(self, dataset, **kw):
        # Perfect, per-item-deterministic transcription (independent of window slicing).
        hyps, gts = [], []
        for i in range(len(dataset)):
            row = dataset[i]
            hyps.append(str(row["transcription"]))
            gts.append(str(row["transcription"]))
        return hyps, gts


class _MockTextEmb:
    model = _Named("mock-text")

    def process_batch(self, texts, show_progress=False, desc=None):
        return np.array([_vec(t) for t in texts], dtype="float32")


_CORPUS = [
    {"doc_id": f"d{i}", "text": f"document number {i} about topic {i % 3}"}
    for i in range(6)
]


def _dataset(n=7):
    samples = [
        AudioSample(
            audio_array=np.zeros(8000, dtype="float32"), sampling_rate=16000,
            transcription=f"query about topic {i % 3} item {i}",
            sample_id=f"q{i}", language="en",
            metadata={"groundtruth_doc_ids": [f"d{i % 6}"]},
        )
        for i in range(n)
    ]
    return AudioSamplesQueryDataset(samples, corpus_entries=list(_CORPUS))


def _run(window_size=None, n=7, cpu_executor=None):
    cache = CacheManager(enabled=False)
    cfg = EvaluationConfig()
    cfg.graph_override = {"template": "asr_text_retrieval"}
    cfg.model.asr_model_type = "whisper"
    cfg.model.text_emb_model_type = "jina_v4"
    cfg.compute_confidence_intervals = True  # exercise the order-sensitive bootstrap CIs
    if window_size is not None:
        cfg.streaming.window_size = window_size
    if cpu_executor is not None:
        cfg.cpu_stage_executor = cpu_executor
    context = EvaluationContext(
        retrieval_pipeline=RetrievalPipeline(
            InMemoryVectorStore(), cache, strategy_config=RetrievalStrategyConfig()
        ),
        asr_pipeline=_MockASR(),
        text_embedding_pipeline=_MockTextEmb(),
        cache_manager=cache, k=5, batch_size=4,
    )
    return run_graph(_dataset(n), context, eval_config=cfg)


# Volatile keys that legitimately differ run-to-run (time / environment / provenance identity).
_VOLATILE = {
    "provenance", "processing_time", "total_processing_time", "timestamp", "duration_s",
    "timing", "evaluation_time", "wall_time_s",
}


def _canon(obj):
    """Strip volatile keys recursively so two runs are comparable on their numeric content."""
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in obj.items() if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_canon(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 9)
    return obj


@pytest.mark.parametrize("window_size", [1, 2, 3, 4])
def test_windowed_run_reproduces_whole_run_report(window_size):
    whole = _run(window_size=None)
    windowed = _run(window_size=window_size)  # 7 items, various splits incl. 1-per-window

    # headline metrics identical
    for key in ("MRR", "MAP", "Recall@5", "NDCG@5", "WER", "CER"):
        assert _canon(whole.get(key)) == _canon(windowed.get(key)), (key, window_size)

    # the full report (per-branch scores + bootstrap CIs) identical modulo provenance/timing
    assert _canon(whole["report"]) == _canon(windowed["report"])


def test_single_window_equals_whole():
    # A window >= n collapses to one window → must match the whole run exactly.
    whole = _run(window_size=None)
    one = _run(window_size=99)
    assert _canon(whole["report"]) == _canon(one["report"])


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_cpu_stage_executor_matches_sync(backend):
    # 4b wiring: the per-item WER/CER map runs through parallel_map; thread/process must
    # produce a report byte-identical to the default sync (order-preserving + pure).
    ref = _run(cpu_executor="sync")
    got = _run(cpu_executor=backend)
    assert _canon(ref["report"]) == _canon(got["report"])
    assert "WER" in got and got["WER"] is not None


def test_windowed_actually_ran_metrics():
    windowed = _run(window_size=2)
    assert windowed["pipeline_mode"] == "asr_text_retrieval"
    assert windowed.get("Recall@5") is not None  # query side reached retrieval + metrics


def _run_corrected(window_size=None, n=7):
    """A run with query correction + the C7 opt-in corrected metrics enabled."""
    from evaluator.evaluation.executor.state import RunFeatures

    cache = CacheManager(enabled=False)
    cfg = EvaluationConfig()
    cfg.graph_override = {"template": "asr_text_retrieval"}
    cfg.model.asr_model_type = "whisper"
    cfg.model.text_emb_model_type = "jina_v4"
    cfg.query_correction.enabled = True
    cfg.query_correction.method = "rule"
    cfg.query_correction.corrected_metrics = True
    if window_size is not None:
        cfg.streaming.window_size = window_size
    context = EvaluationContext(
        retrieval_pipeline=RetrievalPipeline(
            InMemoryVectorStore(), cache, strategy_config=RetrievalStrategyConfig()
        ),
        asr_pipeline=_MockASR(),
        text_embedding_pipeline=_MockTextEmb(),
        cache_manager=cache, k=5, batch_size=4,
        features=RunFeatures(query_correction_config=cfg.query_correction),
    )
    return run_graph(_dataset(n), context, eval_config=cfg)


def test_windowed_run_reproduces_corrected_metrics():
    # R3-P1 regression: streaming's accumulate-set derives from finalize BINDINGS; before
    # the terminal reads were declared, corrected_query_text survived only the last window
    # (whole n=7 vs windowed n=1). Declared reads make the lifetime analysis see it.
    whole = _run_corrected(window_size=None)
    windowed = _run_corrected(window_size=3)
    assert _canon(whole["report"]) == _canon(windowed["report"])
    branches = whole["report"]["branches"]
    assert any("corrected_wer" in m for m in branches.values())  # the C7 metrics really ran


# ── Window-granular checkpoint / resume ──────────────────────────────


class _CrashASR(_MockASR):
    """A mock ASR that raises on its N-th process_dataset call (simulates a mid-run crash)."""

    def __init__(self, crash_on_call):
        self._calls = 0
        self._crash_on_call = crash_on_call

    def process_dataset(self, dataset, **kw):
        self._calls += 1
        if self._crash_on_call and self._calls == self._crash_on_call:
            raise RuntimeError(f"injected crash on asr call {self._calls}")
        return super().process_dataset(dataset, **kw)


def _run_ckpt(cache_dir, *, window_size, asr=None, resume=False):
    cache = CacheManager(cache_dir=str(cache_dir), enabled=False)
    cfg = EvaluationConfig()
    cfg.graph_override = {"template": "asr_text_retrieval"}
    cfg.model.asr_model_type = "whisper"
    cfg.model.text_emb_model_type = "jina_v4"
    cfg.streaming.window_size = window_size
    context = EvaluationContext(
        retrieval_pipeline=RetrievalPipeline(
            InMemoryVectorStore(), cache, strategy_config=RetrievalStrategyConfig()
        ),
        asr_pipeline=asr or _MockASR(),
        text_embedding_pipeline=_MockTextEmb(),
        cache_manager=cache, k=5, batch_size=4,
        experiment_id="streaming_resume",   # enables the journal (+ interval below)
        checkpoint_interval=1,
        resume_from_checkpoint=resume,
    )
    return run_graph(_dataset(7), context, eval_config=cfg)


def test_checkpointing_does_not_change_the_report(tmp_path):
    plain = _run(window_size=3)
    checkpointed = _run_ckpt(tmp_path / "a", window_size=3)
    assert _canon(plain["report"]) == _canon(checkpointed["report"])


def test_windowed_resume_after_crash_matches_full_run(tmp_path):
    reference = _run(window_size=3)  # 7 items → windows [0:3] [3:6] [6:7]

    run_dir = tmp_path / "run"
    # crash on the 2nd window's ASR call → window 0 is checkpointed, window 1 isn't
    with pytest.raises(RuntimeError, match="injected crash"):
        _run_ckpt(run_dir, window_size=3, asr=_CrashASR(crash_on_call=2))

    # resume (same cache dir + experiment id + config ⇒ same journal): finishes the run
    resumed = _run_ckpt(run_dir, window_size=3, resume=True)
    assert _canon(resumed["report"]) == _canon(reference["report"])
