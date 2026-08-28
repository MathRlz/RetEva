"""Regression: a multi-variant TTS-comparison graph (several `tts` nodes sharing one
dataset) silently collapsed to whichever engine ran first.

Root cause: `synthesize_missing_query_audio` decided "does this question need
synthesizing" from a bare `question.audio_path` check on the SHARED Question objects.
Once the first tts node (e.g. `tts_xtts`) set it for all questions, every OTHER tts
node's variant (`tts_m4t`, `tts_piper`) saw audio "already available" and skipped
synthesis entirely — reusing the first engine's audio under a different label. Confirmed
live: a 3-engine comparison came back with byte-identical downstream retrieval metrics
for all 3 "different" engines.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.config.audio_synthesis import AudioSynthesisConfig
from evaluator.pipeline.audio import prepare as prepare_mod
from evaluator.storage.cache.manager import CacheManager


class _FakeSynthesizer:
    """Stands in for AudioSynthesizer: records (provider, text) and writes a real file
    at output_path so CacheManager's exists()-based cache-hit check works for real."""

    calls = []

    def __init__(self, config):
        self.config = config

    def synthesize(self, text, output_path=None):
        _FakeSynthesizer.calls.append((self.config.provider, text))
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake-wav")
        return None


@pytest.fixture(autouse=True)
def _patch_synthesizer(monkeypatch):
    _FakeSynthesizer.calls = []
    # prepare.py does `from evaluator.pipeline.audio.synthesis import AudioSynthesizer`
    # INSIDE the function body, re-fetching the name fresh on every call — patch the
    # source module's attribute so that lazy import picks up the fake.
    monkeypatch.setattr(
        "evaluator.pipeline.audio.synthesis.AudioSynthesizer", _FakeSynthesizer,
    )
    yield


def _questions():
    return [
        SimpleNamespace(question_id="q1", question_text="what is the treatment", audio_path=None),
        SimpleNamespace(question_id="q2", question_text="what is the diagnosis", audio_path=None),
    ]


def test_sibling_tts_variants_each_synthesize_their_own_audio(tmp_path):
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    questions = _questions()
    out_dir = str(tmp_path / "audio")
    cfg_a = AudioSynthesisConfig(provider="xtts_v2", language="en", output_dir=out_dir)
    cfg_b = AudioSynthesisConfig(
        provider="piper", voice="en_US-lessac-medium", language="en", output_dir=out_dir,
    )

    n_a = prepare_mod.synthesize_missing_query_audio(questions, cfg_a, cache_manager=cache)
    n_b = prepare_mod.synthesize_missing_query_audio(questions, cfg_b, cache_manager=cache)

    assert n_a == 2 and n_b == 2
    providers_called = {p for p, _ in _FakeSynthesizer.calls}
    assert providers_called == {"xtts_v2", "piper"}  # THE bug: piper used to never run
    assert len(_FakeSynthesizer.calls) == 4  # 2 questions x 2 providers, no false cache hits


def test_rerunning_the_same_config_is_a_real_cache_hit_not_a_resynthesis(tmp_path):
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    questions = _questions()
    cfg = AudioSynthesisConfig(provider="xtts_v2", language="en", output_dir=str(tmp_path / "audio"))

    n1 = prepare_mod.synthesize_missing_query_audio(questions, cfg, cache_manager=cache)
    assert n1 == 2
    assert len(_FakeSynthesizer.calls) == 2

    n2 = prepare_mod.synthesize_missing_query_audio(questions, cfg, cache_manager=cache)
    assert n2 == 2
    assert len(_FakeSynthesizer.calls) == 2  # still 2 — the second call was a pure cache hit


def test_no_cache_manager_legacy_path_trusts_existing_audio_path(tmp_path):
    # A direct caller with no cache_manager (no multi-variant coordination possible) keeps
    # the original, simpler behavior: an already-set audio_path short-circuits entirely.
    questions = _questions()
    questions[0].audio_path = "/already/there.wav"
    cfg = AudioSynthesisConfig(provider="xtts_v2", language="en", output_dir=str(tmp_path))

    n = prepare_mod.synthesize_missing_query_audio(questions, cfg, cache_manager=None)
    assert n == 1  # only q2 synthesized
    assert {p for p, _ in _FakeSynthesizer.calls} == {"xtts_v2"}
    assert len(_FakeSynthesizer.calls) == 1


def test_no_cache_flag_forces_resynthesis_regardless_of_shared_audio_path(tmp_path):
    cache = CacheManager(cache_dir=str(tmp_path), enabled=False)  # --no-cache
    questions = _questions()
    out_dir = str(tmp_path / "audio")
    cfg_a = AudioSynthesisConfig(provider="xtts_v2", language="en", output_dir=out_dir)
    cfg_b = AudioSynthesisConfig(provider="piper", language="en", output_dir=out_dir)

    prepare_mod.synthesize_missing_query_audio(questions, cfg_a, cache_manager=cache)
    prepare_mod.synthesize_missing_query_audio(questions, cfg_b, cache_manager=cache)

    providers_called = {p for p, _ in _FakeSynthesizer.calls}
    assert providers_called == {"xtts_v2", "piper"}
    assert len(_FakeSynthesizer.calls) == 4
