"""Phase 1.5 (graph-first): ``ModelServiceProvider.get_*`` accept ``**kwargs`` and thread them to
the factory, registry-driven — so a model's own ``Params`` (e.g. ``pooling``) flow through without
the provider knowing them. The kwargs fold into the cache key, so distinct param sets cache as
distinct models (``pooling=mean_abtt`` ≠ ``attention``) while no-kwargs callers key as before.
"""

import evaluator.services.model_provider as mp
from evaluator.services.model_provider import ModelServiceProvider, _kw_key


def _patch_audio_factory(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return object()  # a fresh, distinct model per build

    monkeypatch.setattr(mp, "factory_create_audio_embedding_model", fake)
    return calls


def test_kwargs_thread_to_factory(monkeypatch):
    calls = _patch_audio_factory(monkeypatch)
    p = ModelServiceProvider()
    p.get_audio_embedding_model("attention_pool", "n", "/p", 8, "cpu", pooling="mean_abtt")
    assert calls == [dict(
        model_type="attention_pool", model_name="n", model_path="/p",
        emb_dim=8, device="cpu", pooling="mean_abtt",
    )]


def test_distinct_kwargs_cache_distinctly(monkeypatch):
    calls = _patch_audio_factory(monkeypatch)
    p = ModelServiceProvider()
    base = ("attention_pool", "n", "/p", 8, "cpu")
    m1 = p.get_audio_embedding_model(*base, pooling="mean_abtt")
    m1b = p.get_audio_embedding_model(*base, pooling="mean_abtt")  # same kwargs → cached
    m2 = p.get_audio_embedding_model(*base, pooling="attention")   # different → new model
    assert m1 is m1b              # one build for the repeated param set
    assert m1 is not m2           # pooling=mean_abtt and pooling=attention are different models
    assert len(calls) == 2        # exactly two distinct builds


def test_no_kwargs_keys_unchanged():
    # back-compat: a no-kwargs call contributes no key fragment, so callers cache exactly as before.
    assert _kw_key({}) == ()
    assert _kw_key({"pooling": "mean_abtt"}) != ()
    # order-independent: same dict, different insertion order → identical key fragment.
    assert _kw_key({"a": 1, "b": 2}) == _kw_key({"b": 2, "a": 1})
