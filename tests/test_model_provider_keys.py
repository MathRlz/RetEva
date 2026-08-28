"""B3/F16: ModelServiceProvider builds service-bucket keys via one `_key_for(family, **kw)`.

Pins the per-family tuple shape so a future field reorder can't silently split the get key.
"""

import pytest

from evaluator.services.model_provider import ModelServiceProvider


def test_key_for_tuple_shapes_match_legacy():
    P = ModelServiceProvider
    assert P._key_for(
        "asr", model_type="whisper", model_name="m", adapter_path=None, device="cuda:0"
    ) == ("whisper", "m", None, "cuda:0")
    assert P._key_for("text", model_type="jina", model_name=None, device="cpu") == (
        "jina", None, "cpu",
    )
    assert P._key_for(
        "audio", model_type="ap", model_name="n", model_path=None,
        emb_dim=8, device="cuda:1",
    ) == ("ap", "n", None, 8, "cuda:1")
    assert P._key_for(
        "reranker", model_type="ce", model_name=None, device="cpu",
        batch_size=32, max_length=512,
    ) == ("ce", None, "cpu", 32, 512)


def test_key_for_unknown_family_raises():
    with pytest.raises(ValueError, match="bogus"):
        ModelServiceProvider._key_for("bogus", model_type="x")
