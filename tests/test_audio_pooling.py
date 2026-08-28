"""Selectable pooling for the attention-pool APM (attention / mean / whiten / ABTT)."""

import pytest

torch = pytest.importorskip("torch")

from evaluator.models.a2e import attention_pool as ap  # noqa: E402
from evaluator.models.a2e.postprocessing import abtt_batch, whiten_batch  # noqa: E402


def test_make_pooling_builds_each_and_rejects_unknown():
    for kind in ap.POOLING_CHOICES:
        assert isinstance(ap._make_pooling(kind, 4), torch.nn.Module)
    with pytest.raises(ValueError, match="Unknown pooling"):
        ap._make_pooling("bogus", 4)


def test_abtt_removes_top_component_then_normalizes_single():
    # remove [1,0,0] from [5,2,1] -> [0,2,1], then L2 normalize (matches training abtt_batch)
    out = abtt_batch(torch.tensor([[5.0, 2.0, 1.0]]), torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]))
    expected = torch.tensor([[0.0, 2.0, 1.0]])
    expected = expected / expected.norm()
    assert torch.allclose(out, expected)
    assert torch.allclose(out.norm(dim=-1), torch.ones(1))  # rows are unit-norm


def test_abtt_supports_multiple_components_then_normalizes():
    pc = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = abtt_batch(torch.tensor([[5.0, 2.0, 7.0]]), torch.zeros(3), pc)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0]]))  # [0,0,7] normalized


def test_abtt_matches_training_formula():
    # Byte-for-byte vs apm_new/apm/postprocessing.py:abtt_batch (mu (1,D), pc1 (D,1)).
    def _ref(X, mu, pc1):
        X = X - mu
        X = X - (X @ pc1) @ pc1.T
        return X / (X.norm(dim=1, keepdim=True) + 1e-9)

    e = torch.randn(5, 8)
    mu = e.mean(dim=0, keepdim=True)              # (1, 8)
    _, _, V = torch.svd(e - mu)
    pc1 = V[:, 0:1]                               # (8, 1), unit
    assert torch.allclose(abtt_batch(e, mu, pc1), _ref(e, mu, pc1), atol=1e-5)


def test_abtt_accepts_training_convention_stat_shapes():
    # Real APM checkpoints store mu (1,H) and pc1 (H,1) — must work the same as (H,)/(k,H).
    e = torch.randn(4, 8)
    pc = torch.randn(8)
    pc = pc / pc.norm()
    flat = abtt_batch(e, torch.zeros(8), pc)
    shaped = abtt_batch(e, torch.zeros(1, 8), pc.view(8, 1))
    assert torch.allclose(flat, shaped, atol=1e-5)


def test_whiten_centers_and_transforms():
    out = whiten_batch(torch.tensor([[2.0, 4.0]]), torch.tensor([1.0, 1.0]), torch.eye(2) * 2)
    assert torch.allclose(out, torch.tensor([[2.0, 6.0]]))


class _FakeBackend:
    def __init__(self, hidden_dim=4):
        self.hidden_dim = hidden_dim
        self.encoder = torch.nn.Identity()
        self.processor = object()

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def run_encoder(self, encoder, features, mask):
        return features, mask


@pytest.mark.parametrize(
    "pooling, stats",
    [
        ("mean_abtt", {"mu": torch.zeros(1, 4), "pc1": torch.zeros(4, 1)}),
        ("mean_whiten", {"m": torch.zeros(1, 4), "W": torch.eye(4)}),
    ],
)
def test_load_weights_accepts_training_convention_shapes(monkeypatch, tmp_path, pooling, stats):
    # Regression: strict load_state_dict must accept the trailing-singleton stat shapes.
    monkeypatch.setattr(ap, "_select_backend", lambda name: _FakeBackend(4))
    model = ap.AttentionPoolAudioModel(emb_dim=6, pooling=pooling)
    sd = {f"attn_pool.{k}": v for k, v in stats.items()}
    sd.update({f"proj.{k}": v for k, v in model.projection_head.state_dict().items()})
    ckpt = tmp_path / f"{pooling}.pt"
    torch.save(sd, ckpt)
    model._load_weights(str(ckpt))  # was: size-mismatch RuntimeError
    out = model.encode_from_features(torch.randn(2, 3, 4), torch.ones(2, 3))
    assert out.shape == (2, 6)


class _ParamBackend(_FakeBackend):
    """A backend whose encoder has real params, to exercise audio_enc.* loading."""

    def __init__(self, hidden_dim=4):
        super().__init__(hidden_dim)
        self.encoder = torch.nn.Linear(hidden_dim, hidden_dim)

    def parameters(self):
        return self.encoder.parameters()


def _proj_ckpt_keys(model):
    return {f"proj.{k}": v for k, v in model.projection_head.state_dict().items()}


def test_load_weights_loads_encoder_from_checkpoint(monkeypatch, tmp_path):
    # The checkpoint's audio_enc.encoder.* must be loaded into the encoder (so it matches training).
    monkeypatch.setattr(ap, "_select_backend", lambda name: _ParamBackend(4))
    model = ap.AttentionPoolAudioModel(emb_dim=6, pooling="mean")
    new_w, new_b = torch.arange(16.0).reshape(4, 4), torch.arange(4.0)
    sd = {"audio_enc.encoder.weight": new_w, "audio_enc.encoder.bias": new_b}
    sd.update(_proj_ckpt_keys(model))
    ckpt = tmp_path / "enc.pt"
    torch.save(sd, ckpt)
    model._load_weights(str(ckpt))
    assert torch.allclose(model.backend.encoder.weight, new_w)
    assert torch.allclose(model.backend.encoder.bias, new_b)


def test_load_weights_raises_on_encoder_shape_mismatch(monkeypatch, tmp_path):
    # Wrong Whisper size → encoder shape mismatch must be a loud error, not a silent wrong encoder.
    monkeypatch.setattr(ap, "_select_backend", lambda name: _ParamBackend(4))
    model = ap.AttentionPoolAudioModel(emb_dim=6, pooling="mean")
    sd = {"audio_enc.encoder.weight": torch.randn(8, 8), "audio_enc.encoder.bias": torch.randn(8)}
    sd.update(_proj_ckpt_keys(model))
    ckpt = tmp_path / "bad_enc.pt"
    torch.save(sd, ckpt)
    with pytest.raises((RuntimeError, ValueError)):
        model._load_weights(str(ckpt))


def test_load_weights_raises_on_unmatched_encoder(monkeypatch, tmp_path):
    # An encoder whose keys don't match (structurally different) → loud error.
    monkeypatch.setattr(ap, "_select_backend", lambda name: _ParamBackend(4))
    model = ap.AttentionPoolAudioModel(emb_dim=6, pooling="mean")
    sd = {"audio_enc.encoder.bogus": torch.randn(4)}  # no weight/bias → 0/2 matched
    sd.update(_proj_ckpt_keys(model))
    ckpt = tmp_path / "wrong_enc.pt"
    torch.save(sd, ckpt)
    with pytest.raises(ValueError, match="encoder"):
        model._load_weights(str(ckpt))


def test_load_weights_raises_on_missing_projection(monkeypatch, tmp_path):
    # Projection keys under the wrong prefix (proj.* not proj.proj.*) used to silently leave the
    # head at random init — now it must raise.
    monkeypatch.setattr(ap, "_select_backend", lambda name: _FakeBackend(4))
    model = ap.AttentionPoolAudioModel(emb_dim=6, pooling="mean")
    sd = {f"proj.{k.split('.', 1)[1]}": v for k, v in model.projection_head.state_dict().items()}
    ckpt = tmp_path / "no_proj.pt"
    torch.save(sd, ckpt)
    with pytest.raises(ValueError, match="random init"):
        model._load_weights(str(ckpt))


def test_load_weights_loads_full_attention_structure(monkeypatch, tmp_path):
    # The real attention APM keys (attn_pool.w1/w2/attention.0/3 + proj.proj.*) load fully.
    monkeypatch.setattr(ap, "_select_backend", lambda name: _FakeBackend(4))
    model = ap.AttentionPoolAudioModel(emb_dim=6, pooling="attention")
    sd = {f"attn_pool.{k}": v for k, v in model.pooling.state_dict().items()}
    sd.update(_proj_ckpt_keys(model))
    ckpt = tmp_path / "attn.pt"
    torch.save(sd, ckpt)
    model._load_weights(str(ckpt))  # all pooling + projection keys present → no raise
    out = model.encode_from_features(torch.randn(2, 3, 4), torch.ones(2, 3))
    assert out.shape == (2, 6)


def test_load_weights_no_encoder_keys_keeps_hf_encoder(monkeypatch, tmp_path):
    # A pooling-only checkpoint (no audio_enc.*) must leave the HF-pretrained encoder untouched.
    monkeypatch.setattr(ap, "_select_backend", lambda name: _ParamBackend(4))
    model = ap.AttentionPoolAudioModel(emb_dim=6, pooling="mean")
    before = model.backend.encoder.weight.clone()
    sd = _proj_ckpt_keys(model)
    ckpt = tmp_path / "pool_only.pt"
    torch.save(sd, ckpt)
    model._load_weights(str(ckpt))
    assert torch.allclose(model.backend.encoder.weight, before)  # encoder unchanged
