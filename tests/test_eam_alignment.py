"""EAM alignment model (side/eam AlignmentModel architecture) — audio + text embedding."""

import pytest

torch = pytest.importorskip("torch")

from evaluator.models.a2e import eam_alignment as eam  # noqa: E402
from evaluator.models.factory import (  # noqa: E402
    create_audio_embedding_model,
    create_text_embedding_model,
)


class _FakeAudioBackend:
    def __init__(self, hidden_dim=8):
        self.hidden_dim = hidden_dim
        self.encoder = torch.nn.Identity()
        self.processor = object()

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def preprocess(self, audio_list, sampling_rates):
        return torch.stack(audio_list), torch.ones(len(audio_list), audio_list[0].shape[0])

    def run_encoder(self, encoder, features, mask):
        return features, mask


class _FakeTextBackend:
    def __init__(self, hidden_dim=8):
        self.hidden_dim = hidden_dim
        self.encoder = torch.nn.Identity()
        self.processor = object()

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def preprocess(self, texts):
        batch = len(texts)
        return torch.zeros(batch, 3, dtype=torch.long), torch.ones(batch, 3)

    def run_encoder(self, encoder, input_ids, attention_mask):
        batch = input_ids.shape[0]
        return torch.randn(batch, input_ids.shape[1], self.hidden_dim)


def _patch_backends(monkeypatch, hidden_dim=8):
    monkeypatch.setattr(eam, "_select_backend", lambda name: _FakeAudioBackend(hidden_dim))
    monkeypatch.setattr(eam, "_M4TTextBackend", lambda name: _FakeTextBackend(hidden_dim))


def _full_checkpoint(input_dim=8, embedding_dim=6, num_heads=2, dropout=0.1,
                     include_non_inference=True):
    """Build a real state_dict by instantiating the module classes themselves, so key
    names/shapes are exactly what the model expects."""
    audio_sa = eam._SelfAttentionBlock(input_dim, num_heads, dropout)
    text_sa = eam._SelfAttentionBlock(input_dim, num_heads, dropout)
    pool = eam._AttentionPool(input_dim)
    proj = eam._SharedProjection(input_dim, embedding_dim)
    sd = {f"audio_sa.{k}": v for k, v in audio_sa.state_dict().items()}
    sd.update({f"text_sa.{k}": v for k, v in text_sa.state_dict().items()})
    sd.update({f"pool.{k}": v for k, v in pool.state_dict().items()})
    sd.update({f"proj.{k}": v for k, v in proj.state_dict().items()})
    if include_non_inference:
        # cross_attn/adapters — real side/eam checkpoints carry these (cross_attn is always
        # constructed); must be ignored, not loaded into anything.
        cross_attn = torch.nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        sd.update({f"cross_attn.attn.{k}": v for k, v in cross_attn.state_dict().items()})
    return sd


def test_infers_dims_from_checkpoint_not_hardcoded(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6)
    ckpt = tmp_path / "eam.pt"
    torch.save(sd, ckpt)
    model = eam.EamAlignmentModel(model_path=str(ckpt))
    assert model.core.hidden_dim == 8
    assert model.core.embedding_dim == 6


def test_missing_model_path_raises(monkeypatch):
    _patch_backends(monkeypatch)
    with pytest.raises(ValueError, match="model_path"):
        eam.EamAlignmentModel(model_path=None)


def test_mismatched_encoder_hidden_dim_raises(monkeypatch, tmp_path):
    # Checkpoint trained with input_dim=4 but encoder hidden_dim=8 -> loud error.
    _patch_backends(monkeypatch, hidden_dim=8)
    sd = _full_checkpoint(input_dim=4, embedding_dim=6)
    ckpt = tmp_path / "mismatch.pt"
    torch.save(sd, ckpt)
    with pytest.raises(ValueError, match="input_dim"):
        eam.EamAlignmentModel(model_path=str(ckpt))


def test_truncated_audio_sa_raises(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6)
    del sd["audio_sa.norm.weight"]
    ckpt = tmp_path / "truncated_audio.pt"
    torch.save(sd, ckpt)
    with pytest.raises(ValueError, match="random init"):
        eam.EamAlignmentModel(model_path=str(ckpt))


def test_truncated_text_sa_raises(monkeypatch, tmp_path):
    # text_sa is a real inference-time module (encode_text uses it) — must be strict too.
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6)
    del sd["text_sa.norm.weight"]
    ckpt = tmp_path / "truncated_text.pt"
    torch.save(sd, ckpt)
    with pytest.raises(ValueError, match="random init"):
        eam.EamAlignmentModel(model_path=str(ckpt))


def test_missing_cross_attn_and_adapters_does_not_raise(monkeypatch, tmp_path):
    # cross_attn/adapters are never used at inference — absent (or present) must not fail.
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6, include_non_inference=False)
    ckpt = tmp_path / "no_extras.pt"
    torch.save(sd, ckpt)
    model = eam.EamAlignmentModel(model_path=str(ckpt))
    assert model.core.embedding_dim == 6


def test_encode_audio_shape_and_normalized(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6)
    ckpt = tmp_path / "eam.pt"
    torch.save(sd, ckpt)
    model = eam.EamAlignmentModel(model_path=str(ckpt))
    out = model.encode_from_features(torch.randn(2, 3, 8))
    assert out.shape == (2, 6)
    norms = (out ** 2).sum(axis=-1) ** 0.5
    assert norms == pytest.approx([1.0, 1.0], abs=1e-5)


def test_encode_text_shape_and_normalized(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6)
    ckpt = tmp_path / "eam.pt"
    torch.save(sd, ckpt)
    model = eam.EamAlignmentModel(model_path=str(ckpt))
    out = model.encode(["a question", "another question"])
    assert out.shape == (2, 6)
    norms = (out ** 2).sum(axis=-1) ** 0.5
    assert norms == pytest.approx([1.0, 1.0], abs=1e-5)


def test_audio_and_text_roles_share_one_core_for_the_same_checkpoint(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6)
    ckpt = tmp_path / "eam.pt"
    torch.save(sd, ckpt)
    audio_model = eam.EamAlignmentModel(model_path=str(ckpt))
    text_model = eam.EamAlignmentModel(model_path=str(ckpt))
    assert audio_model.core is text_model.core  # one loaded copy, not two

    other_ckpt = tmp_path / "other.pt"
    torch.save(_full_checkpoint(input_dim=8, embedding_dim=6), other_ckpt)
    other_model = eam.EamAlignmentModel(model_path=str(other_ckpt))
    assert other_model.core is not audio_model.core  # different checkpoint -> different core


def test_factory_round_trip_audio(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6)
    ckpt = tmp_path / "eam.pt"
    torch.save(sd, ckpt)
    model = create_audio_embedding_model("eam_alignment", model_path=str(ckpt), device="cpu")
    assert isinstance(model, eam.EamAlignmentModel)
    assert model.core.embedding_dim == 6


def test_factory_round_trip_text(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    sd = _full_checkpoint(input_dim=8, embedding_dim=6)
    ckpt = tmp_path / "eam.pt"
    torch.save(sd, ckpt)
    model = create_text_embedding_model(
        "eam_alignment", model_name="fake-encoder", model_path=str(ckpt), device="cpu",
    )
    assert isinstance(model, eam.EamAlignmentModel)
    out = model.encode(["hello"])
    assert out.shape == (1, 6)
