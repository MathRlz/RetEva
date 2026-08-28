"""4b: audio augmentation's per-item decode + perturb + write runs through `parallel_map`, so the
`cpu_stage_executor` knob (sync/thread/process) applies — and every backend is byte-identical (the
perturbation is a pure function of (audio, sr, seed), and `_augment_audio_one` is picklable)."""
import functools
import os
import tempfile

import numpy as np
import soundfile as sf

from evaluator.evaluation.executor.cpu_parallel import parallel_map
from evaluator.evaluation.handlers.audio import _augment_audio_one
from evaluator.config.audio_augmentation import AudioAugmentationConfig
from evaluator.pipeline.audio.augmentation import AudioAugmenter


def _make_clips(d, n=6):
    items = []
    for i in range(n):
        x = (np.random.RandomState(i).randn(8000).astype("float32")) * 0.1
        p = os.path.join(d, f"clip{i}.wav")
        sf.write(p, x, 16000)
        items.append((f"q{i}", p))
    return items


def _augment_all(backend, clips, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    aug = AudioAugmenter(
        AudioAugmentationConfig(enabled=True, add_noise=True, snr_db=15.0)
    )
    fn = functools.partial(
        _augment_audio_one, augmenter=aug, n_variants=2, base_seed=42,
        node_id="augment_audio", out_dir=out_dir,
    )
    return parallel_map(fn, clips, backend=backend, workers=2)


def _read(pairs_per_item):
    out = []
    for pairs in pairs_per_item:
        for vid, path in pairs:
            data, sr = sf.read(path)
            out.append((vid, sr, np.asarray(data, dtype="float32")))
    return out


def test_augment_audio_backends_are_byte_identical():
    with tempfile.TemporaryDirectory() as d:
        clips = _make_clips(d)
        base = _read(_augment_all("sync", clips, os.path.join(d, "sync")))
        thr = _read(_augment_all("thread", clips, os.path.join(d, "thread")))
        proc = _read(_augment_all("process", clips, os.path.join(d, "process")))
        # 6 clips × 2 variants, order preserved, lineage ids q·augN
        assert [v for v, _, _ in base] == [
            f"q{i}·aug{n}" for i in range(6) for n in range(2)
        ]
        # same vids+sr across backends, and byte-identical perturbed audio
        assert [(v, sr) for v, sr, _ in thr] == [(v, sr) for v, sr, _ in base]
        assert [(v, sr) for v, sr, _ in proc] == [(v, sr) for v, sr, _ in base]
        for (_, _, a), (_, _, b) in zip(base, thr):
            assert np.array_equal(a, b)
        for (_, _, a), (_, _, b) in zip(base, proc):
            assert np.array_equal(a, b)
