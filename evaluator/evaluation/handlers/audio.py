"""Audio stage handlers: TTS gap-fill + audio-axis augmentation (§4.1 P4).

Split out of ``handlers/source.py`` (B4) — the source module owns dataset/question
concerns; this one owns the audio bricks. Registrations unchanged.
"""

from __future__ import annotations

import time

from ...logging_config import get_logger
from ..executor.state import RunState
from ..audio_refs import audio_refs_from_questions

logger = get_logger(__name__)


def _augment_audio_one(id_path, *, augmenter, n_variants, base_seed, node_id, out_dir):
    """Per-item audio augmentation — the 4b ``parallel_map`` unit: decode one query clip, write its
    ``n_variants`` perturbed WAV(s), return the ``(vid, out_path)`` pairs (lineage ids ``q·augN``
    when n_variants > 1). Top-level + picklable so the ``process`` backend can run it; the
    perturbation is a pure function of (audio, sr, seed), so every backend is byte-identical."""
    import os

    import numpy as np
    import soundfile as sf

    from ...datasets.core import load_audio_file
    from ..provenance import item_seed

    qid, path = id_path
    waveform, sr = load_audio_file(path)
    audio = np.asarray(waveform.squeeze().numpy(), dtype=np.float32)
    pairs = []
    for variant in range(n_variants):
        perturbed = augmenter.augment(
            audio, int(sr), seed=item_seed(base_seed, str(qid), node_id, variant)
        )
        vid = str(qid) if n_variants == 1 else f"{qid}·aug{variant}"
        out_path = os.path.join(out_dir, f"{vid.replace('·', '_')}.wav")
        sf.write(out_path, perturbed, int(sr))
        pairs.append((vid, out_path))
    return pairs


def _stage_augment_audio(s: RunState) -> None:
    """Audio-axis robustness (§4.1 P4): perturb each query clip, republish refs.

    Reads the bus `query_audio` refs, decodes each clip, applies the seeded
    AudioAugmenter (per-item: ``item_seed(seed, query_id, node_id, variant)``),
    writes variant WAVs under ``<output_dir>/augmented_audio/<node_id>/`` and
    publishes the new ref ItemSet (lineage ids ``q·augN`` when n_variants > 1).
    No refs on the bus → no-op (warn).
    """
    import os

    from ...config.audio_augmentation import AudioAugmentationConfig
    from ...pipeline.audio.augmentation import AudioAugmenter
    from ..audio_refs import RefAudioDatasetView  # noqa: F401  (doc link)
    from ..item_set import ItemSet
    from ..provenance import DEFAULT_SEED

    refs = s.keyed_items("query_audio", default=None)
    if not isinstance(refs, ItemSet) or not refs.ids or not all(
        isinstance(v, str) for v in refs.values
    ):
        logger.warning("augment_audio: no audio refs on the bus — no-op")
        return
    params = s.node_params
    node_id = getattr(s.current_node, "id", "augment_audio")
    base = getattr(s.config, "audio_augmentation", None) or AudioAugmentationConfig()
    from dataclasses import replace as dc_replace

    overlay = {
        k: v
        for k, v in params.items()
        if k in ("add_noise", "snr_db", "speed_perturbation", "pitch_shift",
                 "volume_change", "n_variants", "noise_type")
        and v not in (None, "")
    }
    if "snr_db" in overlay:
        overlay["snr_db"] = float(overlay["snr_db"])
    if "n_variants" in overlay:
        overlay["n_variants"] = int(overlay["n_variants"])
    cfg = dc_replace(base, enabled=True, **overlay)
    augmenter = AudioAugmenter(cfg)
    n_variants = max(1, int(getattr(cfg, "n_variants", 1) or 1))

    seed = getattr(getattr(s.config, "audio_synthesis", None), "seed", None)
    base_seed = int(seed) if seed is not None else DEFAULT_SEED
    out_dir = os.path.join(
        getattr(s.config, "output_dir", "evaluation_results"),
        "augmented_audio",
        node_id,
    )
    os.makedirs(out_dir, exist_ok=True)

    import functools

    from ..executor.cpu_parallel import parallel_map, resolve_cpu_backend

    # Per-item decode + perturb + write through the 4b parallel_map (sync default → byte-identical
    # to the serial loop; the perturbation is pure in (audio, sr, seed), so thread/process match).
    backend, workers = resolve_cpu_backend(s.config)
    per_item = parallel_map(
        functools.partial(
            _augment_audio_one, augmenter=augmenter, n_variants=n_variants,
            base_seed=base_seed, node_id=node_id, out_dir=out_dir,
        ),
        list(zip(refs.ids, refs.values)),
        backend=backend, workers=workers,
    )
    out_ids, out_paths = [], []
    for pairs in per_item:
        for vid, out_path in pairs:
            out_ids.append(vid)
            out_paths.append(out_path)
    s.put_artifact("query_audio", ItemSet(out_ids, out_paths))
    logger.info(
        "augment_audio '%s': %d clips -> %d perturbed refs",
        node_id,
        len(refs.ids),
        len(out_ids),
    )


def _stage_tts(s: RunState) -> None:
    """Synthesize query audio from the dataset's text (TTS bridge). Synthesizes only
    the genuinely-missing clips (cached / on-disk hits skipped); no-op when nothing is
    missing. No-op when config is absent (direct callers)."""
    if s.config is None:
        return
    questions = getattr(s.dataset, "questions", None)
    if not questions:
        return
    from ...pipeline.audio.prepare import synthesize_missing_query_audio

    _t = time.perf_counter()
    synthesize_missing_query_audio(
        questions,
        s.config.audio_synthesis,
        log=logger,
        cache_manager=s.cache_manager,
    )
    s.stage_times["tts_s"] = s.stage_times.get("tts_s", 0.0) + (
        time.perf_counter() - _t
    )
    # Publish the now-complete audio refs (P4): questions were gap-filled in place
    # (legacy consumers), the bus carries the same paths for audio-axis nodes.
    refs = audio_refs_from_questions(questions)
    if refs is not None:
        from ..item_set import ItemSet

        ids, paths = refs
        s.put_artifact("query_audio", ItemSet(ids, paths))
