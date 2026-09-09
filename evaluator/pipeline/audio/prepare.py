"""Shared audio-preparation helpers (TTS synthesis for text-only datasets).

Used by both the CLI (``cli/commands.py``) and the webapi service
(``services/evaluation_service.py``) so the two paths synthesize identically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from tqdm import tqdm
from ...utils.progress import progress_disabled
from evaluator.pipeline.audio import synthesis
from evaluator.devices.pool import get_active_pool
from evaluator.config import estimate_model_memory_gb

logger = logging.getLogger(__name__)


def synthesize_missing_query_audio(
    questions: List[Any],
    synth_config: Any,
    *,
    log: Optional[logging.Logger] = None,
    cache_manager: Any = None,
    device_pool: Any = None,
) -> int:
    """Synthesize audio for questions that lack an ``audio_path``, cached through the shared
    CacheManager (content-keyed by text + voice settings) when supplied and enabled, else
    legacy on-disk reuse under ``output_dir``; ``--no-cache`` always re-synthesizes.
    ``device_pool`` allocates/releases the TTS model's device through it (reservation + LRU
    eviction), falling back to the run's active pool or the config device; the model is freed
    here, unlike the long-lived embedders. Returns the number of questions whose audio is now
    available."""
    log = log or logger
    if not questions:
        return 0

    caching = cache_manager is not None and getattr(cache_manager, "enabled", False)
    # cache_manager supplied but disabled = --no-cache: never reuse, always re-synthesize.
    no_cache = cache_manager is not None and not caching

    if cache_manager is not None:
        # A bare `question.audio_path` can't tell "resolved for THIS synth_config" apart from
        # "resolved by a SIBLING tts variant" — shared Question objects carry only one slot, so
        # a naive skip-if-set check silently reused the first variant's audio for every other
        # one (confirmed live: byte-identical metrics across 3 "different" TTS engines). Route
        # every question through the real, provider-discriminating check below instead.
        pending = list(questions)
    else:
        # Direct caller, no multi-variant coordination — legacy: trust audio_path as-is.
        pending = [q for q in questions if not getattr(q, "audio_path", None)]
    if not pending:
        return 0

    import dataclasses

    out_dir = Path(synth_config.output_dir) if synth_config.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    def _existing_path(q) -> Optional[str]:
        """An already-available audio path for q (no TTS model load), or None."""
        if caching:
            hit = cache_manager.get_synthesized_audio(_tts_cache_key(synth_config, q))
            return str(hit) if hit is not None else None
        if not no_cache and out_dir is not None:
            wav = out_dir / f"{q.question_id}.wav"
            return str(wav) if wav.exists() else None
        return None  # --no-cache forces re-synthesis

    # Resolve already-available clips first, so a fully-cached set never loads a TTS model.
    resolved = 0
    still_pending = []
    for q in pending:
        path = _existing_path(q)
        if path is not None:
            q.audio_path = path
            resolved += 1
        else:
            still_pending.append(q)
    pending = still_pending

    if not pending:
        log.info(
            "All query audio already available (%d clip(s)); skipping TTS model load",
            resolved,
        )
        return resolved

    default_lang = getattr(synth_config, "language", "en") or "en"
    log.info(
        "Synthesizing audio for %d text question(s) via TTS provider '%s' (lang=%s)%s",
        len(pending),
        synth_config.provider,
        default_lang,
        " -> cache" if caching else (f" -> {out_dir}" if out_dir else " (in-memory)"),
    )

    # CacheManager (or --no-cache) owns caching, so disable the synthesizer's ad-hoc cache.
    base_cfg = (
        dataclasses.replace(synth_config, skip_cache=True)
        if (caching or no_cache)
        else synth_config
    )

    # Route the TTS model's device through the GPU pool so its footprint is visible to the
    # pool's packing. Providers with a 0.0 estimate (piper: CPU-only) skip the pool entirely.
    pool = device_pool
    if pool is None:
        pool = get_active_pool()
    tts_pool_key: Optional[str] = None
    if pool is not None:
        memory_gb = estimate_model_memory_gb("tts", synth_config.provider)
        if memory_gb > 0:
            tts_pool_key = f"tts:{synth_config.provider}"
            device = pool.allocate(tts_pool_key, memory_gb)
            if device != getattr(base_cfg, "device", None):
                log.info(
                    "TTS provider '%s': pool allocated %s (config device %s)",
                    synth_config.provider, device, getattr(base_cfg, "device", None),
                )
                base_cfg = dataclasses.replace(base_cfg, device=device)

    # One synthesizer per language (a question's own ``language`` wins over the default).
    synthesizers: dict = {}

    def _get_synth(lang: str) -> "synthesis.AudioSynthesizer":
        if lang not in synthesizers:
            cfg = (
                dataclasses.replace(base_cfg, language=lang)
                if lang != default_lang
                else base_cfg
            )
            synthesizers[lang] = synthesis.AudioSynthesizer(cfg)
        return synthesizers[lang]

    done = 0
    failed = 0
    first_error: Optional[str] = None
    try:
        for question in tqdm(pending, desc="TTS synthesis", unit="clip", disable=progress_disabled()):
            text = getattr(question, "question_text", None)
            if not text:
                # A silently-skipped question ends up with no audio_path and every downstream
                # consumer dies far from the cause; count it as a failure so the raise below
                # names the question.
                failed += 1
                if first_error is None:
                    first_error = (
                        f"question {getattr(question, 'question_id', '?')} has empty question_text"
                    )
                log.error(
                    "Question %s has no text — synthesis impossible",
                    getattr(question, "question_id", "?"),
                )
                continue
            lang = getattr(question, "language", None) or default_lang
            if caching:
                key = _tts_cache_key(synth_config, question)
                output_path: Optional[str] = str(cache_manager.synthesized_audio_path(key))
            elif out_dir is not None:
                output_path = str(out_dir / f"{question.question_id}.wav")
            else:
                output_path = None
            try:
                _get_synth(lang).synthesize(text, output_path=output_path)
                question.audio_path = output_path
                if caching and output_path:
                    cache_manager.register_synthesized_audio(key, Path(output_path))
                done += 1
            except Exception as e:  # attempt every clip first, then fail the node (see below)
                failed += 1
                if first_error is None:
                    first_error = f"{type(e).__name__}: {e}"
                log.error(
                    "Failed to synthesize audio for question %s: %s",
                    getattr(question, "question_id", "?"),
                    e,
                )

        for synth in synthesizers.values():
            if hasattr(synth, "log_cache_stats"):
                synth.log_cache_stats()
    finally:
        # Release the TTS model(s) before the caller embeds (co-residence has caused crashes).
        synthesizers.clear()
        _release_torch_memory()
        if pool is not None and tts_pool_key is not None:
            pool.release(tts_pool_key)

    if failed:
        # Fail HERE instead of "succeeding" with holes: a branch with partial audio is
        # incomparable to its siblings, and used to surface later as a misleading ASR crash.
        raise RuntimeError(
            f"TTS provider '{synth_config.provider}' failed for {failed}/{len(pending)} "
            f"question(s); first error: {first_error}"
        )

    log.info("Audio synthesis complete (%d synthesized, %d reused)", done, resolved)
    return resolved + done


def _tts_cache_key(synth_config: Any, question: Any) -> str:
    """Content-based cache key: text + the voice settings that affect the waveform."""
    import hashlib

    lang = (
        getattr(question, "language", None)
        or getattr(synth_config, "language", "en")
        or "en"
    )
    parts = [
        str(getattr(synth_config, "provider", "")),
        str(getattr(synth_config, "voice", "")),
        lang,
        str(getattr(synth_config, "sample_rate", "")),
        str(getattr(synth_config, "speed", "")),
        str(getattr(synth_config, "pitch", "")),
    ]
    # Appended only when SET: an unset speaker_id keeps every existing cache key valid,
    # while a chosen speaker keys its clips apart from the default-speaker ones.
    speaker_id = getattr(synth_config, "speaker_id", None)
    if speaker_id is not None:
        parts.append(f"speaker:{speaker_id}")
    parts.append(getattr(question, "question_text", "") or "")
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _release_torch_memory() -> None:
    """Drop freed torch tensors promptly (GC + CUDA cache)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - cache clear is best-effort
        logger.debug("cuda cache clear skipped: %s", exc)
