"""Shared audio-preparation helpers (TTS synthesis for text-only datasets).

Used by both the CLI (``cli/commands.py``) and the webapi service
(``services/evaluation_service.py``) so the two paths synthesize audio
identically.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def synthesize_missing_query_audio(
    questions: List[Any],
    synth_config: Any,
    *,
    log: Optional[logging.Logger] = None,
    cache_manager: Any = None,
) -> int:
    """Synthesize audio for questions that lack an ``audio_path``.

    Synthesized clips are cached through the shared CacheManager (the ``synthesized_audio``
    category) when one is supplied and enabled — so they appear in cache stats and obey
    ``--no-cache`` / ``--clear-cache`` like every other artifact. Keys are content-based
    (text + voice settings), so an already-cached clip sets the question's ``audio_path``
    without loading a TTS model. Per-question failures are logged and skipped.

    Cache behaviour:
      * cache_manager enabled → reuse/store under the cache (content-keyed);
      * cache_manager disabled (``--no-cache``) → never reuse; always re-synthesize;
      * no cache_manager (direct callers) → legacy: reuse on-disk WAVs in ``output_dir``.

    Args:
        questions: BenchmarkQuestion-like objects with ``question_id``,
            ``question_text`` and a settable ``audio_path``.
        synth_config: AudioSynthesisConfig (provider, voice, output_dir, ...).
        log: Optional logger; falls back to module logger.
        cache_manager: Optional shared CacheManager for synthesized-audio caching.

    Returns:
        Number of questions whose audio is now available (synthesized + reused).
    """
    log = log or logger
    if not questions:
        return 0

    pending = [q for q in questions if not getattr(q, "audio_path", None)]
    if not pending:
        return 0

    import dataclasses
    from evaluator.pipeline.audio.synthesis import AudioSynthesizer

    caching = cache_manager is not None and getattr(cache_manager, "enabled", False)
    # cache_manager supplied but disabled = --no-cache: never reuse, always re-synthesize.
    no_cache = cache_manager is not None and not caching

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
        log.info("All query audio already available (%d clip(s)); skipping TTS model load", resolved)
        return resolved

    default_lang = getattr(synth_config, "language", "en") or "en"
    log.info(
        "Synthesizing audio for %d text question(s) via TTS provider '%s' (lang=%s)%s",
        len(pending), synth_config.provider, default_lang,
        " -> cache" if caching else (f" -> {out_dir}" if out_dir else " (in-memory)"),
    )

    # CacheManager (or --no-cache) owns caching, so disable the synthesizer's ad-hoc cache.
    base_cfg = (
        dataclasses.replace(synth_config, skip_cache=True)
        if (caching or no_cache) else synth_config
    )

    # One synthesizer per language (a question's own ``language`` wins over the default).
    synthesizers: dict = {}

    def _get_synth(lang: str) -> AudioSynthesizer:
        if lang not in synthesizers:
            cfg = dataclasses.replace(base_cfg, language=lang) if lang != default_lang else base_cfg
            synthesizers[lang] = AudioSynthesizer(cfg)
        return synthesizers[lang]

    done = 0
    for question in pending:
        text = getattr(question, "question_text", None)
        if not text:
            log.warning("Question %s has no text, skipping synthesis", getattr(question, "question_id", "?"))
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
        except Exception as e:  # keep going; one bad clip shouldn't abort the run
            log.error("Failed to synthesize audio for question %s: %s", getattr(question, "question_id", "?"), e)

    for synth in synthesizers.values():
        if hasattr(synth, "log_cache_stats"):
            synth.log_cache_stats()

    # Release the TTS model(s) before the caller embeds (co-residence has caused crashes).
    synthesizers.clear()
    _release_torch_memory()

    log.info("Audio synthesis complete (%d synthesized, %d reused)", done, resolved)
    return resolved + done


def _tts_cache_key(synth_config: Any, question: Any) -> str:
    """Content-based cache key: text + the voice settings that affect the waveform."""
    import hashlib
    lang = getattr(question, "language", None) or getattr(synth_config, "language", "en") or "en"
    parts = [
        str(getattr(synth_config, "provider", "")),
        str(getattr(synth_config, "voice", "")),
        lang,
        str(getattr(synth_config, "sample_rate", "")),
        str(getattr(synth_config, "speed", "")),
        str(getattr(synth_config, "pitch", "")),
        getattr(question, "question_text", "") or "",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _release_torch_memory() -> None:
    """Drop freed torch tensors promptly (GC + CUDA cache)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
