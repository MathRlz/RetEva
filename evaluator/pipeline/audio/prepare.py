"""Shared audio-preparation helpers (TTS synthesis for text-only datasets).

Used by both the CLI (``cli/commands.py``) and the webapi service
(``services/evaluation_service.py``) so the two paths synthesize audio
identically.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def synthesize_missing_query_audio(
    questions: List[Any],
    synth_config: Any,
    *,
    log: Optional[logging.Logger] = None,
) -> int:
    """Synthesize audio for questions that lack an ``audio_path``.

    Each pending question gets ``<output_dir>/<question_id>.wav`` (or an
    in-memory waveform with no file when ``output_dir`` is unset) and its
    ``audio_path`` set. The AudioSynthesizer skips re-synthesis on a cache or
    file hit, so existing WAVs are reused. Per-question failures are logged and
    skipped rather than aborting the run.

    Args:
        questions: BenchmarkQuestion-like objects with ``question_id``,
            ``question_text`` and a settable ``audio_path``.
        synth_config: AudioSynthesisConfig (provider, output_dir, ...).
        log: Optional logger; falls back to module logger.

    Returns:
        Number of questions successfully synthesized (0 if none pending).
    """
    log = log or logger
    if not questions:
        return 0

    pending = [q for q in questions if not getattr(q, "audio_path", None)]
    if not pending:
        return 0

    import dataclasses
    from evaluator.pipeline.audio.synthesis import AudioSynthesizer

    out_dir = Path(synth_config.output_dir) if synth_config.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    default_lang = getattr(synth_config, "language", "en") or "en"
    log.info(
        "Synthesizing audio for %d text question(s) via TTS provider '%s' (lang=%s)%s",
        len(pending), synth_config.provider, default_lang,
        f" -> {out_dir}" if out_dir else " (in-memory)",
    )

    # One synthesizer per language. Each question's own ``language`` (if set)
    # wins over the config default, so multilingual datasets synthesize in the
    # right language; same-language datasets reuse a single synthesizer.
    synthesizers: dict = {}

    def _get_synth(lang: str) -> AudioSynthesizer:
        if lang not in synthesizers:
            cfg = dataclasses.replace(synth_config, language=lang) if lang != default_lang else synth_config
            synthesizers[lang] = AudioSynthesizer(cfg)
        return synthesizers[lang]

    done = 0
    for question in pending:
        text = getattr(question, "question_text", None)
        if not text:
            log.warning("Question %s has no text, skipping synthesis", getattr(question, "question_id", "?"))
            continue
        lang = getattr(question, "language", None) or default_lang
        output_path = str(out_dir / f"{question.question_id}.wav") if out_dir else None
        try:
            _get_synth(lang).synthesize(text, output_path=output_path)
            question.audio_path = output_path
            done += 1
        except Exception as e:  # keep going; one bad clip shouldn't abort the run
            log.error("Failed to synthesize audio for question %s: %s", getattr(question, "question_id", "?"), e)

    for synth in synthesizers.values():
        if hasattr(synth, "log_cache_stats"):
            synth.log_cache_stats()
    log.info("Audio synthesis complete (%d/%d questions)", done, len(pending))
    return done
