"""TTS preview + guarded audio streaming routes."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from evaluator.config import AudioSynthesisConfig
from evaluator.webapi.schemas import TtsPreviewRequest

_ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 waveform as WAV bytes via soundfile."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), int(sample_rate), format="WAV")
    return buf.getvalue()


def build_tts_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/tts/preview", summary="Synthesize a short TTS clip")
    def tts_preview(req: TtsPreviewRequest) -> Response:
        """Synthesize ``text`` with the chosen provider and return WAV bytes.

        Registry-driven: any provider registered via ``@register_tts_model``
        (incl. aliases) is selectable.
        """
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")

        from evaluator.pipeline.audio.synthesis import AudioSynthesizer

        synth_config = AudioSynthesisConfig(
            enabled=True,
            provider=req.provider,
            voice=req.voice or AudioSynthesisConfig.voice,
            language=req.language,
            sample_rate=req.sample_rate,
            skip_cache=True,
            output_dir=None,
        )
        try:
            synthesizer = AudioSynthesizer(synth_config)
            audio = synthesizer.synthesize(req.text)
        except ValueError as exc:  # unknown provider
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # provider/runtime failure
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

        return Response(content=_wav_bytes(audio, req.sample_rate), media_type="audio/wav")

    @router.get("/api/audio", summary="Stream an audio file (path-guarded)")
    def get_audio(path: str = Query(..., description="Path to an audio file under the working tree")) -> Response:
        """Stream a WAV/audio file, restricted to the working-directory tree.

        Guards against path traversal: the resolved path must stay within CWD
        and carry an allowed audio suffix.
        """
        root = Path.cwd().resolve()
        try:
            resolved = Path(path).resolve() if os.path.isabs(path) else (root / path).resolve()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid path: {exc}") from exc

        if resolved != root and root not in resolved.parents:
            raise HTTPException(status_code=403, detail="path outside allowed root")
        if resolved.suffix.lower() not in _ALLOWED_AUDIO_SUFFIXES:
            raise HTTPException(status_code=400, detail="unsupported audio type")
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="audio file not found")

        media = "audio/wav" if resolved.suffix.lower() == ".wav" else "application/octet-stream"
        return Response(content=resolved.read_bytes(), media_type=media)

    return router
