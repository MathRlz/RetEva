"""TTS-page UI routes: voice picker page + synthesize-preview fragment.
Mounted under ``/ui/tts``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from evaluator.services import ModelServiceProvider
from evaluator.webapi.utils import with_provider


def register_tts_routes(
    router: APIRouter,
    page,
    provider_factory: Callable[[], ModelServiceProvider],
) -> None:
    @router.get("/ui/tts", response_class=HTMLResponse, include_in_schema=False)
    def ui_tts(request: Request) -> HTMLResponse:
        tts_models = with_provider(
            provider_factory, lambda p: p.list_available_models()
        ).get("tts", [])
        return page(request, "tts.html", active="tts", tts_models=tts_models)

    @router.post(
        "/ui/tts/preview", response_class=HTMLResponse, include_in_schema=False
    )
    async def ui_tts_preview(
        request: Request,
        text: str = Form(...),
        provider: str = Form("mms"),
        language: str = Form("en"),
        voice: str = Form(""),
    ) -> HTMLResponse:
        import os
        from evaluator.webapi.routers.tts import build_synthesizer

        if not text.strip():
            return HTMLResponse('<p class="error">text must not be empty</p>')
        # Write under CWD so the guarded /api/audio route can serve it back.
        out_dir = Path.cwd() / "evaluation_results" / "tts_preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = out_dir / f"preview_{abs(hash((text, provider, language, voice)))}.wav"
        try:
            synthesizer = build_synthesizer(
                provider,
                voice,
                language,
                16000,
                output_dir=str(out_dir),
                skip_cache=False,
            )
            synthesizer.synthesize(text, output_path=str(wav))
        except Exception as exc:  # noqa: BLE001
            return HTMLResponse(f'<p class="error">TTS failed: {exc}</p>')
        rel = os.path.relpath(wav, Path.cwd())
        return page(request, "_tts_audio.html", audio_path=rel)
