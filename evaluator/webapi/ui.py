"""Minimalistic htmx UI for the evaluator.

Server-rendered Jinja2 templates + htmx (CDN) + Plotly (CDN, results only).
No build step, no node. UI routes reuse the same service/config helpers as the
JSON API and return HTML fragments for htmx swaps. Mounted at ``/ui``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from evaluator.services import ModelServiceProvider
from evaluator.webapi.config_helpers import graph_preview
from evaluator.webapi.jobs import JobManager
from evaluator import list_presets
from evaluator.webapi.ui_helpers import (
    _model_section,
    _prepared_config_or_error,
    _preset_form_context,
)


def _default_preset() -> str:
    """First available config preset (the Config page loads prefilled from it)."""
    presets = list_presets()
    return presets[0] if presets else ""
from evaluator.webapi.utils import with_provider

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def build_ui_router(
    provider_factory: Callable[[], ModelServiceProvider],
    jobs: JobManager,
) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"active": ctx.pop("active", ""), **ctx})

    @router.get("/ui", include_in_schema=False)
    def ui_root() -> RedirectResponse:
        return RedirectResponse(url="/ui/config")

    @router.get("/ui/config", response_class=HTMLResponse, include_in_schema=False)
    def ui_config(request: Request) -> HTMLResponse:
        ctx = _preset_form_context(provider_factory, _default_preset())
        return page(request, "config.html", active="config", **ctx)

    @router.get("/ui/models", response_class=HTMLResponse, include_in_schema=False)
    def ui_models(request: Request, pipeline_mode: str = "asr_text_retrieval") -> HTMLResponse:
        return page(
            request, "_models.html",
            model_sections=_model_section(provider_factory, pipeline_mode, {}),
        )

    @router.get("/ui/preset", response_class=HTMLResponse, include_in_schema=False)
    def ui_preset(request: Request, name: str = "") -> HTMLResponse:
        # hx-include sends the whole form as query params; preserve user fields
        # (dataset/paths) the chosen preset doesn't define.
        ctx = _preset_form_context(provider_factory, name, dict(request.query_params))
        return page(request, "_config_form.html", **ctx)

    @router.post("/ui/validate", response_class=HTMLResponse, include_in_schema=False)
    async def ui_validate(request: Request) -> HTMLResponse:
        config, error = _prepared_config_or_error(await request.form())
        if error is not None:
            return error
        return HTMLResponse('<p class="ok">Config valid ✓</p>')

    @router.post("/ui/graph", response_class=HTMLResponse, include_in_schema=False)
    async def ui_graph(request: Request) -> HTMLResponse:
        config, error = _prepared_config_or_error(await request.form())
        if error is not None:
            return error
        preview = graph_preview(config)
        return page(request, "_graph.html", levels=preview.get("levels", []))

    @router.post("/ui/yaml", response_class=HTMLResponse, include_in_schema=False)
    async def ui_yaml(request: Request) -> HTMLResponse:
        """Render the full config the form produces as copy-able YAML."""
        config, error = _prepared_config_or_error(await request.form())
        if error is not None:
            return error
        import html
        import yaml
        text = yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False)
        return HTMLResponse(
            f'<section class="step"><h4>Config (YAML)</h4>'
            f'<pre class="yaml">{html.escape(text)}</pre></section>'
        )

    def _status_response(request: Request, job_id: str) -> HTMLResponse:
        try:
            job = jobs.get(job_id)
        except Exception:
            return HTMLResponse('<p class="error">job not found</p>')
        data = job.to_dict()
        result = None
        if data.get("status") == "completed":
            result = getattr(job, "result", None)
            if hasattr(result, "to_dict"):
                result = result.to_dict()
        log_text = "\n".join(jobs.get_log(job_id, tail=400))
        return page(request, "_status.html", job_id=job_id, job=data, result=result, log_text=log_text)

    @router.post("/ui/run", response_class=HTMLResponse, include_in_schema=False)
    async def ui_run(request: Request) -> HTMLResponse:
        config, error = _prepared_config_or_error(await request.form())
        if error is not None:
            return error
        job = jobs.submit_evaluation(config)
        return _status_response(request, job.job_id)

    @router.get("/ui/jobs", response_class=HTMLResponse, include_in_schema=False)
    def ui_jobs(request: Request) -> HTMLResponse:
        return page(request, "jobs.html", active="jobs")

    @router.get("/ui/jobs/list", response_class=HTMLResponse, include_in_schema=False)
    def ui_jobs_list(request: Request) -> HTMLResponse:
        # Newest first.
        return page(request, "_jobs.html", jobs=list(reversed(jobs.list_jobs())))

    @router.get("/ui/jobs/{job_id}/status", response_class=HTMLResponse, include_in_schema=False)
    def ui_job_status(request: Request, job_id: str) -> HTMLResponse:
        return _status_response(request, job_id)

    @router.get("/ui/results", response_class=HTMLResponse, include_in_schema=False)
    def ui_results(
        request: Request, metric: str = "MRR", output_dir: str = "evaluation_results"
    ) -> HTMLResponse:
        from evaluator.storage import ExperimentStore
        try:
            store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
            metrics = store.available_metrics()
            filters = store.filter_options()
        except Exception:
            metrics, filters = [], {}
        if metric not in metrics:
            metric = metrics[0] if metrics else "MRR"
        return page(
            request, "results.html", active="results",
            metric=metric, metrics=metrics, filters=filters, output_dir=output_dir,
        )

    def _render_leaderboard(
        request, metric, output_dir, name,
        dataset_name, pipeline_mode,
        asr_model_type, text_emb_model_type, audio_emb_model_type,
    ) -> HTMLResponse:
        from evaluator.webapi.routers.leaderboard import leaderboard_rows
        model_filters = {
            "asr_model_type": asr_model_type,
            "text_emb_model_type": text_emb_model_type,
            "audio_emb_model_type": audio_emb_model_type,
        }
        try:
            rows: List[Dict[str, Any]] = leaderboard_rows(
                metric=metric, limit=50, output_dir=output_dir,
                dataset_name=dataset_name or None, pipeline_mode=pipeline_mode or None,
                name=name or None, model_filters=model_filters,
            )
        except Exception:
            rows = []
        labels = [str(r.get("experiment_name") or r.get("run_id") or i) for i, r in enumerate(rows)]
        values = [r.get("metric_value") for r in rows]
        return page(
            request, "_leaderboard.html",
            rows=rows, metric=metric, output_dir=output_dir,
            chart_labels=json.dumps(labels), chart_values=json.dumps(values),
        )

    @router.get("/ui/leaderboard", response_class=HTMLResponse, include_in_schema=False)
    def ui_leaderboard(
        request: Request,
        metric: str = "MRR",
        output_dir: str = "evaluation_results",
        name: str = "",
        dataset_name: str = "",
        pipeline_mode: str = "",
        asr_model_type: str = "",
        text_emb_model_type: str = "",
        audio_emb_model_type: str = "",
    ) -> HTMLResponse:
        return _render_leaderboard(
            request, metric, output_dir, name, dataset_name, pipeline_mode,
            asr_model_type, text_emb_model_type, audio_emb_model_type,
        )

    @router.get("/ui/runs/{run_id}/confirm-delete", response_class=HTMLResponse, include_in_schema=False)
    def ui_confirm_delete(
        request: Request, run_id: int, output_dir: str = "evaluation_results"
    ) -> HTMLResponse:
        return page(request, "_delete_confirm.html", run_id=run_id, output_dir=output_dir)

    @router.post("/ui/runs/{run_id}/delete", response_class=HTMLResponse, include_in_schema=False)
    def ui_delete_run(
        request: Request,
        run_id: int,
        delete_cache: str = Form(""),
        output_dir: str = Form("evaluation_results"),
        metric: str = Form("MRR"),
        name: str = Form(""),
        dataset_name: str = Form(""),
        pipeline_mode: str = Form(""),
        asr_model_type: str = Form(""),
        text_emb_model_type: str = Form(""),
        audio_emb_model_type: str = Form(""),
    ) -> HTMLResponse:
        from evaluator.webapi.routers.leaderboard import delete_run_and_cache
        try:
            delete_run_and_cache(
                run_id, delete_cache=(delete_cache == "on"), output_dir=output_dir
            )
        except Exception:
            pass  # Re-render the (now-updated) list regardless.
        return _render_leaderboard(
            request, metric, output_dir, name, dataset_name, pipeline_mode,
            asr_model_type, text_emb_model_type, audio_emb_model_type,
        )

    @router.get("/ui/runs/{run_id}", response_class=HTMLResponse, include_in_schema=False)
    def ui_run_detail(
        request: Request, run_id: int, output_dir: str = "evaluation_results"
    ) -> HTMLResponse:
        from evaluator.storage import ExperimentStore
        try:
            store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
            run = store.get_run(run_id)
        except Exception:
            run = None
        if run is None:
            return HTMLResponse('<p class="error">run not found</p>')
        return page(request, "_run_detail.html", run=run)

    @router.get("/ui/tts", response_class=HTMLResponse, include_in_schema=False)
    def ui_tts(request: Request) -> HTMLResponse:
        tts_models = with_provider(
            provider_factory, lambda p: p.list_available_models()
        ).get("tts", [])
        return page(request, "tts.html", active="tts", tts_models=tts_models)

    @router.post("/ui/tts/preview", response_class=HTMLResponse, include_in_schema=False)
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
                provider, voice, language, 16000,
                output_dir=str(out_dir), skip_cache=False,
            )
            synthesizer.synthesize(text, output_path=str(wav))
        except Exception as exc:  # noqa: BLE001
            return HTMLResponse(f'<p class="error">TTS failed: {exc}</p>')
        rel = os.path.relpath(wav, Path.cwd())
        return page(request, "_tts_audio.html", audio_path=rel)

    return router
