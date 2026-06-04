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

from evaluator import ConfigurationError
from evaluator.services import ModelServiceProvider
from evaluator.webapi.config_helpers import create_config_options, graph_preview, prepare_run_config
from evaluator.webapi.jobs import JobManager

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Pipeline field -> (family, type form-field, size form-field).
_MODEL_FIELDS = {
    "model.asr_model_type": ("asr", "asr_model_type", "asr_size"),
    "model.text_emb_model_type": ("text_embedding", "text_emb_model_type", "text_emb_size"),
    "model.audio_emb_model_type": ("audio_embedding", "audio_emb_model_type", "audio_emb_size"),
}


def _required_fields(mode: str) -> List[str]:
    from evaluator.pipeline.stage_graph import resolve_pipeline_mode_spec
    try:
        return list(resolve_pipeline_mode_spec(mode).required_model_fields)
    except ValueError:
        return []


def _model_section(provider_factory, mode: str, current: Dict[str, str]) -> List[Dict[str, Any]]:
    """Build the registry-driven model-select descriptors for a pipeline mode."""
    provider = provider_factory()
    try:
        models = provider.list_available_models()
    finally:
        provider.shutdown()
    import evaluator.models  # noqa: F401
    from evaluator.models.registry import FAMILY_REGISTRIES

    sections: List[Dict[str, Any]] = []
    for field in _required_fields(mode):
        spec = _MODEL_FIELDS.get(field)
        if not spec:
            continue
        family, type_field, size_field = spec
        reg = FAMILY_REGISTRIES.get(family)
        selected_type = current.get(type_field, "")
        sizes = list(reg.get_sizes(selected_type).keys()) if (reg and selected_type) else []
        sections.append({
            "family": family,
            "type_field": type_field,
            "size_field": size_field,
            "options": [{"type": e["type"], "name": e["name"]} for e in models.get(family, [])],
            "selected_type": selected_type,
            "sizes": sizes,
            "selected_size": current.get(size_field, ""),
        })
    return sections


def _form_to_config(form: Dict[str, str]) -> Dict[str, Any]:
    """Map flat UI form fields to a nested EvaluationConfig dict (drop blanks)."""
    def s(key: str) -> str:
        return (form.get(key) or "").strip()

    def num(key: str, default):
        v = s(key)
        try:
            return int(v) if v else default
        except ValueError:
            return default

    model: Dict[str, Any] = {"pipeline_mode": s("pipeline_mode") or "asr_text_retrieval"}
    for _field, (_family, type_field, size_field) in _MODEL_FIELDS.items():
        if s(type_field):
            model[type_field] = s(type_field)
        if s(size_field):
            model[size_field] = s(size_field)

    data: Dict[str, Any] = {}
    for k in ("dataset_name", "questions_path", "corpus_path"):
        if s(k):
            data[k] = s(k)
    data["batch_size"] = num("batch_size", 8)
    data["trace_limit"] = num("trace_limit", 0)

    config: Dict[str, Any] = {
        "experiment_name": s("experiment_name") or "webui_experiment",
        "output_dir": s("output_dir") or "evaluation_results/webui",
        "model": model,
        "data": data,
        "vector_db": {
            "type": s("vector_db_type") or "inmemory",
            "k": num("k", 10),
            "retrieval_mode": s("retrieval_mode") or "dense",
        },
        "audio_synthesis": {
            "enabled": s("audio_synthesis_enabled") == "on",
            "provider": s("tts_provider") or "mms",
        },
    }
    return config


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
        options = create_config_options(provider_factory)
        mode = "asr_text_retrieval"
        return page(
            request, "config.html", active="config",
            options=options, mode=mode,
            model_sections=_model_section(provider_factory, mode, {}),
        )

    @router.get("/ui/models", response_class=HTMLResponse, include_in_schema=False)
    def ui_models(request: Request, pipeline_mode: str = "asr_text_retrieval") -> HTMLResponse:
        return page(
            request, "_models.html",
            model_sections=_model_section(provider_factory, pipeline_mode, {}),
        )

    @router.get("/ui/preset", response_class=HTMLResponse, include_in_schema=False)
    def ui_preset(request: Request, name: str = "") -> HTMLResponse:
        from evaluator.config.model_presets import get_preset
        cfg: Dict[str, Any] = {}
        if name:
            try:
                cfg = get_preset(name, auto_devices=False)
            except Exception:
                cfg = {}
        m = cfg.get("model", {})
        data = cfg.get("data", {})
        vdb = cfg.get("vector_db", {})
        mode = m.get("pipeline_mode", "asr_text_retrieval")
        current = {
            "asr_model_type": m.get("asr_model_type", "") or "",
            "asr_size": m.get("asr_size", "") or "",
            "text_emb_model_type": m.get("text_emb_model_type", "") or "",
            "text_emb_size": m.get("text_emb_size", "") or "",
            "audio_emb_model_type": m.get("audio_emb_model_type", "") or "",
            "audio_emb_size": m.get("audio_emb_size", "") or "",
        }
        options = create_config_options(provider_factory)
        return page(
            request, "_config_form.html", options=options, mode=mode,
            preset={
                "experiment_name": cfg.get("experiment_name", "webui_experiment"),
                "output_dir": cfg.get("output_dir", "evaluation_results/webui"),
                "dataset_name": data.get("dataset_name", ""),
                "questions_path": data.get("questions_path", ""),
                "corpus_path": data.get("corpus_path", ""),
                "batch_size": data.get("batch_size", 8),
                "trace_limit": data.get("trace_limit", 0),
                "vector_db_type": vdb.get("type", "inmemory"),
                "k": vdb.get("k", 10),
                "retrieval_mode": vdb.get("retrieval_mode", "dense"),
            },
            model_sections=_model_section(provider_factory, mode, current),
        )

    @router.post("/ui/validate", response_class=HTMLResponse, include_in_schema=False)
    async def ui_validate(request: Request) -> HTMLResponse:
        form = dict(await request.form())
        try:
            prepare_run_config(_form_to_config(form), auto_devices=True)
            return HTMLResponse('<p class="ok">Config valid ✓</p>')
        except (ConfigurationError, ImportError, ValueError) as exc:
            return HTMLResponse(f'<p class="error">Invalid: {exc}</p>')

    @router.post("/ui/graph", response_class=HTMLResponse, include_in_schema=False)
    async def ui_graph(request: Request) -> HTMLResponse:
        form = dict(await request.form())
        try:
            config = prepare_run_config(_form_to_config(form), auto_devices=True)
            preview = graph_preview(config)
            return page(request, "_graph.html", levels=preview.get("levels", []))
        except (ConfigurationError, ImportError, ValueError) as exc:
            return HTMLResponse(f'<p class="error">{exc}</p>')

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
        return page(request, "_status.html", job_id=job_id, job=data, result=result)

    @router.post("/ui/run", response_class=HTMLResponse, include_in_schema=False)
    async def ui_run(request: Request) -> HTMLResponse:
        form = dict(await request.form())
        try:
            config = prepare_run_config(_form_to_config(form), auto_devices=True)
            job = jobs.submit_evaluation(config)
        except (ConfigurationError, ImportError, ValueError) as exc:
            return HTMLResponse(f'<p class="error">{exc}</p>')
        return _status_response(request, job.job_id)

    @router.get("/ui/jobs/{job_id}/status", response_class=HTMLResponse, include_in_schema=False)
    def ui_job_status(request: Request, job_id: str) -> HTMLResponse:
        return _status_response(request, job_id)

    @router.get("/ui/results", response_class=HTMLResponse, include_in_schema=False)
    def ui_results(request: Request, metric: str = "MRR") -> HTMLResponse:
        return page(request, "results.html", active="results", metric=metric)

    @router.get("/ui/leaderboard", response_class=HTMLResponse, include_in_schema=False)
    def ui_leaderboard(request: Request, metric: str = "MRR", output_dir: str = "evaluation_results") -> HTMLResponse:
        from evaluator.storage import ExperimentStore
        rows: List[Dict[str, Any]] = []
        try:
            store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
            for r in store.query_leaderboard(metric_name=metric, limit=50):
                rows.append({
                    "run_id": r.run_id,
                    "experiment_name": r.experiment_name,
                    "dataset_name": r.dataset_name,
                    "pipeline_mode": r.pipeline_mode,
                    "metric_value": r.metric_value,
                })
        except Exception:
            rows = []
        labels = [str(r.get("experiment_name") or r.get("run_id") or i) for i, r in enumerate(rows)]
        values = [r.get("metric_value") for r in rows]
        return page(
            request, "_leaderboard.html",
            rows=rows, metric=metric,
            chart_labels=json.dumps(labels), chart_values=json.dumps(values),
        )

    @router.get("/ui/live", response_class=HTMLResponse, include_in_schema=False)
    def ui_live(request: Request) -> HTMLResponse:
        return page(request, "live.html", active="live")

    @router.get("/ui/tts", response_class=HTMLResponse, include_in_schema=False)
    def ui_tts(request: Request) -> HTMLResponse:
        provider = provider_factory()
        try:
            tts_models = provider.list_available_models().get("tts", [])
        finally:
            provider.shutdown()
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
        from evaluator.config import AudioSynthesisConfig
        from evaluator.pipeline.audio.synthesis import AudioSynthesizer

        if not text.strip():
            return HTMLResponse('<p class="error">text must not be empty</p>')
        # Write under CWD so the guarded /api/audio route can serve it back.
        out_dir = Path.cwd() / "evaluation_results" / "tts_preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = out_dir / f"preview_{abs(hash((text, provider, language, voice)))}.wav"
        try:
            cfg = AudioSynthesisConfig(
                enabled=True, provider=provider, voice=voice or AudioSynthesisConfig.voice,
                language=language, sample_rate=16000, output_dir=str(out_dir),
            )
            AudioSynthesizer(cfg).synthesize(text, output_path=str(wav))
        except Exception as exc:  # noqa: BLE001
            return HTMLResponse(f'<p class="error">TTS failed: {exc}</p>')
        rel = os.path.relpath(wav, Path.cwd())
        return page(request, "_tts_audio.html", audio_path=rel)

    @router.post("/ui/live", response_class=HTMLResponse, include_in_schema=False)
    async def ui_live_query(request: Request) -> HTMLResponse:
        # Placeholder: live query needs a built pipeline; surface a hint for now.
        return HTMLResponse('<p class="muted">Use the JSON API /api/live/query for ad-hoc retrieval.</p>')

    return router
