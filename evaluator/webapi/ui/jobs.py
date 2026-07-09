"""Jobs-page UI routes: submit a run (preset / edited graph) + the jobs list / per-job
status fragments. Mounted under ``/ui/jobs`` (plus the ``/ui/run-*`` submit endpoints).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from evaluator.webapi.jobs import JobManager


def _seconds_since_progress(data: dict):
    """Seconds since the last progress update for a still-running job (else None) — drives the
    "no update for Ns" soft-stall hint. A snapshot recomputed each poll, so it self-clears."""
    if data.get("status") not in ("running", "queued"):
        return None
    lp = data.get("last_progress") or {}
    ts = lp.get("ts")
    if not ts:
        return None
    from datetime import datetime, timezone

    try:
        elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    return max(0, int(elapsed.total_seconds()))


def register_jobs_routes(router: APIRouter, page, jobs: JobManager) -> None:
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
        return page(
            request,
            "_status.html",
            job_id=job_id,
            job=data,
            result=result,
            log_text=log_text,
            stale_seconds=_seconds_since_progress(data),
        )

    @router.post("/ui/run-preset", response_class=HTMLResponse, include_in_schema=False)
    async def ui_run_preset(request: Request) -> HTMLResponse:
        """Run a chosen config preset as-is (the simplified Config & Run flow): no form, just the
        preset name. Editing happens in the builder (Open in builder), not here."""
        import html

        from evaluator import EvaluationConfig, get_preset

        name = (await request.form()).get("name") or ""
        try:
            config = EvaluationConfig.from_dict(get_preset(name))
        except Exception as exc:  # noqa: BLE001 — surface an unknown/invalid preset inline
            return HTMLResponse(
                f'<p class="error">{html.escape(str(exc))}</p>', status_code=400
            )
        job = jobs.submit_evaluation(config)
        return _status_response(request, job.job_id)

    @router.post("/ui/run-graph", response_class=HTMLResponse, include_in_schema=False)
    async def ui_run_graph(request: Request) -> HTMLResponse:
        """Run a Config & Run graph with the user's per-node parameter edits applied. The page posts
        the edited canvas spec ({mode, nodes:[{id,type,params}]}); same translate+validate path as
        /api/jobs/from-graph, but returns the status fragment for htmx."""
        import html

        from evaluator.webapi.form_config import build_validated_run_config

        body = await request.json()
        spec = body.get("spec") or {}
        try:
            # shared translate+validate path with /api/jobs/from-graph (dataset choice, topology,
            # embedding-space check) so a Config&Run edit can't bypass what the API run enforces.
            config = build_validated_run_config(
                spec, experiment_name=body.get("name") or "webui_edited", auto_devices=True
            )
        except Exception as exc:  # noqa: BLE001 — surface a bad edit/graph inline
            return HTMLResponse(
                f'<p class="error">{html.escape(str(exc))}</p>', status_code=400
            )
        job = jobs.submit_evaluation(config)
        return _status_response(request, job.job_id)

    @router.get("/ui/jobs", response_class=HTMLResponse, include_in_schema=False)
    def ui_jobs(request: Request) -> HTMLResponse:
        return page(request, "jobs.html", active="jobs")

    @router.get("/ui/jobs/list", response_class=HTMLResponse, include_in_schema=False)
    def ui_jobs_list(request: Request) -> HTMLResponse:
        # Newest first.
        return page(request, "_jobs.html", jobs=list(reversed(jobs.list_jobs())))

    @router.get(
        "/ui/jobs/{job_id}/status", response_class=HTMLResponse, include_in_schema=False
    )
    def ui_job_status(request: Request, job_id: str) -> HTMLResponse:
        return _status_response(request, job_id)
