"""Jobs-page UI routes: submit a run + the jobs list / per-job status fragments.
Mounted under ``/ui/jobs`` (plus the ``/ui/run`` submit endpoint).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from evaluator.webapi.jobs import JobManager
from evaluator.webapi.form_builder import _prepared_config_or_error


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
        )

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

    @router.get(
        "/ui/jobs/{job_id}/status", response_class=HTMLResponse, include_in_schema=False
    )
    def ui_job_status(request: Request, job_id: str) -> HTMLResponse:
        return _status_response(request, job_id)
