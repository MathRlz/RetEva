"""Results-page UI routes: leaderboard table/chart, run detail, and run delete.
Mounted under ``/ui/results``, ``/ui/leaderboard`` and ``/ui/runs/...``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from evaluator.webapi.ui._common import _graph_view_from_config


def register_results_routes(router: APIRouter, page) -> None:
    @router.get("/ui/results", response_class=HTMLResponse, include_in_schema=False)
    def ui_results(
        request: Request, metric: str = "MRR", output_dir: str = "evaluation_results"
    ) -> HTMLResponse:
        from evaluator.storage import ExperimentStore

        try:
            store = ExperimentStore(
                db_path=str(Path(output_dir) / "leaderboard.sqlite")
            )
            metrics = store.available_metrics()
            filters = store.filter_options()
        except Exception:
            metrics, filters = [], {}
        if metric not in metrics:
            metric = metrics[0] if metrics else "MRR"
        return page(
            request,
            "results.html",
            active="results",
            metric=metric,
            metrics=metrics,
            filters=filters,
            output_dir=output_dir,
        )

    def _render_leaderboard(
        request,
        metric,
        output_dir,
        name,
        dataset_name,
        pipeline_mode,
        asr_model_type,
        text_emb_model_type,
        audio_emb_model_type,
    ) -> HTMLResponse:
        from evaluator.webapi.routers.leaderboard import leaderboard_rows

        model_filters = {
            "asr_model_type": asr_model_type,
            "text_emb_model_type": text_emb_model_type,
            "audio_emb_model_type": audio_emb_model_type,
        }
        try:
            rows: List[Dict[str, Any]] = leaderboard_rows(
                metric=metric,
                limit=50,
                output_dir=output_dir,
                dataset_name=dataset_name or None,
                pipeline_mode=pipeline_mode or None,
                name=name or None,
                model_filters=model_filters,
            )
        except Exception:
            rows = []
        labels = [
            str(r.get("experiment_name") or r.get("run_id") or i)
            for i, r in enumerate(rows)
        ]
        values = [r.get("metric_value") for r in rows]
        return page(
            request,
            "_leaderboard.html",
            rows=rows,
            metric=metric,
            output_dir=output_dir,
            chart_labels=json.dumps(labels),
            chart_values=json.dumps(values),
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
            request,
            metric,
            output_dir,
            name,
            dataset_name,
            pipeline_mode,
            asr_model_type,
            text_emb_model_type,
            audio_emb_model_type,
        )

    @router.get(
        "/ui/runs/{run_id}/confirm-delete",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def ui_confirm_delete(
        request: Request, run_id: int, output_dir: str = "evaluation_results"
    ) -> HTMLResponse:
        return page(
            request, "_delete_confirm.html", run_id=run_id, output_dir=output_dir
        )

    @router.post(
        "/ui/runs/{run_id}/delete", response_class=HTMLResponse, include_in_schema=False
    )
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
            request,
            metric,
            output_dir,
            name,
            dataset_name,
            pipeline_mode,
            asr_model_type,
            text_emb_model_type,
            audio_emb_model_type,
        )

    @router.get(
        "/ui/runs/{run_id}", response_class=HTMLResponse, include_in_schema=False
    )
    def ui_run_detail(
        request: Request, run_id: int, output_dir: str = "evaluation_results"
    ) -> HTMLResponse:
        from evaluator.storage import ExperimentStore

        try:
            store = ExperimentStore(
                db_path=str(Path(output_dir) / "leaderboard.sqlite")
            )
            run = store.get_run(run_id)
        except Exception:
            run = None
        if run is None:
            return HTMLResponse('<p class="error">run not found</p>')
        levels, node_io = _graph_view_from_config(run.get("config") or {})
        return page(
            request, "_run_detail.html", run=run, levels=levels, node_io=node_io
        )
