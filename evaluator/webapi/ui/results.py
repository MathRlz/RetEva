"""Results-page UI routes: leaderboard table/chart, run detail, run compare
and run delete. Mounted under ``/ui/results``, ``/ui/leaderboard``,
``/ui/compare`` and ``/ui/runs/...``.
"""

from __future__ import annotations

import difflib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...logging_config import get_logger

logger = get_logger(__name__)

#: DB-access failures that legitimately yield an empty/None view (missing, locked, or
#: corrupt leaderboard sqlite). Narrower than ``Exception`` so a real bug 500s instead of
#: silently rendering an empty page (C1/F20).
_STORE_ERRORS = (sqlite3.Error, OSError)


def _numeric_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a run's metrics to comparable numbers (one nested level, no CIs)."""
    flat: Dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if key.endswith("_ci") or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flat[key] = float(value)
        elif isinstance(value, dict):
            for sub, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(
                    sub_value, bool
                ):
                    flat[f"{key}.{sub}"] = float(sub_value)
    return flat


def _config_diff(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> List[str]:
    """Unified YAML diff of two runs' configs (canonical key order)."""
    import yaml

    def dump(run: Dict[str, Any]) -> List[str]:
        return yaml.safe_dump(
            run.get("config") or {}, sort_keys=True, default_flow_style=False
        ).splitlines()

    return list(
        difflib.unified_diff(
            dump(run_a),
            dump(run_b),
            fromfile=f"run {run_a['run_id']} ({run_a.get('experiment_name')})",
            tofile=f"run {run_b['run_id']} ({run_b.get('experiment_name')})",
            lineterm="",
        )
    )


from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

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
        except _STORE_ERRORS as exc:
            logger.warning("results page: leaderboard store unavailable (%s): %s",
                           output_dir, exc)
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
        except _STORE_ERRORS as exc:
            logger.warning("leaderboard render: query failed (%s): %s", output_dir, exc)
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

    @router.get("/ui/pareto", response_class=HTMLResponse, include_in_schema=False)
    def ui_pareto(
        request: Request,
        experiment_group: str = "",
        objectives: str = "MRR:max,latency_ms:min",
        output_dir: str = "evaluation_results",
    ) -> HTMLResponse:
        """Cross-run Pareto view for an experiment group (Roadmap 4a): a 2-objective scatter
        with the non-dominated frontier highlighted + a table tagging each comparable run."""
        from evaluator.storage import ExperimentStore
        from evaluator.webapi.routers.leaderboard import pareto_rows

        try:
            groups = ExperimentStore(
                db_path=str(Path(output_dir) / "leaderboard.sqlite")
            ).experiment_groups()
        except _STORE_ERRORS as exc:
            logger.warning("pareto page: group list unavailable (%s): %s", output_dir, exc)
            groups = []
        if not experiment_group and groups:
            experiment_group = groups[0]

        data: Optional[Dict[str, Any]] = None
        error = None
        if experiment_group:
            try:
                data = pareto_rows(experiment_group, objectives=objectives, output_dir=output_dir)
            except ValueError as exc:
                error = str(exc)
            except _STORE_ERRORS as exc:
                logger.warning("pareto render: query failed (%s): %s", output_dir, exc)

        rows = (data or {}).get("rows", [])
        objs = (data or {}).get("objectives", [])
        ykey = objs[0]["metric"] if objs else None
        xkey = objs[1]["metric"] if len(objs) > 1 else ykey  # 1 objective → 1-D strip

        def _trace(on_frontier: bool) -> Dict[str, list]:
            picks = [r for r in rows if r.get("on_frontier") is on_frontier]
            return {
                "x": [r["metrics"].get(xkey) for r in picks],
                "y": [r["metrics"].get(ykey) for r in picks],
                "text": [r.get("experiment_name") for r in picks],
            }

        return page(
            request,
            "_pareto.html",
            groups=groups,
            experiment_group=experiment_group,
            objectives=objectives,
            rows=rows,
            objs=objs,
            xkey=xkey,
            ykey=ykey,
            error=error,
            output_dir=output_dir,
            frontier_pts=json.dumps(_trace(True)),
            dominated_pts=json.dumps(_trace(False)),
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
        except _STORE_ERRORS as exc:
            # Re-render the (now-updated) list regardless, but don't hide the failure.
            logger.warning("delete run %s failed: %s", run_id, exc)
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

    @router.get("/ui/compare", response_class=HTMLResponse, include_in_schema=False)
    def ui_compare(
        request: Request, a: int, b: int, output_dir: str = "evaluation_results"
    ) -> HTMLResponse:
        from evaluator.storage import ExperimentStore

        run_a: Optional[Dict[str, Any]] = None
        run_b: Optional[Dict[str, Any]] = None
        try:
            store = ExperimentStore(
                db_path=str(Path(output_dir) / "leaderboard.sqlite")
            )
            run_a = store.get_run(a)
            run_b = store.get_run(b)
        except _STORE_ERRORS as exc:
            logger.warning("compare runs %s/%s: store read failed: %s", a, b, exc)
        if run_a is None or run_b is None:
            missing = a if run_a is None else b
            return HTMLResponse(f'<p class="error">run {missing} not found</p>')
        metrics_a = _numeric_metrics(run_a.get("metrics") or {})
        metrics_b = _numeric_metrics(run_b.get("metrics") or {})
        metric_rows = [
            {
                "name": name,
                "a": metrics_a.get(name),
                "b": metrics_b.get(name),
                "delta": (
                    metrics_b[name] - metrics_a[name]
                    if name in metrics_a and name in metrics_b
                    else None
                ),
            }
            for name in sorted(set(metrics_a) | set(metrics_b))
        ]
        return page(
            request,
            "compare.html",
            active="results",
            run_a=run_a,
            run_b=run_b,
            metric_rows=metric_rows,
            diff_lines=_config_diff(run_a, run_b),
            output_dir=output_dir,
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
        except _STORE_ERRORS as exc:
            logger.warning("run detail %s: store read failed: %s", run_id, exc)
            run = None
        if run is None:
            return HTMLResponse('<p class="error">run not found</p>')
        levels, node_io = _graph_view_from_config(run.get("config") or {})
        return page(
            request, "_run_detail.html", run=run, levels=levels, node_io=node_io
        )

    @router.get("/ui/runs/{run_id}/config.yaml", include_in_schema=False)
    def ui_run_config_yaml(
        run_id: int, output_dir: str = "evaluation_results"
    ) -> PlainTextResponse:
        """Download a run's resolved config as YAML — reproduce / share / re-run the exact
        experiment (the config stored with the run, byte-for-byte what produced these results)."""
        import yaml

        from evaluator.storage import ExperimentStore

        try:
            store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
            run = store.get_run(run_id)
        except _STORE_ERRORS as exc:
            logger.warning("run config %s: store read failed: %s", run_id, exc)
            run = None
        config = (run or {}).get("config")
        if not config:
            return PlainTextResponse("run config not found", status_code=404)
        text = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
        return PlainTextResponse(text, headers={
            "Content-Disposition": f'attachment; filename="run-{run_id}-config.yaml"',
        })
