"""SQLite-backed experiment run metadata and leaderboard store."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _model_matches(config_json: str, filters: Dict[str, str]) -> bool:
    """True when the run's config model fields equal every requested filter."""
    try:
        model = (json.loads(config_json) or {}).get("model", {}) or {}
    except (TypeError, ValueError):
        return False
    return all(str(model.get(key) or "") == value for key, value in filters.items())


@dataclass(frozen=True)
class LeaderboardRow:
    run_id: int
    experiment_name: str
    dataset_name: str
    pipeline_mode: str
    metric_name: str
    metric_value: float
    duration_seconds: Optional[float]
    created_at: str


class ExperimentStore:
    """Persist evaluation runs and provide sortable leaderboard queries."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    pipeline_mode TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    output_dir TEXT,
                    metrics_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_dataset_mode_created
                ON runs(dataset_name, pipeline_mode, created_at DESC)
                """
            )
            # Run-grouping columns (architecture-improvements §3) — additive, migrated onto
            # pre-existing DBs (SQLite has no ADD COLUMN IF NOT EXISTS, so probe pragma first).
            existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            if "experiment_group" not in existing:
                conn.execute("ALTER TABLE runs ADD COLUMN experiment_group TEXT")
            if "tags" not in existing:
                conn.execute("ALTER TABLE runs ADD COLUMN tags TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_group ON runs(experiment_group)"
            )
            conn.commit()

    def upsert_run(
        self,
        *,
        experiment_name: str,
        dataset_name: str,
        pipeline_mode: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        output_dir: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        experiment_group: Optional[str] = None,
    ) -> int:
        """Insert a run from raw parameters (useful for tests and external ingestion)."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                INSERT INTO runs(
                    experiment_name, dataset_name, pipeline_mode, start_time, end_time,
                    duration_seconds, output_dir, metrics_json, config_json, metadata_json,
                    experiment_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_name,
                    dataset_name,
                    pipeline_mode,
                    start_time,
                    end_time,
                    duration_seconds,
                    output_dir,
                    json.dumps(metrics or {}, default=str),
                    json.dumps(config or {}, default=str),
                    json.dumps(metadata or {}, default=str),
                    experiment_group,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def ingest_result(self, result: Any) -> int:
        config = result.config
        metadata = result.metadata or {}
        tags = metadata.get("tags")
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                INSERT INTO runs(
                    experiment_name, experiment_group, tags, dataset_name, pipeline_mode,
                    start_time, end_time, duration_seconds, output_dir, metrics_json,
                    config_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config.experiment_name,
                    getattr(config, "experiment_group", None),
                    json.dumps(tags) if tags else None,
                    config.data.dataset_name,
                    str(config.model.pipeline_mode),
                    metadata.get("start_time"),
                    metadata.get("end_time"),
                    metadata.get("duration_seconds"),
                    config.output_dir,
                    json.dumps(result.metrics, default=str),
                    json.dumps(config.to_dict(), default=str),
                    json.dumps(metadata, default=str),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def query_leaderboard(
        self,
        *,
        metric_name: str = "MRR",
        limit: int = 20,
        dataset_name: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        name: Optional[str] = None,
        experiment_group: Optional[str] = None,
        model_filters: Optional[Dict[str, str]] = None,
    ) -> List[LeaderboardRow]:
        sql = """
            SELECT id, experiment_name, dataset_name, pipeline_mode, metrics_json,
                   duration_seconds, created_at, config_json
            FROM runs
        """
        clauses: List[str] = []
        params: List[Any] = []
        if dataset_name:
            clauses.append("dataset_name = ?")
            params.append(dataset_name)
        if pipeline_mode:
            clauses.append("pipeline_mode = ?")
            params.append(pipeline_mode)
        if name:
            clauses.append("LOWER(experiment_name) LIKE ?")
            params.append(f"%{name.lower()}%")
        if experiment_group:
            clauses.append("experiment_group = ?")
            params.append(experiment_group)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Fetch a generous pre-limit so Python filtering by metric presence
        # doesn't scan the full archive; actual top-N slice applied after sort.
        pre_limit = max(limit * 10, 500)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(pre_limit)

        active_model_filters = {k: v for k, v in (model_filters or {}).items() if v}

        rows: List[LeaderboardRow] = []
        with closing(self._connect()) as conn, conn:
            result_rows = conn.execute(sql, params).fetchall()
        for run_id, exp_name, ds_name, mode, metrics_json, duration, created_at, config_json in result_rows:
            if active_model_filters and not _model_matches(config_json, active_model_filters):
                continue
            metrics = json.loads(metrics_json)
            value = metrics.get(metric_name)
            if value is None:
                continue
            try:
                metric_value = float(value)
            except (TypeError, ValueError):
                continue
            rows.append(
                LeaderboardRow(
                    run_id=run_id,
                    experiment_name=exp_name,
                    dataset_name=ds_name,
                    pipeline_mode=mode,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    duration_seconds=duration,
                    created_at=created_at,
                )
            )
        rows.sort(key=lambda row: row.metric_value, reverse=True)
        return rows[:limit]

    def group_runs(
        self, experiment_group: str, *, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """All runs in ``experiment_group`` with their full numeric metrics — the multi-metric /
        Pareto cross-run view (Roadmap 4a). Each row: ``{run_id, experiment_name, dataset_name,
        pipeline_mode, created_at, metrics}``."""
        sql = (
            "SELECT id, experiment_name, dataset_name, pipeline_mode, metrics_json, created_at "
            "FROM runs WHERE experiment_group = ? ORDER BY created_at DESC LIMIT ?"
        )
        out: List[Dict[str, Any]] = []
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(sql, (experiment_group, limit)).fetchall()
        for run_id, name, ds_name, mode, metrics_json, created_at in rows:
            try:
                raw = json.loads(metrics_json) or {}
            except (TypeError, ValueError):
                raw = {}
            metrics = {
                k: float(v)
                for k, v in raw.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            out.append({
                "run_id": run_id,
                "experiment_name": name,
                "dataset_name": ds_name,
                "pipeline_mode": mode,
                "created_at": created_at,
                "metrics": metrics,
            })
        return out

    def available_metrics(self) -> List[str]:
        """Return the sorted union of numeric metric names across all runs."""
        names: set = set()
        with closing(self._connect()) as conn, conn:
            for (metrics_json,) in conn.execute("SELECT metrics_json FROM runs").fetchall():
                try:
                    metrics = json.loads(metrics_json)
                except (TypeError, ValueError):
                    continue
                for key, value in (metrics or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        names.add(key)
        return sorted(names)

    def filter_options(self) -> Dict[str, List[str]]:
        """Distinct values for the Results filter dropdowns (datasets/modes/models)."""
        datasets: set = set()
        modes: set = set()
        asr: set = set()
        text_emb: set = set()
        audio_emb: set = set()
        with closing(self._connect()) as conn, conn:
            for ds_name, mode, config_json in conn.execute(
                "SELECT dataset_name, pipeline_mode, config_json FROM runs"
            ).fetchall():
                if ds_name:
                    datasets.add(ds_name)
                if mode:
                    modes.add(mode)
                try:
                    model = (json.loads(config_json) or {}).get("model", {}) or {}
                except (TypeError, ValueError):
                    model = {}
                if model.get("asr_model_type"):
                    asr.add(model["asr_model_type"])
                if model.get("text_emb_model_type"):
                    text_emb.add(model["text_emb_model_type"])
                if model.get("audio_emb_model_type"):
                    audio_emb.add(model["audio_emb_model_type"])
        return {
            "datasets": sorted(datasets),
            "pipeline_modes": sorted(modes),
            "asr_models": sorted(asr),
            "text_emb_models": sorted(text_emb),
            "audio_emb_models": sorted(audio_emb),
        }

    def experiment_groups(self) -> List[str]:
        """Distinct non-empty ``experiment_group`` values (the Pareto view's group picker)."""
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT DISTINCT experiment_group FROM runs "
                "WHERE experiment_group IS NOT NULL AND experiment_group != '' "
                "ORDER BY experiment_group"
            ).fetchall()
        return [g for (g,) in rows]

    def delete_run(self, run_id: int) -> bool:
        """Delete a run row. Returns True if a row was removed."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_runs(
        self,
        *,
        limit: int = 100,
        dataset_name: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT id, experiment_name, dataset_name, pipeline_mode, start_time, end_time,
                   duration_seconds, output_dir, created_at
            FROM runs
        """
        clauses: List[str] = []
        params: List[Any] = []
        if dataset_name:
            clauses.append("dataset_name = ?")
            params.append(dataset_name)
        if pipeline_mode:
            clauses.append("pipeline_mode = ?")
            params.append(pipeline_mode)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with closing(self._connect()) as conn, conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "dataset_name": ds_name,
                "pipeline_mode": mode,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "output_dir": output_dir,
                "created_at": created_at,
            }
            for run_id, experiment_name, ds_name, mode, start_time, end_time, duration_seconds, output_dir, created_at in rows
        ]

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                """
                SELECT id, experiment_name, dataset_name, pipeline_mode, start_time, end_time,
                       duration_seconds, output_dir, metrics_json, config_json, metadata_json, created_at
                FROM runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        (
            run_id,
            experiment_name,
            dataset_name,
            pipeline_mode,
            start_time,
            end_time,
            duration_seconds,
            output_dir,
            metrics_json,
            config_json,
            metadata_json,
            created_at,
        ) = row
        return {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "pipeline_mode": pipeline_mode,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "output_dir": output_dir,
            "metrics": json.loads(metrics_json),
            "config": json.loads(config_json),
            "metadata": json.loads(metadata_json),
            "created_at": created_at,
        }
