"""SQLite-backed experiment run metadata and leaderboard store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        with self._connect() as conn:
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
    ) -> int:
        """Insert a run from raw parameters (useful for tests and external ingestion)."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs(
                    experiment_name, dataset_name, pipeline_mode, start_time, end_time,
                    duration_seconds, output_dir, metrics_json, config_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def ingest_result(self, result: Any) -> int:
        config = result.config
        metadata = result.metadata or {}
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs(
                    experiment_name, dataset_name, pipeline_mode, start_time, end_time,
                    duration_seconds, output_dir, metrics_json, config_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config.experiment_name,
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
    ) -> List[LeaderboardRow]:
        sql = """
            SELECT id, experiment_name, dataset_name, pipeline_mode, metrics_json,
                   duration_seconds, created_at
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
        sql += " ORDER BY created_at DESC"

        rows: List[LeaderboardRow] = []
        with self._connect() as conn:
            result_rows = conn.execute(sql, params).fetchall()
        for run_id, exp_name, ds_name, mode, metrics_json, duration, created_at in result_rows:
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

        with self._connect() as conn:
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
        with self._connect() as conn:
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
