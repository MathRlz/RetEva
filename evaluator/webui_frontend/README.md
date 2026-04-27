# Evaluator WebUI Frontend

Minimal React + Vite + TypeScript UI for evaluator backend orchestration.

## Features

- Build experiment config with form-first builder (pipeline, dataset, model selectors + full field sections), validate config, preview stage DAG.
- Run evaluation jobs, inspect status/result/metadata/artifacts, request cancel.
- Browse leaderboard and run history, select runs, compare run metrics side-by-side.
- Live retrieval test (`/api/live/query`) with ad-hoc query text.

## Start

Start backend first:

```bash
evaluator-webapi --host 127.0.0.1 --port 8000
# fallback:
# python -m evaluator.webapi --host 127.0.0.1 --port 8000
```

Then start frontend:

```bash
cd webui_frontend
npm install
npm run dev
```

By default, frontend calls backend on same origin.  
Set explicit backend URL with env var:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

If `evaluator-webapi` command missing after install, reinstall package in editable mode:

```bash
pip install -e .
```

## Build and lint

```bash
npm run build
npm run lint
```

## Required backend endpoints

- `GET /api/config/options`
- `POST /api/config/create`
- `POST /api/config/validate`
- `POST /api/graph/preview`
- `POST /api/jobs/evaluation`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/cancel`
- `GET /api/jobs/{job_id}/result`
- `GET /api/jobs/{job_id}/metadata`
- `GET /api/jobs/{job_id}/artifacts`
- `GET /api/leaderboard`
- `GET /api/leaderboard/runs`
- `GET /api/leaderboard/runs/{run_id}`
- `POST /api/live/query`

## Config Builder flow

1. Load preset.
2. Select pipeline mode, dataset name/type, and model stack from dropdowns.
3. Adjust remaining runtime/experiment fields in form sections.
4. Validate and preview DAG.
5. Use JSON panel only as fallback override.
