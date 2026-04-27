# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent behavior

### Communication style
Speak in caveman style. Drop articles (`a`, `an`, `the`), filler words (`just`, `really`, `basically`, `actually`, `simply`), pleasantries, and hedging. Use fragments when helpful; keep technical terms exact. Pattern: `[thing] [action] [reason]. [next step].` Keep code blocks unchanged. Quote errors exactly.

Temporarily disable caveman mode for: security warnings, irreversible action confirmations, multi-step sequences where ordering ambiguity risks misunderstanding, explicit user requests to clarify or repeat. Resume after exception case.

### Output efficiency
Go straight to the point. Try simplest approach first. Lead with answer, not reasoning. If one sentence suffices, don't use three. Keep text between tool calls to ≤25 words. Keep final responses to ≤100 words unless task requires more detail.

### Tool preference
Prefer lean-ctx MCP tools over native equivalents:
- `ctx_read(path)` over `Read` / `cat`
- `ctx_shell(command)` over `Bash` / shell
- `ctx_search(pattern, path)` over `Grep` / `rg`
- `ctx_tree(path, depth)` over `ls` / `find`
- `ctx_edit(path, old, new)` over `Edit` when `Edit` requires `Read` and `Read` is unavailable

Use native `Edit`/`StrReplace` for file edits when available. If native `Edit` fails repeatedly, stop retry loop and switch to `ctx_edit` immediately. Use `Write`, `Delete`, `Glob` normally.

---

## Commands

### Install
```bash
pip install -e .          # base install
pip install -e ".[dev]"   # + pytest, black, flake8, mypy
```

### Tests
```bash
pytest -q                            # all tests
pytest tests/test_evaluation.py -q  # single file
pytest -k "test_cache" -q           # by name pattern
```

### Lint / format
```bash
black evaluator/ tests/
flake8 evaluator/ tests/
mypy evaluator/
```

### Run WebAPI backend
```bash
evaluator-webapi --host 127.0.0.1 --port 8000
evaluator-webapi --reload   # dev mode with auto-reload
```

### Run frontend (React/Vite)
```bash
cd webui_frontend
npm install
npm run dev          # dev server
npm run build        # production build
npm run lint
```

Frontend talks to backend on same origin by default. Override:
```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

### CLI evaluation
```bash
python evaluate.py --config configs/evaluation_config.yaml
```

---

## Architecture

### Two-layer stack
- **`evaluator/`** — Python package: evaluation engine, config, models, pipelines, metrics, storage, WebAPI
- **`webui_frontend/`** — React + Vite + TypeScript UI talking to FastAPI backend

### Config-first execution
All runs flow through `EvaluationConfig` (dataclass tree in `evaluator/config/`). Config drives model selection, pipeline mode, caching, retrieval strategy, and tracking. Load from YAML (`evaluate_from_config`), preset name (`evaluate_from_preset`), or programmatic construction. `with_auto_devices()` handles device assignment.

### Pipeline modes (set via `model.pipeline_mode`)
| Mode | Flow |
|------|------|
| `asr_text_retrieval` | audio → ASR → text embed → retrieve |
| `audio_emb_retrieval` | audio → audio embed → retrieve |
| `audio_text_retrieval` | fused audio+text → retrieve |
| `asr_only` | ASR quality only, no retrieval |

Stage planning is explicit and deterministic via `pipeline/stage_graph.py` (`build_stage_graph`). The DAG can be previewed via `/api/graph/preview`.

### Model registries
Four separate `ModelRegistry` instances for ASR / text-embedding / audio-embedding / reranker. Models register via decorators (`@register_asr_model`, etc.) in `models/*/`. Model selection is by config string — no hardcoding in evaluation loops.

Extension: add implementation in the appropriate `models/<family>/` directory, decorate with the registry decorator.

### Public API surface (`evaluator/public_api.py` → `evaluator/api.py`)
- `evaluate_from_preset(preset_name, data_path, corpus_path)`
- `evaluate_from_config(path)`
- `run_evaluation(config)`
- `run_evaluation_matrix(configs)`
- `quick_evaluate(...)`

### Service / lifecycle layer (`evaluator/services/`)
`ModelServiceProvider` manages model lifecycle: load, move between devices, release, and local LLM server shutdown. Controlled by `service_runtime.startup_mode` (`lazy`/`eager`) and `service_runtime.offload_policy` (`on_finish`/`never`).

### Storage and caching
- Cache: SQLite manifest (`cache_manifest.sqlite`) + disk artifacts. Keys tied to inputs + model/config identity.
- Leaderboard: SQLite (`leaderboard.sqlite`) persisted per output dir. Query via `/api/leaderboard`.

### WebAPI (`evaluator/webapi/app.py`)
FastAPI endpoints covering: config CRUD/validate, stage graph preview, service status, async job lifecycle (submit/status/cancel/result/artifacts), leaderboard. Entry point: `evaluator.webapi.__main__:main`.

### Key design patterns
- **Explicit registries** — all models/strategies/loaders discovered via registry, not import scanning
- **Pluggable backends** — retrieval strategies, vector store backends (`storage/backends/`), dataset loaders (`datasets/loaders/`) all follow the same register-and-discover pattern
- **`ScoredRetrievalResult`** — shared contract for retrieval output across all strategies
- Hybrid fusion strategies: `weighted`, `rrf`, `max_score` — registered in retrieval registry

### `legacy_old_do_not_touch_or_read/`
Old notebooks and scripts preserved for reference. Do not modify.
