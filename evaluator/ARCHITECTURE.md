# Evaluator Architecture

## Overview

`evaluator` is a modular framework for benchmarking **audio-query retrieval** pipelines.  
It focuses on reproducible comparisons across:

- model stacks (ASR / text embedding / audio embedding / TTS),
- retrieval strategies (dense / sparse / hybrid),
- evaluation modes and configuration presets.

The public entrypoints are exposed from `evaluator` package root via `public_api.py`.

## High-level flow

```text
Config (YAML/preset/programmatic)
            ↓
   EvaluationConfig validation
            ↓
    Model + pipeline creation
            ↓
 Data loading / optional synthesis
            ↓
      Query processing path
            ↓
 Retrieval + optional judge scoring
            ↓
   Metrics + artifacts + tracking
```

## Core modules

### 1. Public API and orchestration

- `evaluator/public_api.py`  
  Stable API exports (`evaluate_from_config`, `evaluate_from_preset`, `run_evaluation`, etc.).
- `evaluator/api.py`  
  High-level orchestration and convenience wrappers.
- `evaluate.py`  
  CLI entrypoint.

### 2. Configuration system

- `evaluator/config/`  
  Dataclass-based config tree (`EvaluationConfig`) covering models, data, cache, retrieval, tracking, judge, query optimization, and audio synthesis.
- Supports:
  - YAML loading,
  - presets,
  - programmatic overrides,
  - auto device assignment (`with_auto_devices()`),
  - validation with actionable errors.

### 3. Models

- `evaluator/models/asr/` — ASR backends
- `evaluator/models/t2e/` — text embedding models
- `evaluator/models/a2e/` — audio embedding models
- `evaluator/models/tts/` — synthesis backends (`piper`, `xtts_v2`, `mms`)
- `evaluator/models/registry.py` — model registration and discovery

All model families are selected by config type/name, not hardcoded in evaluation loops.

### 4. Pipelines and execution

- `evaluator/pipeline/`  
  Processing pipelines for ASR, embeddings, retrieval, and audio preparation/synthesis.
  Includes DAG-lite stage planning via `pipeline/stage_graph.py`.
- `evaluator/evaluation/`  
  Sample-wise and matrix evaluation runners.
- `evaluator/parallel/`  
  Parallel execution helpers for multi-worker runs.

### 5. Retrieval stack

- `evaluator/models/retrieval/`  
  Retrieval strategies, query optimization, embedding fusion, and advanced RAG components.
- `evaluator/storage/backends/`  
  Vector/index backends used by retrieval pipeline.

### 6. Metrics, analysis, tracking

- `evaluator/metrics/` — IR and ASR metrics
- `evaluator/judge/` — LLM-as-judge scoring
- `evaluator/analysis/` and `evaluator/visualization/` — run comparison and outputs
- `evaluator/tracking/` — experiment tracking integrations

### 7. Data and services

- `evaluator/datasets/` — dataset loaders and adapters
- `evaluator/services/` — orchestration and model-service provider layer
- `evaluator/devices/` — device monitoring/allocation support

Service runtime controls:
- `service_runtime.startup_mode` (`lazy` / `eager`)
- `service_runtime.offload_policy` (`on_finish` / `never`)

`ModelServiceProvider` exposes lifecycle APIs:
- list available models
- move model between devices
- release one model service or all services
- manage local LLM server lifecycle (owned-process-safe shutdown)

## Evaluation modes

Configured through `model.pipeline_mode`:

1. `asr_text_retrieval`  
   Audio → ASR → text embedding → retrieval
2. `audio_emb_retrieval`  
   Audio → audio embedding → retrieval
3. `audio_text_retrieval`  
   Audio + text representations fused for retrieval
4. `asr_only`  
   ASR quality evaluation without retrieval metrics

## Caching and reproducibility

- Cache controls live under `config.cache`.
- Cache keys are tied to inputs and model/config identity to avoid unsafe reuse.
- Cache manifest uses **SQLite metadata index** (`cache_manifest.sqlite`) with key -> artifact path mapping.
- Cache artifacts stay on local disk under cache subdirectories.
- Checkpoint/resume support is built into evaluation config.
- Matrix runs support consistent side-by-side experiment comparisons.
- Service shutdown policy is deterministic through `service_runtime.offload_policy`.

## Runtime planning and retrieval strategy seams

- Retrieval output contracts are explicit via `ScoredRetrievalResult`.
- Hybrid fusion strategies use registry (`weighted`, `rrf`, `max_score`).
- Stage planning is explicit and deterministic per pipeline mode through stage graph levels.

## Web API orchestration surface

`evaluator/webapi/app.py` exposes orchestration-first endpoints:
- config options/create/validate
- stage graph preview (`/api/graph/preview`)
- service status (`/api/services/status`)
- async job lifecycle + metadata + artifact listing
- leaderboard query (`/api/leaderboard`)

## Experiment registry / leaderboard

- `evaluator/storage/leaderboard.py` persists run metadata in SQLite (`leaderboard.sqlite`).
- `run_evaluation` ingests each completed run for sortable metric leaderboards.

## TTS in architecture

Audio synthesis is optional and controlled by `config.audio_synthesis`:

- `provider`: `piper`, `xtts_v2`, or `mms`
- optional language/voice controls
- output is normalized to configured sample rate before downstream processing

This enables text-only datasets to be converted into comparable audio-query benchmarks.

## Extension points

Main extensibility surfaces:

- add model implementations in appropriate `models/*` family + registry wiring,
- add retrieval methods under `retrieval/`,
- add new storage backend under `storage/backends/`,
- add dataset loader under `datasets/loaders/`,
- add new config section under `config/` and thread it through orchestration.

Keep extension contracts aligned with existing config + pipeline patterns to preserve testability.

## Design principles

- **Composable modules** over monolithic scripts
- **Config-first execution** for reproducibility
- **Explicit registries** for discoverability and validation
- **Pluggable backends** for models/retrieval/storage
- **Deterministic experiment artifacts** for comparison and debugging
