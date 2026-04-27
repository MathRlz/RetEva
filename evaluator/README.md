# Medical Speech Retrieval Evaluator

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Framework for evaluating audio retrieval pipelines in medical domain. Supports:
- **ASR → text embedding → retrieval**
- **Audio embedding → retrieval**
- **Hybrid retrieval**, caching, experiment tracking, and configurable evaluation presets

Recent architecture upgrades:
- **DAG-lite stage graph** for pipeline mode planning (`build_stage_graph`)
- **Cache manifest v2**: SQLite metadata index + disk artifacts
- **Local LLM runtime lifecycle** managed by evaluator service provider
- **WebAPI orchestration endpoints** for graph preview, job metadata, artifacts, and leaderboard
- **SQLite leaderboard store** (`output_dir/leaderboard.sqlite`) for run comparison

## Why this project

Use this repo to compare retrieval quality across model stacks, configs, and pipeline modes with consistent metrics (MRR, MAP, NDCG, Recall, WER, CER).

## Quick start

### 1. Install

```bash
git clone <repo-url>
cd evaluator
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"  # tests + lint tooling
```

Base install now includes full runtime stack (web API, Faster-Whisper, notebooks/visualization).
For FAISS GPU acceleration, install `faiss-gpu` manually in your environment.

### 2. Run with preset (Python API)

```python
from evaluator import evaluate_from_preset

results = evaluate_from_preset(
    "whisper_labse",
    data_path="questions.json",
    corpus_path="corpus.json",
)

print(results.summary())
print(results.get_metric("MRR"))
```

### 3. Run with config file (CLI)

```bash
python evaluate.py --config configs/evaluation_config.yaml
```

### 4. Run WebAPI backend

Primary launch command:

```bash
evaluator-webapi --host 127.0.0.1 --port 8000
```

Fallback module launch:

```bash
python -m evaluator.webapi --host 127.0.0.1 --port 8000
```

For development reload:

```bash
evaluator-webapi --reload
```

## Core APIs

- `evaluate_from_preset(...)` — fastest way to run baseline evaluation
- `evaluate_from_config(path)` — load YAML config and run
- `run_evaluation(config)` — run from prepared `EvaluationConfig`
- `run_evaluation_matrix(configs)` — batch matrix runs
- `quick_evaluate(...)` — minimal single-call API

## Supported model families

### ASR
- Whisper
- Wav2Vec2
- Faster-Whisper

### Text embedding
- LaBSE
- Jina
- Nemotron
- BGE-M3
- CLIP (text side)

### Audio embedding
- Attention-pooling style encoders
- CLAP-style encoders

### TTS (audio synthesis for experiments)
- `piper`
- `xtts_v2` (Coqui XTTS v2)
- `mms` (Meta MMS, Hugging Face)

## Typical workflow

1. Prepare dataset files (`questions.json`, `corpus.json`).
2. Start from preset or template config.
3. Run evaluation.
4. Compare runs and inspect metrics.

For custom datasets, use `ConfigTemplates.custom_dataset(...)` from `evaluator.config`.

## Documentation map

- User guide: `docs/source/user_guide/`
- Configuration guide: `docs/source/user_guide/configuration.md`
- Configuration reference: `docs/source/user_guide/configuration_reference.md`
- Models guide: `docs/source/user_guide/models.md`
- Architecture notes: `ARCHITECTURE.md`

Build docs locally:

```bash
cd docs
make html
```

## Project structure

```text
evaluator/
├── evaluator/      # main package
├── configs/        # YAML config examples/presets
├── docs/           # Sphinx docs + user guides
├── tests/          # unit and integration tests
├── evaluate.py     # CLI entrypoint
└── setup.py
```

## Contributing

```bash
pip install -e ".[dev]"
pytest -q
```

Use focused PRs, include tests for behavior changes, keep config/docs updates in same PR when user-facing behavior changes.

## License

MIT
