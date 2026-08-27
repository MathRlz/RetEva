# evaluator — spoken clinical QA evaluation framework

An experiment is a **directed graph**. Typed datasources feed **artifacts**; **nodes** (ASR,
TTS, embedding, correction, retrieval, RAG, metrics, aggregation) are typed functions over
them; **metrics and aggregation are nodes too**. One `EvaluationConfig` YAML describes the
whole graph, and one execution core runs it — from the CLI, the Python API, or the web UI.

Built for the thesis domain: **voice query → ASR → correction → retrieval/RAG over medical
corpora**, with safety-oriented metrics (CEER) and statistically honest branch comparisons
(paired bootstrap CIs, Wilcoxon, Cohen's *d*, Benjamini–Hochberg FDR). The operator algebra
and registries are domain-general; end-to-end validation covers the spoken-clinical path.

```
 datasources ──artifacts──▶ transform nodes ──artifacts──▶ metric nodes ─item_scores─▶ aggregate
  (text/audio/vector,        (asr, tts, embed,            (auto-injected where        (reduce +
   query↔answer,              fusion, correct,             their declared inputs       deltas →
   query↔doc-pair)            retrieve, rag, …)            exist)                      report → sinks)
```

| Document | What it covers |
|---|---|
| **[`evaluator-architecture.md`](evaluator-architecture.md)** | the full design reference (15 sections: principles, artifact algebra, metrics, statistical rigor, execution, config schema, authoring guide) |
| **[`configs/campaign/RUNBOOK.md`](configs/campaign/RUNBOOK.md)** | how to run the thesis experiment campaign on a machine with the datasets/checkpoints |
| **[`baselines/README.md`](baselines/README.md)** | the byte-parity gate and how to re-baseline |
| **[`tests/README.md`](tests/README.md)** | what the test suite guards, and the rules for adding to it |

---

## Install

```bash
pip install -e .            # core
pip install -e .[dev]       # + pytest, flake8, mypy, pytrec-eval-terrier
pip install -e .[webapi]    # + FastAPI/uvicorn for the web UI
pip install -e .[all]       # everything incl. chromadb, qdrant, sonar, notebooks
```

Python ≥ 3.10. Model weights download on first use (HuggingFace cache).

## Quickstart

```bash
python3 -m pytest -q                                      # unit suite — no models needed
evaluator graph --config configs/e2e_pubmed_qa_small.yaml  # print the DAG, load nothing
evaluator run   --config configs/e2e_pubmed_qa_small.yaml  # real e2e (downloads Whisper + LaBSE)
```

The report lands in the config's `experiment.output_dir` as JSON, beside a
`…config_resolved.yaml` sidecar (the exact DAG that ran) and a `leaderboard.sqlite`.

## Core ideas in one screen

- **The graph is the spec.** A config carries `graph.nodes` **and** `graph.edges`; there is no
  mode switch and no `features:` block. Every capability is a node, its settings ride on that
  node, and the run label is *derived* from the node kinds. Generate the edge block with
  `evaluator graph --config X --emit-edges [--write]`.
- **Artifacts are typed and per-producer keyed.** Nodes never call each other; they publish and
  read named artifacts on a per-run bus, keyed by producing node — which is what makes duplicate
  nodes, multi-dataset graphs, and parallel branches possible.
- **Per-item identity (`ItemSet`).** Every per-item artifact carries stable ids, so a node that
  drops or fans out items (augmentation variants, `multi_query`) never corrupts alignment
  downstream — consumers join by id, and variants roll up to their lineage parent before any
  statistic is computed.
- **Branches + auto-CSE.** Declare `graph.branches` (ref / asr / corrected / per-corrector …);
  the builder expands per-branch node copies and collapses identical subgraphs, so the shared
  prefix runs **once** and divergence starts exactly at the first differing node. The terminal
  `aggregate` node computes paired cross-branch deltas.
- **Metrics are nodes, and the registry is the single scalar source.** A metric declares the
  artifacts it consumes and is auto-injected wherever they exist; the flat headline keys
  (`WER`, `MRR`, `Recall@5`, …) are derived aliases of the keyed report.

Full detail: architecture doc §1–§9.

## A minimal config

```yaml
experiment: {name: demo, output_dir: evaluation_results/demo}
dataset:
  id: pubmed_qa
  questions: examples/data/pubmed_qa_small/questions.json
  corpus: examples/data/pubmed_qa_small/corpus.json
  trace_limit: 200          # caps the dataset AND gates per-query traces (see the note below)
graph:
  nodes: [dataset_source, tts, corpus_embedding, vector_db, asr, text_embedding,
          retrieval, transcription_metrics, retrieval_metrics, metrics, finalize]
  edges:                    # port-level: {from, to, input} (+ output: when it differs)
  - {from: dataset_source, to: corpus_embedding, input: corpus}
  - {from: text_embedding, output: text_query_vectors, to: retrieval, input: query_vectors}
  # …
nodes:                      # per-node model config, keyed by graph node id
  asr:            {model: whisper, name: openai/whisper-base, device: cuda:0}
  text_embedding: {model: labse, device: cuda:0}
  vector_db:      {store: inmemory}
  retrieval:      {k: 5}
```

Machine-specific paths belong in the environment — `${VAR}` and `$VAR` expand at load, and an
**unset** variable is left literal so the error names the variable you forgot:

```yaml
nodes:
  audio_embedding: {model_path: ${APM_CHECKPOINT_DIR}/apm_whisper_jina.pt}
```

Unknown/misspelled keys are rejected with a path-named error before any heavy work. Config
schema reference: architecture doc §10.

## CLI

`evaluator <command> --help` shows each command's own flags.

| Command | What it does |
|---|---|
| `run --config X` | run an evaluation. Flags are *operational only* (`--output_dir`, `--devices`, `--no_cache`, `--batch_size`, `--streaming_window_size`, `--cpu_stage_executor`, `--verbosity`) — the experiment lives in the config |
| `graph --config X` | print the DAG without loading models. `--format dot -o f.dot` for a publication figure; `--emit-edges [--write]` generates the `edges:` block; `--emit-metrics` generates a `metrics:` allowlist |
| `presets` | list config presets; `presets show <name>` dumps one |
| `datasets` | list registered dataset descriptors |
| `cache status\|clear` | inspect / clear the artifact cache |
| `leaderboard` | top runs by metric across `leaderboard.sqlite` |
| `sweep --base X --axes AX --out Y` | expand a base config's node params over an axes spec into a multi-variant graph config (see [Sweeping over node parameters](#sweeping-over-node-parameters)) |
| `compare a.json b.json` \| `compare DIR_A DIR_B [DIR_C ...]` | offline significance comparison (BH-FDR, under-powered flags); two flat result files, or 2+ variant/run dirs (baseline-vs-each) with a resolved-config diff + per-query answer diffs |
| `export R -f FMT -o OUT` | `csv`, `excel`, `latex`, `latex-compare`, `samples`, `metrics-table`, `traces`, `traces-parquet`, `provenance`, `mlflow`, `wandb` |
| `branch-report R --out-dir D` | LaTeX branch table + delta plot + per-query failure CSV (`--plot-format pdf\|svg` for vector) |
| `replay --config X --query-id Q` | re-run one query through the full graph with a per-node artifact trace |
| `benchmark` | model micro-benchmarks (latency/memory) |
| `gpu` | GPU status |

## Sweeping over node parameters

There's no `graph.branches` — a config is one explicit graph, and comparing N variants of a
node (different models, different retrieval params, ...) is N distinctly-named node entries
in `graph.nodes`, wired from whatever they share via ordinary edges (see e.g.
`configs/pubmed_qa_fulltext_big_rag_cmp.yaml`: one `text_embedding` shared by five
distinctly-named `retrieval_*` variants). Hand-duplicating those entries for every parameter
combination is tedious, so `evaluator sweep` does it for you.

Write an **axes spec** naming which node/param to vary and over which values:

```yaml
# axes.yaml
name: model_sweep
axes:
  - node: text_embedding   # a node id in the base config's graph.nodes
    param: model            # dotted paths nest, e.g. fusion.method
    values: [jina_v4, labse]
  - node: asr
    param: model
    values: [whisper, wav2vec2]
```

Then expand it against a base config (which must already have explicit `graph.edges` —
`evaluator graph --config X --emit-edges --write` generates them if missing):

```bash
evaluator sweep --base configs/pubmed_qa_rag_fulltext.yaml --axes axes.yaml --dry-run
evaluator sweep --base configs/pubmed_qa_rag_fulltext.yaml --axes axes.yaml --out configs/pubmed_qa_rag_fulltext_sweep.yaml
```

Every node topologically downstream of a varied node (inclusive) gets one distinctly-named
copy per combination (`retrieval_jina_v4_5`, `retrieval_labse_10`, ...); everything upstream
(`dataset_source`, `corpus_embedding`, ...) stays a single shared node referenced by every
variant. The write path re-loads the expanded config through `EvaluationConfig.from_yaml` —
the same chokepoint a real run takes — and reports `validation: OK` or the exact error, so a
bad axis value (wrong model name, unknown node id) is caught before you spend compute on it.
The result runs as one `evaluator run`; compare the variants afterward with `evaluator compare
DIR_A DIR_B [DIR_C ...]` on their output dirs.

## Web UI

```bash
pip install -e .[webapi]
evaluator-webapi --host 127.0.0.1 --port 8000     # then open /ui
```

Server-rendered (Jinja + htmx), **light and dark** (toggle in the header; follows the OS
until you choose). Plotly is vendored, so charts work offline. Fifteen panels across the
leaderboard, Pareto frontier, run detail and compare pages — effect sizes with CIs, an
effect-vs-evidence volcano, paired-sample denominators, rank distribution, failure causes,
per-query WER/CER distributions, WER-vs-recall, retrieval score margins, stage timing, cache
reuse, token budget, and cross-run quality-vs-cost. **Every chart carries a data-table twin**
(the accessible, copy-pasteable equivalent) and explains itself in a sentence when its inputs
are missing. Plus a visual **graph builder** (`/ui/builder`) and **Config & Run** forms. The builder is
registry-driven end to end — node ports, model choices and each model's parameter schema come
from the registries, nothing about a model is hardcoded in the UI. Saved builder graphs persist
via `/api/graphs`. See architecture doc §12/§12.1.

## Running experiments

Committed data (`examples/data/pubmed_qa_small`, 20 q) is for demos and the parity gate. For
real campaigns use the 200-question set and the campaign configs:

```bash
python3 scripts/build_pubmed_campaign.py -n 200        # writes examples/data/pubmed_qa_campaign
evaluator run --config configs/campaign/pubmed200_3branch.yaml
```

> **`trace_limit` does two jobs, and both bite.** It caps the dataset *and* gates
> per-query trace building. Below n=20 no bootstrap CI is emitted at all, and a two-sided
> Wilcoxon signed-rank test on n=5 cannot reach p<0.05 for *any* effect size — so the gate
> configs' `trace_limit: 5` (their baselines depend on it) is for parity, never for results.
> But `0` is not the campaign answer either: it means "whole dataset, **no traces**", which
> silently disables failure analysis, the per-speaker breakdown, the per-query charts, and
> the judge. Set it to the dataset size (200 for the campaign set).

**[`configs/campaign/RUNBOOK.md`](configs/campaign/RUNBOOK.md)** is the entry point for the full
campaign: the three arms (branch comparison, APM/cross-modal embedding variants, LLM arms), the
env vars each needs, and the post-run artifact commands.

LLM arms (judge, answer generation, the `llm` corrector) need a local endpoint:

```bash
ollama serve & ; ollama pull mistral:7b-instruct
evaluator run --config configs/campaign/pubmed200_rag_answer.yaml
```

## What a run produces

```
<output_dir>/
  results_<experiment>_<dataset>_<models>.json     # metrics + the keyed report
  results_….config_resolved.yaml                   # the executed DAG, round-trippable
  leaderboard.sqlite                               # every run, queryable across configs
```

`report.provenance` is a first-class part of the result: `config_hash`, resolved model
identities, run `seed`, library versions, `git_commit`, the determinism flags **actually
enforced**, a dataset *content* fingerprint (so "the numbers moved" separates from "the data
changed"), per-stage cache hit/miss, per-node/branch dropped item ids, LLM token+latency cost,
`data_flow` (which producer actually fed each input port, with fired fallbacks flagged), and
`optimization_fallbacks` when an LLM optimizer fell back to the original query.

Cross-branch deltas carry `mean_delta`, bootstrap `ci`, Wilcoxon `p_value`, BH-FDR
`p_value_fdr`, `cohens_d`, honest paired denominators (`n_paired` / `n_branch` / `n_baseline`)
and a `drop_biased` flag when one-sided exclusions exceed 5%.

## Thesis / paper artifacts

```bash
R=evaluation_results/campaign/results_pubmed200_3branch_*.json
evaluator branch-report "$R" --out-dir figures/ --plot-format pdf   # LaTeX table + vector delta plot
evaluator export "$R" -f provenance    -o figures/provenance.tex    # reproducibility table
evaluator export "$R" -f metrics-table -o figures/metrics.csv       # tidy branch×metric rows
evaluator export "$R" -f traces        -o figures/traces.jsonl      # per-query detail
evaluator graph --config <cfg> --format dot -o figures/dag.dot
dot -Tpdf figures/dag.dot -o figures/dag.pdf                        # pipeline figure
```

From Python / notebooks:

```python
from evaluator import evaluate_from_config
results = evaluate_from_config("configs/campaign/pubmed200_3branch.yaml")
results.get_metric("MRR")     # flat headline scalars
results.metrics["report"]     # branches, deltas, provenance
results.to_dataframe()        # tidy (branch, metric, mean, ci_lower, ci_upper, n)
```

## Verification

Two independent gates, because they catch different things:

```bash
python3 -m pytest -q                                                     # unit suite, model-free
python3 m1c_check.py configs/e2e_pubmed_qa_small.yaml   baselines/m1c_baseline_small.json
python3 m1c_check.py configs/e2e_pubmed_qa_3branch.yaml baselines/m1c_baseline_3branch.json
```

- **Unit suite** — fast and model-free (heavy models are mocked). Includes the golden-graph
  harness that freezes every repo config's built graph, an IR cross-check against `pytrec_eval`
  to 1e-9 per query, CEER pins against a hand-computed table, and the expressiveness suite
  proving every documented experiment shape is authorable.
- **Parity gate** — runs real models and diffs the report against a committed baseline. A
  behavior-preserving change must print `PARITY OK` on both configs. The unit suite cannot see
  report-level drift; this is what catches it.

Contributing rules that matter: a **new config** needs `python3 scripts/regen_graph_golden.py`
and the golden diff must be **strictly additive** (a changed existing entry means the refactor
moved a graph); anything touching **handlers, metrics, wiring, or config resolution** needs the
parity gate before commit.

## Extending

All discovery is via explicit registries — no import scanning, no core edits:

| Add a… | How |
|---|---|
| Model | implement the family contract under `models/<family>/`, decorate with `@register_<family>_model`; the model's inner `Params` dataclass *is* its UI surface (defaults, `SIZES`, `CHOICES`) |
| Dataset | subclass the matching ABC in `datasets/types.py`, implement `from_config`, decorate `@register_eval_dataset(id=…)` |
| Node type | `register_stage_node(...)` (contract) + `@register_stage_handler(...)` (executable) |
| Metric | `register_metric(name, scored=…, gt=…)` — declared inputs drive auto-injection |
| Corrector | `@register_corrector("name")` on `(texts, config, client?) → texts` |

Worked examples: architecture doc §15.

## Repository map

| Path | Contents |
|---|---|
| `evaluator/pipeline/graph/` | node registry, operator catalogue, wiring, CSE, templates, DOT export |
| `evaluator/evaluation/` | executor, run state + artifact bus, stage handlers, metric registry, aggregation |
| `evaluator/models/` | five model registries + implementations (asr, t2e, a2e, tts, retrieval, llm) |
| `evaluator/config/` | the dataclass config tree + node-centric→internal translation |
| `evaluator/datasets/` | descriptors, ABCs, builtins, loaders |
| `evaluator/metrics/` | per-query metric functions: IR (MRR/nDCG/recall), STT (WER/CER), clinical CEER, RAG, diagnostics |
| `evaluator/analysis/` | significance, Pareto, exports, error/branch reports |
| `evaluator/judge/`, `llm_client/`, `tracking/` | LLM-as-judge trace scoring; shared OpenAI-compatible LLM client + cost accounting; MLflow / no-op trackers |
| `evaluator/benchmarks/`, `core/` | hardware-fit model/retrieval benchmarking (off-DAG); shared core dataclasses |
| `evaluator/services/`, `devices/`, `storage/` | execution core, GPU pool, cache + vector stores + leaderboard |
| `evaluator/webapi/` | FastAPI routers + server-rendered UI + visual builder |
| `configs/`, `configs/campaign/` | experiment configs; campaign configs + runbook |
| `baselines/`, `m1c_check.py` | parity baselines and the gate |

## Known limitations

- **CEER is near-zero on PubMedQA** — those questions carry almost no drug/dose language. The
  safety-metric story needs medication speech (admed / hani).
- **LLM-backed nodes are reproducible per server, not absolutely.** `llm.seed` is forwarded and
  `temperature: 0` is the default, but the server's model build/quantization is outside the
  report. Local models (ASR, embedders) are seed-reproducible.
- **The web UI's charts are browser-rendered** (light/dark, via the theme toggle) — fine for
  exploration, not for print. Use `branch-report --plot-format pdf` for publication figures.
- **Control flow that changes the graph shape at run time** (self-RAG-until-confident, adaptive
  hop counts, agentic loops) is deliberately out of scope; the sanctioned escape hatch is a
  composite node with a fixed artifact contract.
- Image modality is designed, not built (architecture doc §13 tracks what is open). The C7
  correctors (rule/kb/phonetic/clinical/llm) are built and measured (C7.7); the patient-context
  grounding leg was cut from scope (`docs/archive/C7_GROUNDED_CORRECTION_PLAN.md`).
