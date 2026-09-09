# Code audit — design (2026-09-09)

Goal: remove unnecessary comments, remove duplicated code, and make the package layering
clean, without changing runtime behavior. Scope is the whole repo: `evaluator/`, `tests/`,
`configs/` (only where tests reference them), and the two top-level docs.

Decisions already taken with the owner:

| Question | Decision |
|---|---|
| Architecture depth | Surgical: keep package layout; break import cycles by moving functions/modules to the right layer; no shims; import sites updated. |
| Named graph-template layer | Remove (`pipeline/graph/modes.py`, `templates.py`, `assembly.py`, webapi "Reload template"). |
| Tests referencing absent fixtures | Unit suite is the only gate. No `examples/`, `baselines/`, `m1c_check.py` restored. Tests worth keeping get a tiny synthetic fixture under `tests/data/`; stale tests are deleted. |
| Comment policy | Aggressive (see Phase 4). |
| Models / backends | All registered models and vector-store backends stay (owner decision from the 2026-07 audit). Duplication is removed by merging into shared bases, never by deleting a model. |
| Branch | `audit/2026-09`, one commit per logical unit. |

## Baseline (measured before any change)

| Metric | Value |
|---|---|
| Package LOC / test LOC | 48 987 / 14 581 |
| Comment + docstring lines vs code | 0.38 (3 044 comments, 8 368 docstring lines, 30 230 code) |
| Files with milestone markers (`B1:`, `T1:`, `M1b`, …) | 90 |
| Package-level import cycles | 10 |
| Function-local imports of internal modules | 513 |
| Cross-module imports of `_private` names | 69 |
| vulture ≥ 60 % candidates | 203 |
| Offline unit suite (`HF_HUB_OFFLINE=1`) | 1140 pass, 29 fail, 1 collection error, 30 s |

The suite must always run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`;
without them some tests reach HuggingFace and hang for 20+ minutes.

## Regression oracle

Behavior preservation is checked three ways after every phase:

1. `pytest` green under the offline env vars. This includes the graph golden
   (`tests/data/graph_golden.json`, `graph_order_golden.json`), which already freezes an
   order-insensitive and an order-sensitive signature of every config's built graph. The
   golden is regenerated only in Phase 0 (additive) and never again during the audit.
2. Resolved-config dump: for every YAML under `configs/`, `EvaluationConfig.from_yaml(...)`
   serialized to a dict is written once in Phase 0 to the session scratch directory
   (not committed) and diffed after each phase. Any difference is a regression.
3. Numeric self-check: tests that exercise metrics against `pytrec_eval` and the hand-computed
   CEER table stay untouched.

`scripts/regen_graph_golden.py` (30 lines, unit-suite tooling documented in the README) is
restored from the original repo. `m1c_check.py` and `baselines/` are not.

No real-model parity gate exists (owner decision).

## Phase 0 — Freeze and triage

1. Add `tests/data/pubmed_qa_mini/{questions,corpus}.json`: synthetic, 4 questions, 6 docs,
   same schema as the PubMed loader expects. Add `tests/data/configs/e2e_mini.yaml` with the
   full TTS→ASR→embed→retrieval→metrics explicit graph (shape copied from the deleted
   `e2e_pubmed_qa_small.yaml`, paths pointed at the mini fixture).
2. Re-point tests that only need "a dataset file exists" or "an e2e-shaped config" at those
   fixtures. Tests whose assertions were specific to a config that no longer exists anywhere
   (`c7_*`, `showcase_*`, `evaluation_config_pubmed_qa`, `local_llm_example`) are deleted.
3. Regenerate the graph golden for the current `configs/` (strictly additive: the new admed
   configs were never frozen).
4. Triage the remaining real failures one by one; each gets a fix or a deletion with a
   one-line reason in the commit message:
   `attention_pool` size resolution, `KeyError: 'variants'`, builder save/load button,
   theme tokens, MRR `None` in the audio-embedding integration test, template endpoint
   (removed in Phase 1).
5. Write the resolved-config dump to the scratch directory.
6. Commit.

## Phase 1 — Dead code

- Delete the named-template layer: `pipeline/graph/modes.py`, `templates.py`, `assembly.py`,
  `GRAPH_TEMPLATE_SPECS`, `GraphTemplateSpec`, `FeatureSet`, `build_stage_graph`,
  `resolve_graph_template`, `_resolve_mode_and_graph` in `executor/run.py`, the webapi
  template endpoint and "Reload template" button, and their tests.
  Survivors move to `pipeline/graph/wiring.py` or `branches.py`: `build_graph_for_config`,
  `complete_structural_plumbing` (its template fallback is replaced by a static structural
  node list), `label_from_graph` / `resolve_template_label` (renamed `graph_label`).
  `EvaluationConfig.graph_template` becomes `EvaluationConfig.graph_label`.
- Vulture candidates: each is checked by grep across `evaluator/`, `tests/`, templates and
  static JS. Decorator-registered handlers, routes and datasets are kept. README-documented
  API (`EvaluationResults.to_dataframe`) is kept. Everything else confirmed unreferenced is
  deleted together with its tests.
- `evaluation/runner.py` (a 41-line alias over `services.evaluation_service._run_core`) is
  removed; its two CLI callers call the service directly.

## Phase 2 — Duplication

Every merge keeps behavior byte-identical for the existing call sites.

| Clone | Resolution |
|---|---|
| `models/a2e/hubert.py` ≈ `wavlm.py` (51 differing lines of 214) | One `HFSpeechEncoderEmbedding` base in `models/a2e/_hf_ssl.py`; `hubert.py` and `wavlm.py` shrink to registration + names/`SIZES`/`CHOICES`. |
| `storage/vector_store.py`: 4 stores repeat search/search_batch/save/load payload plumbing | Base implements payload handling, `search` in terms of `search_batch`, save/load of payloads; subclasses implement only `_build_index`, `_search_vectors`, `_save_index`, `_load_index`. |
| `chromadb_store.py` ≈ `qdrant_store.py` shared chunks | Same base or a shared helper module. |
| `operators_catalog.py:15` ↔ `registry.py:25` constant block | One definition in `pipeline/graph/artifacts.py`, imported by both. |
| Intra-file clones: `cache/io.py`, `cli/benchmark.py`, `benchmarks/model_benchmark.py`, `models/factory.py`, `datasets/loaders/factory.py`, `analysis/significance.py`, `executor/state.py`, `storage/leaderboard.py`, `embedding_fusion/__init__.py` | Extract one private helper per clone pair. |
| `datasets/builtins/{fleurs,local_audio,huggingface_audio}` shared `from_config` body | One helper in `datasets/builtins/_common.py`. |
| resample-to-16k ×6 (prior audit skipped as "sites differ") | One helper with explicit `rate`, `cast`, `downmix` parameters; each site passes its current values. If any site cannot be expressed without changing its output, it stays. |
| MLflow call sequence in `tracking/mlflow_tracker.py` and `analysis/tracking_export.py` | Shared `tracking/_mlflow_calls.py`. |

## Phase 3 — Layering

Target order, leaf to top (a package may import only packages to its left):

```
constants errors logging_config utils
  → core config
  → metrics llm_client storage devices tracking
  → models
  → datasets
  → pipeline
  → evaluation judge analysis benchmarks
  → services
  → cli webapi api public_api
```

Cycle fixes:

| Cycle | Fix |
|---|---|
| `utils → storage` | `utils/cache_helpers.py` moves to `storage/cache/helpers.py`. |
| `evaluation → cli` | `generate_output_filename` moves from `cli/utils.py` to `evaluation/results_io.py`; the CLI imports it from there. |
| `pipeline → evaluation` | Corrector registry (`register_corrector`, `list_correctors`) moves to `models/retrieval/query/correctors.py`; the corrector implementations stay in `evaluation/query_correction.py` and register into it. |
| `datasets → pipeline` | `validate_field_mapping` moves from `pipeline/artifacts.py` to `datasets/descriptor.py` (or `datasets/utils.py`). |
| `config → evaluation`, `config → models`, `config → datasets` | Registry-dependent checks leave `config/validation.py` and join the existing pre-flight chain in `evaluation/validation.py`. `evaluation/node_config.py` moves to `config/node_config.py`. Rollup reducer names become a constant in `config/`. |
| `evaluation ↔ services`, `pipeline ↔ services` | `services/corpus_index.py`, `model_provider.py`, `model_services.py` move to `pipeline/`. `services/` keeps only `evaluation_service.py`. |
| `config → pipeline` | Accepted as the single documented downward edge: `EvaluationConfig.graph` and the node-centric translation need the node vocabulary. The only module config may import from pipeline is the leaf `pipeline/graph/operators.py` (plus `wiring` lazily inside `EvaluationConfig.graph`). All `pipeline → config` reverse edges are cut. |

Then:

- Function-local imports of internal modules are hoisted to module top unless the module is
  on an allowlist with a reason (heavy optional dependency deferral, or `EvaluationConfig.graph`).
- Cross-module `_private` imports: the name is made public (underscore dropped) at its
  definition, or the caller is given a public wrapper.
- `tests/test_import_layering.py` is added: it builds the module-level import graph of
  `evaluator/`, asserts it is acyclic and respects the order above, asserts no cross-package
  import of an underscore name, and asserts function-local internal imports only occur in
  allowlisted modules.

## Phase 4 — Comments

Delete:

- milestone / ticket markers (`B1:`, `T1:`, `M1b`, `R7`, `Roadmap 2d`, dates such as `2026-07-14`)
- history narration ("was X, now Y", "formerly", "no longer", "used to", "legacy" when the legacy path is gone)
- section banners and rulers
- commented-out code
- docstrings that restate the name or signature
- Args / Returns blocks that only repeat the type hints
- doctest-style examples that are not executed
- `# noqa` without a reason; `# type: ignore` that no longer fires

Keep:

- the reason behind a non-obvious choice (invariants, a workaround and the symptom it fixes,
  thread-safety, numerical stability, ordering constraints)
- one-line docstrings on public API and on registries / extension points
- module docstrings of at most three lines stating purpose
- config-field comments in YAML files under `configs/` (out of scope)

Rule for the sweep: if unsure whether a comment carries a reason, keep it. The sweep is done
per package by subagents with this policy verbatim, followed by one reviewer pass over the
diff. Tests get the same policy, lighter. Target: comment + docstring ratio from 0.38 to
about 0.15.

## Phase 5 — Docs

- `README.md`: resolve the `graph.branches` contradiction, remove links to files that do not
  exist here (`configs/campaign/RUNBOOK.md`, `baselines/README.md`, `m1c_check.py`,
  `examples/data/...`), rewrite the Verification section for the unit-suite-only gate,
  remove the "Repository map" rows for deleted modules.
- `evaluator-architecture.md`: remove the template rows in §13's source map and open items,
  `docs/archive` links, container references, and any paragraph describing removed code.
  De-stale only; no rewrite.
- `tests/README.md`: gate instructions and the offline env vars.

## Verification, every phase

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 -m pytest -q
python3 <scratch>/resolved_dump.py --check       # written in Phase 0; empty diff required
```

Final report re-runs the baseline measurements (comment ratio, clone scan, import cycles,
vulture, LOC) and lists them beside the starting values.

## Out of scope

- Correctness, security and performance findings beyond the six pre-existing test failures
  (they are logged, not fixed, unless a refactor step exposes them).
- YAML config duplication under `configs/` (hundreds of repeated edge lines per file).
- Restructuring `config/graph_config.py`'s node-centric → internal translation.
- The three public entry modules (`api.py`, `public_api.py`, `__init__.py`) keep their names.

## Deviations as executed

- **Corrector registry.** No separate `models/retrieval/query/correctors.py`: the whole
  `query_correction` module moved, so `register_corrector`/`list_correctors` and the corrector
  implementations live together in `models/retrieval/query/correction.py`.
- **`validate_field_mapping` stayed in `pipeline/artifacts.py`.** The `datasets → pipeline` edge
  was cut by moving the descriptor data to `config/dataset_descriptors.py` instead, which reaches
  the artifact registry through one allowlisted function-local import.
- **Registry checks reach `config.validate()` through a hook.** `config/validation.py` exposes
  `EXTRA_VALIDATORS`; `evaluation/validation.py` appends `collect_registry_problems` to it at
  import. `config/` still names no registry.
- **`graph_template` → `graph_label`, plus a run-only `run_label`.** The config property is
  `EvaluationConfig.graph_label`; the executor derives its own label from the graph override via
  `pipeline/graph/build.py:run_label`.
- **Bad corrector / combine-strategy names raise `ConfigurationError` from `validate()`**, not the
  `ValueError` the pre-audit call sites raised — the check is now a config-validation problem, not
  a dispatch-time failure.
- **Comment ratio landed at 0.24, not the 0.15 target.** The gap is retained invariants: comments
  that state a non-obvious contract (locking, ordering, fallback rules) were kept rather than cut
  to hit the number.
