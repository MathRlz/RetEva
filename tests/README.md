# Tests

Run: `pytest` (config in `pyproject.toml` → `testpaths = ["tests"]`). Lint: `flake8` (`.flake8`).

These tests are **fast and model-free** — no real ASR/embedding weights are loaded. Heavy model
behavior is covered by mocking the loaders/pipelines, so the whole suite runs in seconds and is
safe to run anywhere.

## What the suite guards

Grouped by what the tests pin. This lists the load-bearing groups rather than every
file — `pytest --collect-only -q` is the authoritative inventory.

| Area | Representative files | Guards |
|------|----------------------|--------|
| **Graph structure (the parity spine)** | `test_graph_golden.py`, `test_explicit_edges.py`, `test_run_graph_parity.py`, `test_cse_guard.py` | every repo config's built graph is frozen (order-insensitive AND order-sensitive signatures); explicit port edges reproduce the auto-wirer; preview and run builders agree; CSE never over-shares |
| **Config translation** | `test_graph_config.py`, `test_configs_smoke.py`, `test_config_roundtrip.py`, `test_config_env_expansion.py` | node-centric → legacy fold; the `configs/apm_tests/**` configs load + build + validate (broad config coverage rides on the graph goldens, row above); canvas round-trip; `${VAR}` expansion |
| **Expressiveness** | `test_expressiveness.py`, `test_typed_port_routing.py` | every §15.3 experiment shape is authorable; type-open OneOf ports + derivation ranking |
| **Metrics correctness** | `test_ir_reference_impl.py`, `test_ceer_pins.py`, `test_ir_metrics.py`, `test_stt_metrics.py`, `test_metric_allowlist.py` | IR metrics match `pytrec_eval` per query to 1e-9; CEER pinned against a hand-computed table; the `metrics:` allowlist computes exactly what it names |
| **Statistical honesty** | `test_variant_rollup.py`, `test_drop_bias.py`, `test_significance_bootstrap.py`, `test_branch_artifact_sharing.py` | variants roll up to their lineage parent before pairing; asymmetric drops flag `drop_biased`; bootstrap RNG hygiene; a branch never inherits another branch's corrected text |
| **Execution engine** | `test_streaming_equivalence.py`, `test_handler_registration.py`, `test_cpu_stage_*.py`, `test_observability.py` | windowed == whole-run report; registration is double-import safe; every CPU-parallel backend is byte-identical; failure attribution |
| **Reproducibility** | `test_reproducibility.py`, `test_data_flow_provenance.py`, `test_llm_seed_and_fallbacks.py` | content fingerprint, `repo@revision` pin, resolved-config sidecar; which producer fed each port; LLM seed forwarding + optimization-fallback surfacing |
| **Models & registries** | `test_model_registries.py`, `test_node_registry.py`, `test_operators.py`, `test_audio_pooling.py` | five registries resolve + expose param schemas; 12 operators' ports/taxonomy/dispatch; ABTT/whiten math |
| **Thesis artifacts** | `test_dag_export.py`, `test_branch_report_artifacts.py`, `test_report_export.py` | DOT DAG export; LaTeX branch + provenance tables; metrics-table/traces shapes |
| **Web + CLI surfaces** | `test_webapi*.py`, `test_builder_run.py`, `test_emit_edges_write.py`, `test_quick_evaluate_shim.py` | builder/canvas round-trips, `--emit-edges --write`, the `quick_evaluate` config shim |

Beyond the unit suite, `m1c_check.py` + `baselines/` gate **runtime** behavior: a
behavior-preserving change must reproduce two committed baselines byte-for-byte (see
`baselines/README.md`). That gate runs real models and lives outside `pytest`.

## How to grow it

- **Every bug fix gets a regression test** here first (write the failing test, then fix).
- Keep tests model-free: mock `create_*_model` / the `*Pipeline` wrappers, or the dataset
  loaders (`monkeypatch.setattr`), as the existing files do.
- Config/graph changes → add a case to `test_graph_config.py`; a NEW config also needs the
  goldens regenerated (`python3 scripts/regen_graph_golden.py`), and the diff must be
  strictly additive — a changed existing entry means the refactor moved a graph.
- Anything touching handlers, metrics, wiring, or config resolution → run the m1c parity
  gate in the container before committing; the unit suite cannot see report-level drift.
