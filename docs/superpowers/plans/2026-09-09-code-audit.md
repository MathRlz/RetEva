# Code Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unnecessary comments, duplicated code and layering violations from `evaluator/` without changing runtime behavior.

**Architecture:** Five sequential phases on branch `audit/2026-09`: freeze the oracle (P0), delete dead code (P1), merge clones (P2), fix layering (P3), sweep comments (P4), de-stale docs (P5). Every task ends with the offline unit suite green and the resolved-config dump unchanged. Surgical scope: functions and modules move, nothing is renamed at the public API surface except `EvaluationConfig.graph_template` → `graph_label`.

**Tech Stack:** Python 3.13, pytest, flake8, vulture 2.16, git. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-09-code-audit-design.md`

## Global Constraints

- Every test run uses: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 -m pytest -q -p no:cacheprovider`. Define once per shell: `export PYT='env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 -m pytest -q -p no:cacheprovider'` and run `$PYT`.
- Baseline after P0 must be: 0 failed, 0 errors. Every later task keeps it there.
- All registered models, datasets, vector-store backends and correctors stay. Duplication is removed by merging into shared code, never by deleting a model.
- No shims, no re-export stubs for moved code. Import sites are updated.
- The graph golden files (`tests/data/graph_golden.json`, `tests/data/graph_order_golden.json`) are regenerated exactly once, in Task 0.3, and never again. A later golden failure is a regression to fix in code.
- The resolved-config dump lives at `$SCRATCH/resolved/` where `SCRATCH=/home/krystian/studia/magisterka/evaluator_clean/.superpowers/sdd/2026-09-09-code-audit/scratch`. `python3 $SCRATCH/resolved_dump.py check` must print `RESOLVED OK` after every task from 0.5 onward.
- Commit message format: `<area>: <what>` on one line, blank line, then the reason if not obvious, then the trailer block:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MjkyXCYRDSYqE867TUXGR2
  ```
- Files under `evaluator/`, `configs/`, `tests/` were root-owned at planning time. Before Task 0.1 verify `[ -w evaluator/api.py ] && echo ok`; if not writable, stop and ask the owner to run `sudo chown -R krystian:krystian /home/krystian/studia/magisterka/evaluator_clean`.
- Comment policy (used by P4 and applied opportunistically in every earlier task that touches a file): delete milestone markers (`B1:`, `T1:`, `M1b`, `R5/`, `E4`, `Roadmap 2d`, dates like `2026-07-14`), history narration ("was", "formerly", "no longer", "used to", "moved from", "the former"), section banner rulers, commented-out code, docstrings that only restate the name or signature, Args/Returns blocks that only repeat type hints, doctest examples that are not executed, `# noqa` without a reason. Keep: the reason behind a non-obvious choice, one-line docstrings on public API and registries, module docstrings of at most three lines. If unsure whether a comment carries a reason, keep it.

---

## Phase 0 — Freeze and triage

### Task 0.1: Test fixture dataset and e2e config

**Files:**
- Create: `tests/data/pubmed_qa_mini/questions.json`
- Create: `tests/data/pubmed_qa_mini/corpus.json`
- Create: `tests/data/configs/e2e_mini.yaml`
- Create: `tests/fixtures_paths.py`

**Interfaces:**
- Produces: `tests/fixtures_paths.py` exporting `MINI_Q: str`, `MINI_C: str` (absolute paths to the two JSON files) and `E2E_MINI: str` (absolute path to the YAML). Every later test edit imports these instead of hard-coding `examples/data/...`.

- [ ] **Step 1: Write the questions fixture** (schema copied from the PubMed loader: list of objects with `question_id`, `question_text`, `groundtruth_doc_ids`, `relevance_grades`, `language`, `metadata`)

```json
[
  {"question_id": "q1", "question_text": "Does aspirin reduce the risk of a second heart attack?",
   "groundtruth_doc_ids": ["d1"], "relevance_grades": {"d1": 1}, "language": "en", "metadata": {"pubid": 1}},
  {"question_id": "q2", "question_text": "Is metformin safe in mild kidney disease?",
   "groundtruth_doc_ids": ["d2"], "relevance_grades": {"d2": 1}, "language": "en", "metadata": {"pubid": 2}},
  {"question_id": "q3", "question_text": "Can statins cause muscle pain?",
   "groundtruth_doc_ids": ["d3"], "relevance_grades": {"d3": 1}, "language": "en", "metadata": {"pubid": 3}},
  {"question_id": "q4", "question_text": "Does vitamin D supplementation prevent falls in the elderly?",
   "groundtruth_doc_ids": ["d4"], "relevance_grades": {"d4": 1}, "language": "en", "metadata": {"pubid": 4}}
]
```

- [ ] **Step 2: Write the corpus fixture** (list of objects with `doc_id`, `title`, `text`; six docs, four relevant plus two distractors)

```json
[
  {"doc_id": "d1", "title": "Aspirin for secondary prevention", "text": "Low-dose aspirin reduces the risk of recurrent myocardial infarction in patients with prior coronary events."},
  {"doc_id": "d2", "title": "Metformin and renal function", "text": "Metformin can be continued in mild chronic kidney disease with dose adjustment and monitoring of eGFR."},
  {"doc_id": "d3", "title": "Statin-associated muscle symptoms", "text": "Myalgia is the most common adverse effect reported by patients taking statins."},
  {"doc_id": "d4", "title": "Vitamin D and falls", "text": "Vitamin D supplementation has a modest effect on fall prevention in older adults with deficiency."},
  {"doc_id": "d5", "title": "Influenza vaccination timing", "text": "Annual influenza vaccination is recommended before the start of the flu season."},
  {"doc_id": "d6", "title": "Hypertension lifestyle changes", "text": "Reducing sodium intake and regular exercise lower blood pressure in most patients."}
]
```

- [ ] **Step 3: Write the e2e config** — the full TTS→ASR→embed→retrieval→metrics explicit graph (shape from the deleted `e2e_pubmed_qa_small.yaml`; edges copied verbatim from `git -C ../evaluator show c6fc9a1^:configs/e2e_pubmed_qa_small.yaml`, paths relative to repo root)

```yaml
experiment:
  name: e2e_mini
  output_dir: evaluation_results/e2e_mini
dataset:
  id: pubmed_qa
  questions: tests/data/pubmed_qa_mini/questions.json
  corpus: tests/data/pubmed_qa_mini/corpus.json
  trace_limit: 4
  batch_size: 2
data:
  num_workers: 0
graph:
  nodes:
  - dataset_source
  - {id: tts, type: tts, params: {provider: mms, voice: en, sample_rate: 16000, seed: 42, output_dir: evaluation_results/e2e_mini/audio}}
  - corpus_embedding
  - vector_db
  - asr
  - text_embedding
  - retrieval
  - transcription_metrics
  - retrieval_metrics
  - metrics
  - build_query_traces
  - finalize
  edges:
  # paste the `edges:` list from `git -C ../evaluator show c6fc9a1^:configs/e2e_pubmed_qa_small.yaml` here, unchanged
nodes:
  asr: {model: whisper, name: openai/whisper-base, device: cpu}
  text_embedding: {model: labse, device: cpu}
  vector_db: {store: inmemory}
  retrieval: {k: 3}
runtime:
  cache: {enabled: false}
```

Run `git -C ../evaluator show c6fc9a1^:configs/e2e_pubmed_qa_small.yaml | sed -n '/^  edges:/,/^nodes:/p'` and paste the edge lines in place of the comment.

- [ ] **Step 4: Write `tests/fixtures_paths.py`**

```python
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI_Q = os.path.join(_ROOT, "tests", "data", "pubmed_qa_mini", "questions.json")
MINI_C = os.path.join(_ROOT, "tests", "data", "pubmed_qa_mini", "corpus.json")
E2E_MINI = os.path.join(_ROOT, "tests", "data", "configs", "e2e_mini.yaml")
```

- [ ] **Step 5: Verify the config loads and builds a graph**

Run: `python3 -c "from evaluator.config import EvaluationConfig; c=EvaluationConfig.from_yaml('tests/data/configs/e2e_mini.yaml'); print(len(c.graph.nodes))"`
Expected: an integer ≥ 12, no exception.

- [ ] **Step 6: Commit**

```bash
git add tests/data tests/fixtures_paths.py
git commit -m "tests: add mini PubMed fixture + e2e_mini config"
```

### Task 0.2: Re-point tests at the fixture; delete stale tests

**Files:**
- Modify: `tests/test_config_roundtrip.py:15-17,55-56,275,294`
- Modify: `tests/test_dag_export.py:9,22`
- Modify: `tests/test_emit_metrics.py:12,23`
- Modify: `tests/test_run_graph_parity.py:18-19`
- Modify: `tests/test_resolved_node_config.py:17-18,119-121`
- Modify: `tests/test_webapi_endpoints.py:24-25,122`
- Modify: `tests/test_builder_run.py:20-21`
- Modify: `tests/test_builder_integration.py:18-19`
- Modify: `tests/test_introspection_schema.py:187-188,231-232,398-399`
- Modify: `tests/test_presets.py:11,15,56-59`
- Modify: `tests/test_graph_store.py:22-23`
- Modify: `tests/test_explicit_edges.py:172,202-203`

- [ ] **Step 1: Replace every `examples/data/pubmed_qa_small/{questions,corpus}.json` literal with the fixture constants.** In each file listed above add `from tests.fixtures_paths import MINI_Q, MINI_C, E2E_MINI` (adjust to what the file uses) and replace the string literals. Where a test builds the path with `_ROOT / "examples/..."`, use the constant directly. Where a test uses a relative string `"examples/data/pubmed_qa_small/questions.json"`, use `os.path.relpath(MINI_Q)`; tests run from repo root so relative paths resolve.

- [ ] **Step 2: Replace `configs/e2e_pubmed_qa_small.yaml` references with `E2E_MINI`** in `test_config_roundtrip.py:15`, `test_dag_export.py:9`, `test_emit_metrics.py:12`, `test_run_graph_parity.py` (search for `e2e_pubmed`), `test_resolved_node_config.py:121`, `test_explicit_edges.py:202`.

- [ ] **Step 3: Delete tests bound to configs that exist nowhere** (grep confirms none of these names exist under `configs/` here or in `../evaluator`):
  - `test_config_roundtrip.py`: the test at line 275 (`e2e_pubmed_qa_branched`, uses removed `graph.branches`) and the test at line 294 (`local_llm_example`). Delete both test functions whole.
  - `test_dag_export.py`: the test at line 22 (`e2e_pubmed_qa_3branch`, `graph.branches`). Delete.
  - `test_emit_metrics.py`: the test at line 23 (`c7_correction_branches`). Delete.
  - `test_resolved_node_config.py:119-121`: drop the two `showcase_*` names from the parametrization; keep `E2E_MINI`.
  - `test_webapi_endpoints.py:122`: the test loading `evaluation_config_pubmed_qa.yaml`. Replace the file read with `yaml.safe_load(open(E2E_MINI))`. If the assertions then fail because they were config-specific, delete the test.
  - `test_explicit_edges.py:172`: remove `showcase_correction_after_optimization.yaml` from the set; `:202-203`: parametrize over `E2E_MINI` and `configs/apm_tests/admed/evaluation_config_admed_selfretr_fusion.yaml` (verify with `ls configs/apm_tests/admed/`), drop the other two.
  - `test_presets.py`: replace `"e2e_pubmed_qa_small"` with `"pubmed_qa_rag_fulltext"` (exists in `configs/`) in all three places.

- [ ] **Step 4: Run the touched files**

Run: `$PYT tests/test_config_roundtrip.py tests/test_dag_export.py tests/test_emit_metrics.py tests/test_run_graph_parity.py tests/test_resolved_node_config.py tests/test_webapi_endpoints.py tests/test_builder_run.py tests/test_builder_integration.py tests/test_introspection_schema.py tests/test_presets.py tests/test_graph_store.py tests/test_explicit_edges.py`
Expected: only failures left are those listed in Task 0.4 (real drift). Record any other failure and fix the fixture path before continuing.

- [ ] **Step 5: Commit**

```bash
git add tests
git commit -m "tests: point fixture-bound tests at tests/data; drop tests for configs that no longer exist"
```

### Task 0.3: Restore the golden regen script and regenerate

**Files:**
- Create: `scripts/regen_graph_golden.py` (copied from `../evaluator/scripts/regen_graph_golden.py`)
- Modify: `tests/data/graph_golden.json`, `tests/data/graph_order_golden.json`

- [ ] **Step 1: Copy the script**

Run: `cp ../evaluator/scripts/regen_graph_golden.py scripts/regen_graph_golden.py && sed -n '1,40p' scripts/regen_graph_golden.py`
Confirm `config_paths()` globs `configs/*.yaml`, `configs/apm_tests/**/*.yaml`, `configs/examples/*.yaml`. Remove the `configs/campaign/*.yaml` glob line (directory does not exist).

- [ ] **Step 2: Regenerate and check the diff is additive**

Run: `python3 scripts/regen_graph_golden.py && git diff --stat tests/data/ && git diff tests/data/graph_golden.json | grep -c '^-  ' `
Expected: the count of removed lines is 0 (only additions: the 39 admed/pubmed configs that were never frozen). If any existing entry changed, stop: something in the current code differs from when the golden was frozen; report it to the owner before continuing.

- [ ] **Step 3: Run the golden tests**

Run: `$PYT tests/test_graph_golden.py`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/regen_graph_golden.py tests/data
git commit -m "tests: restore golden regen script; freeze the 39 unfrozen configs (additive)"
```

### Task 0.4: Triage the six real failures

**Files:** determined per item below.

For each item: reproduce, decide fix-or-delete by the rule "the test pins documented behavior → fix the code; the test pins removed behavior → delete the test", record the decision in the commit body.

- [ ] **Step 1: `tests/test_model_registries.py::test_resolve_model_name_by_size_and_default`** — error `Unknown size 'large-v3' for AudioEmbedding model 'attention_pool'`. Run `grep -n "large-v3" tests/test_model_registries.py evaluator/models/a2e/attention_pool.py`. If the test asks `attention_pool` for a size its `Params.SIZES` never declared, change the test to a size it declares (`grep -n SIZES -A6 evaluator/models/a2e/attention_pool.py`). If `SIZES` was removed from `attention_pool` by mistake (check `git -C ../evaluator log -p -S'large-v3' -- evaluator/models/a2e/attention_pool.py | head -60`), restore it.

- [ ] **Step 2: `tests/test_multi_variant_results.py::test_two_retrieval_variants_both_survive_a_real_graph_run`** — `KeyError: 'variants'`. Run `$PYT tests/test_multi_variant_results.py -x 2>&1 | grep -B5 KeyError`. Find which dict lacks `variants` (report vs test expectation). `grep -rn '"variants"' evaluator/evaluation/aggregate.py evaluator/evaluation/handlers/metrics.py` tells whether the report still emits it. If the code emits it under another key now, update the test; if the code dropped it, restore emission (the README promises variant rollups).

- [ ] **Step 3: `tests/test_builder_integration.py::test_builder_page_has_save_load_ui`** — expects `id="btn-save-graph"`. Run `grep -rn 'btn-save-graph\|save-graph' evaluator/webapi/templates evaluator/webapi/static`. If the button was renamed, update the test to the new id; if it was removed, delete the test.

- [ ] **Step 4: `tests/test_ui_theme.py::test_both_token_sets_ship_and_differ`** — expects `--chart-cat-1: #3987e5`. Run `grep -n 'chart-cat-1' evaluator/webapi/static/*.css`. Update the test to the shipped value if the palette changed intentionally (both light and dark values must still differ); otherwise restore the CSS.

- [ ] **Step 5: `tests/test_integration_audio_emb.py::test_audio_emb_complete_nodelist_graph_produces_metrics`** — `MRR` is `None` and the log says `node audio_embedding failed`. Run `$PYT tests/test_integration_audio_emb.py -x -s 2>&1 | grep -A3 'audio_embedding failed'`. The mocked audio-embedding model probably no longer matches the handler's expected interface. Fix the mock in the test (preferred) or the handler if the handler's contract changed without a test.

- [ ] **Step 6: `tests/test_webapi_endpoints.py::test_graph_template_endpoint_serves_dag_and_404s`** — delete this test; the template endpoint is removed in Task 1.1. Also delete `tests/test_graph_templates.py` and `tests/test_template_loading.py` if they only exercise the named templates (check: `grep -n 'def test' tests/test_graph_templates.py tests/test_template_loading.py`). Any test in them that exercises `build_graph_for_config` on an explicit graph is moved into `tests/test_graph_config.py`.

- [ ] **Step 7: Full suite**

Run: `$PYT 2>&1 | tail -3`
Expected: `N passed, 4 skipped` with 0 failed, 0 errors.

- [ ] **Step 8: Commit** (one commit per item is fine; the body of each names the decision)

```bash
git add -A tests evaluator
git commit -m "tests: triage six pre-existing failures"
```

### Task 0.5: Resolved-config dump

**Files:**
- Create: `$SCRATCH/resolved_dump.py` (not committed)

- [ ] **Step 1: Write the script**

```python
"""write: dump every config's resolved dict to $SCRATCH/resolved/<name>.json.
check: re-dump to a temp dir and diff. Prints RESOLVED OK or the differing names."""
import glob, json, os, sys, tempfile, filecmp
ROOT = "/home/krystian/studia/magisterka/evaluator_clean"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resolved")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import yaml
from evaluator.config import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs, resolved_node_config
from evaluator.config.serialization import to_nested_dict


def configs():
    pats = ["configs/*.yaml", "configs/apm_tests/**/*.yaml", "configs/examples/*.yaml",
            "tests/data/configs/*.yaml"]
    return sorted(p for pat in pats for p in glob.glob(pat, recursive=True))


def dump(path, out_dir):
    raw = yaml.safe_load(open(path))
    cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs(raw), validate=False)
    doc = {"nested": to_nested_dict(cfg), "nodes": resolved_node_config(cfg)}
    name = path.replace("/", "__") + ".json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=str)


def main(mode):
    if mode == "write":
        os.makedirs(OUT, exist_ok=True)
        for p in configs():
            dump(p, OUT)
        print(f"wrote {len(configs())} dumps to {OUT}")
        return
    tmp = tempfile.mkdtemp()
    for p in configs():
        dump(p, tmp)
    _, diff, _ = filecmp.cmpfiles(OUT, tmp, os.listdir(OUT), shallow=False)
    missing = set(os.listdir(OUT)) ^ set(os.listdir(tmp))
    if not diff and not missing:
        print("RESOLVED OK")
    else:
        print("RESOLVED DIFF:", sorted(diff | missing))
        sys.exit(1)


main(sys.argv[1])
```

- [ ] **Step 2: Write and self-check**

Run: `python3 $SCRATCH/resolved_dump.py write && python3 $SCRATCH/resolved_dump.py check`
Expected: `wrote 79 dumps ...` then `RESOLVED OK`.

Note on Phase 1: Task 1.1 renames `graph_template` → `graph_label` in the nested dict. After Task 1.1 completes and its diff is reviewed, re-run `write` once (that is the single sanctioned re-baseline; record it in the Task 1.1 commit body).

---

## Phase 1 — Dead code

### Task 1.1: Remove the named-template layer

**Files:**
- Delete: `evaluator/pipeline/graph/templates.py`, `evaluator/pipeline/graph/assembly.py`
- Create: `evaluator/pipeline/graph/build.py` (the explicit-graph builder, moved out of `modes.py`)
- Delete: `evaluator/pipeline/graph/modes.py` (after its survivors moved)
- Modify: `evaluator/pipeline/graph/__init__.py:33-40`
- Modify: `evaluator/config/evaluation.py:141-161`, `evaluator/config/serialization.py:74-80,104-110`, `evaluator/config/validation.py:197`, `evaluator/config/graph_config.py:453,707-708,805`
- Modify: `evaluator/evaluation/executor/run.py:20-22,191-245`
- Modify: `evaluator/services/evaluation_service.py:318`, `evaluator/pipeline/factory.py:444`, `evaluator/cli/main.py:215,232`
- Modify: `evaluator/webapi/routers/config.py:7,65-80`, `evaluator/webapi/routers/introspection.py:106`, `evaluator/webapi/form_builder.py:839-855`, `evaluator/webapi/ui/builder.py:13`, the builder template (`grep -rn 'Reload template\|graph/template' evaluator/webapi/templates evaluator/webapi/static`)
- Modify tests: `tests/graph_golden_util.py:18`, `tests/graph_test_helpers.py:7`, `tests/test_graph_only_config.py:12`, `tests/test_multi_variant_plumbing.py:12,291`, `tests/test_webapi_registry_contract.py:14-17`, `tests/test_explicit_edges.py:198,225`, `tests/test_resolved_node_config.py:13`, `tests/test_streaming_foundation.py:15`, `tests/test_dag_export.py:5`, `tests/test_graph_modes.py`, `tests/test_introspection_schema.py:41`, `tests/test_expressiveness.py:12`, `tests/test_run_graph_parity.py:13`, `tests/test_node_registry.py:115`

**Interfaces:**
- Produces in `evaluator/pipeline/graph/build.py`:
  - `build_graph_for_config(config) -> StageGraph` (same behavior for explicit graphs; raises `ValueError("a config needs an explicit graph.nodes to build from")` when `graph_override` has no `nodes`)
  - `build_run_graph(graph_override, *, eval_config) -> StageGraph` (explicit graphs only; the old feature-flag keyword arguments are removed — grep callers: `evaluator/evaluation/executor/run.py`, `tests/graph_golden_util.py`, `tests/test_run_graph_parity.py`)
  - `graph_label(graph_override) -> Optional[str]` (was `label_from_graph`; unchanged body)
  - `complete_structural_plumbing(graph: dict) -> None` (unchanged signature)
  - `structural_specs(nodes: list) -> list` (new; replaces `assemble_specs(mode, _infer_features(nodes))` filtered by `is_structural`)
- `EvaluationConfig.graph_label` property replaces `graph_template` (returns `graph_label(self.graph_override)`).

- [ ] **Step 1: Write the failing test for `structural_specs`** in `tests/test_graph_config.py`

```python
def test_structural_specs_mirror_the_old_template_derivation():
    from evaluator.pipeline.graph.build import structural_specs
    # asr + retrieval: transcription + retrieval metrics, report, finalize
    assert structural_specs(["dataset_source", "asr", "text_embedding", "retrieval"]) == [
        "transcription_metrics", "retrieval_metrics", "metrics", "finalize"]
    # asr only: no retrieval metrics
    assert structural_specs(["dataset_source", "asr"]) == [
        "transcription_metrics", "metrics", "finalize"]
    # pure text retrieval: no transcription metrics
    assert structural_specs(["dataset_source", "text_embedding", "retrieval"]) == [
        "retrieval_metrics", "metrics", "finalize"]
    # audio+text fusion: alignment diagnostic
    assert structural_specs(["dataset_source", "audio_embedding", "text_embedding", "fusion", "retrieval"]) == [
        "embedding_alignment_metrics", "retrieval_metrics", "metrics", "finalize"]
    # answer_gen adds answer_metrics; judge adds the trace builder
    assert structural_specs(["dataset_source", "text_embedding", "retrieval", "answer_gen", "answer_judge"]) == [
        "retrieval_metrics", "metrics", "answer_metrics", "build_query_traces", "finalize"]
    # no query head and no retrieval: nothing to derive
    assert structural_specs(["dataset_source", "text_embedding"]) == []
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PYT tests/test_graph_config.py -k structural_specs`
Expected: `ModuleNotFoundError: evaluator.pipeline.graph.build`

- [ ] **Step 3: Create `evaluator/pipeline/graph/build.py`** by moving these functions from `modes.py` verbatim (keep bodies, apply the comment policy): `_spec_type`, `_spec_params`, `_spec_id`, `_edge_key`, `_topo_sort_nodes`, `_terminal_variant_ids`, `_variant_roots_diverge`, `complete_structural_plumbing`, `_complete_with_plumbing_per_variant`, `_dataset_fields_for`, `_derived_source_outputs`, `_attach_dataset_fields`, `_attach_display_fields`, `label_from_graph` (renamed `graph_label`), `build_graph_for_config`, `build_run_graph`. Then add:

```python
def structural_specs(nodes: list) -> list:
    """The plumbing nodes an explicit graph needs, in execution-slot order."""
    from .operators import node_kind

    kinds = {node_kind(_spec_type(n), _spec_params(n)) for n in nodes}
    label = graph_label({"nodes": nodes})
    if label is None and "retrieval" in kinds:
        label = "asr_text_retrieval"
    if label is None:
        return []
    if label == "asr_only":
        return ["transcription_metrics", "metrics", "finalize"]
    out: list = []
    # the old template added transcription_metrics only to the asr_text_retrieval template,
    # and the plumbing step dropped it again when that template was borrowed for a graph
    # without an asr node
    if label == "asr_text_retrieval" and "asr" in kinds:
        out.append("transcription_metrics")
    if label == "audio_text_retrieval" and ("fusion" in kinds or "result_fusion" in kinds):
        out.append("embedding_alignment_metrics")
    if label != "asr_only":
        out.append("retrieval_metrics")
    out.append("metrics")
    if "answer_gen" in kinds:
        out.append("answer_metrics")
    if "answer_judge" in kinds:
        out.append("build_query_traces")
    out.append("finalize")
    return out
```

In `complete_structural_plumbing`, replace the block that computes `structural_template` (the `assemble_specs(mode, _infer_features(nodes))` list comprehension and the `mode` fallback above it) with `structural_template = structural_specs(nodes)`. Keep the duplicate-id guard and everything after unchanged.

Rewrite the two builders without `FeatureSet`:

```python
def _wire_explicit_graph(graph_override, *, config, attach_fields: bool) -> StageGraph:
    from .wiring import _has_port_edges, build_graph_from_spec

    override = graph_override or {}
    if not override.get("nodes"):
        raise ValueError("a config needs an explicit graph.nodes to build from")
    nodes = override["nodes"]
    edges = override.get("edges")
    label = graph_label(override) or "custom"
    if _has_port_edges(edges):
        graph = build_graph_from_spec(nodes, mode=label, edges=edges)
        return _attach_display_fields(graph, config) if attach_fields else graph
    if attach_fields:
        nodes = _attach_dataset_fields(nodes, config)
    return build_graph_from_spec(nodes, mode=label, edges=edges)


def build_graph_for_config(config) -> StageGraph:
    return _wire_explicit_graph(getattr(config, "graph_override", None), config=config, attach_fields=True)


def build_run_graph(graph_override, *, eval_config) -> StageGraph:
    return _wire_explicit_graph(graph_override, config=eval_config, attach_fields=False)
```

- [ ] **Step 4: Delete `modes.py`, `templates.py`, `assembly.py`.** Update `pipeline/graph/__init__.py` to export from `.build` (`build_graph_for_config`, `build_run_graph`, `graph_label`, `complete_structural_plumbing`, `structural_specs`) and drop the `FeatureSet`, `assemble_specs`, `build_stage_graph`, `GRAPH_TEMPLATE_SPECS`, `resolve_graph_template` exports.

- [ ] **Step 5: Update every importer** listed in Files. Specific edits:
  - `config/evaluation.py`: property `graph` imports `from ..pipeline.graph.build import build_graph_for_config`; property `graph_template` becomes `graph_label` returning `graph_label(self.graph_override)` (import from `..pipeline.graph.build`). `grep -rn 'graph_template' evaluator tests` and rename every use to `graph_label`.
  - `config/serialization.py:74-80` `_config_template_label`: return `config.graph_label`.
  - `config/validation.py:197`: delete the check that referenced `GRAPH_TEMPLATE_SPECS` (a template name can no longer be referenced).
  - `config/graph_config.py:707-708`: the branch that assembled a template (`FeatureSet, assemble_specs`) is unreachable for explicit graphs — delete that branch; if it was the else-branch of "no `nodes` given", raise `GraphConfigError("graph.nodes is required")` there instead.
  - `evaluation/executor/run.py`: remove `GRAPH_TEMPLATES` import and `_resolve_mode_and_graph`'s template lookup; call `build_run_graph(graph_override, eval_config=eval_config)`; the run's mode label is `graph_label(graph_override) or "custom"`.
  - `pipeline/factory.py:444` `_config_template`: replace with `config.graph_label`.
  - webapi: delete `graph_template_endpoint` (`routers/config.py:65-80`), `list_graph_templates` uses (`routers/introspection.py:106`, `ui/builder.py:13`), `form_builder.py:839-855` `required_model_fields_for`, and the "Reload template" button + its JS handler (`grep -rn 'template' evaluator/webapi/templates/builder.html evaluator/webapi/static/*.js`).
  - tests: `tests/graph_test_helpers.py` and `tests/test_node_registry.py:115`, `tests/test_webapi_registry_contract.py:14-17`, `tests/test_streaming_foundation.py:15`, `tests/test_graph_modes.py` used `FeatureSet`/`assemble_specs`/`build_stage_graph`/`_wire_mode_graph` to build graphs by feature flags. Rewrite each to build from an explicit node list via `build_graph_from_spec(nodes, mode="custom", edges=emit_edges(nodes))` (both in `evaluator.pipeline.graph.wiring`). If a test's only purpose was the template layer (its name or docstring says template/mode/FeatureSet contract), delete it.

- [ ] **Step 6: Run the suite**

Run: `$PYT 2>&1 | tail -3 && grep -rn 'modes\|templates\|assembly\|FeatureSet\|graph_template' evaluator tests --include='*.py' | grep -v 'retrieval_mode\|_modes\b' | head`
Expected: 0 failed; the grep prints nothing (the `graph_golden` tests must pass without regeneration — the structural derivation is byte-identical).

- [ ] **Step 7: Re-baseline the resolved dump (sanctioned once) and commit**

Run: `python3 $SCRATCH/resolved_dump.py write`

```bash
git add -A evaluator tests
git commit -m "graph: remove named-template layer; explicit graphs derive plumbing statically"
```
Body: "structural_specs replaces assemble_specs(mode, _infer_features) filtered by is_structural; graph_template renamed graph_label; resolved dump re-baselined for the key rename."

### Task 1.2: Remove `graph.branches` expansion

**Files:**
- Delete: `evaluator/pipeline/graph/branches.py`
- Modify: `evaluator/pipeline/graph/__init__.py:19`, `evaluator/webapi/form_builder.py:305-370`, `evaluator/webapi/form_config.py:111`
- Modify tests: `tests/test_explicit_edges.py` (the import of `build_branched_graph`/`expand_branches` and the tests using them)

- [ ] **Step 1: Confirm dead.** Run `grep -rn 'branches' evaluator/config/*.py | grep -v '^\s*#'`. Expected: no line reads a `branches` key from YAML (the loader rejects it). Run `grep -rn 'branches' evaluator/webapi/templates evaluator/webapi/static`. Expected: no JS or template emits a `branches` key.

- [ ] **Step 2: Delete** `branches.py`, the two names from `graph/__init__.py`, the `branches` handling block in `form_builder.py` (`build_canvas_graph`'s `if branches:` paths), `"branches"` from `_GRAPH_SPEC_KEYS`. In `tests/test_explicit_edges.py` delete the tests that import from `branches`.

- [ ] **Step 3: Run**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check`
Expected: 0 failed; `RESOLVED OK`.

- [ ] **Step 4: Commit**: `git commit -am "graph: remove graph.branches expansion (loader never accepted it)"`

### Task 1.3: Vulture-verified deletions

**Files:** the candidates below; each deletion also deletes the tests that only exist to cover it.

- [ ] **Step 1: Produce the candidate list**

Run: `python3 -m vulture evaluator --min-confidence 60 | grep -vE "unused (variable|attribute)" | sort > $SCRATCH/vulture.txt; wc -l $SCRATCH/vulture.txt`

- [ ] **Step 2: For each candidate run the liveness check** and keep a decision line in `$SCRATCH/vulture_decisions.txt` (`KEEP <name> <reason>` or `DELETE <name>`):

```bash
name=<symbol>; grep -rn "\b$name\b" evaluator tests scripts evaluator/webapi/templates evaluator/webapi/static README.md | grep -v "def $name\|class $name" | head -5
```
Decision rules, in order:
  1. Decorated with `@register_*`, `@router.*`, `@app.*`, `@register_stage_handler`, `@register_corrector`, `@pytest.fixture` → KEEP (registry dispatch).
  2. Referenced from README or `evaluator-architecture.md` as public API (`to_dataframe`, `get_metric`, `evaluate_from_*`, `quick_evaluate`, `run_evaluation`) → KEEP.
  3. Any reference outside its own definition and outside tests → KEEP.
  4. Referenced only by tests → DELETE the symbol and those tests.
  5. No reference → DELETE.

Expected DELETEs from the planning scan (verify each with the command above before deleting): `datasets/utils.py: validate_schema, normalize_text, extract_field`; `devices/memory.py: get_peak_memory_usage, get_memory_stats, reset_stats`; `devices/pool.py: managed_allocation`; `benchmarks/report.py: load_benchmark_results, compare_benchmark_results`; `metrics/rag.py: faithfulness, answer_relevance, factual_correctness`; `storage/leaderboard.py: upsert_run, list_runs`; `storage/payload_store.py: get_many`; `evaluation/item_set.py: select`; `evaluation/run_journal.py: restore_state`; `logging_config.py: log_timing`; `models/a2e/clap_style.py: CLAPOutput, freeze_encoders, unfreeze_encoders`; `models/embedding_space.py: register_compatible_spaces`; `models/registry.py: get_params_class`; `models/retrieval/query/optimization.py: _hash_request`; `pipeline/audio/augmentation.py: create_variants`; `webapi/form_builder.py: nested_config`; `webapi/chart_data.py: _nums`; `webapi/jobs.py: request_cancel`; `datasets/loaders/huggingface.py: list_known_datasets`; `evaluation/query_correction.py: correct_one_text`; `models/retrieval/rag/strategies.py:110` unused local `doc_payloads`.

- [ ] **Step 3: Delete the DELETE set**, run `python3 -m flake8 --select=F401 evaluator` and remove imports that became unused.

- [ ] **Step 4: Run**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check`
Expected: 0 failed; `RESOLVED OK`.

- [ ] **Step 5: Commit**: `git add -A evaluator tests && git commit -m "cleanup: delete unreferenced functions/classes (vulture-verified)"` with the decisions file content pasted as the body.

### Task 1.4: Fold `evaluation/runner.py`

**Files:**
- Delete: `evaluator/evaluation/runner.py`
- Modify: `evaluator/cli/commands.py:10,89`, `evaluator/cli/replay.py:58,89`, `evaluator/evaluation/results_io.py:74` (docstring mention)

- [ ] **Step 1:** In `services/evaluation_service.py` add a public function directly under `_run_core`:

```python
def run_from_config(config, *, cache_manager=None, progress_callback=None,
                    service_provider=None, query_ids=None):
    """Build pipelines, prepare data, evaluate; returns the metrics dict."""
    from ..storage.cache import CacheManager

    if cache_manager is None:
        cache_manager = CacheManager(cache_dir=config.cache.cache_dir, enabled=config.cache.enabled)
    metrics, _dataset = _run_core(config, cache_manager=cache_manager, service_provider=service_provider,
                                  progress_callback=progress_callback, query_ids=query_ids)
    return metrics
```

- [ ] **Step 2:** Replace `from evaluator.evaluation.runner import run_evaluation_from_config` with `from evaluator.services.evaluation_service import run_from_config` in `cli/commands.py` and `cli/replay.py`; rename the two call sites. Delete `evaluation/runner.py`. Fix the docstring mention in `results_io.py`.

- [ ] **Step 3: Run**: `$PYT tests/test_replay.py tests/test_cli_and_jobs_misc.py && $PYT 2>&1 | tail -3`
Expected: PASS; 0 failed.

- [ ] **Step 4: Commit**: `git add -A && git commit -m "services: fold evaluation/runner.py alias into run_from_config"`

---

## Phase 2 — Duplication

### Task 2.1: HuBERT / WavLM shared base

**Files:**
- Create: `evaluator/models/a2e/_hf_ssl.py`
- Modify: `evaluator/models/a2e/hubert.py` (213 → ~30 lines), `evaluator/models/a2e/wavlm.py` (214 → ~30 lines)
- Test: `tests/test_audio_pooling.py` (existing; extend)

**Interfaces:**
- Produces: `class HFSpeechEncoderEmbedding(AudioEmbeddingModel)` with class attributes `HF_MODEL_CLASS: str` (e.g. `"HubertModel"`) and `DEFAULT_MODEL_NAME: str`; methods `__init__(model_name=None, pooling="mean")`, `to`, `preprocess_audio`, `encode_from_features`, `_get_feat_extract_output_lengths`, `encode_audio`, `name` — bodies copied from `hubert.py:41-213` with the transformers class looked up by name.

- [ ] **Step 1: Write the failing test** (append to `tests/test_audio_pooling.py`)

```python
def test_hubert_and_wavlm_share_the_encoder_base():
    from evaluator.models.a2e._hf_ssl import HFSpeechEncoderEmbedding
    from evaluator.models.a2e.hubert import HuBERTAudioModel
    from evaluator.models.a2e.wavlm import WavLMAudioModel
    assert issubclass(HuBERTAudioModel, HFSpeechEncoderEmbedding)
    assert issubclass(WavLMAudioModel, HFSpeechEncoderEmbedding)
    assert HuBERTAudioModel.HF_MODEL_CLASS == "HubertModel"
    assert WavLMAudioModel.HF_MODEL_CLASS == "WavLMModel"
    assert HuBERTAudioModel.Params.SIZES["base"] == "facebook/hubert-base-ls960"
    assert WavLMAudioModel.Params.SIZES["base-plus"] == "microsoft/wavlm-base-plus"
    assert HuBERTAudioModel.Params.CHOICES == {"pooling": ["mean", "cls"]}
    assert WavLMAudioModel.Params.CHOICES == {"pooling": ["mean", "cls"]}
```

- [ ] **Step 2: Run it**: `$PYT tests/test_audio_pooling.py -k share_the_encoder_base` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `_hf_ssl.py`.** Copy `hubert.py` lines 41–213 into a class `HFSpeechEncoderEmbedding(AudioEmbeddingModel)`; in `__init__` replace `from transformers import HubertModel, Wav2Vec2FeatureExtractor` and `HubertModel.from_pretrained(...)` with:

```python
import transformers
from transformers import Wav2Vec2FeatureExtractor

model_cls = getattr(transformers, self.HF_MODEL_CLASS)
self.model_name = model_name or self.DEFAULT_MODEL_NAME
self.model = model_cls.from_pretrained(self.model_name)
```
`name()` returns `f"{type(self).__name__} - model:{self.model_name} - pooling:{self.pooling} - dim:{self.hidden_dim}"` (identical strings for both subclasses). Apply the comment policy to the copied body.

- [ ] **Step 4: Rewrite `hubert.py`**

```python
"""HuBERT audio embedding (facebook/hubert-*)."""
from dataclasses import dataclass
from typing import ClassVar, Dict

from ..registry import register_audio_embedding_model
from ._hf_ssl import HFSpeechEncoderEmbedding


@register_audio_embedding_model(
    "hubert", default_name="facebook/hubert-base-ls960",
    description="HuBERT self-supervised audio representation model",
)
class HuBERTAudioModel(HFSpeechEncoderEmbedding):
    HF_MODEL_CLASS = "HubertModel"
    DEFAULT_MODEL_NAME = "facebook/hubert-base-ls960"

    @dataclass
    class Params:
        size: str = "base"
        pooling: str = "mean"
        SIZES: ClassVar[Dict[str, str]] = {
            "base": "facebook/hubert-base-ls960",
            "large": "facebook/hubert-large-ll60k",
        }
        CHOICES: ClassVar[Dict[str, list]] = {"pooling": ["mean", "cls"]}
```

`wavlm.py` identically with `"wavlm"`, `default_name="microsoft/wavlm-base"`, description `"WavLM self-supervised audio representation model from Microsoft"`, `HF_MODEL_CLASS = "WavLMModel"`, `SIZES = {"base": "microsoft/wavlm-base", "base-plus": "microsoft/wavlm-base-plus", "large": "microsoft/wavlm-large"}`, and the same `CHOICES` (wavlm previously lacked it; adding it only affects the builder form, and the registry test in Step 1 pins it).

- [ ] **Step 5: Run**: `$PYT tests/test_audio_pooling.py tests/test_model_registries.py tests/test_introspection_schema.py && $PYT 2>&1 | tail -3`
Expected: PASS; 0 failed.

- [ ] **Step 6: Commit**: `git add -A evaluator/models/a2e tests && git commit -m "models/a2e: share the HF speech-encoder base between hubert and wavlm"`

### Task 2.2: FAISS vector-store base

**Files:**
- Modify: `evaluator/storage/vector_store.py:147-371`
- Test: `tests/test_vector_store.py` (existing)

**Interfaces:**
- Produces: `class _FaissStore(VectorStore)` holding `search` and `search_batch` (bodies from lines 159–179, byte-identical results: `search` returns numpy scalars, `search_batch` returns `float`), plus `_write_payloads(dir)`, `_read_payloads(dir)` helpers for the JSON sidecar. `FaissVectorStore`, `FaissMmapVectorStore`, `FaissGpuVectorStore` subclass it and keep only `__init__`, `build`, `save`, `load` (`FaissMmapVectorStore` also keeps its `_payload_at` override).

- [ ] **Step 1: Write the failing test** (append to `tests/test_vector_store.py`)

```python
def test_faiss_stores_share_search_implementation():
    from evaluator.storage import vector_store as vs
    assert vs.FaissVectorStore.search is vs._FaissStore.search
    assert vs.FaissMmapVectorStore.search_batch is vs._FaissStore.search_batch
    assert vs.FaissGpuVectorStore.search is vs._FaissStore.search
```

- [ ] **Step 2: Run**: `$PYT tests/test_vector_store.py -k share_search` — Expected: `AttributeError: _FaissStore`.

- [ ] **Step 3: Implement.** Insert after `_ensure_faiss`/`VectorStore`:

```python
class _FaissStore(VectorStore):
    index: Any

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return [(p, scores[0][j])
                for j, i in enumerate(indices[0])
                if i >= 0 and (p := self._payload_at(int(i))) is not None]

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[Any, float]]]:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return [[(p, float(s)) for s, i in zip(row_scores, row_indices)
                 if i >= 0 and (p := self._payload_at(int(i))) is not None]
                for row_scores, row_indices in zip(scores, indices)]

    def _write_payloads(self, directory: Path) -> None:
        with open(directory / "payloads.json", "w") as f:
            json.dump(self.payloads, f, indent=2)

    def _read_payloads(self, directory: Path) -> None:
        with open(directory / "payloads.json") as f:
            self.payloads = json.load(f)
```

Then delete `search`/`search_batch` from the three FAISS classes, make them subclass `_FaissStore`, and replace their inline payload JSON blocks in `save`/`load` with `self._write_payloads(save_path)` / `self._read_payloads(load_path)`. `InMemoryVectorStore` also uses the two payload helpers (move them up to `VectorStore` if that reads cleaner; then `_FaissStore` keeps only search).

- [ ] **Step 4: Run**: `$PYT tests/test_vector_store.py tests/test_vector_store_registry.py tests/test_vectorstore_load.py tests/test_off_ram_store.py && $PYT 2>&1 | tail -3`
Expected: PASS; 0 failed.

- [ ] **Step 5: Commit**: `git commit -am "storage: one FAISS search implementation for the three FAISS stores"`

### Task 2.3: chromadb / qdrant shared chunks

**Files:**
- Modify: `evaluator/storage/backends/chromadb_store.py:63-90,270-290`, `evaluator/storage/backends/qdrant_store.py:122-170,358-380`
- Test: `tests/test_vector_store_registry.py` (existing, model-free)

- [ ] **Step 1: Locate the clones**: `python3 $SCRATCH/dup_scan.py evaluator/storage/backends 5` prints the paired line ranges. Read both ranges.

- [ ] **Step 2: Extract** each shared block into a module-level function in `evaluator/storage/backends/_common.py` (e.g. `payload_ids(payloads) -> list[str]`, `hits_to_results(ids, scores, payload_at) -> list[tuple]`, `batched(iterable, size)`), call it from both stores. Intra-file clones (`chromadb_store.py:78` vs `:270`, `qdrant_store.py:156` vs `:358`) become one private method each.

- [ ] **Step 3: Run**: `python3 $SCRATCH/dup_scan.py evaluator/storage 5` shows no pair with ≥3 windows inside `backends/`; `$PYT tests/test_vector_store_registry.py` passes.

- [ ] **Step 4: Commit**: `git commit -am "storage/backends: share chromadb/qdrant hit and payload helpers"`

### Task 2.4: `operators_catalog` ↔ `registry` constants

**Files:**
- Modify: `evaluator/pipeline/graph/operators_catalog.py:15-60`, `evaluator/pipeline/graph/registry.py:25-70`, `evaluator/pipeline/graph/artifacts.py`

- [ ] **Step 1:** `diff <(sed -n 15,60p evaluator/pipeline/graph/operators_catalog.py) <(sed -n 25,70p evaluator/pipeline/graph/registry.py)` to see the duplicated constant block (artifact-name constants).

- [ ] **Step 2:** Keep one definition in `evaluator/pipeline/graph/artifacts.py` (it already holds artifact vocabulary — `sed -n 1,40p` it). Both files import the names from there. If `artifacts.py` imports `registry`, put the constants in a new leaf `evaluator/pipeline/graph/_names.py` instead and import from it in all three.

- [ ] **Step 3: Run**: `$PYT tests/test_node_registry.py tests/test_operators.py tests/test_graph_golden.py && python3 $SCRATCH/resolved_dump.py check`
Expected: PASS; `RESOLVED OK`.

- [ ] **Step 4: Commit**: `git commit -am "graph: single definition of artifact-name constants"`

### Task 2.5: Intra-file clones

**Files (pair → helper name):**
- `evaluator/storage/cache/io.py:65` vs `:106` → `_atomic_write(path, writer)`
- `evaluator/cli/benchmark.py:206` vs `:232` → `_print_benchmark_table(rows, title)`
- `evaluator/benchmarks/model_benchmark.py:147` vs `:336` → `_time_batches(fn, batches)`
- `evaluator/models/factory.py:151` vs `:182` → `_instantiate_registered(registry, name, params, device)`
- `evaluator/datasets/loaders/factory.py:45` vs `:98` → `_resolve_loader(kind, config)`
- `evaluator/analysis/significance.py:36` vs `:69` → `_paired_arrays(a, b)`
- `evaluator/evaluation/executor/state.py:313` vs `:329` → `_copy_scoped_fields(src, dst, scope)`
- `evaluator/storage/leaderboard.py:172` vs `:335` → `_row_to_run(row)`
- `evaluator/models/retrieval/embedding_fusion/__init__.py:124` vs `:136` → `_check_dims(a, b)`
- `evaluator/datasets/builtins/{fleurs,local_audio,huggingface_audio}.py` shared `from_config` body → `evaluator/datasets/builtins/_common.py: audio_dataset_kwargs(config)`

The helper names above are proposals; the implementer reads both ranges and names the helper for what the shared code does.

- [ ] **Step 1:** For each pair: read both ranges (`sed -n`), confirm they are the same logic with different variables, write the helper directly above the first use, call it from both sites.

- [ ] **Step 2: After each file**: run the tests that import it (`grep -rl '<module name>' tests | xargs $PYT`), then the full suite at the end.

- [ ] **Step 3: Verify**: `python3 $SCRATCH/dup_scan.py evaluator 6` — every remaining pair with ≥3 windows must be one the implementer judged not worth merging, listed with a reason in the commit body.

- [ ] **Step 4: Commit** per file: `git commit -am "<module>: extract <helper> (was duplicated at L<a>/L<b>)"`

### Task 2.6: resample-to-16k helper

**Files:**
- Modify: `evaluator/utils/audio.py` (add `to_mono_16k`), `evaluator/models/asr/base_asr.py:119`, `evaluator/models/asr/faster_whisper.py:165`, `evaluator/models/a2e/_hf_ssl.py` (the loop in `preprocess_audio`), `evaluator/models/a2e/clap_style.py:590`, `evaluator/models/a2e/attention_pool.py` (`_resample_to_16k`)
- Test: `tests/test_audio_utils.py` (existing)

- [ ] **Step 1: Write the failing test**

```python
def test_to_mono_16k_matches_the_inline_sites():
    import torch, torchaudio
    from evaluator.utils.audio import to_mono_16k
    stereo = torch.randn(2, 8000)
    out = to_mono_16k(stereo, 8000, downmix="mean", as_numpy=True)
    ref = torchaudio.functional.resample(stereo, 8000, 16000).squeeze().numpy().mean(axis=0)
    assert out.shape == ref.shape and abs(out - ref).max() < 1e-6
    mono = torch.randn(4000)
    assert to_mono_16k(mono, 16000, downmix="mean", as_numpy=False) is mono
```

- [ ] **Step 2: Run**: `$PYT tests/test_audio_utils.py -k to_mono_16k` — Expected: ImportError.

- [ ] **Step 3: Implement** in `evaluator/utils/audio.py`:

```python
def to_mono_16k(audio, sr: int, *, downmix: str = "mean", as_numpy: bool = False):
    """Resample to 16 kHz and collapse channels. Returns the input object unchanged when
    nothing has to be done and ``as_numpy`` is False (keeps the identity the ASR path relies on)."""
    import torch, torchaudio

    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)
    changed = False
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        changed = True
    if not as_numpy and not changed and audio.ndim == 1:
        return audio
    x = audio.squeeze()
    if x.ndim > 1:
        x = x.mean(dim=0) if downmix == "mean" else x[0]
    return x.numpy() if as_numpy else x
```

- [ ] **Step 4: Replace each site** only where the site's observable output is identical: read the six sites (`grep -n 'resample' evaluator/models -r`). A site that casts dtype, keeps stereo, or resamples to another rate keeps its own code and gets a one-line comment naming the difference. Expect 3–4 of the 6 to switch.

- [ ] **Step 5: Run**: `$PYT tests/test_audio_utils.py tests/test_audio_mono.py tests/test_audio_pooling.py tests/test_clap_checkpoint_safety.py && $PYT 2>&1 | tail -3`

- [ ] **Step 6: Commit**: `git commit -am "audio: one to_mono_16k helper for the interchangeable resample sites"`

### Task 2.7: MLflow call sequence

**Files:**
- Create: `evaluator/tracking/mlflow_calls.py`
- Modify: `evaluator/tracking/mlflow_tracker.py`, `evaluator/analysis/tracking_export.py`
- Test: `tests/test_tracking_export.py` (existing)

- [ ] **Step 1:** `diff <(grep -n 'mlflow\.' evaluator/tracking/mlflow_tracker.py) <(grep -n 'mlflow\.' evaluator/analysis/tracking_export.py)` to list the shared sequence (set experiment → start run → log params → log metrics → log artifacts → end run).

- [ ] **Step 2:** Write `mlflow_calls.py` with one function per shared step, e.g. `log_run(mlflow, *, experiment, run_name, params, metrics, artifacts, tags)`; both modules call it. The live tracker keeps its incremental `log_metric` path if it logs during the run.

- [ ] **Step 3: Run**: `$PYT tests/test_tracking_export.py tests/test_export_excel.py`

- [ ] **Step 4: Commit**: `git commit -am "tracking: share the MLflow logging sequence between live tracker and export"`

---

## Phase 3 — Layering

### Task 3.1: Layering test (written first, expected to fail, then made to pass by 3.2–3.7)

**Files:**
- Create: `tests/test_import_layering.py`

**Interfaces:**
- Produces: the ordered layer list `LAYERS` and the lazy-import allowlist `LAZY_ALLOWED` that later tasks edit.

- [ ] **Step 1: Write the test**

```python
"""Package layering: module-level imports form a DAG that respects LAYERS; no cross-package
import of an underscore name; function-local imports of internal modules only where allowed."""
import ast
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "evaluator"

LAYERS = [
    {"constants", "errors", "logging_config", "utils"},
    {"core", "config"},
    {"metrics", "llm_client", "storage", "devices", "tracking"},
    {"models"},
    {"datasets"},
    {"pipeline"},
    {"evaluation", "judge", "analysis", "benchmarks"},
    {"services"},
    {"cli", "webapi", "api", "public_api", "diagnostics"},
]
RANK = {pkg: i for i, layer in enumerate(LAYERS) for pkg in layer}

# module → reason. Function-local imports of evaluator internals are allowed only here.
LAZY_ALLOWED = {
    "config/evaluation.py": "EvaluationConfig.graph builds via pipeline (documented downward edge)",
}


def _pkg(rel):
    parts = rel.split("/")
    return parts[0].removesuffix(".py") if len(parts) == 1 else parts[0]


def _imports(path, rel):
    tree = ast.parse(path.read_text())
    parts = rel.split("/")
    top, local, private = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = parts[:-1][: len(parts) - 1 - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
                target = mod.split(".")[0] if mod else _pkg(rel)
            elif node.module and node.module.startswith("evaluator."):
                target = node.module.split(".")[1]
            else:
                continue
            names = [a.name for a in node.names]
            entry = (target, node.lineno, names)
            (top if node.col_offset == 0 else local).append(entry)
            if target != _pkg(rel):
                private += [(n, node.lineno) for n in names if n.startswith("_") and n != "__version__"]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("evaluator."):
                    entry = (a.name.split(".")[1], node.lineno, [a.name])
                    (top if node.col_offset == 0 else local).append(entry)
    return top, local, private


def _modules():
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if "__pycache__" in rel:
            continue
        yield p, rel


def test_module_level_imports_respect_layer_order():
    bad = []
    for p, rel in _modules():
        src = _pkg(rel)
        top, _, _ = _imports(p, rel)
        for target, line, names in top:
            if target == src or target not in RANK or src not in RANK:
                continue
            if RANK[target] > RANK[src]:
                bad.append(f"{rel}:{line} imports {target} ({names}) from a higher layer")
    assert not bad, "\n".join(bad)


def test_no_cross_package_private_imports():
    bad = []
    for p, rel in _modules():
        _, _, private = _imports(p, rel)
        bad += [f"{rel}:{line} imports private {n}" for n, line in private]
    assert not bad, "\n".join(bad)


def test_function_local_internal_imports_are_allowlisted():
    bad = []
    for p, rel in _modules():
        _, local, _ = _imports(p, rel)
        if local and rel not in LAZY_ALLOWED:
            bad.append(f"{rel}: {len(local)} function-local internal imports (first at line {local[0][1]})")
    assert not bad, "\n".join(bad)
```

- [ ] **Step 2: Run it to see the current violations**: `$PYT tests/test_import_layering.py -x 2>&1 | grep -c 'imports\|function-local'` — Expected: a large number; save the full output to `$SCRATCH/layering_before.txt`.

- [ ] **Step 3: Commit the test marked xfail** so the suite stays green until Task 3.7 flips it:

```python
pytestmark = pytest.mark.xfail(reason="layering cleanup in progress (audit P3)", strict=False)
```
`git add tests/test_import_layering.py && git commit -m "tests: layering contract (xfail until P3 lands)"`

### Task 3.2: `utils → storage` and `evaluation → cli`

**Files:**
- Move: `evaluator/utils/cache_helpers.py` → `evaluator/storage/cache/helpers.py`
- Modify: every importer (`grep -rn 'cache_helpers' evaluator tests`)
- Move function: `generate_output_filename` from `evaluator/cli/utils.py` → `evaluator/evaluation/results_io.py`; `cli/utils.py` and `results_io.py:81` import/use it from there.

- [ ] **Step 1:** `git mv evaluator/utils/cache_helpers.py evaluator/storage/cache/helpers.py`; fix relative imports inside it (`..storage.cache` → `.`); update importers.
- [ ] **Step 2:** Move `generate_output_filename` (cut from `cli/utils.py`, paste into `results_io.py` above its first use; `cli/utils.py` gets `from ..evaluation.results_io import generate_output_filename` if other CLI code uses it — `grep -rn generate_output_filename evaluator`).
- [ ] **Step 3: Run**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check`
- [ ] **Step 4: Commit**: `git commit -am "layering: cache helpers live in storage; output filename lives with results io"`

### Task 3.3: `pipeline → evaluation` and `datasets → pipeline`

**Files:**
- Create: `evaluator/models/retrieval/query/correctors.py` (registry only)
- Modify: `evaluator/evaluation/query_correction.py` (implementations register into the new module), `evaluator/pipeline/graph/operators_catalog.py:332`, `evaluator/config/query_correction.py:73`
- Move function: `validate_field_mapping` from `evaluator/pipeline/artifacts.py` → `evaluator/datasets/descriptor.py`; update `pipeline/artifacts.py` importers (`grep -rn validate_field_mapping evaluator tests`).

- [ ] **Step 1: Write `correctors.py`**

```python
"""Corrector registry: name → callable(texts, config, client=None) -> texts."""
from typing import Callable, Dict, List

_CORRECTORS: Dict[str, Callable] = {}


def register_corrector(name: str):
    def deco(fn):
        _CORRECTORS[name] = fn
        return fn
    return deco


def get_corrector(name: str) -> Callable:
    try:
        return _CORRECTORS[name]
    except KeyError:
        raise KeyError(f"unknown corrector {name!r}; registered: {sorted(_CORRECTORS)}") from None


def list_correctors() -> List[str]:
    return sorted(_CORRECTORS)
```
Read `evaluation/query_correction.py`'s current registry (`grep -n '_CORRECTORS\|def register_corrector\|def list_correctors\|def get_corrector' evaluator/evaluation/query_correction.py`) and match its exact semantics (ordering of `list_correctors`, error type on unknown name) — adjust the code above to be identical, then delete the registry from `query_correction.py` and import from `correctors.py`. Keep `evaluation/query_correction.py` re-registering its four correctors at import time. Because `pipeline/graph/operators_catalog.py` calls `list_correctors()` at graph-build time, it must import `evaluator.evaluation.query_correction` for its side effect — that is a reverse edge. Instead move the four corrector *implementations* too: they are pure `(texts, config, client) -> texts` functions over `models/retrieval/query` helpers (`grep -n '^from\|^import' evaluator/evaluation/query_correction.py` shows what they depend on). If every dependency is at or below `models`, move the whole file to `evaluator/models/retrieval/query/correction.py` and leave `evaluator/evaluation/query_correction.py` deleted, updating importers (`grep -rn 'query_correction import' evaluator tests`). If a dependency is above `models`, keep only the registry move and add `evaluation/handlers/query.py` as the import-for-side-effect site with a comment.

- [ ] **Step 2:** Move `validate_field_mapping` into `datasets/descriptor.py`; `pipeline/artifacts.py` imports it from there if it still needs it (check with grep).

- [ ] **Step 3: Run**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check`
- [ ] **Step 4: Commit**: `git commit -am "layering: corrector registry below pipeline; field-mapping validation in datasets"`

### Task 3.4: `config → evaluation / models / datasets`

**Files:**
- Move: `evaluator/evaluation/node_config.py` → `evaluator/config/node_config.py`
- Modify: `evaluator/config/validation.py:114-115,164,178,265,393`, `evaluator/config/query_correction.py:73`, `evaluator/config/query_optimization.py:68`, `evaluator/evaluation/validation.py` (pre-flight chain gains the moved checks), `evaluator/config/graph_config.py:495`

- [ ] **Step 1:** `git mv evaluator/evaluation/node_config.py evaluator/config/node_config.py`; update importers (`grep -rn 'node_config' evaluator tests`). Its `_KIND_KEEP` becomes `KIND_KEEP` (public; it crosses modules).

- [ ] **Step 2:** In `config/validation.py`, identify each function that imports from `evaluation`, `models` or `datasets` (lines 164, 178, 265, 393 and the `query_correction.py:73` / `query_optimization.py:68` list-checks). Cut those functions (registry-dependent: metric names exist, reducer names exist, model names resolve, dataset ids resolve, corrector/strategy names exist) and paste them into `evaluator/evaluation/validation.py` as steps of `run_pre_flight` (read `grep -n 'def run_pre_flight' -A40 evaluator/evaluation/validation.py` for the chain shape). Rollup reducer names: replace the `_ROLLUP_REDUCERS` import with a constant `ROLLUP_REDUCERS = ("mean", "min", "max")` in `config/evaluation.py`, and make `evaluation/aggregate.py` build its reducer dict from that tuple.

- [ ] **Step 3:** `config/graph_config.py:495` and `config/validation.py:393` look up dataset descriptors (`get_descriptor`, `list_registered_datasets`, `resolve_dataset_descriptor`) while translating/validating. Read `evaluator/datasets/descriptor.py` (`grep -n '^from\|^import\|^class\|^def\|^_REGISTRY\|@dataclass' evaluator/datasets/descriptor.py`). Split it: the descriptor dataclass, the id→descriptor registry dict, `get_descriptor`, `list_registered_datasets` and `resolve_dataset_descriptor` move to `evaluator/config/dataset_descriptors.py` (they are data + lookup, no loader code); `datasets/descriptor.py` keeps the loader binding and imports the registry from the new module; `datasets/builtins/*.py` keep registering through `register_eval_dataset`, which stays in `datasets/` and writes into the config-level registry dict. `graph_config.py` and `validation.py` import from `.dataset_descriptors`.

- [ ] **Step 3b:** `config/vector_db.py:6` imports the constant `RETRIEVAL_DEPTH` from `metrics/ir.py` (a higher layer). Move the constant's definition to `evaluator/constants.py`; `metrics/ir.py` and `config/vector_db.py` both import it from there (`grep -rn RETRIEVAL_DEPTH evaluator tests` for every use).

- [ ] **Step 4:** Ensure every check that moved still runs on `EvaluationConfig.from_yaml(..., validate=True)`: `grep -n 'run_pre_flight\|validate_config' evaluator/config/evaluation.py evaluator/config/loading.py evaluator/services/evaluation_service.py`. `from_yaml(validate=True)` must call the pre-flight chain the same way the CLI does; add the call if it only ran from the service.

- [ ] **Step 5: Run**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check && $PYT tests/test_preflight_model_validation.py tests/test_metric_allowlist.py tests/test_configs_smoke.py -v | tail -5`
- [ ] **Step 6: Commit**: `git commit -am "layering: config no longer imports registries; registry checks run in pre-flight"`

### Task 3.5: `services` ↔ `evaluation` / `pipeline`

**Files:**
- Move: `evaluator/services/corpus_index.py` → `evaluator/pipeline/corpus_index.py`; `evaluator/services/model_provider.py` → `evaluator/pipeline/model_provider.py`; `evaluator/services/model_services.py` → `evaluator/pipeline/model_services.py`
- Modify: importers (`grep -rn 'services.corpus_index\|services.model_provider\|services.model_services\|from \.\.services\|from \.services' evaluator tests`), `evaluator/services/__init__.py`

- [ ] **Step 1:** `git mv` the three files; fix their relative imports (`..` depth changes); update importers.
- [ ] **Step 2:** Check the moved modules import nothing from `evaluation` or `services` at module level: `grep -nE '^from \.\.(evaluation|services)|^from evaluator\.(evaluation|services)' evaluator/pipeline/corpus_index.py evaluator/pipeline/model_provider.py evaluator/pipeline/model_services.py`. Any hit is a function that belongs in `evaluation/` (e.g. a handler helper that ended up in the service) — move that function to `evaluator/evaluation/handlers/_common.py`.
- [ ] **Step 3: Run**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check`
- [ ] **Step 4: Commit**: `git commit -am "layering: corpus index + model provider live in pipeline; services keeps the orchestrator"`

### Task 3.6: Cut `pipeline → config` reverse edges; hoist lazy imports; publicize private names

**Files:**
- Modify: `evaluator/pipeline/audio/prepare.py`, `evaluator/pipeline/factory.py`, `evaluator/pipeline/graph/operators_catalog.py` (their `config` imports), then every module with function-local internal imports (list from `$PYT tests/test_import_layering.py::test_function_local_internal_imports_are_allowlisted 2>&1 | grep function-local`), then every private cross-import (`... ::test_no_cross_package_private_imports`).

- [ ] **Step 1: `pipeline → config`**: `config` is *below* `pipeline` in LAYERS, so `pipeline` importing `config` types at module level is allowed. What must go is `config` importing `pipeline` at module level (none after Task 1.1 except lazily in `config/evaluation.py`). Confirm: `grep -rnE '^from \.\.pipeline|^from evaluator\.pipeline' evaluator/config` prints nothing.

- [ ] **Step 2: Hoist.** For each module in the function-local list: move each internal import to the module top. If moving it causes an `ImportError` (cycle) when running `python3 -c "import evaluator.<module>"`, that is a real cycle inside a package — resolve by moving the imported function to the lower module, not by keeping the lazy import. Lazy imports of *third-party* heavy modules (torch, transformers, faiss, mlflow, chromadb, qdrant) stay; they are not internal and the test ignores them.

- [ ] **Step 3: Publicize.** For each `(module, name)` in the private-import list: drop the leading underscore at the definition and all uses (`grep -rn "\b_name\b" evaluator tests`). If the name is re-exported through a package `__init__`, add it to `__all__`.

- [ ] **Step 4: Run**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check`

- [ ] **Step 5: Commit** in three commits: `layering: hoist function-local internal imports`, `layering: public names for cross-module helpers`.

### Task 3.7: Flip the layering test to strict

- [ ] **Step 1:** Remove the `pytestmark = pytest.mark.xfail(...)` line from `tests/test_import_layering.py`.
- [ ] **Step 2: Run**: `$PYT tests/test_import_layering.py -v` — Expected: 3 passed. Any remaining violation is fixed now, or added to `LAZY_ALLOWED` with a one-line reason (only for heavy-dependency deferral).
- [ ] **Step 3: Full suite + dump**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check`
- [ ] **Step 4: Commit**: `git commit -am "tests: layering contract is now strict"`

---

## Phase 4 — Comments

### Task 4.1: Mechanical sweep per package

**Files:** every `.py` under `evaluator/<pkg>/`, one task per package in this order: `pipeline`, `evaluation`, `config`, `webapi`, `models`, `storage`, `analysis`, `datasets`, `cli`, `services`, `devices`, `benchmarks`, `metrics`, `llm_client`, `judge`, `tracking`, `utils`, `core`, root modules.

Each package task is dispatched to a fresh subagent with this prompt verbatim, followed by the Global Constraints comment policy verbatim:

> Sweep the comments and docstrings in `evaluator/<pkg>/` under the policy below. Do not change code, names, imports, or formatting of code lines. Do not touch YAML. Edit with the native Edit tool. When done, run `python3 -m flake8 --select=E,W,F evaluator/<pkg>` (must be clean or no worse than before: run it first and save the count) and `$PYT` (0 failed). Report: files touched, comment lines removed, and the five comments you were least sure about (kept), quoted.

- [ ] **Step 1: Baseline per package**: `python3 $SCRATCH/comment_stats.py evaluator/<pkg> | head -2`
- [ ] **Step 2: Dispatch the sweep** (subagent).
- [ ] **Step 3: Review the diff yourself**: `git diff --stat && git diff | grep '^-' | grep -vE '^-\s*#|^-\s*"""|^-\s*[A-Z].*\.$|^---' | head` — every removed line that is not a comment/docstring line is a code change and must be reverted.
- [ ] **Step 4: Run**: `$PYT 2>&1 | tail -3 && python3 $SCRATCH/resolved_dump.py check`
- [ ] **Step 5: Commit**: `git commit -am "comments: <pkg> sweep"`

Packages are independent: up to four subagents may run at once on disjoint packages; review and commit each separately.

### Task 4.2: Tests sweep

Same as 4.1 for `tests/`, lighter: remove only milestone markers, history narration and commented-out code; keep every test docstring that states what the test guards.

- [ ] **Step 1: Dispatch**, **Step 2: Review diff**, **Step 3: `$PYT`**, **Step 4: Commit** `comments: tests sweep`.

### Task 4.3: Measure

- [ ] **Step 1:** `python3 $SCRATCH/comment_stats.py evaluator | head -2` — record the ratio (target ≈ 0.15; if above 0.20 the sweep was too timid — dispatch a second pass over the five packages with the highest remaining ratio).
- [ ] **Step 2:** `grep -rnoE '\b(M[0-9][a-z]?|[A-Z][0-9])[:)]' evaluator --include='*.py' | wc -l` — Expected: 0.

---

## Phase 5 — Docs

### Task 5.1: README

**Files:** `README.md`

- [ ] **Step 1:** Remove the "Core ideas" bullet about `graph.branches` (the Sweeping section says there is none, and Task 1.2 deleted the code). Keep the multi-variant explanation.
- [ ] **Step 2:** Remove the document-table rows and links for `configs/campaign/RUNBOOK.md` and `baselines/README.md`; remove the `m1c_check.py` lines and the "Parity gate" bullet from Verification; the Verification section says: unit suite with the three offline env vars, golden regen rule, layering test.
- [ ] **Step 3:** Replace `examples/data/pubmed_qa_small` mentions with `tests/data/pubmed_qa_mini`; remove the campaign paragraph (`scripts/build_pubmed_campaign.py` exists, keep that one line; drop `configs/campaign/...` commands).
- [ ] **Step 4:** Repository map: drop the `baselines/`, `configs/campaign/` rows; `evaluator/services/` row now says "orchestrator"; `evaluator/pipeline/` row gains "corpus index, model provider".
- [ ] **Step 5:** `grep -nE 'campaign|baselines|m1c|examples/data|graph\.branches|template' README.md` — Expected: only the line mentioning that node templates do not exist, if any.
- [ ] **Step 6: Commit**: `git commit -am "docs: README matches the repo (no campaign/parity artifacts, no branches, no templates)"`

### Task 5.2: Architecture doc

**Files:** `evaluator-architecture.md`

- [ ] **Step 1:** §13 source map: delete the "Graph templates (former modes)" row and the "Declarative graph-template assembly" row; update "One execution core" row (`services/evaluation_service.py:_run_core`) unchanged; "Corpus embed" row path → `pipeline/corpus_index.py`; delete the "Deterministic e2e parity-gate check" row.
- [ ] **Step 2:** §13 open items: delete the "Named graph templates" row (done) and every `docs/archive/...` link (directory does not exist here; the sentence "Completed roadmap rows … live in docs/archive" becomes "…are in git history").
- [ ] **Step 3:** `grep -nE 'modes\.py|assembly\.py|templates\.py|FeatureSet|GRAPH_TEMPLATE|graph\.branches|docs/archive|m1c|amazing_curie|examples/data' evaluator-architecture.md` — edit every hit: delete the sentence if it describes removed code, otherwise repoint the path.
- [ ] **Step 4:** Do not rewrite sections; only the hits above.
- [ ] **Step 5: Commit**: `git commit -am "docs: architecture doc de-staled after the audit"`

### Task 5.3: tests/README

**Files:** `tests/README.md`

- [ ] **Step 1:** Run line: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 pytest` with one sentence why. Remove the `m1c_check.py` / `baselines/` paragraph and the last bullet about the parity gate. Add a row for `test_import_layering.py` under a "Layering" area. Add the fixture note: `tests/data/pubmed_qa_mini` + `tests/data/configs/e2e_mini.yaml`.
- [ ] **Step 2: Commit**: `git commit -am "docs: tests README"`

### Task 5.4: Final measurement and report

- [ ] **Step 1:** Run and paste into the final message a table with before/after: `find evaluator -name '*.py' | xargs cat | wc -l`; `python3 $SCRATCH/comment_stats.py evaluator | head -2`; `python3 $SCRATCH/import_graph.py | grep -c '<->'`; `python3 $SCRATCH/dup_scan.py evaluator 6 | head -8`; `python3 -m vulture evaluator --min-confidence 60 | wc -l`; `grep -rnE '^\s{4,}from (\.|evaluator)' evaluator --include='*.py' | wc -l`; `$PYT 2>&1 | tail -1`.
- [ ] **Step 2:** `git log --oneline master..audit/2026-09 | wc -l` and list the commits.
- [ ] **Step 3:** Leave the branch unmerged; the owner merges.
