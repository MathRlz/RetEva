# Evaluator architecture — design

North-star design for the DAG-evaluator, and the ground-truth map of what exists today. An
experiment is a **directed graph**: typed **datasources** feed **artifacts**; **nodes** are
typed functions that transform artifacts; **metrics and aggregation are nodes too**. One
`EvaluationConfig` describes the whole graph; one execution core runs it.

**Status legend** — `[impl]` implemented today · `[planned]` designed (decision locked), not
yet built · `⊘` reclassified won't-do / deferred (with rationale). Everything in this doc is
`[impl]` unless tagged otherwise; the short list of open items lives in §13.

```
 datasources ──artifacts──▶ transform nodes ──artifacts──▶ metric nodes ─item_scores─▶ aggregate
  (typed fields:            (asr, embed, tts,             (auto-injected where         (reduce +
   text/audio/image,         fusion, augment,              its declared inputs          deltas →
   query↔answer,             retrieval, rag, …)            exist)                       report →
   query↔doc-pair)                                                                      sinks)
```

This doc is the **single architecture reference** — all architecture detail (incl. the former
DAG-execution doc) lives here, including the node-centric config schema (§10). Status of each
area is tracked inline in §13 (the completed task-trackers were retired once their work landed).

---

## Table of contents

1. [Principles](#1-principles)
2. [Datasources & typed fields](#2-datasources--typed-fields)
3. [Artifacts — the typed blackboard](#3-artifacts--the-typed-blackboard)
4. [Transform operations — the node algebra](#4-transform-operations--the-node-algebra)
5. [Augmentation nodes](#5-augmentation-nodes)
6. [Retrieval, RAG & query optimization](#6-retrieval-rag--query-optimization)
7. [Metrics as nodes](#7-metrics-as-nodes)
8. [Aggregation & reporting](#8-aggregation--reporting)
9. [Graph construction & execution](#9-graph-construction--execution)
10. [Configuration](#10-configuration)
11. [Models & registries](#11-models--registries)
12. [Cross-cutting: caching, lifecycle, web](#12-cross-cutting-caching-lifecycle-web)
13. [Extension points & status summary](#13-extension-points--status-summary)
14. [Operability & system hardening](#14-operability--system-hardening)
15. [Authoring guide — add a model, add a dataset, build experiments](#15-authoring-guide--add-a-model-add-a-dataset-build-experiments)

---

## 1. Principles

1. **Artifacts are typed, modality-tagged values.** Everything passed between nodes is a named
   artifact with a type (e.g. `query_audio : Audio`, `query_vectors : Vector[s]` where `s` is the
   embedding space). Nodes never call each other; they read/write artifacts on a per-run blackboard.
2. **Nodes are typed functions over artifacts.** A node declares the artifacts it consumes and
   produces; the executable handler is the function. Identity (`id`) is separate from type, so a
   type may appear many times (duplicate/parallel instances).
3. **Edges are derived from data, not hand-wired.** Given an ordered node list, each node
   auto-wires to the earlier nodes that produce its inputs. Parallel branches and merges fall out
   of the data dependencies.
4. **Metrics are nodes.** A metric node is `metric(*artifacts) → item_scores` (ground-truth is one
   *optional* input — reference-free metrics exist). Metric nodes are **auto-injected** wherever
   their declared inputs are present (every satisfiable metric lands in the report; the dataset's
   mode set picks the headline aliases). Reduction (per-item → scalar) is the aggregate node's job, not theirs.
5. **Aggregation is a node.** A terminal `aggregate` node collects all `item_scores` — across
   parallel branches — and reduces them into the final `report` (means, CIs, cross-branch deltas);
   terminal sink nodes then persist it.
6. **One config, one core.** The full `EvaluationConfig` *is* the experiment; every entry point
   (CLI, webapi, API) runs the same execution core.
7. **The graph is the spec.** What runs is the node graph — it drives both pipeline construction
   and handler behaviour. A *named template* (`graph.mode` → `graph_override['template']`) is
   optional sugar that assembles a default graph at config-creation time; there is no
   `pipeline_mode` runtime field, and a config carrying an explicit `graph.nodes` needs no template.

---

## 2. Datasources & typed fields

A **datasource** is a record of **typed fields**. The fields it advertises become the source
artifacts that enter the graph. Modeling a datasource as fields (rather than a fixed shape)
generalizes the system across modalities and task framings.

> The artifact vocabulary (§3) is an **extensible artifact-type registry** —
> `register_artifact(name, modality)` (`pipeline/artifacts.py`), like the model/node/dataset
> registries — with each datasource declaring a `field → registered artifact` mapping
> (`validate_field_mapping`). A new field/modality is additive (register the type + an embed
> op + map the field), no core enum edit.

**Typed dataset columns** `[impl]`. Each `DatasetDescriptor` declares its column schema —
`fields: {column_name → registered artifact}` (defaulted per `dataset_type` from
`FIELDS_BY_DATASET_TYPE`, validated against the artifact registry at registration). The
graph builder injects the schema into every `dataset_source` node (`params.fields`,
`modes.py:_attach_dataset_fields` — multi-source nodes resolve their `datasets:` entry), so:
(a) the node *advertises exactly its declared columns* (`_effective_outputs`, role-narrowed;
an empty role∩fields intersection falls back to the role slice), and (b) every DAG surface
shows `name: modality` columns on dataset nodes (`dataset_columns` — preview/builder cards,
`--print_graph`, run-detail) — an experiment's data is readable from its diagram.

**Dataset picker on the node** `[impl]`. In the builder, `dataset_source`'s `dataset` param
is a picker over the registered datasets (`/api/datasets` + `/api/dataset/{id}/fields`,
which carries the column schema and `required_settings`); picking one populates the node's
columns immediately and surfaces the dataset's **required settings** (descriptor
`required_data_fields`, e.g. pubmed_qa → `questions`/`corpus`, local → `audio_dir`) as
required inputs on the node, plus a **split picker** when the descriptor declares
``splits`` (e.g. huggingface/fleurs: train/validation/test; default = ``default_split``;
an undeclared split fails with the available list named — the check lives in
``DatasetDescriptor.validate_data_config`` so YAML and builder paths share it). At config
translation a node naming a registered dataset gets
a synthesized `data.datasets[<node_id>]` entry from its settings params
(`graph_config._synthesize_dataset_entry` — validated by the descriptor's own validator),
so builder-authored graphs run through the standard multi-source loader unchanged.

**Field / modality types**

| Field | Modality | Example |
|-------|----------|---------|
| `text` | text | question text, document text `[impl]` |
| `audio` | audio | spoken query, recording `[impl]` |
| `image` | image | RTG/scan, figure `[planned]` |
| `vector` | vector | precomputed embedding `[impl]` (§4.1 T4) |

**Relational structures** (how fields relate across records)

| Structure | Meaning | Today |
|-----------|---------|-------|
| query ↔ answer | a query and its reference short answer | `short_answers` `[impl]` |
| query ↔ document-pair | a query and its relevant corpus docs (graded) | `relevant_docs` via `doc_id` `[impl]` |
| query ↔ transcription | a spoken query and its spoken-GT text | `reference_transcription` `[impl]` |
| query ↔ question | a query and the dataset's question text | `reference_text` `[impl]` |
| patient ↔ history | a patient's prior medications + visits, grounding query correction | `patient_context` `[planned]` (C7) |
| corpus | the document collection (each doc = a typed record) | `[impl]` (text); multi-field/image `[planned]` |

Both references are **published as graph artifacts** so metric nodes auto-pair the scored text
against the right one (§3/§7): `reference_text` (= `question_text`, by `dataset_source`) is the
retrieval-side reference; `reference_transcription` (the dataset's `transcription` field, by the
`asr`/`audio_embedding` node) is the ASR-quality reference. The two coincide **only** on
TTS-bridge datasets where the spoken text *is* the question — conflating them was the migration's
semantic trap, now guarded by `tests/test_reference_semantics_parity.py`.

**Dataset types today** (`datasets/types.py`, `descriptor.py`) — capability flags
(`requires_audio/text`, `supports_generation`, `evaluation_mode`, `compatible_pipeline_modes` — now
naming graph templates) drive validation + default metrics:

| Type (`DatasetType`) | Outputs | Eval mode |
|------|---------|-----------|
| `audio_transcription` | `query_audio` + reference transcription | transcription |
| `audio_query_retrieval` | `query_audio` + `corpus` + `relevant_docs` | retrieval |
| `text_query_retrieval` | `query_text` + `corpus` + `relevant_docs` | retrieval |
| `multimodal_qa` | `query_text` + `corpus` + `relevant_docs` + `short_answers` | qa_retrieval |

Builtins `[impl]`: `admed_voice`, `fleurs` (transcription); `local`, `huggingface` (audio
retrieval); `pubmed_qa` (multimodal QA).

**TTS bridge** `[impl]`. Pipeline modes are audio-first; a text dataset with
`supports_generation=True` synthesizes `query_audio` from its text at run time
(`pipeline/audio/prepare.py`) — turning a text corpus into a voice benchmark with no recordings.

**Multiple datasources in one graph** (corpus from A + questions from B):
- A `dataset_source` node carries a `role`: `corpus` / `questions` / `both` (default). Role
  narrows advertised outputs (`stage_graph.py:_effective_outputs`) so downstream nodes wire to
  the right source.
- Config: a `datasets:` map (id → `{questions, corpus, role}`); a node references it via
  `params.dataset: <id>`.
- **Join contract.** A query links to corpus docs by shared **`doc_id`**
  (`evaluation/helpers.py:_build_relevant_from_item`). Cross-dataset IR metrics are meaningful
  only when the two `doc_id` namespaces overlap; a **disjoint** namespace disables the IR metrics
  (`datasets.validate_dataset_join` → `disable_ir_metrics`, so they aren't a misleading 0).
- Per-source runtime loading (`datasets.load_runtime_datasets` → `{id: QueryDataset}`) + per-source
  artifact keying (`corpus_embedding` binds its docs source via `_node_dataset`).

**Loading (runtime).** Dataset loading is **in-graph**, owned by the `dataset_source` node
(`handlers/source.py:_ensure_dataset_loaded`): the first such node to run resolves the
`DatasetDescriptor` (the single source of truth for a dataset's capabilities + loader), loads the
single + multi-source map + the disjoint-join gate + any replay slice, places the live dataset on
`RunState` (a shared resource, like the model pipelines), and surfaces its source artifacts into the
graph. TTS is likewise the in-graph `tts` node. Only **config validation**
(`validate_dataset_runtime_config`) stays pre-graph — no data load before the DAG, so the run is
exactly the graph the config describes.

---

## 3. Artifacts — the typed blackboard

A node publishes artifacts into a per-run blackboard
(`evaluation/run_context.py:RunContext`) keyed by `(producer_node_id, artifact_name)`. A consumer
reads via **bindings** — `(artifact_name, producer_id)` pairs resolved when the graph is built.
Per-producer keying lets two nodes produce the *same* artifact type without collision (the basis
for duplicate nodes, multi-dataset, and multi-branch).

Read semantics (`RunState.get_artifact`): for input `x`, scan this node's bound producers of
`x` **newest→oldest** and return the first that actually published — so a skipped producer (e.g.
fusion bailing) falls back to an earlier one.

Consumers that chain (query-text variants, vector streams) read a `one_of(...)` of distinct
names — the highest-priority alternative an upstream producer actually published — rather than
relying on newest-wins over one mutated name (see "consumption order" below).

| Artifact | Type | Produced by |
|----------|------|-------------|
| `query_text` | text | dataset_source (question text) **or** asr (hypothesis) — the IMMUTABLE base; never overwritten |
| `corrected_query_text` | text | query_correction |
| `augmented_query_text` | text | augmenter (query axis) |
| `optimized_query_text` | text | query_optimization (pre-retrieval rewrite/HyDE) |
| `refined_query_text` | text | query_refine (post-retrieval reformulation — top of the chain) |
| `query_audio` | audio | dataset_source, tts |
| `query_image` | image | dataset_source `[planned]` |
| `corpus` | record set | dataset_source |
| `relevant_docs` | query↔doc grades (GT) | dataset_source (sole producer) |
| `short_answers` | query↔answer (GT) | dataset_source |
| `reference_text` | the dataset's question text (retrieval-side GT) | dataset_source |
| `reference_transcription` | the spoken-transcription text (ASR-quality GT) | dataset_source (sole producer) |
| `patient_context` | per-query patient record: prior meds + visits | dataset_source `[planned]` (C7) |
| `correction_diff` | per-item raw→corrected record | query_correction |
| `audio_query_vectors` / `text_query_vectors` | `V[s]` per-stream query embeddings | audio_embedding / text_embedding |
| `fused_query_vectors` | `V[s]` embedding-level fused query | fusion |
| `query_vectors` | `V[s]` precomputed query column (dataset) / one_of read key | dataset_source (vector column) |
| `embedding_alignment` | audio↔text cosine diagnostic | embedding_alignment_metrics |
| `vector_index` | searchable index `[s]` | vector_db |
| `corpus_vectors` | `V[s]` embedded corpus (vectors + payloads) | corpus_embedding, corpus_merge |
| `retrieved` | ranked `(payload, score)` | retrieval, rerank, mmr, threshold, result_fusion, multi_query_retrieval |
| `generated_answers` | text | answer_gen |
| `transcription_scores` / `retrieval_scores` / `answer_scores` / `judge_scores` | per-comparison summary | the typed metric nodes (`*_metrics`) / answer_judge |
| `query_traces` | per-query retrieval/answer trace | build_query_traces |
| `report` | reduced result: scalars, CIs, cross-branch deltas + provenance | metrics / aggregate node |

**Consumption order (no in-place mutation).** The query-text transforms each emit a DISTINCT
name; consumers (text_embedding, retrieval, answer_gen, …) declare a `one_of(optimized,
augmented, corrected, query_text)` and read the most-processed variant an upstream node
published. Wiring restricts each node to the variants produced before it, so the same chain
expresses every correction/optimization on/off combination. The retrieval vector input is
likewise `one_of(fused, audio, text, query_vectors)`, so a bailing fusion falls back to the
audio stream by construction. Because `query_text` is never overwritten, it is the
un-rewritten ASR hypothesis — `wer`/`cer` score it directly (no `raw_query_text`).

**Ground-truth is first-class.** `relevant_docs` / `short_answers` / `reference_text` /
`reference_transcription` are all published artifacts, so metric nodes (§7) auto-pair a scored
artifact with its reference.

### Per-item identity: the `ItemSet`

Positional alignment is implicit and brittle: any node that changes cardinality — augmentation
emitting N variants/clip, `multi_query` expansion, or a per-item failure that drops an item —
silently breaks index-based pairing for everything downstream. That is why a per-item artifact
is an **`ItemSet`** — a columnar container:

```
ItemSet:
  ids:    ['q1', 'q2', 'q42·aug0', 'q42·aug1', ...]   # stable per-item key, with lineage
  values: <np.ndarray | list, aligned 1:1 to ids>     # the data (vectors, texts, results)
```

Why columnar (ids + aligned values array) rather than `dict[id→value]`: it keeps the **vectorized
batch path** (embedders still receive/return arrays) while making identity explicit. Operations:

- **transform** — a node maps `values → values'` and returns `itemset.with_values(values')`; ids
  ride along unchanged (embedding, ASR, correction).
- **filter / failure** — a node drops failed ids; the result is *sparse*. Downstream and metric
  nodes **join by id**, so gaps are tolerated rather than corrupting alignment (this is also the
  answer to failure semantics). **Failure policy = drop-and-log (no retry):** a per-item
  failure is logged and its id removed from that node's output `ItemSet`; the item simply does not
  reach downstream nodes (and is absent from their metrics). The effective eval-set size therefore
  shrinks; the count of dropped ids (per node) is recorded in the run provenance/`report` so a
  silently-smaller denominator is visible. (Retry/backoff is explicitly out of scope for now.)
- **fan-out (cardinality up)** — augmentation / `multi_query` emit **new ids with lineage**:
  `q42 → q42·aug0, q42·aug1`. The parent id is recoverable, so variants can later be rolled up
  (mean / worst-case) or aligned across branches.
- **join** — keyed consumers (the metric registry / `aggregate`) align two `ItemSet`s by id
  via `align` (not by position); a producer that published a subset still lines up.

`RunContext` stores `ItemSet`s; the per-producer keying is unchanged. Corpus-side artifacts
(`corpus`, `vector_index`) are not per-item and stay as-is. This is the **prerequisite for paired
cross-branch deltas** (§8): two branches can only be compared per query because both carry the
same `query_id`.

**Reading the bus.** Three accessors on `RunState`: `get_artifact` returns the plain values
(an `ItemSet` is unwrapped — positional consumers like embedders see arrays); `get_items`
returns an `ItemSet` (a plain-list publish is wrapped with index ids, best-effort);
`keyed_items` returns an `ItemSet` **only if the producer published true keyed identity** —
this is the per-item-identity source (no positional wrap), and a node that needs ids takes
them from the keyed artifact it reads (`ItemSet.ids` rides along).

**One bus.** The keyed `ctx`/`ItemSet` bus is the **single cross-node carrier** (it builds the
report + powers branching); the former flat per-branch accumulators (`all_hypotheses`,
`all_retrieved`, …) were retired by the staged M1 migration (2026-06 audit, parity-gated
byte-identical). Only per-branch *control* attrs stay on `RunState` — `current_node`, the
transiently-swapped pipelines, `query_opt_bypassed` — each scope-marked
(`executor/state.py:per_branch_field_names` drives `_NodeView` isolation; an unclassified field
fails a test). The migration's semantic trap (spoken *transcription* vs `question_text` as the
WER reference) is resolved by the `reference_transcription` artifact and pinned by
`tests/test_reference_semantics_parity.py`.

---

## 4. Transform operations — the node algebra

A transform node is a typed function `f : inputs → outputs` over artifacts. Writing the
signatures over modalities makes the algebra explicit (T=text, A=audio, I=image,
V[s]=vector in embedding space `s`; dimension implied by the space):

| Node | Signature | Notes | Status |
|------|-----------|-------|--------|
| `tts` | `T → A` | synthesize query audio from text | `[impl]` |
| `asr` | `A → T` | transcribe; emits the immutable `query_text` hypothesis (never overwritten) | `[impl]` |
| `text_embedding` | `T → V[s]` | `u = E_text(t)` in space `s` | `[impl]` |
| `audio_embedding` | `A → V[s]` | `u = E_audio(a)` in space `s` | `[impl]` |
| `image_embedding` | `I → V[s]` | `u = E_image(x)` in space `s` | `[planned]` |
| `fusion` | `V[s₁] × … × V[sₙ] → V[s']` | combine query embeddings (binary today; n-ary `[planned]`) | `[impl]` |
| `query_optimization` | `T → T` | pre-retrieval rewrite / HyDE (pure) → `optimized_query_text` | `[impl]` |
| `query_refine` | `T × retrieved → T` | post-retrieval reformulation (rewrite-with-context / relevance_feedback / self_rag_critique) → `refined_query_text` | `[impl]` |
| `multi_query_retrieval` | `T → retrieved` | RAG-fusion composite: expand → embed → retrieve → fuse (decompose / multi_query) | `[impl]` |
| `query_correction` | `T → T` | post-ASR domain correction (rule / KB-fuzzy / LLM) + `correction_diff` | `[impl]` |
| `augmenter` | `T → T` | robustness perturbation (ASR-homophone / dose-unit corruption, seeded) | `[impl]` |
| `corpus_embedding` | `corpus → V[s]` | embed corpus docs (text, or audio via TTS) | `[impl]` |
| `corpus_merge` | `V[s] × … → V[s]` | concatenate embedded corpora (set union; space/dim validated) | `[impl]` |
| `vector_db` | `V[s] → vector_index[s]` | build searchable index; owns the store backend choice per node (§6) | `[impl]` |

Procedurally, per query `i` over a batch:

```
text_embedding:  uᵢ = normalize( E_text(textᵢ) ) ∈ ℝ^d
audio_embedding: uᵢ = normalize( E_audio(audioᵢ) )
fusion:          uᵢ = Fuse(uᵢ_audio, uᵢ_text)        # dims validated; method-dependent
corpus_embedding: v_j = normalize( E(doc_j) ) for all docs;  vector_db: index I = {(v_j, payload_j)}
```

Embeddings are L2-normalized so that the dense retrieval inner product equals cosine similarity
(`storage/vector_store.py`).

### 4.1 The minimal op algebra (the lego set)

Every node above is an instance of one of **eleven orthogonal op shapes** — the minimum
brick set from which any experiment in this framework is assembled. The named nodes are
deliberate: **named bricks over generic nodes** — a scientist must read the experiment from
the diagram (`corpus_embedding` reads instantly; a generic `embed(axis=…)` does not), so
genericity lives in shared *cores* underneath the names, never in the node vocabulary.

| Op shape | Signature | Today's bricks | Status |
|----------|-----------|----------------|--------|
| **source** | — → typed columns | `dataset_source` | `[impl]` |
| **convert** (modality) | `X → Y` | `tts` (T→A), `asr` (A→T) | `[impl]` |
| **transform** (type-preserving) | `T → T` | `query_correction`, `query_optimization`, `augmenter` | `[impl]` both axes (`axis: docs` on augmenter; correction/optimization query-only until needed) |
| **embed** | `X → V[s]` | `text_embedding`, `audio_embedding`, `corpus_embedding` (+`image_embedding` `[planned]`) | `[impl]` |
| **combine per item** (align ids) | `V[s₁] × V[s₂] → V[s']` | `fusion` | `[impl]` |
| **union of sets** (⊎ disjoint ids) | `Set × … → Set` | `corpus_merge` (corpus axis); `dataset_union` (query axis) | `[impl]` |
| **index** | `V[s] → index[s]` | `vector_db` | `[impl]` |
| **search** | `(V[s], index[s]) → ranked` | `retrieval` | `[impl]` |
| **refine** | `ranked → ranked` | `rerank`, `mmr`, `threshold` (composable chain) | `[impl]` |
| **generate** (LLM) | `context → T` | `answer_gen` | `[impl]` |
| **measure / reduce / sink** | artifacts → scores → report → ∅ | `metrics` (auto-injected), `aggregate`, `*_sink` | `[impl]` |

**The axis insight.** `text_embedding` and `corpus_embedding` run the *same op* (`T → V[s]`,
same model registry); what differs is the **data axis** — per-query (`ItemSet` keyed by
`query_id`) vs per-document (keyed by `doc_id`) — plus the container (`query_vectors`
ItemSet vs `CorpusVectors`, the corpus-axis specialization carrying payloads) and the cache
granularity (per-item vs whole-corpus store-agnostic key). The same axis distinction
separates `fusion` (combine vectors *of the same item*, aligned by id) from `corpus_merge`
(union of item *sets*, ids disjoint). Keeping both axes first-class — rather than special-
casing the corpus as a blob — is what makes the algebra closed: any per-item op can, in
principle, run on either axis.

**Expressiveness gaps being closed** (the experiments a researcher could not author):

1. **Corpus-side transforms** `[impl]` (T2) — the `axis: docs` node param (the
   `augmenter` first; `_DOCS_AXIS_CAPABLE` lists the capable bricks):
   `_effective_inputs`/`_effective_outputs` flip the instance to `corpus → corpus`, the
   handler perturbs each doc's text with the same per-item seeding
   (`item_seed(seed, doc_id, node_id, variant)`), and `corpus_embedding` reads the newest
   corpus producer (bus-first). Showcase: `configs/showcase_corpus_robustness.yaml`.
2. **Generic set union** `[impl]` (T3, unblocked by P4) — `dataset_union` unions every
   bound producer's question-axis ItemSets (audio refs / texts / GT) with **disjoint
   query_ids enforced** (duplicate → `ConfigurationError`); `corpus_merge` remains the
   corpus-axis sibling. Possible because `query_audio` now rides the bus as **refs**
   (paths, never waveforms): ASR/audio-embedding resolve refs bus-first via
   `evaluation/audio_refs.py:RefAudioDatasetView` (metadata joined by query_id across
   all loaded datasets) and pass the dataset object through untouched when the refs
   match it (byte-identical parity).
3. **Precomputed-vector columns** `[impl]` (T4) — a dataset column mapped to
   `query_vectors`/`corpus_vectors` (modality `vector`, §2) feeds retrieval directly —
   third-party embeddings without an embed brick. The descriptor declares the columns in
   `fields` + an `embedding_space` id (V[s] pairing); `dataset_source` publishes them when
   every item carries an `embedding` (`handlers/source.py:_publish_vector_columns`), so
   `dataset_source → vector_db → retrieval` is a complete embed-free graph.

Prerequisite for axis-generic ops (T1, `[impl]`): the corpus travels the bus as a
**doc_id-keyed `ItemSet`** like every per-item artifact (`handlers/source.py:
_publish_corpus_itemset`; positional consumers unaffected via the `get_artifact`
unwrap, §3).

**Embedding-space typing (the `V[s]` tag).** A vector is only comparable to another vector from
the *same embedding space* `s` (same model/projection). Dense retrieval requires the query
embedder and the corpus embedder to share `s`; otherwise the inner product is meaningless. The
design therefore tags vector artifacts with their space and the graph validator must **reject a
retrieval whose query and corpus vectors carry different `s`**. Consequently `image_embedding`
only "drops into" existing retrieval when it shares the corpus space (a cross-modal model, e.g.
CLIP-style) — it is *not* free polymorphism. The space id is **model-declared** (`embedding_space`;
default unique per model, joint/cross-modal models share one) and the graph builder hard-fails a
retrieval whose query/corpus spaces differ (`models/embedding_space.py`,
`validate_embedding_spaces` wired into `_run_core`).

**Compatible-space registry + runtime guard** `[impl]`. Two *distinct* ids can be declared
cross-comparable when a model puts both modalities in one geometry — `register_compatible_spaces` /
`spaces_compatible` (`models/embedding_space.py`); the pre-flight validators accept a registered
pair, not just identical ids. A defense-in-depth **runtime guard** (`RetrievalPipeline.
assert_query_space`, fed by `validation.resolve_query_space`) re-checks the bound query stream
against the index's tagged space right before the dot product, so a path that slips past pre-flight
raises a loud `EmbeddingSpaceMismatch` instead of returning silent garbage.

**Cross-modal audio projection (APM)** `[impl]`. `attention_pool` / `attention_pool_m4t` project
audio into a *text* embedder's space (so audio queries retrieve text corpus); they share that space
id. The APM's checkpoint loader is strict-by-contract (`models/a2e/attention_pool.py`): it loads the
**encoder from the checkpoint's `audio_enc.*`** (so the Whisper/M4T encoder matches what training
used, not just the HF name — a size mismatch is a loud error), **fails loudly** if the pooling or
projection weights are absent (no silent random init), and applies the post-pool transform matching
training — **ABTT is L2-normalized, whitening is not** (`models/a2e/postprocessing.py`; the asymmetry
is load-bearing: the projection head was fit on unit-norm post-ABTT inputs).

---

## 5. Augmentation nodes

**Principle: an augmentation is a transform op that preserves type.** A node consumes artifact
`X` and produces `X'` of the *same* type, so it slots transparently in front of any consumer of
`X` — no downstream change. Domain-specificity is just parameterization.

```
augment_audio: A → A     # perturb the waveform
augment_text:  T → T     # perturb the text / inject ASR-style errors
augment_image: I → I     # perturb the image
```

| Augmentation | Op | Examples | Status |
|--------------|----|----------|--------|
| Audio | `A→A` | the `augment_audio` NODE: noise (white/pink/brown), speed, pitch, volume; N variants per clip with lineage ids; perturbed REFS republished on the bus (`pipeline/audio/augmentation.py:AudioAugmenter`; showcase `configs/showcase_audio_robustness.yaml`) | `[impl]` |
| Text / ASR-error | `T→T` | ASR-confusion homophone swap, dose/unit corruption (`mg`↔`mcg`), `max_edits` cap (`pipeline/text_augmentation.py:TextAugmenter`, the `augmenter` node) | `[impl]` |
| Domain-specific | `T→T` / `A→A` | medical dose/unit corruption to stress safety + correction (the dangerous-class swaps drive CEER / drug-dosage-safety) | `[impl]` |
| Image | `I→I` | crop/noise/contrast | `[planned]` |

Because augmenters are type-preserving nodes, a robustness experiment is just a branch:
`query_audio → augment_audio → asr → …` evaluated beside the clean branch (see §8 multi-branch).

**Determinism.** Augmenters are seeded per item, not globally:
`seed_i = hash(global_seed, query_id, node_id, variant_idx)`. A given item's variant is then
reproducible regardless of batch order or parallelism, and the `global_seed` is recorded in
`report.provenance`. Fan-out variants get lineage ids (`q42·aug0…`, §3) so they roll up cleanly.

---

## 6. Retrieval, RAG & query optimization

`[impl]` unless noted. Shared output contract: `ScoredRetrievalResult`
(`models/retrieval/contracts.py`).

**Retrieval modes** (`strategy.py`, `vector_db.retrieval_mode`):
- `dense` — `score(qᵢ, d_j) = ⟨ûᵢ, v̂_j⟩` (cosine via normalized vectors), top-k.
- `sparse` — lexical score.
- `hybrid` — fuse dense + sparse via a strategy:

```
weighted:  s(d) = w·denseₙ(d) + (1−w)·sparseₙ(d)        # min-max normalized per list
rrf:       s(d) = Σ_lists 1 / (rrf_k + rank_list(d))
max_score: s(d) = max( w·denseₙ(d), (1−w)·sparseₙ(d) )
```

(`models/retrieval/fusion_registry.py`; `w = hybrid_dense_weight`.)

**Embedding fusion** (`fusion` node, `embedding_fusion/`): combine audio + text *query*
embeddings before retrieval (audio_text_retrieval); dims validated. **Deferred:** fusion stays
binary (`V×V→V`); n-ary fusion (`fusion(*V)→V`) is revisited when image / 3-modality lands. `[planned]`

**Corpus split: `corpus_embedding` + `vector_db`** `[impl]`. The §4 split — the only corpus
path since the flip (the combined `corpus_index` node was removed; old YAMLs naming it fail
with a hint): `corpus_embedding` publishes `corpus_vectors`
(`services/corpus_index.py:CorpusVectors` — vectors + aligned payloads + space tag), cached under
a **store-agnostic** key (`cache_keys.py:corpus_embeddings_manifest_key`); the `vector_db` node
builds the index and **owns the backend choice per node** — `store` + backend essentials
(faiss_gpu `gpu_id`, chromadb `path`, qdrant `url`/`collection`) overlay the global `vector_db`
config transiently (`pipeline/factory.py:effective_vector_db_config`, never mutates the global).
Multiple corpora go into one DB via the explicit `corpus_merge` node (`corpus_vectors × … → corpus_vectors`, fusion's corpus-side sibling): it concatenates every bound producer's set — space/dim mismatches fail loud (`services/corpus_index.py:merge_corpus_vectors`) — and the following `vector_db` indexes the union. Two branches differing only in store share ONE `corpus_embedding` via CSE. Default pipeline
modes emit the pair, so every mode-derived diagram shows the vector DB node. Audio corpora go
through `corpus_embedding`'s audio path (`embed_corpus_audio`: TTS-synthesize + audio-embed,
uncached). Per-node optional backends are pre-flighted (`check_graph_backend_dependencies`
in `_run_core`).

**Reranking / refinement — three composable nodes** (each `retrieved → retrieved`, present
only when configured, chained in this order so the last feeds metrics/answer_gen):
- `rerank` — reorder by `cross_encoder` (Reranker registry) or `token_overlap` (lexical).
- `mmr` — diversity re-selection (greedy):

```
pick d* = argmax_{d∉S} [ λ·rel(d) − (1−λ)·max_{d'∈S} sim(d, d') ]   # λ = mmr_lambda
```

- `threshold` — drop results below a similarity cutoff.

Each holds per-instance config (`rerank`: model/mode/top_k; `mmr`/`threshold`: `k`) applied
transiently. They share a target depth via `_refine_inputs`; `rerank` keeps the larger
`fetch_k` pool when an `mmr` node follows. (The former single bundled rerank node was split
into this chain — see §9 catalogue.) The chain is **declared + reorderable** `[impl]` via
`vector_db.refine_ops` (an ordered list over `{rerank, mmr, threshold}`, repeats allowed for
cascades): unset reproduces the canonical rerank→mmr→threshold order byte-for-byte
(`pipeline/graph/assembly.py:_refine_chain`).

**Vector backends** (`storage/vector_store.py` + `storage/backends/`): `inmemory`, `faiss`,
`faiss_gpu`, `faiss_mmap` (off-RAM: mmap index + by-id Parquet payloads, `storage/payload_store.py`;
search byte-identical to `faiss`), `chromadb`, `qdrant` (`vector_db.type`).

**Query optimization** (`query_optimization` node): pure text→text `rewrite` / `hyde` before
embedding. The fan-out methods `decompose` / `multi_query` (expand → retrieve per sub-query →
merge via `combine_strategy`) live in the explicit `multi_query_retrieval` composite node —
their sub-query count is runtime-variable so they cannot be static DAG nodes.

**RAG answer generation** (`answer_gen` node, `evaluation/answer_gen.py`): answer from retrieved
context; methods `simple` / `chain_of_thought` / `multi_query`; `context_docs`,
`context_max_chars`, prompt templates.

**RAG-grounded entity correction** `[planned]` (C7). Beyond the rule / KB-fuzzy / LLM correctors
(§4, C1), the medical `query_correction` is grounded in a knowledge base by retrieval, with
**constrained decoding** — snap the ASR hypothesis to a *validated* entity, never free-generate
(free generation risks hallucinating a non-existent drug). Two ordered stages:
1. **Drug name.** A **phonetic** index (Soundex / Metaphone; ASR errors are phonetic, not semantic)
   retrieves nearest valid names from a drug registry; the corrector picks *among the retrieved
   candidates* — no out-of-registry invention.
2. **Dose.** For the resolved drug, retrieve its admissible strengths/forms; reject infeasible
   values (e.g. paracetamol `500 mcg` → `500 mg`), normalize units, range-check against the
   therapeutic / max-daily dose. Stage order matters: the admissible-dose set depends on the drug.

No confident candidate → the item is flagged for **human review**, not guessed (a clinical-safety
requirement). Retrieval spans **two sources** — the general drug registry and the per-patient
history (`patient_context`: prior meds + visits, §2/§3) — weighting the personalized source higher,
since a drug or dose from a prior visit strongly disambiguates an unclear hypothesis. Evaluated as a
branch beside the uncorrected and reference branches (§8), scored by **CEER** + **drug-dosage-safety**;
the safety metric penalizes errors the correction **introduces** (a correct entity flipped to a wrong
one) harder than ones it leaves, since a new error is clinically more dangerous than an existing one.

---

## 7. Metrics as nodes

A **metric node** consumes **N artifacts (ground-truth optional)** and emits scores. It is inert
w.r.t. the pipeline (read-only) and tags its output with **provenance** (which producer / branch
it scored). The narrow `(scored, gt) → metric` shape is a special case — many useful metrics are
**reference-free** (latency, retrieval-score distribution, embedding norms, answer length,
faithfulness/self-consistency), so the signature is `metric(*artifacts) → scores`.

**Per-item vs aggregate are separate.** A metric node emits an `item_scores` series (e.g. WER per
query); the `aggregate` node (§8) reduces it to a scalar. Cross-stage metrics (e.g. WER↔Recall
correlation) consume **other metrics' `item_scores`** — i.e. metric→metric edges — and are placed
after their inputs by the same topological wiring.

**Auto-injection rule.** A **metric-spec registry** declares, per metric, the artifact pattern it
needs: `register_metric(name, inputs=(scored_type, gt_type?), …)`. The builder scans the graph
and injects a metric node wherever its declared inputs are all present:

```
query_text(asr hypothesis) + reference_text  ⇒  wer / cer / ceer   [transcription_metrics]
retrieved        + relevant_docs   ⇒  recall@k, precision@k, mrr, ndcg, ap  [retrieval_metrics]
generated_answers + short_answers  ⇒  rougeL, llm_judge
retrieved        (no GT)           ⇒  score distribution, retrieval_failure_rate
item_scores(wer) + item_scores(recall5)  ⇒  wer↔recall correlation
```

The registry is what makes "auto-inject where appropriate" concrete — without it there is no way
to match scored artifacts to metrics (`evaluation/metric_registry.py`, `register_metric` +
`compute_metric(s)`).

- The report computes **every metric whose inputs are satisfiable** (collect-all is the
  effective behavior of both report paths); the dataset's `evaluation_mode` set
  (`METRICS_BY_MODE`) defines which of them surface as the flat headline aliases.

The registry-selected metrics run per branch (the aggregate scans the ctx per branch, computes
the metric ItemSets, and reduces). **The registry is the single scalar source**: the flat
headline keys (`WER`/`MRR`/`Recall@k`/…) are **report-derived aliases** (`_derive_bare_keys`;
WER/CER gated to ASR modes). Because `query_text` is the IMMUTABLE ASR hypothesis (correction/
optimization emit distinct names), `wer`/`cer` always measure ASR quality against
`reference_transcription` — the former `raw_wer`/`raw_cer` (a separate raw snapshot) are
subsumed. A "corrected WER" is now expressible by registering a metric that scores
`corrected_query_text`. The metric computation is decomposed into typed comparison nodes
(`transcription_metrics`, `retrieval_metrics`); the `metrics` node assembles the report from
them. CEER (critical drug/dose/unit entity error) superseded the term-weighted `TW_WER` as the
safety-relevant instrument. The
`report` is the canonical, complete source the leaderboard + UI read (§8, §13). Shared reduction
utilities (`_branch_scores` / `_run_provenance` / `_attach_report` / `_retrieval_wer_impact`,
`evaluation/handlers/metrics.py`) serve both the single-branch metrics node and the multi-branch
aggregate.

**Mode → default metrics** (`datasets/descriptor.py`):

```
transcription → wer, cer
retrieval     → mrr, ndcg, precision, recall
qa_retrieval  → wer, cer, mrr, ndcg, precision, recall
qa            → mrr, ndcg, llm_judge
ranking       → ndcg, precision
```

### Metric taxonomy + formulas

**Component (single-stage)** — score one node against its reference. `[impl]`
```
WER = (S + D + I) / N_ref           # word substitutions/deletions/insertions over ref length
CER = same over characters          # scored on query_text (the IMMUTABLE ASR hypothesis)
# a "corrected WER" is opt-in: register a metric scoring corrected_query_text vs reference
CEER = critical-entity (drug/dose/unit) error rate   # metrics/clinical.py
embedding_alignment = mean cosine(audio_emb, text_emb)   # fused mode; computed by the fusion node
```

**Retrieval (ranking)** — score `retrieved` vs `relevant_docs`. `[impl]` (`metrics/ir.py`)
```
Recall@k    = |retrieved_k ∩ rel| / |rel|
Precision@k = |retrieved_k ∩ rel| / k
MRR         = mean_i ( 1 / rank_i(first relevant) )
DCG@k       = Σ_{j=1..k} (2^{grade_j} − 1) / log₂(j + 1)
NDCG@k      = DCG@k / IDCG@k
AP          = mean of Precision@j at each relevant rank j
```
Aggregated at k ∈ {1,5,10} (`compute_ir_metrics`, `metrics/ir_aggregate.py`).

**Cross-stage (whole-pipeline)** — consume other metrics / multiple artifacts. `[impl]`
(`metrics/diagnostics.py`, `analysis/`)
```
wer_recall_correlation        = Pearson r over per-query (werᵢ, recall5ᵢ)
first_relevant_rank_distribution + retrieval_failure_rate
categorize_failures           # ASR-caused vs embedding vs not-in-corpus
per_speaker_breakdown         # metrics sliced by speaker (accent robustness)
bootstrap CIs on MRR/Recall/NDCG (analysis/significance.py)
```

**RAG / judge.** `[impl]` answer-gen ROUGE-L; **LLM judge** (`judge/`, modes
`retrieval`/`answer_quality`/`both`); `judge_calibration` correlates judge score with IR.

**Medical/voice metrics.** `[impl]`: **Retrieval-WER-Impact** (`Recall(ref) − Recall(branch)`, in
the aggregate report, via multi-branch §8), **CEER** (critical-entity error on drug/dose/unit
terms, `metrics/clinical.py`), per-branch WER/CER + wer↔recall correlation in the report;
**RAG-generation metrics** (`metrics/rag.py`) — **drug-dosage-safety** + **context-recall**
(heuristic, surfaced in `answer_gen` output) and **faithfulness / answer-relevance / factual**
(LLM-judged, injected client).

---

## 8. Aggregation & reporting

A terminal **`aggregate` / `report` node** consumes *all* `metric` artifacts in the graph and
produces the final result object. `[impl]` (`_stage_aggregate` + `aggregate.py:build_report`).

- **Within a run**: collect every `metric` (whatever ran) → one result; persist + ingest into the
  leaderboard (`leaderboard.sqlite`, `[impl]`).
- **Across branches**: when the graph fans out into parallel branches that share a source — e.g.
  `ref_text → retrieval`, `audio → asr → retrieval`, `audio → asr → query_correction → retrieval`
  — each branch ends in its own metric nodes; the aggregator computes **delta metrics** across
  branches (ΔRecall@k, Retrieval-WER-Impact, ΔSafety@k). This is the experiment harness for the
  thesis hypothesis (`configs/e2e_pubmed_qa_3branch.yaml`, real ref/asr/corr report with
  `asr_vs_ref`/`corr_vs_ref` deltas validated on amazing_curie).
- **Across runs**: per-run aggregates roll up into the leaderboard / comparison surface
  (`/api/leaderboard`). `[impl]` for storage. Multi-run promotion `[impl]`: declarative
  **sweeps** (`analysis/sweep.py`, `evaluator sweep` — a base config × axes expands via
  `GridSearch` to a tagged run-group); the leaderboard carries `experiment_group`/`tags` with a
  group filter so a sweep's runs query/pivot as one experiment; and the offline `compare`
  (`analysis/significance.py:compare_experiments`) applies **BH-FDR** across the metric panel +
  flags **under-powered** comparisons (n < 20). A **cross-run Pareto frontier** over a run group
  `[impl]` surfaces the non-dominated trade-off set across objectives (`analysis/pareto.py`
  multi-objective non-domination; `ExperimentStore.group_runs`; `GET /api/leaderboard/pareto` +
  a server-rendered `/ui/pareto` scatter/table; objectives like `MRR:max,latency_ms:min`). A
  sweep-submit form + report export to **MLflow/W&B** (`evaluator export -f mlflow|wandb`,
  `analysis/tracking_export.py`) round out the cross-run surface.

### Statistical rigor on the deltas

The cross-branch comparison is the thesis claim, so the deltas are *statistically honest*, not bare
means (`aggregate.py`, `analysis/significance.py`):
- **Honest paired denominators** — `paired_delta` reports `n_paired` / `n_branch` / `n_baseline` /
  `n_only_branch` / `n_only_baseline` + `denominator_policy`, so a shrinking, asymmetric paired
  sample can't hide behind a silent intersection.
- **Significance per delta** — a seeded bootstrap CI on the mean delta, a Wilcoxon signed-rank
  p-value, paired **Cohen's d**, and a **Benjamini–Hochberg FDR** q-value across the whole family of
  delta tests (so a panel of ΔRecall/ΔWER comparisons can't accumulate false positives).
- **Reproducibility** — RNGs are seeded globally + per item at run start (`set_global_determinism`
  in `_run_core`; `item_seed(seed, query_id, node_id, variant)`), and the actual determinism state
  (seed, `PYTHONHASHSEED`, cuDNN / deterministic-algorithm flags — recorded honestly, GPU caveats
  and all) lands in `report.provenance.determinism`. Same config + seed → identical metrics.

```
            ┌─ branch A (ref)        → ir_metric ─┐
dataset_src ┼─ branch B (asr)        → ir_metric ─┼─▶ aggregate ─▶ report (+ deltas) ─▶ leaderboard
            └─ branch C (asr+correct)→ ir_metric ─┘
```

### In-graph branching: template + auto-CSE

Hand-duplicating asr/embed/retrieval/metric nodes per branch is verbose and **recomputes shared
prefixes** (one ASR pass per branch). Instead, branches are authored once and expanded at
graph-build time; the executor stays a plain topological DAG (no new runtime).

1. **Author** a base sub-graph + a `branches:` variant set — each variant overrides only what
   differs:
   ```yaml
   branches:
     - { id: ref,  query_text: reference }          # oracle: use reference text
     - { id: asr,  asr: whisper }                    # transcribe
     - { id: corr, asr: whisper, query_correction: rules }   # transcribe + correct
   ```
2. **Expand** — the builder emits namespaced node copies per branch (`asr@asr`, `asr@corr`, …).
3. **Auto-CSE (common-subexpression elimination)** — nodes with identical
   `(type, params, resolved input producers)` collapse to one. So `dataset_source → asr@asr` and
   the `asr` inside `corr` are the *same* node (identical params + inputs) and **run once**;
   divergence begins exactly at the first differing node (`query_correction` in `corr`, the
   reference substitution in `ref`). The shared prefix is computed once by construction.

   **The identity key is the correctness-critical part.** Two nodes collapse iff a *canonical*
   key matches: `key = hash(type, normalize(params), sorted(input bindings (artifact, producer_id)))`.
   `normalize(params)` must **resolve defaults and impose a canonical key order** before hashing —
   otherwise `{asr: whisper}` and `{asr: whisper, size: <default>}` hash differently and the
   shared ASR silently **runs twice** (lost reuse); conversely a too-loose key could **over-share**
   nodes that should differ (wrong results). CSE is applied bottom-up so that input-producer
   identity is itself already canonicalized when a node's key is computed
   (`stage_graph.py:collapse_common_subexpressions`; `_freeze_params` resolves a node's declared
   `param_defaults` before hashing, so explicit-vs-default twins collapse).
4. **Provenance** — every node (hence every terminal `metric`) carries its branch label from the
   namespace, so `aggregate` knows which series belongs to which branch.

**Delta join (where keyed items §3 × branching meet).** `aggregate` groups each metric's `item_scores` by
`query_id`, pivots by branch label, and computes **paired** deltas
`ΔRecall@k(q) = Recall_corr(q) − Recall_asr(q)`, then reduces (mean + bootstrap CI). Pairing is
only sound because every branch carries the same `query_id` (the `ItemSet`, §3).

**Results ownership.** `aggregate` owns computation and emits `report`; terminal
`leaderboard_sink` / `tracking_sink` nodes consume `report` and persist it (mirrors `dataset_sink`),
so the DAG stays the single source of truth and I/O is an explicit node (`evaluation/sinks.py`;
opt-in, not in the default modes).

**Provenance (reproducibility).** `report.provenance` is a first-class part of the result — an eval
framework's job is comparable numbers. It records (`evaluation/provenance.py:build_provenance`):
`config_hash`, resolved model identities + versions, the run `seed`, library/runtime versions (same
data as the `RUNTIME` log line), per-node `timing`, the `git_commit`, and — added since — the
`determinism` block (§8 statistical rigor), the `cache` hit/miss counts per stage (§14),
`dropped_by_node` / `dropped_by_branch` (which items each node/branch dropped, §14), and the
LLM `cost` block (tokens/latency per component, §14). Combined with the deterministic per-item
seeds, a rerun reproduces the same numbers.

Two reproducibility fields are **content-addressed, not just config-addressed** (R1): `dataset` is
a content fingerprint (`{corpus_docs, corpus_sha256, questions}` from `dataset_content_fingerprint`)
so a reader can tell whether two runs evaluated *the same data* — "the numbers moved" becomes "the
corpus changed" (or not), independent of the config hash; and `failure_analysis` (present only when
items were dropped) gives per-item attribution — total dropped, per-node counts, top error types,
examples — so a shrinking sample is explainable, not opaque (§14, R7). Alongside the report, the CLI
writes a `…config_resolved.yaml` sidecar — the **executed DAG** as node-centric YAML
(`graph: {nodes, edges}` with per-node resolved model params, no template/`pipeline_mode`), so it
round-trips and can never list a model for a node the run never had — and the HuggingFace loader
accepts `repo@revision` to pin an exact dataset snapshot. Determinism stays
on-by-default (`set_global_determinism`, opt-out `EVALUATOR_NONDETERMINISM=1`).

---

## 9. Graph construction & execution

Two registries: **node type** — `register_stage_node(stage, category=, domain=, model_field=,
inputs=, outputs=, optional_inputs=)` (`pipeline/graph/`, the package re-exported via
`pipeline/stage_graph.py`) declares the taxonomy class (§ Node taxonomy) + data contract;
**handler** — `@register_stage_handler("name")`
(`evaluation/stage_registry.py`; the handler functions live in `evaluation/handlers/`, one module
per stage family) is the executable. A pre-flight check (`validate_graph_handlers`, run before
dispatch) fails a typo'd/unregistered node type before any heavy work.

**Formal model.** A graph `G = (V, E)`. Each node `v` has type `τ(v)`, params `θ(v)`, effective
outputs `Out*(v)` (role-scoped for `dataset_source`). For each input artifact `x` of `v`, the
binding `β(v, x)` is the ordered list of earlier producers `p` with `x ∈ Out*(p)`. The blackboard
`B` is a partial map `(id, name) ↦ value`. `read(v, x)` returns `B[(p*, x)]` for the newest
`p ∈ β(v,x)` that has published `x`.

**Build** (`build_graph_from_spec`): auto-wire each node to producers of its required + present-
optional inputs (`edges` add ordering not implied by data); `validate_graph_artifacts` checks
every required input is satisfiable in topological order; `topological_levels()` yields
deterministic levels and detects cycles. `build_graph_for_config` uses an explicit `graph.nodes`
override, else expands the config's **graph template** (`graph_override['template']`, set from
`graph.mode`) with its feature flags. The run uses the sibling `build_run_graph` (same assembly
tail, `_wire_mode_graph`), which sources its feature flags from the *built pipelines* + run
features rather than the config. There is no `pipeline_mode` field — the graph is the spec.

**Execute** (`run_from_bundle` → `run_graph`, `evaluation/executor/`). `run_graph(dataset,
context, *, service_provider, offload_policy, eval_config, load_info, graph_override)` takes an
`EvaluationContext` (`executor/state.py`) as its single contract for the pipelines + execution
params (cache, k, batch_size, trace_limit, checkpoints, `features`); `run_from_bundle` builds it
from the `PipelineBundle` + config:
```
B ← ∅
for level in topological_levels(G):          # nodes in a level are independent
    for v in level:
        set current_node = v
        inputs  = { x: read(v, x) for x in In(τ(v)) }
        outputs = handler[τ(v)](inputs, θ(v))
        for (name, value) in outputs: B[(v.id, name)] = value
        release v's model if no later node needs it      # offload policy
```

**Partial failure & cardinality.** The clean model above hides two real concerns: (a) a node may
fail for *some items*, so per-item artifacts can be sparse and downstream nodes + metrics must
tolerate missing items (this is why `query_id`-keyed artifacts beat positional lists, §3); (b)
some nodes change cardinality (augmentation N-variants, `multi_query` expansion), so "per query
`i`" is not invariant across the graph. Both are handled by the `ItemSet` (§3): keyed,
sparse-tolerant, lineage-carrying.

**Windowed streaming execution** `[impl]` (opt-in `streaming.window_size`, `executor/streaming.py`).
For corpus-scale runs, a windowed driver replaces the whole-dataset pass: the **prelude** (source +
corpus embed/index) runs once and is shared; the **query producers** (asr/embed/retrieve/refine) run
per query window; only the finalize-bound per-item bus slots (`query_text`, `retrieved`) accumulate
across windows while the heavy query vectors are released each window (bounded RAM); the **finalize**
nodes (per-item metrics + report + bootstrap CIs) run once over the full accumulated set. The phase
split is taxonomy-driven (`partition_phases`); accumulation preserves dataset order, so the report —
metrics and rankings — is byte-identical to a whole run (container-gated on real models; raw scores
differ only ≤2e-7 from per-window embedding batch-shape). Checkpoint/resume is window-granular
(`_setup_window_journal`: a crashed run resumes at the first incomplete window). **CPU-stage
parallelism** `[impl]`: a `cpu_stage_executor` knob (sync/thread/process, `executor/cpu_parallel.py`)
runs a stage's per-item map over an order-preserving, determinism-neutral `parallel_map` — wired
into the WER/CER fold so far; default `sync` is the serial path.

**Node taxonomy** (`pipeline/graph/taxonomy.py`). Every node declares two orthogonal axes at
registration (both required + validated): **category** — the data-flow *role*
(`source` / `model` / `transform` / `metric` / `sink`), load-bearing for lifecycle, validation
and execution (`stages_in_category("model")` is the device/offload-managed set); and **domain** —
the functional *area* (`ingest` / `query` / `transcription` / `embedding` / `retrieval` /
`refine` / `fusion` / `generation` / `robustness` / `scoring` / `reporting` / `export`), used for
UI grouping + DAG coloring + docs. The catalogue below is **drift-guarded** against the registry
by `tests/test_node_catalogue_doc.py`.

**Operator vocabulary (the registered node types).** The DAG is built from **11 generic
operators** (`pipeline/graph/operators.py`, `DAG_OPERATOR_ABSTRACTION.md`); a **field**
selects the specific behavior, so the registry holds one node type per op-shape instead of one
per parameter combination. The classic node names (`asr`, `text_embedding`, `rerank`, …) are
**load-time aliases** (`ALIASES`): authoring `{type: corpus_embedding}` or `nodes.rerank:`
expands at the wiring chokepoint (`wiring._normalize_spec_item`) to `(operator, fields)` while
**keeping the node id** = the legacy name (so keyed artifacts + branch ids are stable). Ports /
category / domain / model_field are **field-resolved** (`registry._resolve` over a per-instance
`params`); `node_kind(operator, params)` is the reverse, used wherever runtime still keys on a
concrete kind (device map, offload, V[s] validation, handler dispatch). The catalogue below is
**drift-guarded** against the registry by `tests/test_node_catalogue_doc.py`.

<!-- node-catalogue:begin (keep operator names in sync with pipeline/graph/operators_catalog.py — tests/test_node_catalogue_doc.py) -->

| Operator | Category | Selecting fields | Collapses (legacy nodes) |
|----------|----------|------------------|--------------------------|
| `source` | source | `union`, `role`, `dataset`, `fields` | dataset_source, dataset_union |
| `convert` | model / transform* | `op` (asr/tts) | asr, tts |
| `transform` | transform | `op` (correct/optimize/refine/perturb), `axis`, `modality` | query_correction, query_optimization, query_refine, augmenter, augment_audio |
| `embed` | model | `axis` (query/corpus), `modality` (text/audio) | text_embedding, audio_embedding, corpus_embedding |
| `combine` | transform | `level` (embedding/result/set) | fusion, result_fusion, corpus_merge |
| `index` | transform | `store` | vector_db |
| `search` | transform | `mode` (dense/sparse/hybrid), `method` (fan-out) | retrieval, multi_query_retrieval |
| `refine` | transform | `op` (rerank/mmr/threshold) | rerank, mmr, threshold |
| `generate` | transform | prompt/method (context optional → closed-book) | answer_gen |
| `measure` | metric / transform* | `family` (transcription/retrieval/answer/alignment/judge/report), `trace` | transcription_metrics, retrieval_metrics, answer_metrics, embedding_alignment_metrics, metrics, answer_judge, build_query_traces |
| `sink` | sink | `target` (finalize/aggregate/dataset/leaderboard/tracking) | finalize, aggregate, dataset_sink, leaderboard_sink, tracking_sink |

<!-- node-catalogue:end -->

`*` `convert` and `measure` carry a **callable category** (convert: model for asr, transform for
tts; measure: metric, or transform for the `trace` builder). Each operator's per-field artifact
contract (the exact consumes → produces) is declared per operator in
`pipeline/graph/operators_catalog.py` (the `register_stage_node` blocks) and field-resolved by
`pipeline/graph/registry.py`; the friendly per-instance label (e.g. `embed{axis:corpus}` →
"Corpus embedding") comes from `pipeline/graph/display.py:display_label`.

> **Graph package layout.** `pipeline/graph/registry.py` holds the registration machinery +
> validation (`StageNode`, `StageNodeDef`, `register_stage_node`, `_resolve`, `validate_graph_artifacts`);
> the **artifact-name vocabulary** (`ARTIFACT_*`, `OneOf`, `SOURCE_ARTIFACTS`, …) lives in
> `pipeline/graph/artifacts.py`; the **operator catalogue** (the per-operator helper fns + the
> `register_stage_node(...)` calls) lives in `pipeline/graph/operators_catalog.py`, imported by
> `registry.py` to populate the node registry. `registry.py` re-exports the vocabulary, so
> `from …graph.registry import ARTIFACT_CORPUS` (and the package `__init__` surface) is unchanged.

**Builder-settable params (two sources, by design).** The pipeline builder keeps node
forms minimal: a registry-backed node shows only its **model select** (+ `device`) — every
model-specific field appears *after* a model is chosen and is **declared by the model
author** on the model's `Params` dataclass (defaults, `SIZES` → the size dropdown — e.g.
whisper tiny/base/small/medium/large-v2/large-v3 — and `CHOICES` for enumerated values,
§11). Genuinely *node-scoped* switches are declared in the node registry
(`register_stage_node(param_spec=…)`) and rendered as typed widgets:

| Node | Node-scoped switches (`param_spec`) |
|------|------------------------------------|
| `dataset_source` | `dataset` (picker: registered dataset id, or a datasets-map id), `role` (both/questions/corpus), + the picked dataset's required settings |
| `asr` | `oracle` (bool — this branch uses reference transcriptions, R2) |
| `retrieval` | `k`, `mode` (dense/sparse/hybrid), `distance`, `gpu_id`, `vectors` (pin one stream — result-fusion / per-hop) |
| `vector_db` | `store` (inmemory/faiss/faiss_gpu/chromadb/qdrant) + conditional backend fields via `show_if` — `gpu_id` (faiss_gpu), `path` (chromadb), `url`/`collection` (qdrant/chromadb) |
| `rerank` | `mode` (none/token_overlap/cross_encoder), `k`, `top_k`; model picker from the Reranker registry |
| `mmr` | `k` (diversity re-selection; `mmr_lambda` from the global config) |
| `threshold` | `k` (similarity-cutoff filter) |
| `query_correction` | `enabled`, `method` (rule/kb/llm), `use_default_rules`, `kb_max_distance`, `kb_terms` (json), `replacements` (json) |
| `augmenter` | `homophones`, `unit_corruption` (bools), `char_swap_prob`, `max_edits` |
| `query_optimization` | `method` (rewrite/hyde), `temperature`, `max_iterations` — transient overlay on the global config (branch A rewrite vs branch B HyDE) |
| `query_refine` | `method` (rewrite_with_context/relevance_feedback/self_rag_critique), `context_top_k` |
| `multi_query_retrieval` | `method` (multi_query/decompose), `combine_strategy` (rrf/weighted/union/intersection), `k` |

The catalogue endpoint (`/api/graph/nodes`) serializes both sources as `node_params`;
the advanced node-centric YAML keys (`name`/`adapter`/`dim`/…) remain settable via the
builder's free-form "+ add param" row and the YAML config (§10) — they are intentionally
not default form fields.

**Designed additions** `[planned]`: `image_embedding` (image modality).

**Graph templates** (the former pipeline *modes*) are assembled declaratively
(`pipeline/graph/assembly.py:assemble_specs`, driven by a `FeatureSet` dataclass of ~14 capability
flags derived from the config — no hardcoded per-template node list). A template is a
config-creation **skeleton**, not a runtime field: a config selects one via `graph.mode` (a
build-time `graph_override['template']` reference; `pipeline/graph/templates.py` also serves them to
the web canvas and can emit an embeddable `{nodes, edges}` block via `template_graph_spec`) or
carries an explicit `graph.nodes`. The executed graph then
drives both building (`factory._graph_build_plan`) and handler behaviour — there is no
`pipeline_mode`; `GraphTemplateSpec`/`resolve_graph_template` only validate a template name + its
required model fields. The named templates are `asr_only`, `asr_text_retrieval`,
`audio_emb_retrieval`, `audio_text_retrieval`; feature flags slot in the optional nodes:
`tts`, `augment_audio`, `query_correction`, `query_optimization`, `query_refine`,
`multi_query_retrieval`, `rerank`/`mmr`/`threshold`, `fusion`/`result_fusion`,
`embedding_alignment_metrics`, `answer_gen`/`answer_metrics`, `build_query_traces`,
`answer_judge`, `dataset_sink`. Preview: `--print_graph`, `/api/graph/preview`, `/ui/graph`.

**Composable / iterative RAG.** A RAG flow is a sequence of pure nodes — improve the query →
embed → retrieve → reformulate on the retrieved docs → retrieve again. The loop is a strict
DAG, so it is **unrolled** into distinct instances rather than cycled: `config.rag.rounds=N`
(`RagFlowConfig`) emits `text_embedding@h1, retrieval@h1, query_refine@h1, text_embedding@h2,
…, retrieval@hN`. No per-hop artifact names are needed — `_wire_nodes` binds each instance
only to **prior** producers and the `one_of` chains read the highest-priority *published*
variant, so hop `k`'s embedder automatically picks `refine@h(k-1)`'s `refined_query_text` and
hop `k`'s retrieval picks its own hop's vectors. `rounds=1` is byte-identical to a single
pass. The RAG-fusion fan-out (`decompose`/`multi_query`) — whose sub-query count is
runtime-variable and so cannot be static nodes — is the one explicit composite,
`multi_query_retrieval` (`query_text → retrieved`); it replaces the old in-handler
`query_opt_bypassed` short-circuit.

---

## 10. Configuration

The full `EvaluationConfig` (dataclass tree in `config/`) is the experiment. Node-centric YAML
loads through `config/graph_config.py:to_legacy_dict` — the single chokepoint (CLI/API/presets/
webapi).

```yaml
experiment: { name: whisper_jina, output_dir: evaluation_results }
datasets:                                   # multi-source; or single `dataset:` block
  docs: { corpus: corpus.json, role: corpus }
  qa:   { questions: q.json, role: questions }
graph:
  mode: asr_text_retrieval                  # a graph template; or explicit { nodes:[…], edges:[…] }
nodes:
  asr:            { model: whisper, size: large }
  text_embedding: { model: labse }
  vector_db:      { store: faiss }              # canonical store home
  retrieval:      { k: 5, mode: hybrid, reranker: { enabled: true } }
  answer_gen:     { enabled: true }
branches:                                   # expand+CSE (§8); each overrides only diffs
  - { id: ref,  query_text: reference }
  - { id: asr,  asr: whisper }
  - { id: corr, asr: whisper, query_correction: rules }
runtime: { cache: { enabled: true }, tracking: { backend: mlflow }, parallel_enabled: true }
```

Config surface `[impl]`: `augmenter`, `query_correction`, and the `aggregate` node are expressed as
graph/branch nodes with `params`;
`parallel_enabled` turns on intra-level branch concurrency (§14). **Unknown/misspelled keys are
rejected** with a path-named `ConfigurationError` before any heavy work (§14). Sub-configs
(`config/*.py`): `data`, `model`, `vector_db`, `cache`, `logging`, `tracking`, `service_runtime`,
`llm`(+`llm_server`), `dataset_sink`, and features (`judge`, `answer_generation`,
`query_optimization`, `query_correction`, `embedding_fusion`, `audio_synthesis`,
`audio_augmentation`, `device_pool`).

---

## 11. Models & registries

Models register via decorators into five `ModelRegistry` instances (`models/registry.py`);
selection is by config string. **Third-party plugins** `[impl]`: each registry's first-lookup hook
also runs `importlib.metadata` entry-point discovery (`evaluator/plugins.py`, groups
`evaluator.models` / `nodes` / `handlers` / `metrics` / `datasets`), so an external package registers
a model/node/metric/dataset without editing core (no-op when none are installed).

| Registry | Decorator | Registered types |
|----------|-----------|------------------|
| ASR | `@register_asr_model` | `whisper`, `faster_whisper`, `wav2vec2`, `seamless_m4t` |
| TextEmbedding | `@register_text_embedding_model` | `labse`, `jina_v4`, `bge_m3`, `clip`, `nemotron`, `sonar` |
| AudioEmbedding | `@register_audio_embedding_model` | `attention_pool`, `attention_pool_m4t`, `clap_style`, `hubert`, `wavlm`, `sonar_speech` |
| Reranker | `@register_reranker_model` | `cross_encoder` |
| TTS | `@register_tts_model` | `piper`, `xtts_v2`, `m4t`/`m4t_v2`, `mms`, `seamless_m4t` |

**Model-author param declaration.** A model's inner `Params` dataclass is the single
source of its tunable surface: field defaults, `SIZES` (size name → checkpoint; becomes
the builder's size dropdown), and an optional `CHOICES: ClassVar[Dict[str, List]]`
enumerating valid values per field (e.g. hubert `{"pooling": ["mean", "cls"]}`).
`registry.get_params_schema` serializes this for the UI — whoever adds a model decides
which parameters appear and what values they offer (§9 builder-settable params). Two opt-in
load-time knobs flow to the model loaders: a **quantization** field (`config/model.py`, global +
per-family) folded into a model's `__init__` via `**extra_params` when it accepts it (else a
full-precision warning) `[impl]` (1a), and **warm-up batch sizing** that estimates the ASR batch
from a one-sample GPU memory delta instead of the static `batch_size` (`devices/memory.py:
warm_up_batch_size`, opt-in, no-op on CPU) `[impl]` (1b).

The `query_correction` correctors live in a **Corrector registry**
(`evaluation/query_correction.py`: `@register_corrector("rule"|"kb"|"llm")`, a corrector is
`(texts, config, client?) → corrected texts`); the node's `method` choices and
`QueryCorrectionConfig` validation both read the registry, so a custom corrector is valid
config + a builder choice the moment it registers — no core edit (C7 groundwork). An **ImageEmbedding registry** (`I→V`) is `[planned]` (image modality).
LLM nodes call an OpenAI-compatible endpoint or a local LLM server (`models/llm`,
`config/llm_backend.py`, `config/llm_server.py`); per-component token/latency cost is metered (§14).

---

## 12. Cross-cutting: caching, lifecycle, web

- **Caching / storage** (`storage/`): SQLite manifest (`cache_manifest.sqlite`) + disk artifacts;
  content-keyed categories (`asr_features`, `transcriptions`, text/audio `embeddings`, vector DB,
  `synthesized_audio`); `--no-cache`/`--clear-cache` honored. Vector-DB cache rebuilds the index
  from cached corpus embeddings (`build_corpus_index`). Leaderboard `leaderboard.sqlite`. Keys now
  fold the **model version** and paths are stored **cache-dir-relative** (portable across
  machines/containers); hit/miss is reported in `report.provenance.cache` (§14). The manifest
  self-compacts at startup (rows whose artifact file is gone are dropped — bounded even with the
  size limit disabled); `CACHE_SCHEMA_VERSION` carries a documented bump policy.
- **Service & devices** (`services/`, `devices/`): `ModelServiceProvider` loads/moves/releases
  models + local LLM servers; `service_runtime.startup_mode` (lazy|eager),
  `offload_policy` (on_finish | never | **`on_finish_soft_cpu`** `[impl]` — instead of freeing a
  model after its last use, park it warm on host RAM in a bounded LRU+TTL pool so a later
  stage/run re-acquires it with a CPU→device move, not a full reload; offload events recorded in
  `report.provenance.offload`). `devices/capability.py:usable_gpu_indices` filters CUDA
  devices whose arch is absent from `torch.cuda.get_arch_list()` (AMD iGPUs / unbuilt `sm_XX`).
  TTS runs + offloads before embedders load (no co-resident native runtimes).
- **WebAPI & UI** (`webapi/`): FastAPI + routers; server-rendered Jinja2 + htmx at `/ui`
  (Plotly from CDN). Endpoints: config/options/schema/presets, `/api/models…`, `/api/datasets…`,
  `/api/graph/preview`, jobs (subprocess-run), `/api/leaderboard`, `/api/tts/preview`. Visual
  builder `/ui/builder` (Drawflow, no build step) + `/api/graph/nodes` (catalogue) +
  `/api/graph/build` (validate canvas → levels). The builder is **registry-driven end to
  end**: node ports are labeled with their artifact names (optional inputs marked), and each
  node's param form offers its model choices from `/api/models` plus that model's `Params`
  schema (sizes/choices/defaults via `/api/models/{family}/{type}/params`) — no model list or
  param map is hardcoded in the UI. What each node lets the user set is specified in §9
  ("Builder-settable params"): minimal node forms, model fields only after model selection
  (model-author-declared, §11), node switches from `register_stage_node(param_spec=…)`.

---

## 13. Extension points & status summary

**Add a…** (all discovery via explicit registries — no import scanning):

| Add | Do |
|-----|----|
| Dataset | subclass an ABC in `datasets/types.py`, implement `from_config` (+ `__len__`/`__getitem__`/`get_corpus`), `@register_eval_dataset(id=…)`; `supports_generation=True` unlocks audio via TTS |
| Model | impl under `models/<family>/`, decorate with the registry decorator |
| Node type | `register_stage_node(...)` (contract, `pipeline/graph/operators_catalog.py`) + `@register_stage_handler(...)` (executable, `evaluation/handlers/`). The two are cross-checked by `stage_registry.validate_node_handler_consistency` (a drift-guard test fails if one is added without the other) |
| Metric | `register_metric(name, inputs=…)` (GT optional): `metric(*artifacts) → item_scores`; declared inputs drive auto-injection; `aggregate` reduces |
| Augmentation | a type-preserving node `X → X` |
| Corrector | `@register_corrector("name")` on `(texts, config, client?) → texts` in `evaluation/query_correction.py`; select via the node's `method` param |
| Retrieval / fusion / backend | register in the corresponding registry |

**Source map** (where each concern lives):

| Concern | File |
|---------|------|
| Node types + sockets, artifact auto-wiring (`build_graph_from_spec`), validation | `pipeline/graph/` (re-exported via `pipeline/stage_graph.py`) |
| Node taxonomy (category/domain axes + validation) | `pipeline/graph/taxonomy.py` |
| Declarative graph-template assembly (`FeatureSet` → node spec list) | `pipeline/graph/assembly.py` |
| Graph templates (former modes) + spec validation | `pipeline/graph/templates.py`, `graph/modes.py` |
| Handler registry (executable node functions + timing) | `evaluation/stage_registry.py` |
| Run state + executor (engine, views, parallel, offload, run entry points) | `evaluation/executor/` |
| Stage handlers (one module per stage family) + shared keying helper (`_common.py:publish_keyed_or_plain`) + retrieval debug logger (`retrieval_debug.py`) | `evaluation/handlers/` |
| Keyed cross-node artifact bus | `evaluation/run_context.py` (`RunContext`) + `item_set.py` (`ItemSet`) |
| Metric-spec registry + aggregate/report reducer | `evaluation/metric_registry.py`, `evaluation/aggregate.py` |
| Shared metric-reduction utilities (both report paths) | `evaluation/handlers/metrics.py` (`_branch_scores`, `_run_provenance`, `_attach_report`, `_retrieval_wer_impact`) |
| Per-branch scope markers (drives `_NodeView` isolation) | `evaluation/executor/state.py:per_branch_field_names` |
| Deterministic e2e parity-gate snapshot | `scripts/report_snapshot.py` |
| One execution core (validate → pipelines → evaluate; the dataset loads in-graph) | `services/evaluation_service.py:_run_core` |
| Pre-flight validation chain (determinism, cost budget, V[s] config+graph, store backends) | `evaluation/validation.py:run_pre_flight` |
| Audio bricks (tts, augment_audio) + the audio-ref bus | `evaluation/handlers/audio.py`, `evaluation/audio_refs.py` |
| Corpus embed (`corpus_embedding`) + index build (`vector_db`) | `services/corpus_index.py` (`embed_corpus`, `embed_corpus_audio`, `build_index_from_vectors`; `build_corpus_index` = back-compat wrapper) |
| Dataset descriptor / registry | `datasets/descriptor.py` |
| TTS synthesis helper | `pipeline/audio/prepare.py` |
| Graph preview API | `webapi/form_builder.py:graph_preview` |

**Open items** — everything else in this doc is `[impl]`:

| Item | Status |
|------|--------|
| Image modality: `image_embedding` node + ImageEmbedding registry + n-ary fusion | `[planned]` (revisited when a 3rd modality lands, §4/§6) |
| ~~T1: corpus as doc_id-keyed ItemSet~~ ~~T2: corpus-side transforms~~ | done (2026-06-12, §4.1) |
| ~~T3/P4: query-set union + audio-ref bus + augment_audio brick~~ | done (2026-06-12, §4.1) |
| ~~T4: precomputed-vector dataset columns~~ | done (2026-06-12, §4.1) |
| ~~Flip default modes to the corpus split~~ | done (2026-06-12); parity baselines in amazing_curie need one regeneration |
| RAG-grounded entity correction (phonetic drug-name → dose validation, `patient_context` retrieval, constrained decode) | `[planned]` (C7, §6) |
| ~~Corrector registry~~ | done (2026-06-12, §11) |
| ~~Dataset loading + TTS as graph nodes~~ — both **done**: TTS is the in-graph `tts` node, and dataset *loading* is now owned by the `dataset_source` node (`handlers/source.py:_ensure_dataset_loaded`; the run is exactly the DAG, no pre-graph load — only config validation stays pre-graph) | `[impl]` (2026-06-18, §2) |
| Pareto / visual cross-run leaderboard views | `[impl]` (2026-06-17, Roadmap 4a): `analysis/pareto.py` (multi-objective non-domination), `ExperimentStore.group_runs` / `experiment_groups`, `GET /api/leaderboard/pareto` (objectives like `MRR:max,latency_ms:min`), and a server-rendered `/ui/pareto` view (frontier scatter + tagged table, `templates/_pareto.html`) on the Results page. A sweep-submit form remains. |
| Streaming / out-of-core evaluation (windowed driver + off-RAM corpus/index) | 3a windowed query-side driver `[impl]` (2026-06-17, opt-in `streaming.window_size`): prelude (source + corpus index) runs once, the query producers (asr/embed/retrieve/refine) run per window, only the finalize-bound per-item slots (`query_text`, `retrieved`) accumulate while query vectors are released per window, and the metric/report/CI nodes run once over the full set. Mock equivalence test proves windowed == whole report (incl. bootstrap CIs) for window sizes 1–4; window-granular checkpoint/resume landed (`_setup_window_journal` — a crashed run resumes at the first incomplete window, re-running the cache-fast prelude and restoring the accumulator; a resume-after-crash test reproduces the full report). **Container m1c gate PASSED (2026-06-17, `amazing_curie`, real models):** on `e2e_pubmed_qa_small`, a windowed run (`--streaming_window_size 2`) reproduces the whole run's **metrics and rankings byte-for-byte** (MRR/MAP/Recall/NDCG/WER/CER + per-query doc order); the only divergence is ≤2.1e-7 in raw similarity scores, because the windowed run embeds fewer query texts per call (a different matmul shape → oneDNN float round-off — `data.batch_size` alone has 0.0 effect, confirming it's the per-window embedding-call size, not a logic difference). Intra-window parallelism remains. 3b off-RAM corpus/index: first increment `[impl]` (2026-06-17) — `vector_db.type: faiss_mmap` (`storage/vector_store.py:FaissMmapVectorStore`) memory-maps the FAISS index + fetches payloads by id from a Parquet store (`storage/payload_store.py:ParquetPayloadStore`, one row group resident), so neither the index nor the corpus is bounded by one box's RAM; search is byte-identical to `faiss` (container gate: faiss_mmap == faiss full report, sole diff the store-name echo). IVF on-disk indexes for huge corpora + a remote HTTP backend remain. |
| ~~Typed embedding spaces~~ (compatible-space registry + runtime guard, 2b, §4.1) · ~~composable retrieval operators~~ (`refine_ops`, 2a, §6) · ~~plugin entry-points~~ (`evaluator/plugins.py`, 1c, §11) | `[impl]` (2026-06-17) |
| ~~LRU+TTL model cache + soft-CPU offload~~ (`on_finish_soft_cpu`, 2c, §12) · ~~quantization knob~~ (1a, §11) · ~~warm-up batch sizing~~ (1b, §11) | `[impl]` (2026-06-17). Async CPU stages (4b, §9) `[impl]`: the order-preserving, determinism-neutral `parallel_map` primitive + `cpu_stage_executor` knob (sync/thread/process), now wired into the per-item WER/CER map (`handlers/metrics.py:_asr_item_scores`, the first GIL-bound stage) — a mock-equivalence test proves thread/process give a byte-identical report to the default sync, and the container gate confirms `--cpu_stage_executor process` == sync byte-for-byte (full report incl. scores + CIs) on the real e2e. Wiring the remaining stages (correction/augmentation) follows the same picklable-pure-fn pattern. |
| ~~MLflow/W&B export bridge~~ (`analysis/tracking_export.py`, 1d, §8) · ~~item replay by query id~~ (`evaluator replay`, 2d, §14) | `[impl]` (2026-06-17) |
| ~~APM weight loading + whiten/ABTT correctness~~ (`models/a2e/attention_pool.py`, `postprocessing.py`, §4.1) | `[impl]` (2026-06-17): the encoder is loaded from the checkpoint's `audio_enc.*` so it matches training (size mismatch → loud error), a missing pooling/projection weight raises instead of silently using random init, and ABTT is L2-normalized to match the training transform (whitening is not). Verified against `apm_new/apm` + an end-to-end load on a real whisper-large encoder in `amazing_curie`. |
| Per-node offload of branch models | `⊘ deferred` (memory opt) |

The completed task-trackers and audit docs that drove the `[impl]` work have been retired
(their history is in git).

---

## 14. Operability & system hardening

An eval framework's output must be *trustworthy, reproducible, and affordable to run*. Beyond the
statistical rigor on the deltas (§8), the following operability layer is in place:

| Concern | What | Where |
|---------|------|-------|
| **Per-item failure isolation + attribution** | One bad item does not abort the run — embed/retrieval drop the failing id (log + placeholder that keeps batch shape), the keyed report excludes it, `report.provenance.dropped_by_node` records which node dropped which ids, and `report.provenance.failure_analysis` (present only on drops) gives the *why* — per-node counts, top error types, examples (R7) | `evaluation/item_isolation.py` (`DropSink.failure_summary`, `isolate_batch`) |
| **Live, machine-readable progress** | Node-lifecycle events stream to a callback + JSONL (`EVALUATOR_PROGRESS_FILE`); the typed `ProgressEvent` dataclass is the stable contract external dashboards consume. Opt-in `EVALUATOR_DUMP_ARTIFACTS=node,…` dumps a node's published ItemSets to JSONL for mid-run inspection (R7). `evaluator replay --query-id q42` re-runs a single item through the full graph (corpus kept whole) with that dump hook + a printed per-node trace — per-item seeding makes the replay reproduce the original run | `evaluation/progress.py` (`ProgressEvent`, `ProgressSink`), `evaluation/artifact_dump.py`, `cli/replay.py` |
| **Tidy result export** | The nested report flattens to a stable metrics table (`branch,metric,mean,ci,n`) + per-query trace export (CSV/JSONL/Parquet); same shapes via CLI `evaluator export -f metrics-table\|traces` and `POST /api/report/metrics-table` (§6) | `analysis/report_export.py` |
| **Vector-index integrity** | A reloaded index whose payload sidecar count mismatches the vector count fails loudly at load (`_verify_payload_count`), and a stale per-hit index is dropped+logged at search time (the H3 guard) | `storage/vector_store.py` |
| **Config validation** | An unknown/misspelled key raises a path-named `ConfigurationError` (`model.asr_modal_type`) before any heavy work — no silent default → wrong experiment | `config/loading.py:_construct_subconfig`, `graph_config.to_legacy_dict` |
| **Cache correctness + portability** | Model **version** folded into the cache keys (stale weights under the same name invalidate); paths stored **cache-dir-relative** (shareable across machines/containers); per-stage **hit/miss** in `report.provenance.cache` | `storage/cache_keys.py`, `storage/cache/` (pkg), `utils/model_version.py` |
| **Intra-level parallel execution** | Independent same-level nodes (the ref/asr/corr branches) run concurrently when `parallel_enabled` — each on a private `_NodeView` whose isolation set derives from the RunState scope markers (§3), serialized per device (no single-GPU contention). The DAG executor is the single in-process parallel path; single-branch stays serial | `evaluation/executor/engine.py:_execute_stage_graph` + `executor/views.py:_NodeView` / `executor/parallel.py` |
| **Checkpoint / resume** | Crash at node 8/12 does not recompute everything — the resumable state (ctx bus + control attrs) is snapshotted at each **level** boundary (keyed by config+graph) and a matching rerun resumes at the first incomplete level | `evaluation/run_journal.py` |
| **OOM-resilient batching** | An oversized batch halves-and-retries on CUDA OOM instead of crashing (down to one item, then a real capacity error); `suggest_batch_size(free_gb, per_item_gb)` additionally *estimates* a safe batch from live free memory (a one-sample warm-up feeds `per_item_gb`) rather than discovering it by failing (§8) | `devices/memory.py:run_with_oom_backoff`, `suggest_batch_size` |
| **LLM cost** | Per-component token + latency accounting (judge / answer_gen / query_correction / query_optimization) in `report.provenance.cost`, with an optional `max_tokens_budget` that aborts a runaway sweep | `llm/cost.py` |
| **Live observability** | Node-granular `node_start` / `node_complete` events (with per-node duration) stream to the progress callback **and** an optional JSONL file (`EVALUATOR_PROGRESS_FILE`) any observer can tail | `evaluation/progress.py:ProgressSink` |

These compose cleanly: the executor gains parallel branches → checkpoint/resume at level boundaries
→ live per-node progress, with per-item drop-and-log and OOM backoff underneath — all surfaced,
where relevant, in the one auditable `report.provenance` (§8).

---

## 15. Authoring guide — add a model, add a dataset, build experiments

This section is the practical "how to extend it." It shows the three things a user does most:
register a **model**, register a **dataset**, and compose **experiments** out of nodes. Each
follows the same rule: declare yourself to a registry, decorate, and the framework discovers you
— no edit to a central dispatch.

### 15.1 Add a model

A model registers itself with one decorator and declares its tunables on an inner `Params`
dataclass. The four pickable families are `asr` / `text_embedding` / `audio_embedding` / `reranker`
(plus `tts`); each has a `register_<family>_model(model_type, default_name=None, **metadata)`
decorator (`models/registry.py:_make_register`). Implement the family's base-class contract:

| Family | Decorator | Base class (`models/base.py`) | Implement |
|--------|-----------|-------------------------------|-----------|
| ASR | `@register_asr_model` | `ASRModel` | `transcribe(audio, sampling_rates, language=None) → List[str]`, `name()`, `to(device)` |
| Text embedding | `@register_text_embedding_model` | `TextEmbeddingModel` | `encode(texts, show_progress=False) → np.ndarray`, `name()` |
| Audio embedding | `@register_audio_embedding_model` | `AudioEmbeddingModel` | `encode_audio(audio, sampling_rates) → np.ndarray`, `name()` |
| Reranker | `@register_reranker_model` | `BaseReranker` (`models/retrieval/rag/reranker.py`) | `rerank(query, documents, top_k=None) → List[(doc, score)]`, `name()` |
| TTS | `@register_tts_model` | `BaseTTSModel` (`models/tts/base_tts.py`) | `synthesize(text) → np.ndarray` (float32 mono) |

The **builder/UI gets its fields for free** from the `Params` dataclass (BUILDER_UX, §9): `SIZES`
(a `{label: hf_name}` map) becomes the **size dropdown**; `CHOICES` (a `{param: [values]}` map)
becomes per-param selects; every other `Params` field becomes a typed widget with its default. This
is the *model author declares their own params* contract — nothing about the model is hardcoded in
the evaluation loop or the form.

```python
# evaluator/models/t2e/myjina.py  — a new text embedder
from dataclasses import dataclass
from typing import ClassVar, Dict, List
import numpy as np, torch
from ..base import TextEmbeddingModel
from ..registry import register_text_embedding_model

@register_text_embedding_model(
    "myjina", default_name="myorg/myjina-v1",
    description="My Jina embeddings", embedding_space="myjina_space",
)
class MyJinaModel(TextEmbeddingModel):
    @dataclass
    class Params:
        size: str = "v1"
        pooling: str = "mean"
        SIZES: ClassVar[Dict[str, str]] = {"v1": "myorg/myjina-v1", "v1-large": "myorg/myjina-v1-large"}
        CHOICES: ClassVar[Dict[str, list]] = {"pooling": ["mean", "cls", "last"]}

    def __init__(self, model_name="myorg/myjina-v1", pooling="mean"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model_name, self.pooling = model_name, pooling

    def to(self, device): self.model.to(device); return self
    def encode(self, texts: List[str], show_progress=False, desc="Embedding") -> np.ndarray:
        return self.model.encode(texts)  # → (n, dim) float array
    def name(self) -> str: return f"MyJina[{self.model_name}/{self.pooling}]"
```

Two rules complete the wiring:
- **Make the module import.** Add `from .myjina import MyJinaModel` to the family `__init__.py`
  (e.g. `models/t2e/__init__.py`). The registry imports the family module lazily on first lookup
  (`FAMILY_REGISTRIES`), which runs the decorator — so the *only* edits are the new file + that one
  import line.
- **Embedding space (`V[s]`, §4.1).** A vector is comparable only to vectors from the same space.
  Declare `embedding_space="…"` in the decorator (default: `"<type>:<name>"`, so distinct models
  never collide); a cross-modal model (CLIP-style audio+text) shares one id so its query and corpus
  vectors validate against each other. The graph builder hard-fails a retrieval whose query and
  corpus spaces differ.

Use it from a config by naming the `model_type`: `nodes.text_embedding.model: myjina`,
`nodes.text_embedding.size: v1-large`, `nodes.text_embedding.pooling: cls`.

### 15.2 Add a dataset

Subclass the per-type ABC in `datasets/types.py` that matches the data, implement `from_config`,
and decorate with `@register_eval_dataset(id=…)` — the `DatasetDescriptor` (capabilities + loader,
§ Datasets) is **derived from the class**, so there is no second registration.

| Subclass (`datasets/types.py`) | For | Native modes |
|--------------------------------|-----|--------------|
| `AudioTranscriptionDataset` | spoken clips + transcripts | `asr_only`, `asr_text_retrieval` |
| `AudioRetrievalDataset` | spoken queries + a doc corpus | all four |
| `TextRetrievalDataset` | text queries + a doc corpus | `()` — text only until TTS unlocks audio |
| `MultimodalQADataset` | QA (question + corpus + answers) | `()` — unlock via TTS |

`from_config(cls, data: DataConfig) → QueryDataset` returns a dataset whose `__getitem__(i)` yields a
per-item dict (`audio_array`, `sampling_rate`, `transcription`/`question_text`, `question_id`,
`groundtruth_doc_ids`, `relevance_grades`, `language`, `metadata`) and whose `get_corpus()` returns
`[{doc_id, text, …}]`. The ready-made adapters `runtime.py:AudioSamplesQueryDataset` /
`LazyAudioQueryDataset` cover most loaders, so `from_config` is usually just "parse files → adapter."

**The TTS bridge — `supports_generation = True`.** Set this class attribute on a *text* dataset and
its `compatible_pipeline_modes()` gains the audio modes: the query text is **synthesized to audio at
run time** (the `tts` node / pre-graph synthesis), so a text-only corpus can be evaluated through the
full `asr → embed → retrieve` audio pipeline. This is what makes a text QA set a spoken-retrieval
benchmark without any audio assets.

```python
# evaluator/datasets/builtins/<your_dataset>.py  (one module per dataset; import it in builtins/__init__.py)
from pathlib import Path
from typing import List, TYPE_CHECKING
from .types import TextRetrievalDataset, register_eval_dataset
from .runtime import AudioSamplesQueryDataset, _load_corpus_entries
from .loaders.base import AudioSample
if TYPE_CHECKING:
    from ..config.data import DataConfig
    from .core import QueryDataset

@register_eval_dataset(id="my_text_retrieval", description="Text retrieval, voiced via TTS")
class MyTextRetrievalDataset(TextRetrievalDataset):
    supports_generation = True                       # ← unlock the audio modes
    required_data_fields = ("questions_path", "corpus_path")

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        import json
        questions = json.loads(Path(data.questions_path).read_text())
        samples = [
            AudioSample(audio_array=None, sampling_rate=16000,   # audio synthesized at run time
                        transcription=q["text"], sample_id=str(q["id"]), language="en", metadata={})
            for q in questions
        ]
        return AudioSamplesQueryDataset(
            samples, trace_limit=getattr(data, "trace_limit", 0),
            corpus_entries=_load_corpus_entries(data.corpus_path),
        )
```

Override `validate(cls, data) → List[str]` for custom pre-flight checks (the default checks
`required_data_fields`). The function form `register_dataset(DatasetDescriptor(...))`
(`datasets/descriptor.py`) stays available for built-ins/advanced cases that want to hand-build the
descriptor. Select it with `dataset.id: my_text_retrieval` (+ `questions` / `corpus` paths).

### 15.3 Build experiments by composing nodes

An experiment **is** its config: a node-centric YAML that mirrors the DAG (§10). The `to_legacy_dict`
loader (`config/graph_config.py`) is the one chokepoint that translates it. A run picks its node set
two ways — a **named template** (the common case) or an **explicit graph** (full control).

**(a) Named template + per-node settings.** `graph.mode` picks a template skeleton; `nodes.<type>`
blocks carry each node's model + params; optional features (correction, rerank, mmr, fusion,
answer_gen, judge, audio/corpus augmentation) slot in from config flags (`assembly.py:FeatureSet`,
§9). The real `configs/showcase_hybrid_rerank_mmr.yaml`:

```yaml
experiment: {name: showcase_hybrid_rerank_mmr, output_dir: evaluation_results/showcase}
dataset:  {id: pubmed_qa, questions: …/questions.json, corpus: …/corpus.json, trace_limit: 5}
graph:    {mode: asr_text_retrieval}
nodes:
  asr:            {model: whisper, size: base, device: cuda:0}
  text_embedding: {model: labse, device: cuda:0}
  vector_db:      {store: inmemory}
  retrieval:                      # dense + sparse fused, then reranked, then diversified
    k: 5
    mode: hybrid
    fusion:   {method: rrf, rrf_k: 60}
    reranker: {enabled: true, mode: token_overlap, top_k: 10}
    mmr:      {enabled: true, lambda: 0.7}
compute_confidence_intervals: true
audio_synthesis: {enabled: true, provider: mms, voice: en, sample_rate: 16000}
```

**(b) Explicit graph — arbitrary nodes + edges.** Replace the implicit node list with your own under
`graph.nodes` — **no template needed**; the graph itself drives building + behaviour (the run derives
a display label from the built pipelines). A node is a bare type string (id = type) or
`{id, type, params}`; `graph.edges` adds ordering not implied by data (auto-wiring handles the data
edges by artifact name). This is how the **same node type appears multiple times** with
distinct params — corpus-side robustness (`augmenter` with `axis: docs`), two retrievals fused by
`result_fusion`, a `rerank → mmr → threshold` refine chain, or multi-source graphs:

```yaml
graph:                                      # no template — the nodes ARE the spec
  nodes:
    - {id: corpus_src, type: dataset_source, params: {dataset: docs, role: corpus}}
    - {id: qa_src,     type: dataset_source, params: {dataset: qa,   role: questions}}
    - corpus_embedding
    - vector_db
    - tts
    - asr
    - text_embedding
    - retrieval
    - metrics
    - finalize
```

**(c) Branches — variant comparison with auto-CSE.** `graph.branches` declares N variants over a
shared template; each is `{id, <node>: <override>}`. Branches expand to per-branch node instances
(`asr@whisper_base`, …), the **un-overridden shared prefix collapses to one run via CSE** (so the
corpus is embedded once), and a terminal `aggregate` fans in every branch → per-branch metrics +
**paired statistics** (Wilcoxon / Cohen's d / CIs, §8). The real `showcase_asr_model_compare.yaml`:

```yaml
graph:
  mode: asr_text_retrieval
  branches:
    - {id: whisper_base, asr: {model: whisper, size: base}}
    - {id: whisper_tiny, asr: {model: whisper, size: tiny}}
```

**(d) Iterative RAG.** `config.rag.rounds = N` (`RagFlowConfig`) unrolls the loop into distinct hop
instances `text_embedding@h1, retrieval@h1, query_refine@h1, text_embedding@h2, …` — no per-hop
artifact names needed: the `one_of` chains read the newest published variant, so hop *k*'s embedder
automatically picks hop *k−1*'s `refined_query_text` (§9).

**What you can author** (composition → experiment; bracketed file is a runnable showcase):

| Experiment | Node composition | Showcase |
|------------|------------------|----------|
| Plain spoken retrieval | `dataset_source → corpus_embedding → vector_db → tts → asr → text_embedding → retrieval → metrics` | `evaluation_config_pubmed_qa_*.yaml` |
| Hybrid retrieval | `… → retrieval(mode: hybrid, fusion.method: rrf)` (dense+sparse, one node) | `evaluation_config_pubmed_qa_hybrid_rrf.yaml` |
| Hybrid + rerank + MMR | `… → retrieval(reranker, mmr)` (or explicit `rerank → mmr → threshold` chain) | `showcase_hybrid_rerank_mmr.yaml` |
| Query-text robustness | `… → asr → augmenter(axis: query) → text_embedding → …` | `showcase_robustness_augmenter.yaml` |
| Corpus robustness | `… → augmenter(axis: docs) → corpus_embedding → …` | `showcase_corpus_robustness.yaml` |
| Audio robustness | `… → tts → augment_audio(snr_db, …) → asr → …` | `showcase_audio_robustness.yaml` |
| Model A/B (paired stats) | `branches: [{asr: base}, {asr: tiny}]` + CSE + `aggregate` | `showcase_asr_model_compare.yaml`, `e2e_pubmed_qa_branched.yaml` |
| Reference vs ASR branch | `branches: [{asr: {oracle: true}}, {asr: {}}]` | `e2e_pubmed_qa_branched.yaml` |
| Multi-dataset join | `dataset_source(role: corpus)` + `dataset_source(role: questions)` (or `dataset_union`) | `showcase_multi_dataset_join.yaml` |
| Query optimization / correction | `… → asr → query_optimization(rewrite/hyde)` or `query_correction(rule/kb/llm) → text_embedding` | flags in config (§4/§6) |
| Iterative RAG | `rag.rounds: N` → unrolled `…@h1 → query_refine@h1 → …@h2` | `RagFlowConfig` (§9) |
| RAG answer + judge | `… → retrieval → answer_gen → answer_metrics → answer_judge` | `evaluation_config_pubmed_qa_hybrid_judge.yaml` |
| Report sinks / leaderboard | `… → metrics → aggregate → leaderboard_sink + dataset_sink` | `showcase_report_sinks.yaml` |

**Preview before running** (no models loaded): `evaluator graph --config <yaml>` (or
`--preset <name>`) prints the topological levels + each node's `inputs → outputs [deps]`;
`POST /api/graph/preview` and `/ui/graph` render the same DAG in the web builder. Every node carries
its declared `category` (DAG color) and `domain` (grouping) from the taxonomy (§9).
