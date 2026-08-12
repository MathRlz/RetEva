# Example Configurations

Five ready-to-run configs demonstrating common evaluation shapes. Each uses the
current node-centric schema: the pipeline is an explicit DAG under `graph.nodes` +
`graph.edges` (there is no `pipeline_mode` switch and no `features:` block — every
capability is a node), per-node model settings live under `nodes:`, and
`experiment` / `dataset` / `runtime` carry the run-level settings.

## Running

```bash
evaluator graph --config configs/examples/basic_asr_retrieval.yaml   # print the DAG, load nothing
evaluator run   --config configs/examples/basic_asr_retrieval.yaml   # run it
```

To adapt one: copy it, then regenerate the edge block after changing nodes with
`evaluator graph --config <yaml> --emit-edges --write`.

## The examples

| Config | Demonstrates |
|---|---|
| `basic_asr_retrieval.yaml` | The standard ASR path: `asr` (Whisper medium) → `text_embedding` (LaBSE) → dense `retrieval` (k=5, in-memory store) on `admed_voice`, with transcription (WER/CER) + retrieval metrics. The recommended starting point. |
| `audio_embedding_only.yaml` | Direct audio retrieval, no ASR node: `audio_embedding` (`clap_style`, your checkpoint via `model_path`) feeds `retrieval` (k=10); corpus side embeds with `clap_text`. Retrieval metrics only — there is no transcript to score. |
| `hybrid_retrieval.yaml` | Duplicate nodes of one type via the `{id, type, params}` node form: a dense `retrieval` + a sparse `retrieval_sparse` (BM25) fused by `result_fusion` (RRF, `rrf_k: 60`), then `rerank` (cross-encoder, top 30). Whisper large-v3 + jina_v4 across two GPUs, on `pubmed_qa`. |
| `fast_development.yaml` | The basic graph shrunk for iteration: whisper-tiny + LaBSE on CPU, batch 4, DEBUG console logging, 5 GB cache cap, checkpoint every 10 items. Runs anywhere. |
| `multi_gpu_production.yaml` | The hybrid graph at production scale on full `pubmed_qa` (`trace_limit: 0`): batch 64, 4 data workers, `faiss_gpu` vector store, reranker top 50 on a second GPU, checkpoint every 200. Includes a (disabled) `audio_synthesis` block for TTS-bridging a text dataset. |

## Config anatomy (common to all five)

- `experiment:` — run name + `output_dir` (report JSON, resolved-config sidecar).
- `dataset:` — dataset `id` (+ `questions`/`corpus` paths for file-backed sets), `batch_size`.
- `graph.nodes` / `graph.edges` — the DAG. Nodes are bare names
  (`dataset_source`, `asr`, `text_embedding`, `retrieval`, `metrics`, `finalize`, …)
  or `{id, type, params}` mappings when one type appears twice.
- `nodes:` — per-node model choice + params (`model`, `name`, `device`, retrieval
  `k`/`mode`/`fusion`/`reranker`, vector `store`).
- `runtime:` — cache toggles (`cache_transcriptions`, `cache_embeddings`, …) and logging levels.
- `checkpoint_*` — periodic checkpointing + resume.

## See also

- [`../../README.md`](../../README.md) — project overview + quickstart
- [`../../evaluator-architecture.md`](../../evaluator-architecture.md) — full architecture (graph, artifacts, metrics)
- [`../campaign/RUNBOOK.md`](../campaign/RUNBOOK.md) — the campaign configs + how to run them
