"""One-line help for the config-form / builder fields whose names don't explain themselves.

A small glossary, exposed via ``/api/introspection/schema`` (so the builder can reuse it) and
rendered as field tooltips on the config form. Keyed by the form ``name`` / config field, so a
template can look up ``field_help.get(name)`` directly.
"""

FIELD_HELP = {
    "pipeline_mode": "Graph template to start from (asr_text_retrieval, audio_emb_retrieval, …).",
    "retrieval_mode": "dense = embedding similarity · sparse = BM25 keywords · hybrid = fuse both.",
    "reranker_mode": "Re-score the top-k: none · token_overlap · cross_encoder.",
    "vector_db_type": "Index backend: inmemory (small) · faiss (fast) · chromadb/qdrant (server).",
    "k": "How many documents to retrieve per query — the @k metrics (Recall@k, …) use this.",
    "trace_limit": "How many per-query retrieval traces to keep for inspection (0 = none).",
    "batch_size": "Items processed per batch — trade memory for throughput.",
    "hybrid_dense_weight": "Hybrid only: the dense score's weight in [0, 1] (1 = all dense).",
    "quick_test": "Cap the dataset to ~20 samples for a fast smoke run.",
    "output_dir": "Where results and the leaderboard sqlite are written.",
    "dataset_name": "A registered dataset id; its column schema + default metrics come with it.",
    # ── builder node-param switches (operator discriminators + knobs) ──
    "family": "Which measurement this node performs (transcription / retrieval / answer / judge).",
    "op": "Which operation this node runs — re-resolves the form's fields to match.",
    "method": "The algorithm variant for this operation.",
    "mode": "The node's operating mode.",
    "modality": "Which signal this node consumes — text / audio.",
    "axis": "query = embed the query · corpus = embed the documents (an embedder needs one).",
    "store": "Vector index backend: inmemory (small) · faiss (fast) · chromadb/qdrant (server).",
    "distance": "Vector similarity metric: cosine · dot · euclidean.",
    "combine_strategy": "How parallel query variants' result lists are merged.",
    "rrf_k": "Reciprocal-rank-fusion constant — larger flattens the rank weighting.",
    "weight": "This stream's weight when fusing scores (higher = more influence).",
    "top_k": "How many items this node keeps / emits.",
    "context_top_k": "How many retrieved docs to feed the generator/refiner as context.",
    "n_variants": "How many query variants to fan out (multi-query retrieval).",
    "oracle": "Use ground-truth relevant docs as the retrieval result (an upper-bound baseline).",
    "trace": "Emit per-query traces from this node (for inspection / the judge).",
    "target": "Which terminal effect this sink performs (finalize / aggregate / dataset / …).",
    "gpu_id": "Which GPU the index lives on (faiss_gpu only).",
    "temperature": "LLM sampling temperature — 0 is deterministic.",
    # ── model Params fields (declared on a model's inner Params; help shared across models) ──
    "pooling": "How frame embeddings are pooled into one vector: mean · cls · attention.",
    "compute_type": "faster-whisper precision: int8 (fast/small) · float16 · float32.",
    "tgt_lang": "Target language code the model decodes into (e.g. eng).",
    "source_lang": "Source language code of the input text (e.g. eng_Latn).",
    "emb_dim": "Output embedding dimensionality.",
    "dropout": "Dropout probability used by the pooling head.",
    "model_path": "Filesystem path to a local checkpoint / weights directory.",
}
