"""Corpus embedding + retrieval-index build (extracted from evaluation_service)."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..config import EvaluationConfig
from ..config.types import enum_to_str
from ..errors import ConfigurationError
from ..logging_config import get_logger
from ..storage.cache import CacheManager
from ..storage.cache_keys import (
    dataset_fingerprint,
    manifest_fingerprint,
    model_fingerprint,
    preprocessing_fingerprint,
    retrieval_fingerprint,
    vector_db_manifest_key,
)


def _resolve_dataset_identity(config: EvaluationConfig) -> tuple[str, Dict[str, Any]]:
    source: Dict[str, Any]
    if config.data.prepared_dataset_dir:
        dataset_dir = Path(config.data.prepared_dataset_dir).resolve()
        source = {"prepared_dataset_dir": str(dataset_dir)}
        return f"prepared:{dataset_dir}", source
    if config.data.questions_path:
        questions_path = Path(config.data.questions_path).resolve()
        corpus_path = (
            Path(config.data.corpus_path).resolve() if config.data.corpus_path else None
        )
        source = {
            "questions_path": str(questions_path),
            "corpus_path": str(corpus_path) if corpus_path else None,
        }
        return f"questions:{questions_path}|corpus:{corpus_path}", source
    return "unknown", {}


def _corpus_signature(corpus: List[Any]) -> str:
    normalized = []
    for doc in corpus:
        if isinstance(doc, dict):
            normalized.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "text": doc.get("text"),
                }
            )
        else:
            normalized.append(str(doc))
    return manifest_fingerprint({"corpus": normalized})


def _retrieval_strategy_dict(vector_db) -> Dict[str, Any]:
    """Retrieval-strategy fields that affect the vector-DB cache fingerprint."""
    return {
        "mode": vector_db.retrieval_mode,
        "hybrid_dense_weight": vector_db.hybrid_dense_weight,
        "hybrid_fusion_method": vector_db.hybrid_fusion_method,
        "rrf_k": vector_db.rrf_k,
        "bm25_k1": vector_db.bm25_k1,
        "bm25_b": vector_db.bm25_b,
        "reranker_mode": vector_db.reranker_mode,
        "reranker_enabled": vector_db.reranker_enabled,
        "reranker_model": vector_db.reranker_model,
        "reranker_weight": vector_db.reranker_weight,
        "use_mmr": vector_db.use_mmr,
        "mmr_lambda": vector_db.mmr_lambda,
        "min_similarity_threshold": vector_db.min_similarity_threshold,
        "distance_metric": vector_db.distance_metric,
    }


def _vector_db_cache_key(
    config: EvaluationConfig, retrieval_pipeline: Any, corpus: List[Any]
) -> str:
    dataset_identity, source = _resolve_dataset_identity(config)
    dataset_fp = dataset_fingerprint(
        dataset_identity=dataset_identity,
        trace_limit=config.data.trace_limit,
        source={
            **source,
            "corpus_signature": _corpus_signature(corpus),
            "corpus_size": len(corpus),
        },
    )
    model_fp = model_fingerprint(
        model_name=config.model.text_emb_model_name
        or config.model.text_emb_model_type
        or "unknown",
        model_type=config.model.text_emb_model_type,
        inference={"device": config.model.text_emb_device},
    )
    retrieval_fp = retrieval_fingerprint(
        vector_store_type=enum_to_str(config.vector_db.type),
        retrieval_strategy=_retrieval_strategy_dict(config.vector_db),
    )
    preprocess_fp = preprocessing_fingerprint(
        {
            "pipeline_mode": config.model.pipeline_mode,
            "query_optimization_enabled": config.query_optimization.enabled,
        }
    )
    return vector_db_manifest_key(
        dataset_fp=dataset_fp,
        model_fp=model_fp,
        retrieval_fp=retrieval_fp,
        preprocessing_fp=preprocess_fp,
    )


def build_corpus_index(
    config: EvaluationConfig,
    dataset,
    retrieval_pipeline=None,
    text_emb_pipeline=None,
    audio_emb_pipeline=None,
    cache_manager: CacheManager | None = None,
    load_info: Dict[str, Any] | None = None,
):
    """Embed the corpus and build the retrieval index when the mode retrieves.

    Runs after the model pipelines are built (TTS already offloaded).
    """
    from ..pipeline.audio.synthesis import AudioSynthesizer

    logger = get_logger(__name__)

    if retrieval_pipeline is not None and hasattr(dataset, "get_corpus"):
        corpus = dataset.get_corpus()
        if not corpus:
            raise ConfigurationError(
                "Retrieval mode requires non-empty corpus. Set data.corpus_path "
                "to JSON/JSONL corpus file with doc_id/text fields."
            )
        if corpus and text_emb_pipeline is not None:
            corpus_texts = [doc.get("text", str(doc)) for doc in corpus]
            can_use_vector_cache = (
                cache_manager is not None
                and config.cache.enabled
                and config.cache.cache_vector_db
            )
            vector_cache_key = (
                _vector_db_cache_key(config, retrieval_pipeline, corpus)
                if can_use_vector_cache
                else None
            )
            if can_use_vector_cache and vector_cache_key:
                cached_db = cache_manager.get_vector_db(vector_cache_key)
                if cached_db is not None:
                    vectors, payloads = cached_db
                    logger.info(
                        "Loaded cached retrieval index artifacts: vectors=%d",
                        len(vectors),
                    )
                    retrieval_pipeline.build_index(
                        embeddings=vectors, metadata=payloads
                    )
                    if load_info is not None:
                        load_info["vector_cache_hit"] = True
                        load_info["vector_cache_key"] = vector_cache_key
                        load_info["corpus_size"] = len(corpus)
                    return dataset
                if load_info is not None:
                    load_info["vector_cache_hit"] = False
                    load_info["vector_cache_key"] = vector_cache_key
                    load_info["corpus_size"] = len(corpus)

            logger.info(
                "STAGE load_dataset: corpus embedding START (%d docs)",
                len(corpus_texts),
            )
            corpus_embeddings = text_emb_pipeline.process_batch(
                corpus_texts, show_progress=True, desc="Embedding corpus docs"
            )
            logger.info("STAGE load_dataset: corpus embedding DONE; building index")
            vectors = np.array(corpus_embeddings)
            retrieval_pipeline.build_index(embeddings=vectors, metadata=corpus)
            if can_use_vector_cache and vector_cache_key:
                cache_manager.set_vector_db(vector_cache_key, vectors, corpus)
                logger.info("Cached retrieval index artifacts for reuse")
                if load_info is not None:
                    load_info["vector_cache_written"] = True
                    load_info["vector_cache_key"] = vector_cache_key
                    load_info["corpus_size"] = len(corpus)
            elif load_info is not None:
                load_info["vector_cache_hit"] = False
                load_info["corpus_size"] = len(corpus)
        elif corpus and audio_emb_pipeline is not None:
            # audio_emb_retrieval: synthesize TTS audio for corpus docs, then embed via audio model
            from ..datasets.core import load_audio_file

            synthesizer = AudioSynthesizer(config.audio_synthesis)
            corpus_audio_dir = (
                config.audio_synthesis.output_dir + "/corpus"
                if config.audio_synthesis.output_dir
                else None
            )
            if corpus_audio_dir:
                import os

                os.makedirs(corpus_audio_dir, exist_ok=True)

            audio_arrays, sampling_rates = [], []
            for doc in corpus:
                doc_id = doc.get("doc_id", "")
                audio_path = doc.get("audio_path")
                if not audio_path:
                    text = (
                        doc.get("text")
                        or doc.get("abstract")
                        or doc.get("title")
                        or str(doc)
                    )
                    if corpus_audio_dir:
                        audio_path = os.path.join(corpus_audio_dir, f"{doc_id}.wav")
                    if audio_path and os.path.exists(audio_path):
                        doc["audio_path"] = audio_path
                    else:
                        try:
                            synthesizer.synthesize(text, output_path=audio_path)
                            doc["audio_path"] = audio_path
                        except Exception as e:
                            logger.warning(f"TTS failed for corpus doc {doc_id}: {e}")
                            continue
                if audio_path:
                    try:
                        waveform, sr = load_audio_file(audio_path)
                        audio_arrays.append(waveform.squeeze().numpy())
                        sampling_rates.append(sr)
                    except Exception as e:
                        logger.warning(f"Failed to load corpus audio {audio_path}: {e}")

            if audio_arrays:
                corpus_embeddings = audio_emb_pipeline.process_batch(
                    audio_arrays, sampling_rates
                )
                vectors = np.array(corpus_embeddings)
                retrieval_pipeline.build_index(
                    embeddings=vectors, metadata=corpus[: len(audio_arrays)]
                )
                logger.info(
                    "Built corpus audio embedding index: %d vectors", len(audio_arrays)
                )
                synthesizer.log_cache_stats()
            else:
                logger.warning(
                    "No corpus audio could be synthesized; retrieval index is empty"
                )
    return dataset
