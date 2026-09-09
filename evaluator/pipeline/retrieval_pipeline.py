"""Retrieval orchestration: vector store + strategy modules glued into one pipeline.

Strategy logic lives in ``models/retrieval/``; this module only sequences index build,
candidate fetch, and refinement."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm import tqdm
from ..utils.progress import progress_disabled

from ..storage.cache import CacheManager
from ..devices.memory import get_memory_manager
from ..logging_config import get_logger, TimingContext
from ..models.retrieval.strategy import RetrievalStrategyConfig
from ..models.retrieval.fusion_registry import fuse_hybrid_results
from ..models.retrieval.rag.strategies import DistanceMetric
from ..models.retrieval.refine import apply_mmr, apply_threshold, rerank_results
from ..models.retrieval.sparse import SparseBM25Index
from ..models.embedding_space import assert_spaces_compatible

if TYPE_CHECKING:
    from ..models.retrieval.rag.reranker import CrossEncoderReranker

logger = get_logger(__name__)


class RetrievalPipeline:
    """Pipeline for vector database operations and retrieval: dense, sparse (BM25), and hybrid
    modes, with optional cross-encoder reranking, MMR diversity, and threshold filtering."""

    def __init__(
        self,
        vector_store: Any,
        cache_manager: Optional[CacheManager] = None,
        strategy_config: Optional[RetrievalStrategyConfig] = None,
        reranker: Optional["CrossEncoderReranker"] = None,
    ) -> None:
        """Build the pipeline from a :class:`RetrievalStrategyConfig` (the sole knob source);
        ``None`` falls back to an all-defaults config (dense, no rerank, cosine)."""
        self.vector_store = vector_store
        self.cache = cache_manager
        self.reranker = reranker

        self.strategy_config = strategy_config or RetrievalStrategyConfig()
        self.strategy_config.validate()

        self.distance_metric = self._parse_distance_metric(
            self.strategy_config.post_processing.distance_metric
        )

        # Store embeddings for MMR (populated during build_index)
        self._index_embeddings: Optional[np.ndarray] = None
        self._index_payloads: Optional[List[Any]] = None
        # Embedding space the index holds. Tagged at index build from the corpus vectors'
        # resolved space; None = unknown (the runtime space guard then no-ops).
        self.index_space_id: Optional[str] = None

        self.sparse_index: Optional[SparseBM25Index] = None
        if self.strategy_config.core.mode in {"sparse", "hybrid"}:
            self.sparse_index = SparseBM25Index(
                k1=self.strategy_config.core.bm25_k1,
                b=self.strategy_config.core.bm25_b,
            )

        reranker_info = ""
        if reranker is not None:
            reranker_info = f", cross-encoder reranker: {reranker.name()}"
        mmr_info = (
            f", MMR (λ={self.strategy_config.post_processing.mmr_lambda})"
            if self.strategy_config.post_processing.use_mmr
            else ""
        )
        threshold_info = (
            f", threshold={self.strategy_config.post_processing.min_similarity_threshold}"
            if self.strategy_config.post_processing.min_similarity_threshold is not None
            else ""
        )
        logger.info(
            f"Retrieval pipeline initialized with vector store: "
            f"{type(vector_store).__name__}{reranker_info}{mmr_info}{threshold_info}"
        )

    @staticmethod
    def _parse_distance_metric(metric: str) -> DistanceMetric:
        """Parse a distance-metric string to the enum (``dot`` aliases ``dot_product``)."""
        key = metric.lower()
        try:
            return DistanceMetric("dot_product" if key == "dot" else key)
        except ValueError:
            raise ValueError(
                f"Unknown distance metric: {metric}. Choose from: cosine, euclidean, dot_product"
            )

    def build_index(
        self, embeddings: np.ndarray, metadata: Optional[List[Any]] = None
    ) -> None:
        with TimingContext(
            f"Building vector index ({len(embeddings)} vectors)", logger
        ):
            payloads = (
                metadata
                if metadata is not None
                else [str(i) for i in range(len(embeddings))]
            )
            self.vector_store.build(embeddings, payloads)
            if self.sparse_index is not None:
                self.sparse_index.build(payloads)

        if self.strategy_config.post_processing.use_mmr:
            self._index_embeddings = np.asarray(embeddings, dtype=np.float32)
            self._index_payloads = payloads

    def assert_query_space(self, query_space: Optional[str]) -> None:
        """Runtime space guard: raise if ``query_space`` is incompatible with the index's space.
        No-op when either is unknown — a defense-in-depth backstop behind pre-flight validators."""

        assert_spaces_compatible(
            self.index_space_id, query_space, where="retrieval"
        )

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[Any, float]]:
        """Search for similar items using a single query embedding.

        Only supports dense mode. For sparse/hybrid, use search_batch.
        """
        if self.strategy_config.core.mode != "dense":
            raise ValueError(
                "search() supports only dense mode. "
                "Use search_batch(..., query_texts=...) for sparse/hybrid"
            )
        if self.strategy_config.reranking.mode != "none":
            raise ValueError(
                "search() does not support reranking. Use search_batch(..., query_texts=...)"
            )
        return self.vector_store.search(query_embedding, k)

    def _rerank(
        self, query_text: str, results: List[Tuple[Any, float]], k: int
    ) -> List[Tuple[Any, float]]:
        """Rerank via the extracted strategy (cross-encoder / token-overlap / none)."""
        return rerank_results(
            query_text,
            results,
            k,
            reranking=self.strategy_config.reranking,
            reranker=self.reranker,
        )

    def _apply_advanced_strategies(
        self,
        query_emb: np.ndarray,
        results: List[Tuple[Any, float]],
        k: int,
    ) -> List[Tuple[Any, float]]:
        """Apply similarity threshold then MMR diversity: threshold must run on the original
        retrieval scores first, since MMR replaces the score with ``λ·rel − (1−λ)·div``, not
        comparable to a similarity cutoff."""
        post = self.strategy_config.post_processing
        # Confidence filter on the original retrieval scores (no-op when threshold unset).
        results = apply_threshold(results, post=post)
        # MMR for diversity over the surviving candidates (before k truncation).
        if post.use_mmr:
            results = apply_mmr(
                query_emb,
                results,
                k,
                post=post,
                index_embeddings=self._index_embeddings,
                index_payloads=self._index_payloads,
                metric=self.distance_metric,
            )
        else:
            results = results[:k]

        return results

    def _needs_reranking(self) -> bool:
        return (
            self.strategy_config.reranking.mode != "none" or self.reranker is not None
        )

    def _needs_advanced(self) -> bool:
        return (
            self.strategy_config.post_processing.use_mmr
            or self.strategy_config.post_processing.min_similarity_threshold is not None
        )

    @property
    def needs_refinement(self) -> bool:
        """Whether a post-retrieval refine step (rerank / MMR / threshold) applies.
        Drives whether the DAG includes a separate ``rerank`` node."""
        return self._needs_reranking() or self._needs_advanced()

    def _fetch_k(self, k: int) -> int:
        fetch_k = k
        if self._needs_reranking():
            fetch_k = max(fetch_k, self.strategy_config.reranking.top_k)
        if self._needs_advanced():
            fetch_k = max(fetch_k, k * 3)
        return fetch_k

    def retrieve_candidates(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        query_texts: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Stage 1 of retrieval: raw dense/sparse/hybrid candidates (pre-refinement); the dense
        fast path returns top-k directly when no refinement applies. ``mode`` overrides the
        configured mode (so the hybrid DAG runs dense+sparse as separate nodes). Pair with
        :meth:`refine_candidates`."""
        memory_manager = get_memory_manager()
        candidates: List[List[Tuple[Any, float]]] = []
        fetch_k = self._fetch_k(k)
        mode = mode or self.strategy_config.core.mode

        if mode == "dense":
            if not self.needs_refinement:
                batch_results = self.vector_store.search_batch(
                    query_embeddings, fetch_k
                )
                for br in batch_results:
                    candidates.append(br[:k])
                    memory_manager.record_operation()
            else:
                for query_emb in tqdm(
                    query_embeddings, desc="Retrieval", unit="query",
                    disable=progress_disabled(),
                ):
                    candidates.append(self.vector_store.search(query_emb, fetch_k))
                    memory_manager.record_operation()
            return candidates

        if query_texts is None:
            raise ValueError(f"retrieval_mode={mode} requires query_texts")
        if self.sparse_index is None:
            raise RuntimeError(
                f"Sparse index is not initialized for retrieval_mode={mode}"
            )

        if mode == "sparse":
            for query_text in tqdm(
                query_texts, desc="Retrieval", unit="query",
                disable=progress_disabled(),
            ):
                candidates.append(self.sparse_index.search(query_text, fetch_k))
                memory_manager.record_operation()
            return candidates

        # hybrid
        for query_emb, query_text in tqdm(
            zip(query_embeddings, query_texts),
            total=len(query_texts),
            desc="Retrieval",
            unit="query",
            disable=progress_disabled(),
        ):
            dense_res = self.vector_store.search(query_emb, fetch_k)
            sparse_res = self.sparse_index.search(query_text, fetch_k)
            candidates.append(
                fuse_hybrid_results(
                    self.strategy_config.core.hybrid_fusion_method,
                    dense_res,
                    sparse_res,
                    dense_weight=self.strategy_config.core.hybrid_dense_weight,
                    top_k=fetch_k,
                    rrf_k=self.strategy_config.core.rrf_k,
                )
            )
            memory_manager.record_operation()
        return candidates

    def refine_candidates(
        self,
        candidates: List[List[Tuple[Any, float]]],
        query_embeddings: np.ndarray,
        k: int = 10,
        query_texts: Optional[List[str]] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Stage 2 of retrieval: rerank (cross-encoder/token-overlap) + advanced
        strategies (MMR, threshold) + truncate to k. Inverse-paired with
        :meth:`retrieve_candidates`."""
        mode = self.strategy_config.core.mode
        needs_reranking = self._needs_reranking()
        fetch_k = self._fetch_k(k)
        results: List[List[Tuple[Any, float]]] = []
        for idx, cand in tqdm(
            enumerate(candidates),
            total=len(candidates),
            desc="Rerank",
            unit="query",
            disable=progress_disabled(),
        ):
            if mode == "dense":
                refined = cand
                if needs_reranking:
                    if query_texts is None:
                        raise ValueError("Reranking requires query_texts")
                    refined = self._rerank(query_texts[idx], refined, fetch_k)
            else:
                # sparse/hybrid call _rerank unconditionally (no-op truncate when no
                # reranker) with the top_k argument.
                top = fetch_k if self._needs_advanced() else k
                refined = self._rerank(query_texts[idx], cand, top)
            results.append(
                self._apply_advanced_strategies(query_embeddings[idx], refined, k)
            )
        return results

    # `refine_candidates` bundled (rerank -> MMR/[:k] -> threshold) as independent steps so
    # the DAG can compose them; rerank keeps the larger fetch_k pool only when MMR follows.
    def rerank_only(
        self, candidates, query_texts, k: int
    ) -> List[List[Tuple[Any, float]]]:
        """Rerank each candidate list, keeping the top ``k`` (pass fetch_k as ``k`` to keep
        the MMR pool). Identical to ``refine_candidates``'s rerank stage."""
        return [
            self._rerank(query_texts[i] if query_texts else "", cand, k)
            for i, cand in enumerate(candidates)
        ]

    def mmr_only(
        self, candidates, query_embeddings, k: int
    ) -> List[List[Tuple[Any, float]]]:
        """MMR-select ``k`` diverse results per query (the MMR stage alone)."""

        post = self.strategy_config.post_processing
        return [
            apply_mmr(
                query_embeddings[i],
                cand,
                k,
                post=post,
                index_embeddings=self._index_embeddings,
                index_payloads=self._index_payloads,
                metric=self.distance_metric,
            )
            for i, cand in enumerate(candidates)
        ]

    def threshold_only(self, candidates, k: int) -> List[List[Tuple[Any, float]]]:
        """Truncate to ``k`` then drop below the similarity threshold per query."""

        post = self.strategy_config.post_processing
        return [apply_threshold(cand[:k], post=post) for cand in candidates]

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        query_texts: Optional[List[str]] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Search with multiple query embeddings -> per-query (payload, score) lists;
        composition of :meth:`retrieve_candidates` then :meth:`refine_candidates` (the two
        halves the DAG runs as the ``retrieval`` and ``rerank`` nodes)."""
        candidates = self.retrieve_candidates(query_embeddings, k, query_texts)
        return self.refine_candidates(candidates, query_embeddings, k, query_texts)

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "vector_count": (
                len(self.vector_store.embeddings)
                if hasattr(self.vector_store, "embeddings")
                else 0
            )
        }
