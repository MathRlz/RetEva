"""Retrieval orchestration: vector store + strategy modules glued into one pipeline.

Strategy logic lives in ``models/retrieval/`` (``sparse`` BM25, ``refine``
rerank/MMR/threshold, ``fusion_registry`` hybrid fusion, ``scoring`` shared
primitives); this module only sequences index build → candidate fetch →
refinement, mapping the DAG's ``retrieval``/``rerank`` nodes onto
:meth:`RetrievalPipeline.retrieve_candidates` / :meth:`refine_candidates`.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from ..storage.cache import CacheManager
from ..devices.memory import get_memory_manager
from ..logging_config import get_logger, TimingContext
from ..models.retrieval.strategy import (
    CoreRetrievalConfig,
    PostProcessingConfig,
    RerankingConfig,
    RetrievalStrategyConfig,
)
from ..models.retrieval.fusion_registry import fuse_hybrid_results
from ..models.retrieval.contracts import (
    ScoredRetrievalResult,
    normalize_search_results,
    normalize_batch_search_results,
)
from ..models.retrieval.rag.strategies import DistanceMetric
from ..models.retrieval.refine import apply_mmr, apply_threshold, rerank_results
from ..models.retrieval.sparse import SparseBM25Index

if TYPE_CHECKING:
    from ..models.retrieval.rag.reranker import BaseReranker

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    Pipeline for vector database operations and retrieval.

    Handles building vector DB from embeddings and performing searches.
    Supports dense, sparse (BM25), and hybrid retrieval modes.

    Hybrid mode supports two fusion strategies:
    - "weighted": Linear combination with min-max normalization
    - "rrf": Reciprocal Rank Fusion

    Supports optional cross-encoder reranking after initial retrieval.

    Advanced features:
    - MMR (Maximal Marginal Relevance) for diverse results
    - Similarity threshold filtering for quality control
    - Configurable distance metrics (cosine, euclidean, dot_product)
    """

    def __init__(
        self,
        vector_store: Any,
        cache_manager: Optional[CacheManager] = None,
        strategy_config: Optional[RetrievalStrategyConfig] = None,
        retrieval_mode: str = "dense",
        hybrid_dense_weight: float = 0.5,
        hybrid_fusion_method: str = "weighted",
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        reranker_mode: str = "none",
        reranker_top_k: int = 20,
        reranker_weight: float = 0.5,
        reranker: Optional["BaseReranker"] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        min_similarity_threshold: Optional[float] = None,
        distance_metric: str = "cosine",
    ) -> None:
        self.vector_store = vector_store
        self.cache = cache_manager
        self.reranker = reranker

        self.strategy_config = strategy_config or RetrievalStrategyConfig(
            core=CoreRetrievalConfig(
                mode=retrieval_mode,
                hybrid_dense_weight=hybrid_dense_weight,
                hybrid_fusion_method=hybrid_fusion_method,
                rrf_k=rrf_k,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
            ),
            reranking=RerankingConfig(
                mode=reranker_mode,
                top_k=reranker_top_k,
                weight=reranker_weight,
            ),
            post_processing=PostProcessingConfig(
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                min_similarity_threshold=min_similarity_threshold,
                distance_metric=distance_metric,
            ),
        )
        self.strategy_config.validate()

        self.distance_metric = self._parse_distance_metric(
            self.strategy_config.post_processing.distance_metric
        )

        # Store embeddings for MMR (populated during build_index)
        self._index_embeddings: Optional[np.ndarray] = None
        self._index_payloads: Optional[List[Any]] = None

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
            f"Retrieval pipeline initialized with vector store: {type(vector_store).__name__}{reranker_info}{mmr_info}{threshold_info}"
        )

    @staticmethod
    def _parse_distance_metric(metric: str) -> DistanceMetric:
        """Parse distance metric string to enum."""
        metric_lower = metric.lower()
        if metric_lower == "cosine":
            return DistanceMetric.COSINE
        elif metric_lower == "euclidean":
            return DistanceMetric.EUCLIDEAN
        elif metric_lower in ("dot_product", "dot"):
            return DistanceMetric.DOT_PRODUCT
        else:
            raise ValueError(
                f"Unknown distance metric: {metric}. Choose from: cosine, euclidean, dot_product"
            )

    def build_index(
        self, embeddings: np.ndarray, metadata: Optional[List[Any]] = None
    ) -> None:
        """
        Build the retrieval index from embeddings.

        Args:
            embeddings: Array of embedding vectors
            metadata: Optional list of metadata/payloads for each embedding
        """
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

            # Store embeddings and payloads for MMR
        if self.strategy_config.post_processing.use_mmr:
            self._index_embeddings = np.asarray(embeddings, dtype=np.float32)
            self._index_payloads = payloads

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        Search for similar items using a single query embedding.

        Only supports dense mode. For sparse/hybrid, use search_batch.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (payload, score) tuples
        """
        if self.strategy_config.core.mode != "dense":
            raise ValueError(
                "search() supports only dense mode. Use search_batch(..., query_texts=...) for sparse/hybrid"
            )
        if self.strategy_config.reranking.mode != "none":
            raise ValueError(
                "search() does not support reranking. Use search_batch(..., query_texts=...)"
            )
        return self.vector_store.search(query_embedding, k)

    def search_records(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[ScoredRetrievalResult]:
        """Typed retrieval output contract for single-query dense search."""
        return normalize_search_results(self.search(query_embedding, k))

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
        """Apply MMR and threshold filtering to results.

        Order: MMR first (needs more candidates), then threshold filtering.
        """
        post = self.strategy_config.post_processing
        # Apply MMR for diversity (before k truncation for better selection)
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

        return apply_threshold(results, post=post)

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
        """Stage 1 of retrieval: raw dense/sparse/hybrid candidates (pre-refinement).

        Fetches ``fetch_k`` candidates (enough for downstream rerank/MMR); the dense
        fast path returns top-k directly when no refinement applies. ``mode`` overrides
        the configured retrieval mode (so the hybrid DAG runs the dense + sparse arms as
        separate retrieval nodes). Pair with :meth:`refine_candidates`."""
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
                for query_emb in tqdm(query_embeddings, desc="Retrieval", unit="query", disable=None):
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
            for query_text in tqdm(query_texts, desc="Retrieval", unit="query", disable=None):
                candidates.append(self.sparse_index.search(query_text, fetch_k))
                memory_manager.record_operation()
            return candidates

        # hybrid
        for query_emb, query_text in tqdm(
            zip(query_embeddings, query_texts),
            total=len(query_texts),
            desc="Retrieval",
            unit="query",
            disable=None,
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
            disable=None,
        ):
            if mode == "dense":
                refined = cand
                if needs_reranking:
                    if query_texts is None:
                        raise ValueError("Reranking requires query_texts")
                    refined = self._rerank(query_texts[idx], refined, fetch_k)
            else:
                # sparse/hybrid call _rerank unconditionally (no-op truncate when no
                # reranker) with the historical top_k argument.
                top = fetch_k if self._needs_advanced() else k
                refined = self._rerank(query_texts[idx], cand, top)
            results.append(
                self._apply_advanced_strategies(query_embeddings[idx], refined, k)
            )
        return results

    # ── refine sub-steps, exposed as the rerank / mmr / threshold nodes ──────────
    # `refine_candidates` is the bundled (rerank → MMR/[:k] → threshold) path; these are
    # the same operations as independent steps so the DAG can compose them. ``k`` is the
    # single consistent target (the retrieve node's k); rerank keeps the larger fetch_k pool
    # only when MMR follows (MMR re-selects k diverse from it).
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
        from ..models.retrieval.refine import apply_mmr

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
        from ..models.retrieval.refine import apply_threshold

        post = self.strategy_config.post_processing
        return [apply_threshold(cand[:k], post=post) for cand in candidates]

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        query_texts: Optional[List[str]] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Search with multiple query embeddings → per-query (payload, score) lists.

        Composition of :meth:`retrieve_candidates` then :meth:`refine_candidates`
        (the two halves the DAG runs as the ``retrieval`` and ``rerank`` nodes)."""
        candidates = self.retrieve_candidates(query_embeddings, k, query_texts)
        return self.refine_candidates(candidates, query_embeddings, k, query_texts)

    def search_batch_records(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        query_texts: Optional[List[str]] = None,
    ) -> List[List[ScoredRetrievalResult]]:
        """Typed retrieval output contract for batch search."""
        return normalize_batch_search_results(
            self.search_batch(query_embeddings, k, query_texts=query_texts)
        )

    def save(self, path: str) -> None:
        """Save the index to disk."""
        with TimingContext(f"Saving vector store to {path}", logger):
            self.vector_store.save(path)

    def load(self, path: str) -> None:
        """Load the index from disk."""
        with TimingContext(f"Loading vector store from {path}", logger):
            self.vector_store.load(path)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get index/cache statistics.

        Returns:
            Dictionary with index statistics including vector count
        """
        return {
            "vector_count": (
                len(self.vector_store.embeddings)
                if hasattr(self.vector_store, "embeddings")
                else 0
            )
        }

    def reset_cache_stats(self) -> None:
        """Reset cache statistics (no-op for retrieval pipeline)."""
        pass
