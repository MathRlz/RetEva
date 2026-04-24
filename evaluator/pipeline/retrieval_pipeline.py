from typing import Any, Dict, List, Optional, Tuple, TypeVar, TYPE_CHECKING

import numpy as np
from collections import Counter, defaultdict

from ..storage.cache import CacheManager
from ..constants import MIN_NORM_THRESHOLD
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
from ..models.retrieval.rag.strategies import (
    DistanceMetric,
    mmr_rerank,
    threshold_filter,
)

if TYPE_CHECKING:
    from ..models.retrieval.rag.reranker import BaseReranker

logger = get_logger(__name__)

K = TypeVar('K')


def _payload_text(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("text", ""))
    return str(payload)


class SparseBM25Index:
    """Minimal BM25 index for lexical retrieval over payload texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.payloads: List[Any] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.doc_term_freqs: List[Counter] = []
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_count: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in text.lower().split() if tok]

    def build(self, payloads: List[Any]) -> None:
        self.payloads = payloads
        self.doc_term_freqs = []
        self.doc_lens = []
        self.doc_freq = defaultdict(int)

        for payload in payloads:
            tokens = self._tokenize(_payload_text(payload))
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            for token in tf.keys():
                self.doc_freq[token] += 1

        self.doc_count = len(payloads)
        self.avgdl = (sum(self.doc_lens) / self.doc_count) if self.doc_count > 0 else 0.0

    def search(self, query_text: str, k: int = 10) -> List[Tuple[Any, float]]:
        if self.doc_count == 0:
            return []

        q_tokens = self._tokenize(query_text)
        if not q_tokens:
            return []

        scores = np.zeros(self.doc_count, dtype=np.float32)
        n = self.doc_count

        for token in q_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue

            idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))

            for i, tf in enumerate(self.doc_term_freqs):
                f = tf.get(token, 0)
                if f == 0:
                    continue

                dl = self.doc_lens[i]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, MIN_NORM_THRESHOLD)))
                scores[i] += idf * (f * (self.k1 + 1.0)) / max(denom, MIN_NORM_THRESHOLD)

        top_idx = np.argsort(-scores)[:k]
        return [(self.payloads[i], float(scores[i])) for i in top_idx if scores[i] > 0.0]


def _payload_key(payload: Any) -> str:
    if isinstance(payload, dict):
        if payload.get("doc_id") is not None:
            return str(payload["doc_id"])
        if payload.get("text") is not None:
            return str(payload["text"])
    return str(payload)


def _min_max_norm(scores: Dict[K, float]) -> Dict[K, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < MIN_NORM_THRESHOLD:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

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
        logger.info(f"Retrieval pipeline initialized with vector store: {type(vector_store).__name__}{reranker_info}{mmr_info}{threshold_info}")
    
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
            raise ValueError(f"Unknown distance metric: {metric}. Choose from: cosine, euclidean, dot_product")

    def build_index(
        self, 
        embeddings: np.ndarray, 
        metadata: Optional[List[Any]] = None
    ) -> None:
        """
        Build the retrieval index from embeddings.
        
        Args:
            embeddings: Array of embedding vectors
            metadata: Optional list of metadata/payloads for each embedding
        """
        with TimingContext(f"Building vector index ({len(embeddings)} vectors)", logger):
            payloads = metadata if metadata is not None else [str(i) for i in range(len(embeddings))]
            self.vector_store.build(embeddings, payloads)
            if self.sparse_index is not None:
                self.sparse_index.build(payloads)
            
            # Store embeddings and payloads for MMR
        if self.strategy_config.post_processing.use_mmr:
            self._index_embeddings = np.asarray(embeddings, dtype=np.float32)
            self._index_payloads = payloads

    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
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
            raise ValueError("search() supports only dense mode. Use search_batch(..., query_texts=...) for sparse/hybrid")
        if self.strategy_config.reranking.mode != "none":
            raise ValueError("search() does not support reranking. Use search_batch(..., query_texts=...)")
        return self.vector_store.search(query_embedding, k)

    def search_records(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[ScoredRetrievalResult]:
        """Typed retrieval output contract for single-query dense search."""
        return normalize_search_results(self.search(query_embedding, k))

    @staticmethod
    def _token_overlap_score(query_text: str, payload: Any) -> float:
        q = set(SparseBM25Index._tokenize(query_text))
        d = set(SparseBM25Index._tokenize(_payload_text(payload)))
        if not q or not d:
            return 0.0
        inter = len(q & d)
        union = len(q | d)
        return inter / max(union, 1)

    def _rerank(self, query_text: str, results: List[Tuple[Any, float]], k: int) -> List[Tuple[Any, float]]:
        """Apply reranking to initial retrieval results.
        
        Supports multiple reranking modes:
        - "none": No reranking, just truncate to k
        - "token_overlap": Simple Jaccard token overlap scoring  
        - "cross_encoder": Use cross-encoder model (requires self.reranker)
        """
        reranker_mode = self.strategy_config.reranking.mode
        if reranker_mode == "none" and self.reranker is None:
            return results[:k]

        limited = results[: max(k, self.strategy_config.reranking.top_k)]
        
        # Cross-encoder reranking takes precedence if reranker is set
        if self.reranker is not None:
            return self.reranker.rerank(query_text, limited, top_k=k)

        if reranker_mode == "token_overlap":
            base_scores = {idx: float(score) for idx, (_, score) in enumerate(limited)}
            base_norm = _min_max_norm(base_scores)
            rerank_scores = {
                idx: self._token_overlap_score(query_text, payload)
                for idx, (payload, _) in enumerate(limited)
            }
            rerank_norm = _min_max_norm(rerank_scores)

            merged = []
            for idx, (payload, _) in enumerate(limited):
                score = (
                    (1.0 - self.strategy_config.reranking.weight) * base_norm.get(idx, 0.0)
                    + self.strategy_config.reranking.weight * rerank_norm.get(idx, 0.0)
                )
                merged.append((payload, float(score)))

            merged.sort(key=lambda x: x[1], reverse=True)
            return merged[:k]
        
        if reranker_mode not in {"none", "token_overlap", "cross_encoder"}:
            raise ValueError(f"Unsupported reranker_mode: {reranker_mode}")
        
        return results[:k]

    def _apply_mmr(
        self,
        query_emb: np.ndarray,
        results: List[Tuple[Any, float]],
        k: int,
    ) -> List[Tuple[Any, float]]:
        """Apply MMR reranking for diversity.
        
        Args:
            query_emb: Query embedding
            results: Initial retrieval results
            k: Number of results to return
            
        Returns:
            MMR-reranked results
        """
        if not self.strategy_config.post_processing.use_mmr or not results:
            return results[:k]
        
        # Get embeddings for retrieved documents
        if self._index_embeddings is None or self._index_payloads is None:
            logger.warning("MMR enabled but index embeddings not stored. Skipping MMR.")
            return results[:k]
        
        # Map payloads to indices
        payload_to_idx = {_payload_key(p): i for i, p in enumerate(self._index_payloads)}
        
        # Get embeddings for results
        result_indices = []
        valid_results = []
        for payload, score in results:
            key = _payload_key(payload)
            if key in payload_to_idx:
                result_indices.append(payload_to_idx[key])
                valid_results.append((payload, score))
        
        if not valid_results:
            return results[:k]
        
        doc_embs = self._index_embeddings[result_indices]
        
        return mmr_rerank(
            query_emb=query_emb,
            results=valid_results,
            doc_embs=doc_embs,
            k=k,
            lambda_param=self.strategy_config.post_processing.mmr_lambda,
            metric=self.distance_metric,
        )

    def _apply_threshold(
        self,
        results: List[Tuple[Any, float]],
    ) -> List[Tuple[Any, float]]:
        """Apply similarity threshold filtering.
        
        Args:
            results: Retrieval results
            
        Returns:
            Filtered results
        """
        if self.strategy_config.post_processing.min_similarity_threshold is None:
            return results
        
        return threshold_filter(
            results,
            min_score=self.strategy_config.post_processing.min_similarity_threshold,
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
        # Apply MMR for diversity (before k truncation for better selection)
        if self.strategy_config.post_processing.use_mmr:
            results = self._apply_mmr(query_emb, results, k)
        else:
            results = results[:k]
        
        # Apply threshold filtering
        results = self._apply_threshold(results)
        
        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        query_texts: Optional[List[str]] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """
        Search for similar items using multiple query embeddings.
        
        Args:
            query_embeddings: Array of query embedding vectors
            k: Number of results to return per query
            query_texts: Optional query texts for sparse/hybrid retrieval and reranking
            
        Returns:
            List of result lists, each containing (payload, score) tuples
        """
        memory_manager = get_memory_manager()
        results: List[List[Tuple[Any, float]]] = []
        needs_reranking = self.strategy_config.reranking.mode != "none" or self.reranker is not None
        needs_advanced = (
            self.strategy_config.post_processing.use_mmr
            or self.strategy_config.post_processing.min_similarity_threshold is not None
        )
        
        # Determine how many candidates to fetch for advanced strategies
        fetch_k = k
        if needs_reranking:
            fetch_k = max(fetch_k, self.strategy_config.reranking.top_k)
        if needs_advanced:
            # Fetch more candidates for MMR diversity selection
            fetch_k = max(fetch_k, k * 3)

        if self.strategy_config.core.mode == "dense":
            if not needs_reranking and not needs_advanced:
                # Fast path: vectorized batch search (single call to backend)
                batch_results = self.vector_store.search_batch(query_embeddings, fetch_k)
                for br in batch_results:
                    results.append(br[:k])
                    memory_manager.record_operation()
            else:
                # Slow path: per-query reranking/MMR
                for idx, query_emb in enumerate(query_embeddings):
                    dense = self.vector_store.search(query_emb, fetch_k)
                    if needs_reranking:
                        if query_texts is None:
                            raise ValueError("Reranking requires query_texts")
                        dense = self._rerank(query_texts[idx], dense, fetch_k)
                    query_results = self._apply_advanced_strategies(query_emb, dense, k)
                    results.append(query_results)
                    memory_manager.record_operation()
            return results

        if query_texts is None:
            raise ValueError(
                f"retrieval_mode={self.strategy_config.core.mode} requires query_texts"
            )

        if self.sparse_index is None:
            raise RuntimeError(
                f"Sparse index is not initialized for retrieval_mode={self.strategy_config.core.mode}"
            )

        if self.strategy_config.core.mode == "sparse":
            for idx, query_text in enumerate(query_texts):
                sparse = self.sparse_index.search(query_text, fetch_k)
                sparse = self._rerank(query_text, sparse, fetch_k if needs_advanced else k)
                
                # Apply advanced strategies
                query_results = self._apply_advanced_strategies(query_embeddings[idx], sparse, k)
                results.append(query_results)
                memory_manager.record_operation()
            return results

        # hybrid mode
        for idx, (query_emb, query_text) in enumerate(zip(query_embeddings, query_texts)):
            dense_res = self.vector_store.search(query_emb, fetch_k)
            sparse_res = self.sparse_index.search(query_text, fetch_k)

            hybrid_results = fuse_hybrid_results(
                self.strategy_config.core.hybrid_fusion_method,
                dense_res,
                sparse_res,
                dense_weight=self.strategy_config.core.hybrid_dense_weight,
                top_k=fetch_k,
                rrf_k=self.strategy_config.core.rrf_k,
            )

            hybrid_results = self._rerank(query_text, hybrid_results, fetch_k if needs_advanced else k)
            
            # Apply advanced strategies
            query_results = self._apply_advanced_strategies(query_emb, hybrid_results, k)
            results.append(query_results)
            memory_manager.record_operation()

        return results

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
            "vector_count": len(self.vector_store.embeddings) if hasattr(self.vector_store, 'embeddings') else 0
        }
    
    def reset_cache_stats(self) -> None:
        """Reset cache statistics (no-op for retrieval pipeline)."""
        pass
