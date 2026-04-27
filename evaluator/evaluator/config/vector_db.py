"""Vector database configuration."""
from dataclasses import dataclass, field
from typing import Optional, Union

from ..config.types import VectorDBType, to_enum


# ── Focused sub-configs ──────────────────────────────────────────────


@dataclass
class RerankerConfig:
    """Reranking settings."""
    mode: str = "none"  # none | token_overlap | cross_encoder
    top_k: int = 20
    weight: float = 0.5
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: Optional[str] = None


@dataclass
class HybridSearchConfig:
    """Hybrid (dense + sparse) retrieval settings."""
    dense_weight: float = 0.5
    fusion_method: str = "weighted"  # weighted | rrf | max_score
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class DiversityConfig:
    """MMR and diversity settings."""
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    diversity_penalty: float = 0.0
    min_similarity_threshold: Optional[float] = None
    max_similarity_threshold: Optional[float] = None


@dataclass
class AdvancedRetrievalConfig:
    """Multi-vector, query expansion, pseudo-relevance feedback, adaptive fusion."""
    multi_vector_enabled: bool = False
    vectors_per_doc: int = 3
    multi_vector_strategy: str = "max_sim"  # max_sim | avg_sim | late_interaction
    query_expansion_enabled: bool = False
    query_expansion_method: str = "synonyms"  # synonyms | embeddings | llm
    expansion_terms: int = 5
    pseudo_feedback_enabled: bool = False
    pseudo_feedback_top_k: int = 3
    pseudo_feedback_weight: float = 0.3
    adaptive_fusion_enabled: bool = False
    confidence_threshold: float = 0.7


@dataclass
class BackendConfig:
    """Backend-specific connection settings."""
    chromadb_path: Optional[str] = None
    chromadb_collection_name: str = "documents"
    qdrant_url: Optional[str] = None
    qdrant_path: Optional[str] = None
    qdrant_collection_name: str = "documents"
    qdrant_api_key: Optional[str] = None


# ── Main config (backward-compatible flat facade) ────────────────────


@dataclass
class VectorDBConfig:
    """
    Configuration for vector database and retrieval settings.
    
    Controls vector store backend, retrieval strategies, reranking, and advanced
    retrieval options like MMR and hybrid search.
    
    Attributes:
        type: Vector database type. Default: "inmemory".
            Options: "inmemory", "faiss", "faiss_gpu", "chromadb", "qdrant".
        k: Number of documents to retrieve. Default: 5.
        gpu_id: GPU device ID for FAISS GPU. Default: 0.
        
        retrieval_mode: Retrieval strategy. Default: "dense".
            Options: "dense", "sparse", "hybrid".
        hybrid_dense_weight: Weight for dense retrieval in hybrid mode (0.0-1.0). Default: 0.5.
        hybrid_fusion_method: Method for combining hybrid results. Default: "weighted".
            Options: "weighted", "rrf", "max_score".
        rrf_k: RRF k parameter (higher = more weight to top results). Default: 60.
        
        bm25_k1: BM25 k1 parameter. Default: 1.5.
        bm25_b: BM25 b parameter. Default: 0.75.
        
        reranker_mode: Reranking strategy. Default: "none".
            Options: "none", "token_overlap", "cross_encoder".
        reranker_top_k: Number of candidates to rerank. Default: 20.
        reranker_weight: Weight for reranker scores (0.0-1.0). Default: 0.5.
        reranker_enabled: Whether cross-encoder reranking is enabled. Default: False.
        reranker_model: Cross-encoder model identifier. Default: "cross-encoder/ms-marco-MiniLM-L-6-v2".
        reranker_device: Device for reranker (None = auto-detect).
        
        use_mmr: Enable Maximal Marginal Relevance for diversity. Default: False.
        mmr_lambda: MMR lambda parameter (0-1, higher = more relevance). Default: 0.5.
        min_similarity_threshold: Filter results below this score. Optional.
        distance_metric: Distance metric for similarity. Default: "cosine".
            Options: "cosine", "euclidean", "dot_product".
        
        chromadb_path: Path for ChromaDB persistent storage (None = in-memory).
        chromadb_collection_name: Collection name in ChromaDB. Default: "documents".
        
        qdrant_url: URL for Qdrant server (e.g., "http://localhost:6333"). Optional.
        qdrant_path: Path for Qdrant local storage (None = in-memory). Optional.
        qdrant_collection_name: Collection name in Qdrant. Default: "documents".
        qdrant_api_key: API key for Qdrant Cloud. Optional.
    
    Examples:
        >>> # Dense retrieval with FAISS
        >>> config = VectorDBConfig(type="faiss", k=10)
        >>> 
        >>> # Hybrid retrieval with reranking
        >>> config = VectorDBConfig(
        ...     type="chromadb",
        ...     retrieval_mode="hybrid",
        ...     hybrid_dense_weight=0.7,
        ...     reranker_enabled=True,
        ...     reranker_top_k=50
        ... )
        >>> 
        >>> # MMR for diversity
        >>> config = VectorDBConfig(
        ...     type="qdrant",
        ...     use_mmr=True,
        ...     mmr_lambda=0.6
        ... )
    """
    type: Union[str, VectorDBType] = "inmemory"  # inmemory | faiss | faiss_gpu | chromadb | qdrant
    k: int = 5
    gpu_id: int = 0
    retrieval_mode: str = "dense"  # dense | sparse | hybrid
    hybrid_dense_weight: float = 0.5
    hybrid_fusion_method: str = "weighted"  # weighted | rrf | max_score
    rrf_k: int = 60  # RRF k parameter (higher = more weight to top results)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reranker_mode: str = "none"  # none | token_overlap | cross_encoder
    reranker_top_k: int = 20
    reranker_weight: float = 0.5
    # Cross-encoder reranker settings
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: Optional[str] = None  # Auto-detect if None
    # Advanced retrieval strategies
    use_mmr: bool = False  # Enable MMR for diversity
    mmr_lambda: float = 0.5  # 0-1, higher = more relevance, lower = more diversity
    min_similarity_threshold: Optional[float] = None  # Filter results below this score
    max_similarity_threshold: Optional[float] = None  # Filter results above this score (for diversity)
    distance_metric: str = "cosine"  # cosine | euclidean | dot_product
    diversity_penalty: float = 0.0  # Penalty for similar documents (0.0-1.0)
    
    # Multi-vector retrieval settings
    multi_vector_enabled: bool = False
    vectors_per_doc: int = 3  # Number of vectors per document (e.g., sentence-level)
    multi_vector_strategy: str = "max_sim"  # max_sim | avg_sim | late_interaction
    
    # Query expansion settings
    query_expansion_enabled: bool = False
    query_expansion_method: str = "synonyms"  # synonyms | embeddings | llm
    expansion_terms: int = 5  # Number of expansion terms to add
    
    # Pseudo-relevance feedback settings
    pseudo_feedback_enabled: bool = False
    pseudo_feedback_top_k: int = 3  # Number of top docs to use for feedback
    pseudo_feedback_weight: float = 0.3  # Weight for feedback terms (0.0-1.0)
    
    # Adaptive fusion settings
    adaptive_fusion_enabled: bool = False
    confidence_threshold: float = 0.7  # Threshold for switching fusion strategies
    
    # ChromaDB-specific settings
    chromadb_path: Optional[str] = None
    chromadb_collection_name: str = "documents"
    # Qdrant-specific settings
    qdrant_url: Optional[str] = None
    qdrant_path: Optional[str] = None
    qdrant_collection_name: str = "documents"
    qdrant_api_key: Optional[str] = None

    def __post_init__(self):
        """Normalize type to enum and validate settings."""
        if isinstance(self.type, str):
            self.type = to_enum(self.type, VectorDBType)
        
        # Validate weights and thresholds
        valid_retrieval_modes = {"dense", "sparse", "hybrid"}
        if self.retrieval_mode not in valid_retrieval_modes:
            raise ValueError(
                f"retrieval_mode must be one of {valid_retrieval_modes}, got {self.retrieval_mode}"
            )

        valid_hybrid_fusion = {"weighted", "rrf", "max_score"}
        if self.hybrid_fusion_method not in valid_hybrid_fusion:
            raise ValueError(
                f"hybrid_fusion_method must be one of {valid_hybrid_fusion}, got {self.hybrid_fusion_method}"
            )

        if not 0.0 <= self.hybrid_dense_weight <= 1.0:
            raise ValueError(f"hybrid_dense_weight must be in [0, 1], got {self.hybrid_dense_weight}")
        
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {self.mmr_lambda}")
        
        if not 0.0 <= self.diversity_penalty <= 1.0:
            raise ValueError(f"diversity_penalty must be in [0, 1], got {self.diversity_penalty}")
        
        if not 0.0 <= self.pseudo_feedback_weight <= 1.0:
            raise ValueError(f"pseudo_feedback_weight must be in [0, 1], got {self.pseudo_feedback_weight}")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        
        # Validate multi-vector settings
        if self.multi_vector_enabled and self.vectors_per_doc < 1:
            raise ValueError(f"vectors_per_doc must be >= 1, got {self.vectors_per_doc}")
        
        valid_mv_strategies = {"max_sim", "avg_sim", "late_interaction"}
        if self.multi_vector_strategy not in valid_mv_strategies:
            raise ValueError(
                f"multi_vector_strategy must be one of {valid_mv_strategies}, "
                f"got {self.multi_vector_strategy}"
            )
        
        # Validate query expansion settings
        valid_expansion_methods = {"synonyms", "embeddings", "llm"}
        if self.query_expansion_method not in valid_expansion_methods:
            raise ValueError(
                f"query_expansion_method must be one of {valid_expansion_methods}, "
                f"got {self.query_expansion_method}"
            )

    # ── Sub-config accessors (read-only views over flat fields) ──────

    @property
    def reranker(self) -> RerankerConfig:
        return RerankerConfig(
            mode=self.reranker_mode, top_k=self.reranker_top_k,
            weight=self.reranker_weight, enabled=self.reranker_enabled,
            model=self.reranker_model, device=self.reranker_device,
        )

    @property
    def hybrid(self) -> HybridSearchConfig:
        return HybridSearchConfig(
            dense_weight=self.hybrid_dense_weight,
            fusion_method=self.hybrid_fusion_method,
            rrf_k=self.rrf_k, bm25_k1=self.bm25_k1, bm25_b=self.bm25_b,
        )

    @property
    def diversity(self) -> DiversityConfig:
        return DiversityConfig(
            use_mmr=self.use_mmr, mmr_lambda=self.mmr_lambda,
            diversity_penalty=self.diversity_penalty,
            min_similarity_threshold=self.min_similarity_threshold,
            max_similarity_threshold=self.max_similarity_threshold,
        )

    @property
    def advanced(self) -> AdvancedRetrievalConfig:
        return AdvancedRetrievalConfig(
            multi_vector_enabled=self.multi_vector_enabled,
            vectors_per_doc=self.vectors_per_doc,
            multi_vector_strategy=self.multi_vector_strategy,
            query_expansion_enabled=self.query_expansion_enabled,
            query_expansion_method=self.query_expansion_method,
            expansion_terms=self.expansion_terms,
            pseudo_feedback_enabled=self.pseudo_feedback_enabled,
            pseudo_feedback_top_k=self.pseudo_feedback_top_k,
            pseudo_feedback_weight=self.pseudo_feedback_weight,
            adaptive_fusion_enabled=self.adaptive_fusion_enabled,
            confidence_threshold=self.confidence_threshold,
        )

    @property
    def backend(self) -> BackendConfig:
        return BackendConfig(
            chromadb_path=self.chromadb_path,
            chromadb_collection_name=self.chromadb_collection_name,
            qdrant_url=self.qdrant_url, qdrant_path=self.qdrant_path,
            qdrant_collection_name=self.qdrant_collection_name,
            qdrant_api_key=self.qdrant_api_key,
        )
