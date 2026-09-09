"""Vector database configuration."""
from dataclasses import dataclass
from typing import List, Optional, Union

from ..config.types import VectorDBType, to_enum
from ..constants import RETRIEVAL_DEPTH

# The composable post-retrieval refine vocabulary (each is one retrieved→retrieved node).
REFINE_OPS = ("rerank", "mmr", "threshold")


@dataclass
class VectorDBConfig:
    """Configuration for vector database and retrieval settings: store backend, retrieval
    strategy, reranking, and advanced options (MMR, hybrid search). Field comments below
    carry per-field semantics."""
    type: Union[str, VectorDBType] = "inmemory"  # inmemory | faiss | faiss_gpu | chromadb | qdrant
    # floored at the deepest report cutoff (@10) so a smaller k cannot silently cap recall@10
    k: int = RETRIEVAL_DEPTH
    gpu_id: int = 0
    retrieval_mode: str = "dense"  # dense | sparse | hybrid
    hybrid_dense_weight: float = 0.5  # 0.0-1.0
    hybrid_fusion_method: str = "weighted"  # weighted | rrf | max_score
    rrf_k: int = 60  # RRF k parameter (higher = more weight to top results)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reranker_mode: str = "none"  # none | token_overlap | cross_encoder
    reranker_top_k: int = 20
    reranker_weight: float = 0.5  # 0.0-1.0
    # Cross-encoder reranker settings
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: Optional[str] = None  # Auto-detect if None
    # Advanced retrieval strategies
    use_mmr: bool = False  # Enable MMR for diversity
    mmr_lambda: float = 0.5  # 0-1, higher = more relevance, lower = more diversity
    min_similarity_threshold: Optional[float] = None  # Filter results below this score
    distance_metric: str = "cosine"  # cosine | euclidean | dot_product
    # Explicit, ordered post-retrieval refine chain (one of REFINE_OPS per entry; repeats
    # allowed → cascades). None derives the chain from reranker/use_mmr/threshold in the
    # canonical rerank→mmr→threshold order.
    refine_ops: Optional[List[str]] = None

    # ChromaDB-specific settings
    chromadb_path: Optional[str] = None  # None = in-memory
    chromadb_collection_name: str = "documents"
    # Qdrant-specific settings
    qdrant_url: Optional[str] = None
    qdrant_path: Optional[str] = None  # None = in-memory
    qdrant_collection_name: str = "documents"
    qdrant_api_key: Optional[str] = None

    def __post_init__(self):
        """Normalize type to enum and validate settings."""
        if isinstance(self.type, str):
            self.type = to_enum(self.type, VectorDBType)

        valid_retrieval_modes = {"dense", "sparse", "hybrid"}
        if self.retrieval_mode not in valid_retrieval_modes:
            raise ValueError(
                f"retrieval_mode must be one of {valid_retrieval_modes}, got {self.retrieval_mode}"
            )

        valid_hybrid_fusion = {"weighted", "rrf", "max_score"}
        if self.hybrid_fusion_method not in valid_hybrid_fusion:
            raise ValueError(
                f"hybrid_fusion_method must be one of {valid_hybrid_fusion}, "
                f"got {self.hybrid_fusion_method}"
            )

        if not 0.0 <= self.hybrid_dense_weight <= 1.0:
            raise ValueError(
                f"hybrid_dense_weight must be in [0, 1], got {self.hybrid_dense_weight}")

        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {self.mmr_lambda}")

        if self.refine_ops is not None:
            bad = [op for op in self.refine_ops if op not in REFINE_OPS]
            if bad:
                raise ValueError(
                    f"refine_ops entries must be one of {REFINE_OPS}, got {bad}"
                )
