"""Sparse retrieval using BM25."""

from typing import List, Tuple, Optional
from rank_bm25 import BM25Plus

from ....logging_config import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """BM25-based sparse retriever using rank_bm25 library.
    
    Uses BM25Plus variant which handles high-frequency terms better than BM25Okapi.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        """Initialize BM25 retriever.
        
        Args:
            k1: BM25 k1 parameter controlling term frequency saturation.
            b: BM25 b parameter controlling document length normalization.
            delta: BM25Plus delta parameter (lower bound for term frequency normalization).
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self._bm25: Optional[BM25Plus] = None
        self._texts: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return [tok for tok in text.lower().split() if tok]
    
    def index(self, texts: List[str]) -> None:
        """Build the BM25 index from a list of documents.
        
        Args:
            texts: List of document texts to index.
        """
        self._texts = texts
        self._tokenized_corpus = [self._tokenize(text) for text in texts]
        self._bm25 = BM25Plus(self._tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
        logger.debug(f"BM25 index built with {len(texts)} documents")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search for documents matching the query.
        
        Args:
            query: Query string to search for.
            k: Number of top results to return.
            
        Returns:
            List of (document_index, score) tuples sorted by score descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call index() first.")
        
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0.0]
    
    def get_text(self, idx: int) -> str:
        """Get the original text for a document index."""
        return self._texts[idx]
    
    @property
    def doc_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._texts)
