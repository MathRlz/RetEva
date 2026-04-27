# Retrieval Strategies

Guide to vector stores, retrieval strategies, and advanced search techniques.

## Vector Stores

### In-Memory Store

Simple in-memory vector storage (default).

**Type**: `inmemory`

**Best for**: Small corpora (<10K documents), development

```yaml
vector_db:
  type: inmemory
```

```python
from evaluator.storage.backends import InMemoryVectorStore

store = InMemoryVectorStore()
store.build(embeddings, payloads)
results = store.search(query_embedding, k=5)
```

**Pros**:
- Simple, no setup
- Fast for small datasets
- No dependencies

**Cons**:
- Limited scalability
- No persistence
- High memory usage

### FAISS (CPU)

Facebook AI Similarity Search on CPU.

**Type**: `faiss`

**Best for**: Medium corpora (10K-1M documents)

```yaml
vector_db:
  type: faiss
  index_type: IVF  # Flat, IVF, IVFPQ, HNSW
  nlist: 100  # Number of clusters
  nprobe: 10  # Clusters to search
```

```python
from evaluator.storage.backends import FaissVectorStore

store = FaissVectorStore(dim=embeddings.shape[1])
store.build(embeddings, payloads)
```

**Index Types**:

| Type | Speed | Accuracy | Memory | Best For |
|------|-------|----------|--------|----------|
| `Flat` | Slow | Perfect | High | <100K docs, exact search |
| `IVF` | Fast | Good | Medium | 100K-1M docs, approximate |
| `IVFPQ` | Fastest | Lower | Low | >1M docs, compressed |
| `HNSW` | Fast | Very Good | High | High recall needed |

### FAISS (GPU)

GPU-accelerated FAISS.

**Type**: `faiss_gpu`

**Best for**: Large corpora (>1M documents), fast search

```yaml
vector_db:
  type: faiss_gpu
  gpu_id: 0
```

```python
from evaluator.storage.backends import FaissGpuVectorStore

store = FaissGpuVectorStore(dim=embeddings.shape[1], gpu_id=0)
store.build(embeddings, payloads)
```

**Installation**: `pip install faiss-gpu`

**Performance**: 10-100x faster than CPU FAISS

### ChromaDB

Persistent vector database with rich metadata.

**Type**: `chromadb`

**Best for**: Production, metadata filtering, persistence

```yaml
vector_db:
  type: chromadb
  chromadb_path: ./chroma_db
  chromadb_collection: medical_docs
  distance_metric: cosine  # cosine, l2, ip
```

```python
from evaluator.storage.backends import ChromaDBVectorStore

store = ChromaDBVectorStore(
    persist_path="./chroma_db",
    collection_name="medical_docs"
)

# Build with payloads (metadata or other payload objects)
store.build(embeddings, payloads)

# Search with filters
results = store.search(
    query_embedding,
    k=5,
    where={"specialty": "cardiology"}
)
```

**Installation**: `pip install chromadb`

### Qdrant

High-performance vector search engine.

**Type**: `qdrant`

**Best for**: Production, filtering, scalability

```yaml
vector_db:
  type: qdrant
  qdrant_path: ./qdrant_db  # or qdrant_url for remote
  qdrant_collection: medical_docs
  distance_metric: cosine
```

```python
from evaluator.storage.backends import QdrantVectorStore

store = QdrantVectorStore(
    path="./qdrant_db",
    collection_name="medical_docs"
)

# Build with payloads (metadata or other payload objects)
store.build(embeddings, payloads)
```

**Installation**: `pip install qdrant-client`

## Retrieval Strategies

### Dense Retrieval

Standard semantic search with dense vectors.

```yaml
retrieval:
  strategy: dense
  similarity_metric: cosine  # cosine, dot, l2
```

```python
from evaluator.retrieval import DenseRetriever

retriever = DenseRetriever(
    vector_store=store,
    similarity="cosine"
)

results = retriever.search(query_embedding, k=10)
```

### Sparse Retrieval

BM25-style keyword search.

```yaml
retrieval:
  strategy: sparse
  sparse_method: bm25  # bm25, tfidf
```

```python
from evaluator.retrieval import SparseRetriever

retriever = SparseRetriever(method="bm25")
retriever.index_documents(corpus_texts)
results = retriever.search(query_text, k=10)
```

**Best for**: Keyword-based queries, exact matches

### Hybrid Retrieval

Combines dense and sparse retrieval.

```yaml
retrieval:
  strategy: hybrid
  dense_weight: 0.7
  sparse_weight: 0.3
  fusion_method: rrf  # rrf (reciprocal rank fusion), weighted, or max_score
```

```python
from evaluator.retrieval import HybridRetriever

retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    dense_weight=0.7,
    sparse_weight=0.3,
    fusion_method="rrf"
)

results = retriever.search(query, k=10)
```

**Benefits**:
- Better recall than dense-only
- Better precision than sparse-only
- Robust to different query types

### Hybrid Fusion Methods

Evaluator hybrid fusion registry supports:
- `weighted`: weighted sum on normalized dense/sparse scores
- `rrf`: reciprocal rank fusion
- `max_score`: max weighted normalized branch score per document

### Re-ranking

Two-stage retrieval with re-ranking.

```yaml
retrieval:
  strategy: rerank
  first_stage_k: 100  # Retrieve candidates
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  final_k: 10  # Final results after reranking
```

```python
from evaluator.retrieval import ReRanker

# First stage: Fast retrieval
candidates = dense_retriever.search(query, k=100)

# Second stage: Rerank
reranker = ReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
final_results = reranker.rerank(query, candidates, k=10)
```

**Best for**: Maximizing precision, when latency allows

## Advanced Search Techniques

### Metadata Filtering

Filter by document attributes:

```python
results = store.search(
    query_embedding,
    k=10,
    filter={
        "specialty": "cardiology",
        "year": {"$gte": 2020},
        "language": "en"
    }
)
```

### Multi-Vector Search

Search with multiple query vectors:

```python
from evaluator.retrieval import MultiVectorRetriever

retriever = MultiVectorRetriever(store)

# Multiple query representations
query_vectors = [
    embedding_model.encode(original_text),
    embedding_model.encode(expanded_text),
    embedding_model.encode(reformulated_text)
]

results = retriever.search_multi(query_vectors, k=10, aggregation="max")
```

### Query Expansion

Expand queries before retrieval:

```python
from evaluator.retrieval import QueryExpander

expander = QueryExpander(method="llm")  # or "synonyms", "pseudo_relevance"
expanded_query = expander.expand("diabetes symptoms")
# "diabetes symptoms blood sugar insulin resistance hyperglycemia"

results = retriever.search(expanded_query, k=10)
```

### Diversity Ranking

Ensure result diversity:

```python
from evaluator.retrieval import DiversityRanker

ranker = DiversityRanker(diversity_lambda=0.5)
diverse_results = ranker.rerank(results, k=10)
```

## Performance Optimization

### Index Optimization

**FAISS IVF tuning**:

```yaml
vector_db:
  type: faiss_gpu
  nlist: 256  # sqrt(num_docs) is a good starting point
  nprobe: 16  # Higher = more accurate but slower
```

**Guidelines**:
- `nlist = sqrt(num_docs)` for IVF
- `nprobe = 10-20` for good accuracy/speed tradeoff
- Use HNSW for highest accuracy

### Batch Search

Search multiple queries at once:

```python
# Instead of:
results = [store.search(q, k=5) for q in queries]  # Slow

# Do:
results = store.batch_search(queries, k=5)  # Fast
```

### Caching

Cache frequent queries:

```yaml
retrieval:
  cache_queries: true
  cache_size: 10000
```

```python
from evaluator.retrieval import CachedRetriever

retriever = CachedRetriever(
    base_retriever=dense_retriever,
    cache_size=10000
)
```

### Pre-filtering

Filter before computing similarity:

```python
# Slow: Compute similarity for all, then filter
results = store.search(query, k=1000)
filtered = [r for r in results if r.metadata["specialty"] == "cardiology"][:10]

# Fast: Filter first, then compute similarity
results = store.search(
    query,
    k=10,
    filter={"specialty": "cardiology"}
)
```

## Evaluation Metrics

### Retrieval Quality

```python
from evaluator.evaluation import RetrievalMetrics

metrics = RetrievalMetrics(predictions, ground_truth)

print(f"MRR: {metrics.mrr():.4f}")
print(f"MAP: {metrics.map():.4f}")
print(f"NDCG@10: {metrics.ndcg(k=10):.4f}")
print(f"Recall@10: {metrics.recall(k=10):.4f}")
```

### Retrieval Latency

```python
from evaluator.benchmarks import measure_retrieval_latency

latency = measure_retrieval_latency(
    retriever=retriever,
    queries=test_queries,
    k=10
)

print(f"P50 latency: {latency.p50:.2f}ms")
print(f"P95 latency: {latency.p95:.2f}ms")
print(f"P99 latency: {latency.p99:.2f}ms")
```

## Example Configurations

### High Accuracy Setup

```yaml
vector_db:
  type: faiss_gpu
  index_type: HNSW
  
retrieval:
  strategy: rerank
  first_stage_k: 100
  reranker_model: cross-encoder/ms-marco-MiniLM-L-12-v2
  final_k: 10
```

### High Speed Setup

```yaml
vector_db:
  type: faiss_gpu
  index_type: IVF
  nlist: 100
  nprobe: 5  # Lower nprobe for speed
  
retrieval:
  strategy: dense
  batch_size: 128  # Large batches
```

### Production Setup

```yaml
vector_db:
  type: qdrant
  qdrant_url: http://qdrant-server:6333
  qdrant_collection: medical_docs
  
retrieval:
  strategy: hybrid
  dense_weight: 0.7
  sparse_weight: 0.3
  cache_queries: true
  cache_size: 50000
```

### Memory-Constrained Setup

```yaml
vector_db:
  type: faiss
  index_type: IVFPQ  # Product quantization for compression
  nlist: 100
  m: 8  # Subquantizers
  nbits: 8  # Bits per subquantizer
```

## Next Steps

- Learn about [Visualization](visualization.md) for analyzing retrieval results
- See [Configuration](configuration.md) for all retrieval options
- Try [GPU Management](gpu_management.md) for accelerating search
