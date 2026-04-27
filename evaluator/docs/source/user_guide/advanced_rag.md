# Advanced RAG Techniques

## Overview

This guide covers advanced Retrieval-Augmented Generation (RAG) techniques implemented in the evaluator framework, including multi-vector retrieval, query expansion, pseudo-relevance feedback, adaptive fusion, and diversity enhancement. These techniques can significantly improve retrieval quality in specialized domains like medical literature.

## Multi-Vector Retrieval

### Concept

Instead of using a single embedding per document, multi-vector retrieval creates multiple embeddings for different aspects or chunks of a document, enabling more nuanced matching.

### Strategies

#### 1. Chunk-Level Retrieval

Split documents into semantic chunks and retrieve the most relevant chunks:

```yaml
vector_db:
  multi_vector:
    enabled: true
    strategy: "chunk"
    chunk_size: 512  # tokens
    chunk_overlap: 50
    max_chunks_per_doc: 10
    aggregation: "max"  # max, mean, or first
```

**When to use**:
- Long documents (medical articles, clinical guidelines)
- When relevant information is localized in specific sections
- To improve precision for specific queries

#### 2. Aspect-Based Retrieval

Create separate embeddings for different document aspects:

```yaml
vector_db:
  multi_vector:
    enabled: true
    strategy: "aspect"
    aspects:
      - "title"
      - "abstract"
      - "methods"
      - "results"
      - "conclusions"
    aspect_weights:
      title: 0.3
      abstract: 0.4
      methods: 0.1
      results: 0.1
      conclusions: 0.1
```

**When to use**:
- Structured documents (research papers)
- When different sections have different relevance
- For aspect-specific search (e.g., "find papers with novel methods")

#### 3. Entity-Focused Retrieval

Extract and embed key entities separately:

```yaml
vector_db:
  multi_vector:
    enabled: true
    strategy: "entity"
    entity_types:
      - "DISEASE"
      - "DRUG"
      - "PROCEDURE"
      - "SYMPTOM"
    entity_weight: 0.3
    context_weight: 0.7
```

**When to use**:
- Medical domain with important entities (diseases, drugs)
- When entity matching is critical
- For structured knowledge extraction

### Aggregation Methods

**Max Pooling**: Use highest similarity score across all vectors
```yaml
aggregation: "max"  # Best for finding any relevant chunk
```

**Mean Pooling**: Average similarity scores
```yaml
aggregation: "mean"  # Best for overall document relevance
```

**Weighted Combination**: Custom weights per vector type
```yaml
aggregation: "weighted"
weights: [0.3, 0.5, 0.2]  # Per chunk/aspect
```

## Query Expansion

### Overview

Query expansion augments the original query with related terms, synonyms, or concepts to improve recall and handle vocabulary mismatch.

### Synonym-Based Expansion

Expand queries using medical thesaurus (UMLS, MeSH):

```yaml
query_expansion:
  enabled: true
  method: "synonym"
  source: "umls"  # or "mesh", "snomed"
  max_synonyms: 5
  synonym_weight: 0.3  # Original query: 1.0, synonyms: 0.3
```

**Example**:
- Original: "heart attack"
- Expanded: "heart attack OR myocardial infarction OR MI OR cardiac arrest OR coronary event"

**Configuration**:
```yaml
query_expansion:
  enabled: true
  method: "synonym"
  sources:
    - name: "umls"
      weight: 0.4
    - name: "mesh"
      weight: 0.3
  max_terms: 10
  boost_original: 2.0  # Give more weight to original terms
```

### Embedding-Based Expansion

Use embedding similarity to find related terms:

```yaml
query_expansion:
  enabled: true
  method: "embedding"
  embedding_model: "biogpt"
  num_similar_terms: 5
  similarity_threshold: 0.7
```

**How it works**:
1. Embed query terms
2. Find similar terms in embedding space
3. Add high-similarity terms to query

**Example**:
- Original: "diabetes treatment"
- Expanded: "diabetes treatment insulin metformin glucose_control glycemic_management"

### LLM-Based Expansion

Use LLM to generate related concepts:

```yaml
query_expansion:
  enabled: true
  method: "llm"
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
  expansion_prompt: |
    Expand this medical query with related terms and concepts:
    "{query}"
    
    Return a list of related terms (comma-separated):
```

### Pseudo-Relevance Feedback (PRF)

Expand query using terms from top retrieved documents:

```yaml
query_expansion:
  enabled: true
  method: "prf"
  num_feedback_docs: 3
  num_expansion_terms: 5
  term_selection: "tfidf"  # or "embedding_similarity"
  feedback_weight: 0.4
```

**How it works**:
1. Retrieve top K documents with original query
2. Extract most important terms from these documents
3. Add extracted terms to query
4. Re-retrieve with expanded query

**When to use**:
- When initial retrieval is reasonably good
- To improve recall on second pass
- For exploratory search

**Example**:
```python
# First pass
query = "COPD symptoms"
docs = retrieve(query, top_k=3)

# Extract terms from top docs
expansion_terms = extract_key_terms(docs)  # ["dyspnea", "wheezing", "bronchodilator"]

# Second pass
expanded_query = query + " " + " ".join(expansion_terms)
final_docs = retrieve(expanded_query, top_k=10)
```

## Adaptive Fusion

### Dynamic Weight Adjustment

Automatically adjust fusion weights based on query characteristics:

```yaml
adaptive_fusion:
  enabled: true
  strategy: "query_based"
  adjustment_method: "confidence"
```

### Query-Based Adaptation

Adjust weights based on query type:

```yaml
adaptive_fusion:
  enabled: true
  rules:
    - condition: "query_length < 5"
      audio_weight: 0.3
      text_weight: 0.7
      reason: "Short queries benefit from text semantics"
    
    - condition: "contains_medical_terms"
      audio_weight: 0.4
      text_weight: 0.6
      reason: "Medical terms better in text"
    
    - condition: "is_acoustic_query"
      audio_weight: 0.8
      text_weight: 0.2
      reason: "Acoustic features prioritized"
```

### Confidence-Based Adaptation

Adjust weights based on retrieval confidence:

```yaml
adaptive_fusion:
  enabled: true
  strategy: "confidence"
  confidence_threshold: 0.8
  low_confidence_action: "increase_text_weight"
```

**How it works**:
1. Perform initial retrieval
2. Compute confidence score (e.g., top similarity score)
3. If confidence < threshold, adjust weights
4. Re-retrieve with adjusted weights

### Result-Based Adaptation

Learn optimal weights from retrieval results:

```yaml
adaptive_fusion:
  enabled: true
  strategy: "result_based"
  learning_rate: 0.1
  update_frequency: 10  # Update weights every 10 queries
```

## Diversity Enhancement

### Maximum Marginal Relevance (MMR)

Balance relevance and diversity in results:

```yaml
diversity:
  enabled: true
  method: "mmr"
  lambda_param: 0.7  # 1.0 = pure relevance, 0.0 = pure diversity
  similarity_threshold: 0.9  # Remove near-duplicates
```

**How it works**:
```python
# MMR algorithm
selected = []
while len(selected) < k:
    # Score = relevance - diversity_penalty
    scores = lambda * relevance_scores - (1-lambda) * max_similarity_to_selected
    next_doc = argmax(scores)
    selected.append(next_doc)
```

**When to use**:
- When documents are highly similar
- For comprehensive information gathering
- To avoid redundant results

### Clustering-Based Diversity

Ensure results from different clusters:

```yaml
diversity:
  enabled: true
  method: "clustering"
  num_clusters: 5
  docs_per_cluster: 2
  clustering_algorithm: "kmeans"
```

### Aspect-Based Diversity

Ensure diverse aspects are covered:

```yaml
diversity:
  enabled: true
  method: "aspect"
  aspects:
    - "causes"
    - "symptoms"
    - "diagnosis"
    - "treatment"
    - "prevention"
  min_docs_per_aspect: 1
```

## Complete Configuration Examples

### Example 1: Advanced Medical Retrieval

```yaml
# configs/advanced_medical_rag.yaml

# Multi-vector retrieval
vector_db:
  multi_vector:
    enabled: true
    strategy: "chunk"
    chunk_size: 512
    chunk_overlap: 50
    aggregation: "max"

# Query expansion
query_expansion:
  enabled: true
  method: "prf"
  num_feedback_docs: 3
  num_expansion_terms: 5
  feedback_weight: 0.4

# Adaptive fusion
adaptive_fusion:
  enabled: true
  strategy: "query_based"
  rules:
    - condition: "query_length < 5"
      audio_weight: 0.2
      text_weight: 0.8

# Diversity
diversity:
  enabled: true
  method: "mmr"
  lambda_param: 0.7
```

### Example 2: Entity-Focused Medical Search

```yaml
# configs/entity_focused_search.yaml

vector_db:
  multi_vector:
    enabled: true
    strategy: "entity"
    entity_types:
      - "DISEASE"
      - "DRUG"
      - "PROCEDURE"
    entity_weight: 0.4
    context_weight: 0.6

query_expansion:
  enabled: true
  method: "synonym"
  source: "umls"
  max_synonyms: 5
  synonym_weight: 0.3

diversity:
  enabled: true
  method: "aspect"
  aspects:
    - "diagnosis"
    - "treatment"
    - "prognosis"
```

### Example 3: High-Recall Exploration

```yaml
# configs/high_recall_exploration.yaml

# Multi-vector for comprehensive coverage
vector_db:
  multi_vector:
    enabled: true
    strategy: "aspect"
    aspects:
      - "title"
      - "abstract"
      - "full_text"
    aggregation: "mean"

# PRF for expansion
query_expansion:
  enabled: true
  method: "prf"
  num_feedback_docs: 5
  num_expansion_terms: 10
  feedback_weight: 0.5

# Clustering for diversity
diversity:
  enabled: true
  method: "clustering"
  num_clusters: 5
  docs_per_cluster: 3
```

## Performance Considerations

### Latency Impact

| Technique | Overhead | Mitigation |
|-----------|----------|------------|
| Multi-vector | +50-100% | Cache embeddings, batch processing |
| Query expansion (PRF) | +100% | Async first-pass retrieval |
| Query expansion (LLM) | +1-3s | Use local LLM, caching |
| Adaptive fusion | +5-10% | Pre-compute query features |
| Diversity (MMR) | +20-30% | Efficient similarity computation |

### Memory Usage

| Technique | Memory Impact | Mitigation |
|-----------|---------------|------------|
| Multi-vector | +2-10x | Store only top chunks |
| Query expansion | Minimal | Stream processing |
| Adaptive fusion | Minimal | Stateless rules |
| Diversity | +10-20% | In-place reranking |

### Quality vs Cost

```yaml
# High quality (slower, more expensive)
advanced_rag:
  multi_vector: true
  query_expansion: "llm"
  adaptive_fusion: true
  diversity: true

# Balanced (recommended)
advanced_rag:
  multi_vector: true
  query_expansion: "prf"
  diversity: true

# Fast (lower quality)
advanced_rag:
  query_expansion: "synonym"
```

## Integration Patterns

### Full Stack RAG

```yaml
# All features enabled
vector_db:
  retrieval_mode: "hybrid"
  embedding_fusion:
    enabled: true
    method: "weighted"
  multi_vector:
    enabled: true
    strategy: "chunk"

query_optimization:
  enabled: true
  method: "rewrite"

query_expansion:
  enabled: true
  method: "prf"

adaptive_fusion:
  enabled: true

diversity:
  enabled: true
  method: "mmr"

reranking:
  enabled: true
  model: "cross-encoder"
```

### Pipeline Order

Recommended order for combining techniques:

1. **Query Optimization** (rewrite/HyDE/decompose)
2. **Query Expansion** (synonyms/PRF)
3. **Multi-Vector Retrieval** (chunk/aspect/entity)
4. **Adaptive Fusion** (adjust weights)
5. **Diversity Enhancement** (MMR/clustering)
6. **Reranking** (cross-encoder)

## Best Practices

### 1. Start Simple, Add Complexity

```yaml
# Start with baseline
retrieval_mode: "dense"

# Add one technique at a time
+ query_expansion: "synonym"
+ diversity: "mmr"
+ multi_vector: "chunk"
```

Measure impact of each addition.

### 2. Use PRF Carefully

PRF assumes initial retrieval is good. If not:
- Use query optimization first
- Increase num_feedback_docs
- Lower feedback_weight

### 3. Tune Diversity Parameter

Too much diversity = irrelevant results:
```yaml
diversity:
  lambda_param: 0.7  # Start here
  # Increase for more relevance: 0.8-0.9
  # Decrease for more diversity: 0.5-0.6
```

### 4. Medical Domain Tuning

For medical queries:
```yaml
# Use medical thesaurus
query_expansion:
  method: "synonym"
  source: "umls"

# Entity-focused retrieval
multi_vector:
  strategy: "entity"
  entity_types: ["DISEASE", "DRUG"]

# Ensure diverse medical aspects
diversity:
  method: "aspect"
  aspects: ["symptoms", "treatment", "diagnosis"]
```

### 5. Monitor Performance

Track metrics for each technique:
```python
results = {
    "baseline": evaluate(baseline_config),
    "+expansion": evaluate(with_expansion),
    "+multiVector": evaluate(with_multi_vector),
    "+diversity": evaluate(full_config)
}
```

## Troubleshooting

### Issue: PRF degrading performance

**Causes**:
- Initial retrieval is poor
- Feedback weight too high
- Too many expansion terms

**Solutions**:
1. Improve initial retrieval first (query optimization)
2. Reduce feedback_weight: `0.4 -> 0.2`
3. Reduce num_expansion_terms: `10 -> 5`
4. Increase num_feedback_docs: `3 -> 5`

### Issue: High latency with multi-vector

**Solutions**:
1. Reduce max_chunks_per_doc: `10 -> 5`
2. Use "first" or "max" aggregation (faster than "mean")
3. Cache chunk embeddings
4. Use approximate nearest neighbor search

### Issue: Over-diversification

**Symptoms**: Results less relevant, lower precision

**Solutions**:
1. Increase lambda_param: `0.7 -> 0.8`
2. Reduce num_clusters or docs_per_cluster
3. Disable diversity for high-precision queries

### Issue: Query expansion adding noise

**Solutions**:
1. Increase similarity_threshold
2. Reduce max_synonyms/num_expansion_terms
3. Boost original query terms
4. Use more conservative expansion method

## See Also

- [Embedding Fusion](embedding_fusion.md) - Multi-modal embedding combination
- [Query Optimization](query_optimization.md) - LLM-based query enhancement
- [Configuration Reference](configuration_reference.md) - Complete config options
- [Tutorial Notebook 17](../../notebooks/17_hybrid_retrieval_deep_dive.ipynb) - Hybrid retrieval experiments
- [Tutorial Notebook 19](../../notebooks/19_full_rag_ablation_study.ipynb) - Full RAG stack evaluation
