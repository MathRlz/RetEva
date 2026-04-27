# Configuration Reference

## Overview

This document provides a complete reference for all configuration options in the evaluator framework. All configurations are specified in YAML format and loaded via `EvaluationConfig.from_yaml()`.

## Configuration Structure

```yaml
# Top-level configuration structure
experiment_name: "my_experiment"
dataset: {...}
vector_db: {...}
query_optimization: {...}
evaluation: {...}
judge: {...}
output: {...}
logging: {...}
```

## Dataset Configuration

```yaml
dataset:
  name: "pubmed_qa"          # Dataset name
  path: "data/pubmed_qa"     # Path to dataset files
  loader: "pubmed"           # Loader type: pubmed, custom, local
  
  # Optional filters
  subset: "train"            # Dataset subset
  max_samples: 1000          # Limit number of samples
  shuffle: true              # Shuffle dataset
  random_seed: 42            # Seed for reproducibility
  
  # Query/document fields
  query_field: "question"    # Field containing queries
  doc_field: "context"       # Field containing documents
  answer_field: "answer"     # Field containing ground truth
  
  # Audio-specific
  audio_field: "audio_path"  # Field with audio file paths
  transcript_field: "text"   # Field with transcripts
```

## Vector Database Configuration

### Basic Settings

```yaml
vector_db:
  type: "chroma"                    # chroma, faiss, or pinecone
  collection_name: "medical_docs"   # Collection identifier
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dimension: 384          # Embedding size
  
  # Retrieval settings
  retrieval_mode: "dense"           # dense, sparse, or hybrid
  top_k: 10                         # Number of results to retrieve
  similarity_metric: "cosine"       # cosine, euclidean, or dot_product
```

### Embedding Fusion Configuration

```yaml
vector_db:
  embedding_fusion:
    enabled: true                   # Enable multi-modal fusion
    method: "weighted"              # weighted, concatenate, max_pool, average
    
    # Weights (for weighted method)
    audio_weight: 0.6
    text_weight: 0.4
    
    # Normalization
    normalize: true                 # L2 normalize embeddings
    
    # Dimension handling
    use_pca: false                  # Use PCA for dimension reduction
    pca_dimensions: 512             # Target dimension after PCA
    project_to_common_dim: null     # Project to common dimension
    
    # Model specification
    audio_model: "wav2vec2-base"
    text_model: "all-MiniLM-L6-v2"
```

### Hybrid Retrieval Configuration

```yaml
vector_db:
  retrieval_mode: "hybrid"
  
  # Hybrid parameters
  hybrid_alpha: 0.7                 # Dense weight (0-1), sparse = 1-alpha
  fusion_method: "weighted"         # weighted, rrf, or adaptive
  
  # RRF parameters
  rrf_k: 60                         # RRF constant (default: 60)
  
  # BM25 parameters (sparse retrieval)
  bm25_k1: 1.5                      # Term frequency saturation
  bm25_b: 0.75                      # Length normalization
  bm25_epsilon: 0.25                # IDF floor
  
  # Adaptive fusion
  adaptive_alpha:
    enabled: false
    method: "query_length"          # query_length, confidence, or learned
```

### Multi-Vector Retrieval

```yaml
vector_db:
  multi_vector:
    enabled: true
    strategy: "chunk"               # chunk, aspect, or entity
    
    # Chunk-level settings
    chunk_size: 512                 # Tokens per chunk
    chunk_overlap: 50               # Overlap between chunks
    max_chunks_per_doc: 10          # Max chunks to store
    
    # Aggregation
    aggregation: "max"              # max, mean, weighted, or first
    aggregation_weights: [0.3, 0.5, 0.2]  # For weighted aggregation
    
    # Aspect-based settings
    aspects:
      - "title"
      - "abstract"
      - "full_text"
    aspect_weights:
      title: 0.3
      abstract: 0.5
      full_text: 0.2
    
    # Entity-based settings
    entity_types:
      - "DISEASE"
      - "DRUG"
      - "PROCEDURE"
    entity_weight: 0.4
    context_weight: 0.6
```

### Reranking Configuration

```yaml
vector_db:
  use_reranking: true
  reranking_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  reranking_top_k: 100              # Initial retrieval size
  final_top_k: 10                   # After reranking
  reranking_batch_size: 32
```

### Diversity Enhancement

```yaml
vector_db:
  diversity:
    enabled: true
    method: "mmr"                   # mmr, clustering, or aspect
    
    # MMR parameters
    lambda_param: 0.7               # Relevance vs diversity (0-1)
    similarity_threshold: 0.9       # Near-duplicate threshold
    
    # Clustering parameters
    num_clusters: 5
    docs_per_cluster: 2
    clustering_algorithm: "kmeans"
    
    # Aspect-based diversity
    aspects:
      - "symptoms"
      - "diagnosis"
      - "treatment"
    min_docs_per_aspect: 1
```

## Query Optimization Configuration

```yaml
query_optimization:
  enabled: true
  method: "rewrite"                 # rewrite, hyde, decompose, or multi_query
  
  # LLM configuration
  llm:
    provider: "openai"              # openai, vllm, ollama, or anthropic
    model: "gpt-3.5-turbo"
    api_key: "${OPENAI_API_KEY}"
    api_base: null                  # For local LLMs
    temperature: 0.7
    max_tokens: 256
    timeout: 30                     # Request timeout
  
  # Query rewriting
  num_iterations: 2                 # Iterative refinement steps
  rewrite_prompt: |
    Given query: "{query}"
    Context: {context}
    Rewrite query:
  
  # HyDE
  hyde_prompt: |
    Write a detailed answer to: "{query}"
  
  # Query decomposition
  max_sub_queries: 3
  decompose_prompt: |
    Decompose "{query}" into sub-queries.
    Return JSON: ["sub1", "sub2", ...]
  
  # Multi-query generation
  num_queries: 3
  fusion_method: "rrf"
  
  # Caching
  cache_llm_calls: true
  cache_ttl: 86400                  # Cache TTL in seconds
  
  # Sampling (for development)
  sample_rate: 1.0                  # 1.0 = all queries
```

## Query Expansion Configuration

```yaml
query_expansion:
  enabled: true
  method: "prf"                     # synonym, embedding, llm, or prf
  
  # Synonym-based expansion
  source: "umls"                    # umls, mesh, snomed, or wordnet
  max_synonyms: 5
  synonym_weight: 0.3
  boost_original: 2.0
  
  # Embedding-based expansion
  embedding_model: "biogpt"
  num_similar_terms: 5
  similarity_threshold: 0.7
  
  # LLM-based expansion
  llm: {...}                        # Same as query_optimization.llm
  expansion_prompt: |
    Expand query "{query}" with related medical terms:
  
  # Pseudo-relevance feedback
  num_feedback_docs: 3
  num_expansion_terms: 5
  term_selection: "tfidf"           # tfidf or embedding_similarity
  feedback_weight: 0.4
```

## Adaptive Fusion Configuration

```yaml
adaptive_fusion:
  enabled: true
  strategy: "query_based"           # query_based, confidence, or result_based
  
  # Query-based adaptation
  rules:
    - condition: "query_length < 5"
      audio_weight: 0.3
      text_weight: 0.7
      reason: "Short queries favor text"
    
    - condition: "contains_medical_terms"
      audio_weight: 0.4
      text_weight: 0.6
    
    - condition: "is_acoustic_query"
      audio_weight: 0.8
      text_weight: 0.2
  
  # Confidence-based adaptation
  confidence_threshold: 0.8
  low_confidence_action: "increase_text_weight"
  
  # Result-based adaptation (learning)
  learning_rate: 0.1
  update_frequency: 10              # Updates per N queries
```

## Evaluation Configuration

```yaml
evaluation:
  metrics:
    - "mrr"                         # Mean Reciprocal Rank
    - "ndcg"                        # Normalized Discounted Cumulative Gain
    - "map"                         # Mean Average Precision
    - "recall"                      # Recall@K
    - "precision"                   # Precision@K
  
  # Metric-specific parameters
  ndcg_k: 10                        # NDCG@10
  recall_k: [5, 10, 20]             # Recall at multiple K
  precision_k: 10
  
  # Statistical testing
  run_significance_tests: true
  significance_test: "paired_t_test"
  confidence_level: 0.95
  
  # Per-sample tracking
  track_per_sample: true
  save_predictions: true
```

## Judge Configuration

```yaml
judge:
  enabled: true
  
  # LLM provider
  provider: "openai"                # openai, vllm, ollama, anthropic
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  api_base: null
  temperature: 0.3                  # Lower for consistency
  max_tokens: 256
  
  # System prompt
  system_prompt: |
    You are evaluating medical literature retrieval quality.
  
  # Aspects to evaluate
  aspects:
    - name: "relevance"
      weight: 0.4
      description: "How relevant is this to the query?"
      scoring_guidelines: |
        5: Highly relevant
        4: Mostly relevant
        3: Partially relevant
        2: Minimally relevant
        1: Not relevant
    
    - name: "accuracy"
      weight: 0.3
      description: "Medical accuracy"
    
    - name: "completeness"
      weight: 0.2
      description: "Completeness of information"
    
    - name: "clarity"
      weight: 0.1
      description: "Clarity of presentation"
  
  # Score aggregation
  aggregation_method: "weighted_average"  # weighted_average, minimum, harmonic_mean
  
  # Few-shot learning
  few_shot_examples:
    - query: "What are symptoms of diabetes?"
      document: "Diabetes symptoms include increased thirst and urination."
      scores:
        relevance: 5
        accuracy: 5
        completeness: 3
      explanation: "Relevant and accurate but incomplete."
  
  # Consistency & calibration
  consistency_test:
    enabled: false
    num_trials: 5
    max_std_dev: 0.5
  
  calibration:
    enabled: false
    test_cases: [...]
  
  # Cost management
  sample_rate: 1.0                  # Judge all results
  cache_results: true
  batch_size: 1                     # Batch multiple judgments
  require_explanation: true
```

## Output Configuration

```yaml
output:
  # Output directory
  results_dir: "evaluation_results"
  experiment_subdir: true           # Create subdirs per experiment
  
  # File formats
  save_json: true
  save_csv: true
  save_html: true
  
  # Detailed outputs
  save_predictions: true
  save_embeddings: false            # Large files
  save_intermediate_results: false
  
  # Visualization
  generate_plots: true
  plot_formats: ["png", "pdf"]
  
  # Compression
  compress_outputs: false
  compression_format: "gzip"
```

## Audio Synthesis Configuration

```yaml
audio_synthesis:
  enabled: false
  provider: "piper"                 # piper, xtts_v2, mms
  voice: "en_US-lessac-medium"      # Piper voice / XTTS speaker wav / MMS model id
  language: "en"                    # Language code
  sample_rate: 16000
  speed: 1.0
  pitch: 1.0
  volume: 1.0
  seed: 42
  output_dir: "prepared_benchmarks/audio"
  cache_dir: null
```

Provider-specific notes:
- `piper`: requires Piper CLI and local voice ONNX files
- `xtts_v2`: requires Coqui `TTS` package, supports multilingual voice cloning
- `mms`: uses `transformers` + `torch` with `facebook/mms-tts-*` checkpoints

## Service Runtime Configuration

```yaml
service_runtime:
  startup_mode: "lazy"        # lazy | eager
  offload_policy: "on_finish" # on_finish | never
```

Policy notes:
- `startup_mode`: controls startup behavior for model services (`eager` logs and warms pipeline-bound services immediately).
- `offload_policy`: controls service shutdown memory behavior (`on_finish` offloads; `never` keeps models resident until process exit).

## Logging Configuration

```yaml
logging:
  level: "INFO"                     # DEBUG, INFO, WARNING, ERROR
  log_file: "logs/evaluation.log"
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # Console output
  console_level: "INFO"
  colored_logs: true
  
  # Progress tracking
  show_progress_bars: true
  progress_bar_style: "tqdm"
  
  # Detailed logging
  log_gpu_memory: false
  log_timing: true
  log_intermediate_results: false
```

## Configuration Presets

The framework includes 18 built-in configuration presets:

### Basic Presets

```yaml
# preset: baseline_dense
vector_db:
  retrieval_mode: "dense"
  
# preset: baseline_sparse
vector_db:
  retrieval_mode: "sparse"
  
# preset: baseline_hybrid
vector_db:
  retrieval_mode: "hybrid"
  hybrid_alpha: 0.7
```

### Fusion Presets

```yaml
# preset: fusion_balanced
vector_db:
  embedding_fusion:
    enabled: true
    method: "weighted"
    audio_weight: 0.5
    text_weight: 0.5

# preset: fusion_audio_emphasis
vector_db:
  embedding_fusion:
    enabled: true
    audio_weight: 0.7
    text_weight: 0.3

# preset: fusion_text_emphasis
vector_db:
  embedding_fusion:
    enabled: true
    audio_weight: 0.3
    text_weight: 0.7
```

### Query Optimization Presets

```yaml
# preset: query_rewrite_iterative
query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 2

# preset: query_hyde
query_optimization:
  enabled: true
  method: "hyde"

# preset: query_decompose
query_optimization:
  enabled: true
  method: "decompose"
  max_sub_queries: 3
```

### Advanced Presets

```yaml
# preset: advanced_rag_full
vector_db:
  retrieval_mode: "hybrid"
  embedding_fusion:
    enabled: true
  multi_vector:
    enabled: true
  diversity:
    enabled: true
query_optimization:
  enabled: true
judge:
  enabled: true

# preset: medical_domain_best
# Optimized for medical retrieval
vector_db:
  retrieval_mode: "hybrid"
  hybrid_alpha: 0.7
  embedding_fusion:
    enabled: true
    audio_weight: 0.4
    text_weight: 0.6
query_expansion:
  enabled: true
  method: "synonym"
  source: "umls"
```

## Loading Configurations

### From YAML File

```python
from evaluator.config import EvaluationConfig

config = EvaluationConfig.from_yaml("configs/my_config.yaml")
```

### From Preset

```python
config = EvaluationConfig.from_preset("medical_domain_best")
```

### Modifying Configuration

```python
config = EvaluationConfig.from_yaml("configs/base.yaml")

# Override specific values
config.vector_db.retrieval_mode = "hybrid"
config.query_optimization.enabled = True
config.judge.model = "gpt-3.5-turbo"

# Save modified config
config.to_yaml("configs/modified.yaml")
```

### Merging Configurations

```python
base = EvaluationConfig.from_yaml("configs/base.yaml")
overrides = EvaluationConfig.from_yaml("configs/overrides.yaml")

config = base.merge(overrides)
```

## Environment Variables

Use environment variables for sensitive data:

```yaml
query_optimization:
  llm:
    api_key: "${OPENAI_API_KEY}"

judge:
  api_key: "${OPENAI_API_KEY}"
```

Set variables:
```bash
export OPENAI_API_KEY="sk-..."
```

## Validation

Configurations are validated on load:

```python
try:
    config = EvaluationConfig.from_yaml("configs/invalid.yaml")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

Common validation errors:
- Missing required fields
- Invalid enum values
- Inconsistent settings (e.g., hybrid_alpha not in [0, 1])
- Incompatible feature combinations

## Configuration Schema

Full schema in JSON Schema format:

```python
from evaluator.config import EvaluationConfig

schema = EvaluationConfig.schema()
print(schema)
```

## Examples

### Minimal Configuration

```yaml
# configs/minimal.yaml
experiment_name: "minimal_test"
dataset:
  name: "pubmed_qa"
  path: "data/pubmed_qa"
vector_db:
  type: "chroma"
  collection_name: "docs"
```

### Full-Featured Configuration

```yaml
# configs/full_featured.yaml
experiment_name: "full_rag_experiment"

dataset:
  name: "medical_audio"
  path: "data/medical"
  max_samples: 1000

vector_db:
  type: "chroma"
  retrieval_mode: "hybrid"
  hybrid_alpha: 0.7
  
  embedding_fusion:
    enabled: true
    method: "weighted"
    audio_weight: 0.6
    text_weight: 0.4
  
  multi_vector:
    enabled: true
    strategy: "chunk"
    chunk_size: 512
  
  use_reranking: true
  reranking_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  
  diversity:
    enabled: true
    method: "mmr"
    lambda_param: 0.7

query_optimization:
  enabled: true
  method: "rewrite"
  num_iterations: 2
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
    api_key: "${OPENAI_API_KEY}"

query_expansion:
  enabled: true
  method: "prf"
  num_feedback_docs: 3

judge:
  enabled: true
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  aspects:
    - name: "relevance"
      weight: 0.6
    - name: "accuracy"
      weight: 0.4

output:
  results_dir: "results"
  save_json: true
  generate_plots: true

logging:
  level: "INFO"
  show_progress_bars: true
```

## See Also

- [Embedding Fusion](embedding_fusion.md) - Multi-modal embedding configuration
- [Query Optimization](query_optimization.md) - LLM-based query enhancement
- [Advanced RAG](advanced_rag.md) - Multi-vector and expansion
- [LLM Judge](llm_judge.md) - LLM-based evaluation
- [Quick Start Guide](quickstart.md) - Getting started tutorial
