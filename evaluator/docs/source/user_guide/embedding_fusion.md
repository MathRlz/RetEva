# Embedding Fusion

## Overview

Embedding fusion combines multiple embedding modalities (e.g., audio and text) to create richer semantic representations for retrieval. This technique is particularly valuable in medical audio datasets where both acoustic features and transcribed text provide complementary information.

The evaluator framework supports multiple fusion methods with configurable weights, allowing you to optimize retrieval performance for your specific use case.

## Why Embedding Fusion?

In medical audio datasets:
- **Audio embeddings** capture acoustic features like tone, rhythm, and pronunciation that may indicate medical conditions or urgency
- **Text embeddings** capture semantic meaning from transcribed medical terminology and concepts
- **Combined embeddings** leverage both modalities for more robust retrieval

### Use Cases

1. **Medical Audio Retrieval**: Combine acoustic features with transcribed medical terms
2. **Multi-modal Search**: Enable search by both sound characteristics and semantic content
3. **Robust Matching**: Improve retrieval accuracy by leveraging complementary signals
4. **Domain Adaptation**: Weight modalities based on data quality (e.g., noisy audio vs clean transcripts)

## Available Fusion Methods

### 1. Weighted Combination

Combines embeddings with learnable or fixed weights:

```python
fused = audio_weight * audio_embedding + text_weight * text_embedding
```

**When to use**:
- When one modality is more reliable than the other
- For domain adaptation (adjust weights based on data characteristics)
- When you want explicit control over modality importance

**Configuration**:
```yaml
embedding_fusion:
  enabled: true
  method: "weighted"
  audio_weight: 0.7
  text_weight: 0.3
  normalize: true
```

### 2. Concatenation

Concatenates audio and text embeddings into a single vector:

```python
fused = [audio_embedding, text_embedding]
```

**When to use**:
- When preserving all information from both modalities is critical
- For downstream models that can learn optimal combinations
- When dimension mismatch is not a concern

**Configuration**:
```yaml
embedding_fusion:
  enabled: true
  method: "concatenate"
  normalize: true
```

**Dimension Handling**: If embeddings have different dimensions, consider using PCA:
```yaml
embedding_fusion:
  enabled: true
  method: "concatenate"
  use_pca: true
  pca_dimensions: 512  # Target dimension after concatenation
```

### 3. Max Pooling

Takes element-wise maximum across embeddings:

```python
fused = max(audio_embedding, text_embedding)
```

**When to use**:
- For highlighting salient features from either modality
- When you want a sparse representation
- For robustness to noise in one modality

**Configuration**:
```yaml
embedding_fusion:
  enabled: true
  method: "max_pool"
  normalize: true
```

### 4. Average Pooling

Takes element-wise average across embeddings:

```python
fused = (audio_embedding + text_embedding) / 2
```

**When to use**:
- As a simple baseline (equivalent to weighted with 0.5/0.5)
- When both modalities are equally reliable
- For quick experiments

**Configuration**:
```yaml
embedding_fusion:
  enabled: true
  method: "average"
  normalize: true
```

## Configuration Guide

### Basic Setup

```yaml
vector_db:
  embedding_fusion:
    enabled: true
    method: "weighted"
    audio_weight: 0.5
    text_weight: 0.5
    normalize: true
```

### Audio-Emphasis Configuration

Use when acoustic features are more important (e.g., detecting voice anomalies):

```yaml
embedding_fusion:
  enabled: true
  method: "weighted"
  audio_weight: 0.8
  text_weight: 0.2
  normalize: true
```

### Text-Emphasis Configuration

Use when semantic content is more important (e.g., medical term matching):

```yaml
embedding_fusion:
  enabled: true
  method: "weighted"
  audio_weight: 0.2
  text_weight: 0.8
  normalize: true
```

### Balanced Multi-modal Configuration

Use when both modalities are equally important:

```yaml
embedding_fusion:
  enabled: true
  method: "average"
  normalize: true
```

## Dimension Mismatch Handling

When audio and text embeddings have different dimensions:

### Option 1: Concatenate with PCA

```yaml
embedding_fusion:
  enabled: true
  method: "concatenate"
  use_pca: true
  pca_dimensions: 512
  normalize: true
```

### Option 2: Project to Common Space

```yaml
embedding_fusion:
  enabled: true
  method: "weighted"
  project_to_common_dim: 384  # Project both to same dimension
  audio_weight: 0.5
  text_weight: 0.5
  normalize: true
```

### Option 3: Use Max/Average (Requires Same Dimensions)

Max and average pooling require embeddings of the same dimension. If dimensions differ, either:
1. Use different embedding models with matching dimensions
2. Pre-process embeddings with projection layers
3. Use weighted or concatenate methods instead

## Performance Considerations

### Memory Usage

- **Weighted/Max/Average**: Same memory as single embedding
- **Concatenate**: 2x memory (sum of both embedding dimensions)
- **PCA**: Additional overhead for transformation matrix

### Computational Cost

- **Weighted**: Minimal overhead (scalar multiplication + addition)
- **Concatenate**: Low overhead (array concatenation)
- **Max/Average**: Low overhead (element-wise operations)
- **PCA**: Moderate overhead (matrix multiplication)

### Retrieval Speed

Fusion happens at indexing time, so retrieval speed is unaffected. However:
- Larger embeddings (concatenate) may slow down distance computations
- Consider using dimensionality reduction for production systems

## Medical Domain Best Practices

### 1. Start with Balanced Weights

```yaml
embedding_fusion:
  enabled: true
  method: "weighted"
  audio_weight: 0.5
  text_weight: 0.5
  normalize: true
```

Run ablation studies to find optimal weights for your dataset.

### 2. Account for Transcription Quality

If using ASR-generated transcripts with high error rates:

```yaml
embedding_fusion:
  enabled: true
  method: "weighted"
  audio_weight: 0.7  # Trust audio more
  text_weight: 0.3
  normalize: true
```

### 3. Medical Terminology Emphasis

For queries focused on medical terms:

```yaml
embedding_fusion:
  enabled: true
  method: "weighted"
  audio_weight: 0.3
  text_weight: 0.7  # Trust text embeddings for technical terms
  normalize: true
```

### 4. Enable Normalization

Always normalize embeddings for consistent similarity computation:

```yaml
embedding_fusion:
  normalize: true  # Recommended for all methods
```

## Tuning Fusion Weights

### Grid Search

Test multiple weight combinations:

```python
weights = [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]

for audio_w, text_w in weights:
    config.vector_db.embedding_fusion.audio_weight = audio_w
    config.vector_db.embedding_fusion.text_weight = text_w
    
    results = evaluate(config)
    print(f"Weights {audio_w}/{text_w}: MRR={results.mrr:.4f}")
```

### Validation-Based Tuning

Use a validation set to find optimal weights:

```python
best_mrr = 0
best_weights = None

for audio_w in np.arange(0.1, 1.0, 0.1):
    text_w = 1.0 - audio_w
    # Evaluate on validation set
    val_mrr = evaluate_on_validation(audio_w, text_w)
    
    if val_mrr > best_mrr:
        best_mrr = val_mrr
        best_weights = (audio_w, text_w)

print(f"Best weights: {best_weights}, MRR: {best_mrr:.4f}")
```

## Troubleshooting

### Issue: Poor retrieval performance after fusion

**Solutions**:
1. Check if embeddings are normalized: Set `normalize: true`
2. Try different fusion methods (start with weighted)
3. Verify embedding dimensions match (for max/average)
4. Run ablation study to compare against single-modality baselines

### Issue: Dimension mismatch errors

**Solutions**:
1. Use `concatenate` method (handles different dimensions)
2. Enable PCA: `use_pca: true, pca_dimensions: 512`
3. Use models with same output dimensions
4. Add projection layers to align dimensions

### Issue: High memory usage

**Solutions**:
1. Use weighted/max/average instead of concatenate
2. Reduce embedding dimensions with PCA
3. Use lower-dimensional embedding models
4. Enable quantization if supported

## Examples

### Example 1: Medical Audio Dataset

```yaml
# configs/medical_audio_fusion.yaml
vector_db:
  embedding_fusion:
    enabled: true
    method: "weighted"
    audio_weight: 0.6  # Slightly favor acoustic features
    text_weight: 0.4
    normalize: true
```

### Example 2: High-Quality Transcripts

```yaml
# configs/clean_transcripts_fusion.yaml
vector_db:
  embedding_fusion:
    enabled: true
    method: "weighted"
    audio_weight: 0.3
    text_weight: 0.7  # Trust clean transcripts more
    normalize: true
```

### Example 3: Exploration Mode

```yaml
# configs/fusion_exploration.yaml
vector_db:
  embedding_fusion:
    enabled: true
    method: "concatenate"  # Preserve all information
    use_pca: true
    pca_dimensions: 768
    normalize: true
```

## Integration with Other Features

Embedding fusion works seamlessly with:
- **Hybrid Retrieval**: Fused embeddings for dense retrieval + BM25 for sparse
- **Query Optimization**: Optimize queries before fusion-based retrieval
- **Reranking**: Use fused embeddings for initial retrieval, then rerank

```yaml
# Full stack configuration
vector_db:
  retrieval_mode: "hybrid"
  embedding_fusion:
    enabled: true
    method: "weighted"
    audio_weight: 0.6
    text_weight: 0.4
  
query_optimization:
  enabled: true
  method: "rewrite"
```

## See Also

- [Query Optimization](query_optimization.md) - Optimize queries before retrieval
- [Advanced RAG](advanced_rag.md) - Multi-vector and expansion techniques
- [Configuration Reference](configuration_reference.md) - Complete config options
- [Tutorial Notebook 15](../../notebooks/15_embedding_fusion_experiments.ipynb) - Interactive experiments
