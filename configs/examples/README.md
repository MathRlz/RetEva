# Example Configuration Files

This directory contains documented example configurations for common evaluation scenarios. Each configuration is heavily commented to explain the available options and their effects.

## Quick Start

Copy an example config and modify it for your needs:

```bash
cp configs/examples/basic_asr_retrieval.yaml configs/my_evaluation.yaml
python evaluate.py --config configs/my_evaluation.yaml
```

## Available Configurations

### 1. `basic_asr_retrieval.yaml` - Recommended Starting Point

**Use case:** First-time users, standard evaluations

The traditional ASR → Text Embedding → Retrieval pipeline:
- Speech is transcribed to text using Whisper
- Text is embedded using LaBSE
- Embeddings are used for retrieval

**Advantages:**
- Interpretable intermediate results (transcribed text)
- Easy to debug ASR and retrieval separately
- Well-understood pipeline

**Key settings:**
```yaml
model:
  pipeline_mode: asr_text_retrieval
  asr_model_type: whisper
  asr_model_name: openai/whisper-medium
  text_emb_model_type: labse
```

---

### 2. `audio_embedding_only.yaml` - Direct Audio Retrieval

**Use case:** End-to-end audio models, bypassing ASR

Directly embeds audio without intermediate text:
- Audio is encoded to embeddings by a single model
- No transcription step (no ASR errors)

**Advantages:**
- No ASR error propagation
- Can capture prosodic and acoustic features
- Potentially faster inference

**Requirements:**
- Pre-trained audio embedding model (CLAP-style or custom)
- Compatible text embeddings for the corpus

**Key settings:**
```yaml
model:
  pipeline_mode: audio_text_retrieval
  audio_emb_model_type: clap_style
  audio_emb_model_path: path/to/your/model.pt
  text_emb_model_type: clap_text
```

---

### 3. `hybrid_retrieval.yaml` - Maximum Accuracy

**Use case:** Production benchmarks, research evaluations

Combines multiple retrieval strategies:
- Dense retrieval (semantic similarity)
- Sparse retrieval (BM25 lexical matching)
- RRF fusion (Reciprocal Rank Fusion)
- Cross-encoder reranking

**Advantages:**
- Best retrieval accuracy
- Handles both semantic and lexical queries
- Reranking improves precision

**Trade-offs:**
- Slower than single-strategy retrieval
- More complex setup

**Key settings:**
```yaml
vector_db:
  retrieval_mode: hybrid
  hybrid_fusion_method: rrf
  rrf_k: 60
  reranker_mode: cross_encoder
  reranker_enabled: true
```

---

### 4. `fast_development.yaml` - Quick Iteration

**Use case:** Development, debugging, testing changes

Minimal configuration for fast feedback:
- Small models (whisper-tiny)
- CPU-only (no GPU required)
- Small batch sizes
- Verbose logging

**Advantages:**
- Runs on any machine
- Fast iteration cycles
- Easy debugging

**Trade-offs:**
- Lower accuracy
- Not representative of production

**Key settings:**
```yaml
model:
  asr_model_name: openai/whisper-tiny
  asr_device: cpu
  text_emb_device: cpu
data:
  batch_size: 4
logging:
  console_level: DEBUG
```

---

### 5. `multi_gpu_production.yaml` - Full-Scale Evaluation

**Use case:** Production runs on GPU servers

Optimized for multi-GPU throughput:
- Models distributed across GPUs
- Large batch sizes
- GPU-accelerated FAISS
- All caching enabled
- Best-quality models

**Requirements:**
- 2+ NVIDIA GPUs
- 16GB+ VRAM per GPU
- Significant storage for caches

**Key settings:**
```yaml
model:
  asr_device: cuda:0
  text_emb_device: cuda:1
data:
  batch_size: 64
  num_workers: 4
vector_db:
  type: faiss_gpu
  gpu_id: 0
```

---

## Configuration Reference

### Pipeline Modes

| Mode | Description | Metrics |
|------|-------------|---------|
| `asr_text_retrieval` | Speech → ASR → Text → Embeddings → Retrieval | WER, CER, MRR, MAP, NDCG, Recall |
| `audio_text_retrieval` | Speech → Audio Embeddings → Retrieval | MRR, MAP, NDCG, Recall |
| `asr_only` | Speech → ASR → Text (no retrieval) | WER, CER |

### ASR Models

| Type | Model Name | Size | Notes |
|------|------------|------|-------|
| `whisper` | `openai/whisper-tiny` | 39M | Fastest |
| `whisper` | `openai/whisper-base` | 74M | Fast |
| `whisper` | `openai/whisper-small` | 244M | Balanced |
| `whisper` | `openai/whisper-medium` | 769M | Recommended |
| `whisper` | `openai/whisper-large-v3` | 1.5B | Best accuracy |
| `wav2vec2` | `facebook/wav2vec2-large-960h` | 315M | English |
| `wav2vec2` | `jonatasgrosman/wav2vec2-large-xlsr-53-polish` | 315M | Polish |

### Text Embedding Models

| Type | Model Name | Dimensions | Notes |
|------|------------|------------|-------|
| `labse` | `sentence-transformers/LaBSE` | 768 | Multilingual |
| `jina_v4` | `jinaai/jina-embeddings-v4` | 1024 | Best semantic |
| `bge_m3` | `BAAI/bge-m3` | 1024 | Multilingual |
| `nemotron` | NVIDIA Nemotron | 1024 | High quality |

### Retrieval Modes

| Mode | Description |
|------|-------------|
| `dense` | Neural embedding similarity only |
| `sparse` | BM25 lexical matching only |
| `hybrid` | Combination of dense + sparse |

### Fusion Methods (for hybrid)

| Method | Description |
|--------|-------------|
| `weighted` | Linear combination: `α * dense + (1-α) * sparse` |
| `rrf` | Reciprocal Rank Fusion: combines rankings |

---

## Tips

### Device Configuration

The framework auto-configures devices if you use `config.with_auto_devices()`. For manual control:

```yaml
# Single GPU
asr_device: cuda:0
text_emb_device: cuda:0

# Multi-GPU
asr_device: cuda:0
text_emb_device: cuda:1

# CPU only
asr_device: cpu
text_emb_device: cpu
```

### Memory Management

If running out of GPU memory:
1. Reduce `batch_size`
2. Use smaller models
3. Distribute models across GPUs
4. Use CPU for some components

### Caching Strategy

For repeated experiments with the same models:
```yaml
cache:
  enabled: true
  cache_transcriptions: true  # Reuse ASR results
  cache_embeddings: true      # Reuse embeddings
```

For testing different models:
```yaml
cache:
  enabled: true
  cache_transcriptions: false  # Different ASR models
  cache_embeddings: false      # Different embedding models
  cache_vector_db: false       # Different retrieval settings
```

---

## See Also

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Full system documentation
- [README.md](../../README.md) - Project overview
- [evaluate.py](../../evaluate.py) - Main entry point
