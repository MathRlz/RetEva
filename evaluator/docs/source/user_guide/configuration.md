# Configuration

Complete guide to configuring the Medical Speech Retrieval Evaluator.

## Configuration Methods

### 1. YAML Config Files

The most flexible method for production use:

```yaml
# config.yaml
model:
  pipeline_mode: asr_text_retrieval
  asr_model_type: whisper
  asr_model_name: openai/whisper-base
  asr_device: cuda:0
  text_emb_model_type: labse
  text_emb_device: cuda:1

data:
  questions_path: data/questions.json
  corpus_path: data/corpus.json
  batch_size: 32
  max_samples: null  # Process all samples

evaluation:
  k: 5
  metrics: [mrr, map, ndcg, recall]

cache:
  enabled: true
  cache_dir: .cache/evaluator
  cache_embeddings: true
  cache_asr_results: true

vector_db:
  type: faiss_gpu
  index_type: IVF
  nlist: 100

output:
  save_dir: evaluation_results
  save_predictions: true
  verbose: true

tracking:
  enabled: false
  backend: mlflow
  experiment_name: medical_retrieval
```

Load and use:

```python
from evaluator import evaluate_from_config

results = evaluate_from_config("config.yaml")
```

### 2. Python API

Configure programmatically:

```python
from evaluator.api import EvaluationConfig, EvaluationRunner

config = EvaluationConfig(
    model_config={
        "pipeline_mode": "asr_text_retrieval",
        "asr_model_type": "whisper",
        "text_emb_model_type": "labse"
    },
    data_config={
        "questions_path": "questions.json",
        "corpus_path": "corpus.json",
        "batch_size": 32
    }
)

runner = EvaluationRunner(config)
results = runner.run()
```

### 3. Presets

Use predefined configurations:

```python
from evaluator.api import EvaluationConfig

# Load preset and customize
config = EvaluationConfig.from_preset("whisper_labse")
config.data.batch_size = 64
config.cache.enabled = True
```

## Configuration Sections

### Model Configuration

```yaml
model:
  # Pipeline mode: asr_text_retrieval, audio_emb_retrieval, asr_only, text_only
  pipeline_mode: asr_text_retrieval
  
  # ASR Model
  asr_model_type: whisper  # whisper, faster_whisper, wav2vec2
  asr_model_name: openai/whisper-base
  asr_device: cuda:0
  asr_batch_size: 16
  
  # Text Embedding Model
  text_emb_model_type: labse  # labse, jina_v4, bge_m3, nemotron, clip
  text_emb_model_name: sentence-transformers/LaBSE
  text_emb_device: cuda:1
  text_emb_batch_size: 32
  
  # Audio Embedding Model (for audio_emb_retrieval mode)
  audio_emb_model_type: clap_style  # clap_style, attention_pool
  audio_emb_device: cuda:0
```

### Data Configuration

```yaml
data:
  # Input paths
  questions_path: data/questions.json
  corpus_path: data/corpus.json
  
  # Processing options
  batch_size: 32
  max_samples: null  # null = process all, or specify number
  shuffle: false
  num_workers: 4
  
  # Audio preprocessing
  audio_sample_rate: 16000
  audio_max_length: 30.0  # seconds
  normalize_audio: true
```

### Evaluation Configuration

```yaml
evaluation:
  # Retrieval parameters
  k: 5  # Number of results to retrieve
  
  # Metrics to compute
  metrics:
    - mrr
    - map
    - ndcg
    - recall
    - precision
  
  # Advanced options
  compute_sample_metrics: true  # Per-sample metrics
  return_predictions: true  # Include full predictions in output
```

### Cache Configuration

```yaml
cache:
  enabled: true
  cache_dir: .cache/evaluator
  
  # What to cache
  cache_embeddings: true
  cache_asr_results: true
  cache_predictions: false
  
  # Cache invalidation
  cache_ttl: 86400  # Time to live in seconds (1 day)
  ignore_cache: false  # Force recompute
```

### Audio Synthesis (TTS) Configuration

```yaml
audio_synthesis:
  enabled: true
  provider: xtts_v2          # piper, xtts_v2, mms
  language: en               # language code (used by xtts_v2/mms)
  voice: en_US-lessac-medium # Piper voice OR XTTS speaker wav path OR MMS model id
  sample_rate: 16000
  speed: 1.0
  pitch: 1.0
  cache_dir: .tts_cache
  output_dir: prepared_benchmarks/audio
```

Provider notes:
- `piper`: local/offline CLI-based synthesis with Piper voices
- `xtts_v2`: Coqui multilingual synthesis with optional speaker voice cloning (`voice=/path/to/speaker.wav`)
- `mms`: Hugging Face Meta MMS checkpoints (language-driven or explicit `voice=facebook/mms-tts-xxx`)

### Service Runtime Configuration

```yaml
service_runtime:
  startup_mode: lazy      # lazy | eager
  offload_policy: on_finish  # on_finish | never
```

Notes:
- `startup_mode=eager` touches all pipeline-bound services at startup for deterministic warm-up logs.
- `offload_policy=on_finish` frees service-held model memory when evaluation ends.

### Vector Database Configuration

```yaml
vector_db:
  type: faiss_gpu  # inmemory, faiss, faiss_gpu, chromadb, qdrant
  
  # FAISS-specific options
  index_type: IVF  # Flat, IVF, IVFPQ, HNSW
  nlist: 100  # Number of clusters for IVF
  nprobe: 10  # Number of clusters to search
  
  # ChromaDB options
  chromadb_path: ./chroma_db
  chromadb_collection: medical_docs
  
  # Qdrant options
  qdrant_path: ./qdrant_db
  qdrant_collection: medical_docs
```

### GPU Management Configuration

```yaml
gpu:
  # GPU pool for automatic device allocation
  gpu_pool:
    enabled: true
    device_ids: [0, 1, 2, 3]
    allocation_strategy: round_robin  # round_robin, memory_balanced, least_loaded
    
  # Memory management
  max_memory_per_gpu: 0.8  # Use up to 80% of GPU memory
  empty_cache_between_batches: false
```

### Tracking Configuration

```yaml
tracking:
  enabled: true
  backend: mlflow  # mlflow, wandb, tensorboard
  
  # MLflow settings
  mlflow:
    tracking_uri: ./mlruns
    experiment_name: medical_retrieval
    run_name: null  # Auto-generate if null
    
  # Log parameters
  log_config: true
  log_metrics: true
  log_artifacts: true
```

### Output Configuration

```yaml
output:
  save_dir: evaluation_results
  save_format: json  # json, yaml, pickle
  
  # What to save
  save_predictions: true
  save_sample_metrics: true
  save_config: true
  
  # Logging
  verbose: true
  log_level: INFO  # DEBUG, INFO, WARNING, ERROR
  log_file: null  # null or path to log file
```

## Complete Example

```yaml
# production_config.yaml
model:
  pipeline_mode: asr_text_retrieval
  asr_model_type: faster_whisper
  asr_model_name: large-v3
  asr_device: cuda:0
  text_emb_model_type: jina_v4
  text_emb_model_name: jinaai/jina-embeddings-v4
  text_emb_device: cuda:1

data:
  questions_path: /data/medical/questions.json
  corpus_path: /data/medical/corpus.json
  batch_size: 64
  num_workers: 8

evaluation:
  k: 10
  metrics: [mrr, map, ndcg, recall, precision]
  compute_sample_metrics: true

cache:
  enabled: true
  cache_dir: /cache/evaluator
  cache_embeddings: true
  cache_asr_results: true

vector_db:
  type: faiss_gpu
  index_type: IVF
  nlist: 256
  nprobe: 16

gpu:
  gpu_pool:
    enabled: true
    device_ids: [0, 1]
    allocation_strategy: memory_balanced

tracking:
  enabled: true
  backend: mlflow
  mlflow:
    tracking_uri: /mlruns
    experiment_name: production_eval

output:
  save_dir: /results/production
  save_predictions: true
  verbose: true
  log_level: INFO
```

## Command Line Overrides

Override config values from command line:

```bash
python evaluate.py --config config.yaml \
    --model.asr_device cuda:0 \
    --data.batch_size 64 \
    --evaluation.k 10 \
    --cache.enabled true
```

## Environment Variables

Set defaults via environment variables:

```bash
export EVALUATOR_DATA_DIR=/data/medical
export EVALUATOR_CACHE_DIR=/cache/evaluator
export EVALUATOR_RESULTS_DIR=/results
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Use in config:

```yaml
data:
  questions_path: ${EVALUATOR_DATA_DIR}/questions.json
  corpus_path: ${EVALUATOR_DATA_DIR}/corpus.json

cache:
  cache_dir: ${EVALUATOR_CACHE_DIR}

output:
  save_dir: ${EVALUATOR_RESULTS_DIR}
```

## Configuration Validation

Validate your config before running:

```python
from evaluator.api import EvaluationConfig

# Load and validate
config = EvaluationConfig.from_yaml("config.yaml")
config.validate()  # Raises error if invalid

# Check for issues
issues = config.check()
if issues:
    for issue in issues:
        print(f"Warning: {issue}")
```

## Best Practices

1. **Use presets as starting points**: Start with a preset and customize
2. **Enable caching**: Dramatically speeds up repeated evaluations
3. **Set appropriate batch sizes**: Balance speed and memory
4. **Use GPU pooling**: For multi-GPU systems
5. **Save configs with results**: For reproducibility
6. **Enable tracking**: For experiment management
7. **Validate configs**: Before running long evaluations

## Next Steps

- See [Models](models.md) for available model options
- Learn about [Pipeline Modes](pipelines.md)
- Configure [GPU Management](gpu_management.md)
- Set up [Retrieval Strategies](retrieval.md)
