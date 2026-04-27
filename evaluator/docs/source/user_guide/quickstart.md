# Quick Start

This guide will help you run your first evaluation in minutes.

## Basic Evaluation

The simplest way to run an evaluation is using presets:

```python
from evaluator import evaluate_from_preset

# Run evaluation with Whisper ASR + LaBSE embeddings
results = evaluate_from_preset(
    "whisper_labse",
    data_path="questions.json",
    corpus_path="corpus.json"
)

# Print results
print(f"MRR: {results['MRR']:.4f}")
print(f"WER: {results['WER']:.4f}")
print(f"Recall@5: {results['Recall@5']:.4f}")
```

## Using the New API

The modern API provides a cleaner interface with the `EvaluationResults` class:

```python
from evaluator.api import EvaluationRunner, EvaluationConfig

# Create configuration
config = EvaluationConfig.from_preset("whisper_labse")
config.data.questions_path = "questions.json"
config.data.corpus_path = "corpus.json"

# Run evaluation
runner = EvaluationRunner(config)
results = runner.run()

# Access results with better formatting
print(results.summary())
print(f"MRR: {results.metrics.mrr:.4f}")
print(f"MAP: {results.metrics.map:.4f}")

# Save results
results.save("output/results.json")
results.to_dataframe().to_csv("output/results.csv")
```

## Available Presets

List all available presets:

```python
from evaluator import list_presets

presets = list_presets()
print("Available presets:", presets)
# ['whisper_labse', 'wav2vec_jina', 'audio_only', 'fast_dev']
```

### Preset Descriptions

| Preset | ASR Model | Embedding | Best For |
|--------|-----------|-----------|----------|
| `whisper_labse` | Whisper Base | LaBSE | Balanced accuracy and speed |
| `wav2vec_jina` | Wav2Vec2 | Jina V4 | Multilingual support |
| `audio_only` | None | CLAP | Direct audio embedding |
| `fast_dev` | Whisper Tiny | LaBSE | Quick testing (limited samples) |

## Using Config Files

For more control, use YAML configuration:

```yaml
# config.yaml
model:
  pipeline_mode: asr_text_retrieval
  asr_model_type: whisper
  asr_model_name: openai/whisper-base
  text_emb_model_type: labse
  
data:
  questions_path: questions.json
  corpus_path: corpus.json
  batch_size: 32
  
evaluation:
  k: 5
  metrics: [mrr, map, ndcg, recall]
```

Run with config:

```python
from evaluator import evaluate_from_config

results = evaluate_from_config("config.yaml")
print(results.summary())
```

## Command Line Usage

Run evaluations from the command line:

```bash
# Using a preset
python evaluate.py --preset whisper_labse \
    --questions questions.json \
    --corpus corpus.json

# Using a config file
python evaluate.py --config configs/evaluation_config.yaml

# Override specific parameters
python evaluate.py --config base.yaml \
    --asr_model_type whisper \
    --text_emb_model_type jina_v4 \
    --k 10
```

## Quick One-Liner

For rapid prototyping:

```python
from evaluator import quick_evaluate

results = quick_evaluate(
    audio_dir="audio_files/",
    model="whisper",
    embedding="labse",
    k=5
)
```

## Data Format

### Questions File (questions.json)

```json
[
  {
    "id": "q1",
    "audio_path": "audio/question1.wav",
    "text": "What are the symptoms of diabetes?",
    "relevant_doc_ids": ["doc123", "doc456"]
  },
  {
    "id": "q2",
    "audio_path": "audio/question2.wav",
    "text": "How to treat hypertension?",
    "relevant_doc_ids": ["doc789"]
  }
]
```

### Corpus File (corpus.json)

```json
[
  {
    "id": "doc123",
    "text": "Diabetes symptoms include increased thirst, frequent urination..."
  },
  {
    "id": "doc456",
    "text": "Type 2 diabetes is characterized by high blood sugar levels..."
  },
  {
    "id": "doc789",
    "text": "Hypertension treatment involves lifestyle changes and medication..."
  }
]
```

## Understanding Results

### Metrics Explained

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first relevant result
- **MAP (Mean Average Precision)**: Mean of average precision across all queries
- **NDCG@k**: Normalized Discounted Cumulative Gain at rank k
- **Recall@k**: Proportion of relevant documents in top-k results
- **WER (Word Error Rate)**: ASR transcription error rate
- **CER (Character Error Rate)**: Character-level transcription error

### Result Structure

```python
{
    "result_format_version": "1.0",
    "experiment_name": "whisper_labse_eval",
    "pipeline_mode": "asr_text_retrieval",
    "asr": "WhisperModel - openai/whisper-base",
    "embedder": "LaBseModel - sentence-transformers/LaBSE",
    "WER": 0.15,
    "CER": 0.08,
    "MRR": 0.75,
    "MAP": 0.68,
    "NDCG@5": 0.72,
    "Recall@5": 0.85,
    "timestamp": "2024-03-31T10:30:00",
    "config": {...}
}
```

## Next Steps

- Learn about [Configuration](configuration.md) options
- Explore available [Models](models.md)
- Understand [Pipeline Modes](pipelines.md)
- Set up [GPU Management](gpu_management.md) for multi-GPU systems
- Try the [Visualization](visualization.md) notebooks
