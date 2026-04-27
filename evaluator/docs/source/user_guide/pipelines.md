# Pipeline Modes

The evaluator supports multiple pipeline modes for different evaluation scenarios.

## Overview

| Mode | Flow | Use Case | Metrics |
|------|------|----------|---------|
| `asr_text_retrieval` | Audio→ASR→Text→Embedding→Retrieval | Standard evaluation | WER, CER, MRR, MAP, NDCG, Recall |
| `audio_emb_retrieval` | Audio→Audio Embedding→Retrieval | Direct audio retrieval | MRR, MAP, NDCG, Recall |
| `text_only` | Text→Embedding→Retrieval | Text-only baseline | MRR, MAP, NDCG, Recall |
| `asr_only` | Audio→ASR→Text | ASR quality testing | WER, CER |

## ASR + Text Retrieval Pipeline

**Mode**: `asr_text_retrieval` (default)

### Flow

```
Audio Files → ASR Model → Transcribed Text → Text Embeddings → Vector Search → Results
```

### Configuration

```yaml
model:
  pipeline_mode: asr_text_retrieval
  
  # ASR component
  asr_model_type: whisper
  asr_model_name: openai/whisper-base
  asr_device: cuda:0
  
  # Text embedding component
  text_emb_model_type: labse
  text_emb_device: cuda:1
  
evaluation:
  k: 5
  metrics: [wer, cer, mrr, map, ndcg, recall]
```

### Example

```python
from evaluator.api import EvaluationConfig, EvaluationRunner

config = EvaluationConfig.from_preset("whisper_labse")
runner = EvaluationRunner(config)
results = runner.run()

print(f"ASR Quality - WER: {results.metrics.wer:.4f}")
print(f"Retrieval Quality - MRR: {results.metrics.mrr:.4f}")
```

### Metrics

- **WER (Word Error Rate)**: Measures ASR transcription quality
- **CER (Character Error Rate)**: Character-level ASR quality
- **MRR (Mean Reciprocal Rank)**: Retrieval quality
- **MAP (Mean Average Precision)**: Retrieval quality
- **NDCG@k**: Normalized discounted cumulative gain
- **Recall@k**: Proportion of relevant docs retrieved

### When to Use

- Evaluating complete speech retrieval system
- Understanding ASR impact on retrieval
- Comparing ASR models
- Medical QA over audio queries

## Audio Embedding Retrieval Pipeline

**Mode**: `audio_emb_retrieval`

### Flow

```
Audio Files → Audio Embedding Model → Audio Embeddings → Vector Search → Results
```

Bypasses ASR entirely, embedding audio directly.

### Configuration

```yaml
model:
  pipeline_mode: audio_emb_retrieval
  
  # Audio embedding component
  audio_emb_model_type: clap_style
  audio_emb_device: cuda:0
  
  # Text embedding for corpus
  text_emb_model_type: labse
  text_emb_device: cuda:1
  
evaluation:
  k: 5
  metrics: [mrr, map, ndcg, recall]
```

### Example

```python
config = EvaluationConfig.from_preset("audio_only")
runner = EvaluationRunner(config)
results = runner.run()

print(f"Direct Audio Retrieval - MRR: {results.metrics.mrr:.4f}")
```

### Metrics

- **MRR, MAP, NDCG@k, Recall@k**: Retrieval metrics only
- No ASR metrics (WER/CER) since ASR is not used

### When to Use

- Investigating audio retrieval without ASR
- Low-latency requirements (no ASR step)
- ASR-independent baselines
- Audio-text alignment research

## Text-Only Retrieval Pipeline

**Mode**: `text_only`

### Flow

```
Ground Truth Text → Text Embeddings → Vector Search → Results
```

Uses gold-standard transcriptions, not audio.

### Configuration

```yaml
model:
  pipeline_mode: text_only
  
  # Only text embedding needed
  text_emb_model_type: labse
  text_emb_device: cuda:0
  
evaluation:
  k: 5
  metrics: [mrr, map, ndcg, recall]
```

### Example

```python
config = EvaluationConfig(
    model_config={"pipeline_mode": "text_only", "text_emb_model_type": "jina_v4"},
    data_config={"questions_path": "questions.json", "corpus_path": "corpus.json"}
)

runner = EvaluationRunner(config)
results = runner.run()

print(f"Upper Bound Retrieval - MRR: {results.metrics.mrr:.4f}")
```

### When to Use

- Establishing upper-bound performance (perfect transcription)
- Comparing embedding models without ASR noise
- Text-only retrieval systems
- Quantifying ASR impact: `text_only` MRR - `asr_text` MRR

## ASR-Only Pipeline

**Mode**: `asr_only`

### Flow

```
Audio Files → ASR Model → Transcribed Text
```

No retrieval, only transcription quality.

### Configuration

```yaml
model:
  pipeline_mode: asr_only
  
  # Only ASR needed
  asr_model_type: whisper
  asr_model_name: openai/whisper-large-v3
  asr_device: cuda:0
  
evaluation:
  metrics: [wer, cer]
```

### Example

```python
config = EvaluationConfig(
    model_config={"pipeline_mode": "asr_only", "asr_model_type": "faster_whisper"},
    data_config={"questions_path": "questions.json"}
)

runner = EvaluationRunner(config)
results = runner.run()

print(f"ASR WER: {results.metrics.wer:.4f}")
print(f"ASR CER: {results.metrics.cer:.4f}")
```

### When to Use

- Benchmarking ASR models
- Medical speech recognition research
- No corpus available (transcription-only)

## Pipeline Comparison

### Full Evaluation (Recommended)

Run all pipeline modes for comprehensive analysis:

```python
from evaluator.api import EvaluationConfig, EvaluationRunner

modes = ["asr_only", "text_only", "asr_text_retrieval"]
results = {}

for mode in modes:
    config = EvaluationConfig.from_preset("whisper_labse")
    config.model.pipeline_mode = mode
    
    runner = EvaluationRunner(config)
    results[mode] = runner.run()

# Compare
print(f"ASR Quality: WER = {results['asr_only'].metrics.wer:.4f}")
print(f"Upper Bound: MRR = {results['text_only'].metrics.mrr:.4f}")
print(f"Real System: MRR = {results['asr_text_retrieval'].metrics.mrr:.4f}")
print(f"ASR Impact: {results['text_only'].metrics.mrr - results['asr_text_retrieval'].metrics.mrr:.4f}")
```

### Ablation Study

Understand component contributions:

```python
from evaluator.ablation import AblationRunner

ablation = AblationRunner(base_config)
ablation.add_component("asr_model", ["whisper-tiny", "whisper-base", "whisper-large"])
ablation.add_component("embedding", ["labse", "jina_v4", "bge_m3"])

results = ablation.run()
results.plot_component_impact()
```

## Advanced Pipeline Features

### Custom Processing

Insert custom processing steps:

```python
from evaluator.pipeline import ASRPipeline, TextEmbeddingPipeline

class CustomASRPipeline(ASRPipeline):
    def post_process_transcription(self, text: str) -> str:
        # Custom post-processing
        text = text.lower()
        text = remove_filler_words(text)
        return text

# Use in config
config.pipeline_overrides = {
    "asr_pipeline_class": CustomASRPipeline
}
```

### Multi-Stage Retrieval

Combine pipelines for re-ranking:

```python
from evaluator.retrieval import HybridRetriever

# First stage: Fast retrieval
config1 = EvaluationConfig(pipeline_mode="audio_emb_retrieval")
stage1 = EvaluationRunner(config1)
candidates = stage1.retrieve_candidates(k=100)

# Second stage: Re-rank with ASR+text
config2 = EvaluationConfig(pipeline_mode="asr_text_retrieval")
stage2 = EvaluationRunner(config2)
final_results = stage2.rerank(candidates, k=10)
```

### Streaming Pipeline

Process audio in streaming mode:

```python
from evaluator.pipeline import StreamingPipeline

pipeline = StreamingPipeline(
    asr_model="faster_whisper",
    embedding_model="labse"
)

for audio_chunk in audio_stream:
    partial_transcription = pipeline.process_chunk(audio_chunk)
    embedding = pipeline.embed_partial(partial_transcription)
    results = pipeline.search(embedding)
    yield results
```

## Pipeline Selection Guide

### Choose `asr_text_retrieval` when:

- Evaluating real-world speech retrieval systems
- Need both ASR and retrieval metrics
- Have labeled audio with ground truth text
- Building medical voice assistants

### Choose `audio_emb_retrieval` when:

- ASR errors are problematic
- Need low latency
- Investigating audio-text alignment
- Researching ASR-free retrieval

### Choose `text_only` when:

- Establishing performance upper bounds
- Comparing embedding models
- No audio data available
- Pure text retrieval baseline

### Choose `asr_only` when:

- Benchmarking ASR models
- No corpus for retrieval
- Focusing on transcription quality

## Next Steps

- Configure [GPU Management](gpu_management.md) for pipeline components
- Explore [Retrieval Strategies](retrieval.md)
- Learn about [Visualization](visualization.md) for pipeline analysis
