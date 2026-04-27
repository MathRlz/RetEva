# Evaluator Usage Examples

This directory contains runnable examples demonstrating how to use the evaluator framework.

## Examples Overview

| File | Description | Difficulty |
|------|-------------|------------|
| `basic_evaluation.py` | Quick start guide using `quick_evaluate()` and presets | Beginner |
| `config_file_usage.py` | Working with YAML configuration files | Beginner |
| `custom_model.py` | Adding new models using the registry system | Advanced |

## Prerequisites

Before running examples, ensure you have:

1. **Python Environment**: Python 3.9+ with the evaluator package installed:
   ```bash
   cd /path/to/evaluator
   pip install -e .
   ```

2. **Dependencies**: Core dependencies including PyTorch, transformers, sentence-transformers:
   ```bash
   pip install torch transformers sentence-transformers
   ```

3. **Dataset** (for actual evaluation runs):
   - Prepared dataset with audio files and ground truth labels
   - See `PUBMED_DATASET_FORMAT.md` in the repository root for format details

4. **GPU** (recommended but not required):
   - CUDA-capable GPU for faster inference
   - Examples auto-detect hardware and work on CPU too

## Quick Start

```python
# Simplest usage - run from the repository root
from evaluator import quick_evaluate

results = quick_evaluate(
    audio_dir="path/to/your/data/",
    model="whisper",
    embedding="labse",
    trace_limit=50  # Limit samples for testing
)

print(f"MRR: {results['MRR']:.4f}")
```

## Example Descriptions

### 1. basic_evaluation.py

Demonstrates the simplest ways to run evaluations:

- **`quick_evaluate()`**: Minimal configuration, ideal for prototyping
- **`evaluate_from_preset()`**: Pre-configured model combinations
- **Result interpretation**: Understanding WER, MRR, Recall@k metrics

```bash
python examples/basic_evaluation.py
```

### 2. config_file_usage.py

Shows how to work with YAML configuration files:

- Loading configurations from YAML
- Common configuration patterns (ASR+text, audio-only, hybrid retrieval)
- Overriding config values programmatically
- Saving configs for reproducibility
- Hardware-specific configurations (CPU, single GPU, multi-GPU)

```bash
python examples/config_file_usage.py
```

### 3. custom_model.py

Advanced example for extending the framework with custom models:

- Subclassing `TextEmbeddingModel`, `ASRModel`, `AudioEmbeddingModel`
- Using `@register_*_model` decorators
- Implementing required abstract methods
- Using custom models in evaluation

```bash
python examples/custom_model.py
```

## Common Configuration Patterns

### Pattern A: ASR + Text Embedding (Most Common)

```yaml
model:
  pipeline_mode: asr_text_retrieval
  asr_model_type: whisper
  asr_model_name: openai/whisper-base
  text_emb_model_type: labse
  text_emb_model_name: sentence-transformers/LaBSE
```

### Pattern B: Direct Audio Embedding

```yaml
model:
  pipeline_mode: audio_emb_retrieval
  audio_emb_model_type: attention_pool
  audio_emb_model_name: facebook/wav2vec2-base-960h
```

### Pattern C: Quick Development

```yaml
model:
  asr_model_name: openai/whisper-tiny  # Smallest model
data:
  trace_limit: 50  # Only 50 samples
  batch_size: 8
```

## Available Presets

Use presets for tested model combinations:

```python
from evaluator import list_presets, evaluate_from_preset

print(list_presets())  # ['whisper_labse', 'wav2vec_jina', 'audio_only', 'fast_dev']

results = evaluate_from_preset("whisper_labse", data_path="...")
```

| Preset | Description |
|--------|-------------|
| `whisper_labse` | Whisper ASR + LaBSE embedding (good general-purpose) |
| `wav2vec_jina` | Wav2Vec2 ASR + Jina V4 embedding (multilingual) |
| `audio_only` | Direct audio embedding, no ASR |
| `fast_dev` | Small models for quick testing |

## Tips

1. **Start small**: Use `trace_limit=50` to test with a subset of data
2. **Use presets**: Start with a preset, then customize
3. **Enable caching**: Set `cache.enabled: true` to speed up re-runs
4. **Auto-detect hardware**: Call `config.with_auto_devices()` or let the API handle it
5. **Check logs**: Logs are saved to `logs/` directory by default

## Troubleshooting

**Out of memory errors**:
- Reduce `batch_size` in configuration
- Use smaller models (e.g., `whisper-tiny` instead of `whisper-large`)
- Use CPU: set `*_device: cpu` in config

**Model download issues**:
- Ensure internet connectivity
- Models are cached in `~/.cache/huggingface/`

**Import errors**:
- Ensure the package is installed: `pip install -e .`
- Check Python version (requires 3.9+)

## Further Reading

- `README.md` - Project overview
- `ARCHITECTURE.md` - Detailed architecture documentation
- `configs/` - Example configuration files
