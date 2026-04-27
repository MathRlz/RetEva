# Text-to-Speech (TTS) Setup Guide

This guide explains how to set up and use text-to-speech synthesis for converting text questions to audio in evaluations.

## Overview

The TTS integration allows you to:
- Convert text-only datasets into audio for speech retrieval evaluation
- Synthesize audio from questions automatically during evaluation
- Cache synthesized audio to avoid re-synthesis
- Support multiple languages and voices

## Installation

### Piper TTS (Recommended)

Piper is a fast, local TTS engine with good quality and multi-language support.

```bash
# Install Piper TTS
pip install piper-tts

# Or install from source
git clone https://github.com/rhasspy/piper.git
cd piper/src/python
pip install -e .
```

### Download Voice Models

Piper requires voice models to be downloaded separately:

1. Visit https://github.com/rhasspy/piper/releases
2. Download a voice model (e.g., `en_US-lessac-medium.onnx`)
3. Place it in one of these directories:
   - `~/.local/share/piper/voices/`
   - `/usr/share/piper/voices/`
   - `./voices/` (in your project directory)

**Recommended voices:**
- **English (US):** `en_US-lessac-medium` (general purpose, clear)
- **English (US, female):** `en_US-amy-medium`
- **English (GB):** `en_GB-alba-medium`
- **Spanish:** `es_ES-mls_9972-low`
- **French:** `fr_FR-siwis-medium`
- **German:** `de_DE-thorsten-medium`

See full list: https://rhasspy.github.io/piper-samples/

## Configuration

### Basic Configuration

Add to your YAML config:

```yaml
audio_synthesis:
  enabled: true
  provider: piper
  voice: en_US-lessac-medium
  sample_rate: 16000
  speed: 1.0
  cache_dir: .tts_cache
  output_dir: prepared_benchmarks/audio
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | false | Enable TTS synthesis |
| `provider` | string | "piper" | TTS provider (piper, coqui, gtts, espeak) |
| `voice` | string | "en_US-lessac-medium" | Voice model name |
| `sample_rate` | int | 16000 | Audio sample rate in Hz |
| `speed` | float | 1.0 | Speech speed multiplier (0.5-2.0) |
| `pitch` | float | 1.0 | Pitch multiplier (reserved) |
| `volume` | float | 1.0 | Volume multiplier (reserved) |
| `language` | string | "en" | Language code |
| `cache_dir` | string | null | Cache directory (null = no caching) |
| `output_dir` | string | "prepared_benchmarks/audio" | Output directory for WAV files |
| `api_key` | string | null | API key for cloud providers |

### Web UI Configuration

1. Navigate to the Experiment page
2. Open the **Models** tab
3. Scroll to **Audio Synthesis (Text-to-Speech)**
4. Check "Enable Text-to-Speech"
5. Configure settings:
   - **Provider:** Select Piper
   - **Voice:** Enter voice model name
   - **Speed:** Adjust speech speed (0.5x - 2.0x)
   - **Cache Directory:** Set cache location

## Usage

### Automatic Synthesis During Evaluation

When `audio_synthesis.enabled = true`, the evaluator automatically:
1. Checks each question for existing audio (`audio_path`)
2. If no audio exists, synthesizes from `question_text`
3. Saves synthesized audio to `output_dir`
4. Caches the audio in `cache_dir` (if set)

**Example workflow:**

```python
from evaluator import evaluate_from_config

# Load config with TTS enabled
results = evaluate_from_config("config_with_tts.yaml")

# Questions without audio will be automatically synthesized
print(f"MRR: {results.get_metric('MRR'):.4f}")
```

### Manual Synthesis

```python
from evaluator.pipeline.audio.synthesis import AudioSynthesizer
from evaluator.config import AudioSynthesisConfig

# Create config
config = AudioSynthesisConfig(
    enabled=True,
    provider="piper",
    voice="en_US-lessac-medium",
    cache_dir=".tts_cache"
)

# Create synthesizer
synthesizer = AudioSynthesizer(config)

# Synthesize single text
audio = synthesizer.synthesize("What is hypertension?")
print(f"Generated audio shape: {audio.shape}")

# Synthesize batch
texts = [
    "What causes diabetes?",
    "Describe symptoms of pneumonia",
    "How is cancer treated?"
]
audio_arrays = synthesizer.synthesize_batch(texts, output_dir="synth_audio/")
print(f"Synthesized {len(audio_arrays)} audio files")
```

## Performance & Caching

### Cache Benefits

With caching enabled:
- **First synthesis:** ~0.5-1.0 seconds per question
- **Cached synthesis:** ~0.01 seconds (instant)
- **Cache hit rate:** >90% for repeated evaluations

### Cache Location

Caches are stored as `.npy` files with SHA256 hash filenames:

```
.tts_cache/
├── 3a7b2c9d... .npy  # "What is hypertension?"
├── 8f4e1a6c... .npy  # "What causes diabetes?"
└── ...
```

Cache key includes: text + provider + voice + speed + pitch

### Cache Management

```bash
# Check cache size
du -sh .tts_cache

# Clear cache
rm -rf .tts_cache

# Clear old cache (older than 30 days)
find .tts_cache -name "*.npy" -mtime +30 -delete
```

## Troubleshooting

### Voice Model Not Found

**Error:**
```
FileNotFoundError: Voice model not found: en_US-lessac-medium
```

**Solution:**
1. Download voice model from https://github.com/rhasspy/piper/releases
2. Place in `~/.local/share/piper/voices/`
3. Ensure filename ends with `.onnx`

### Piper Not Found

**Error:**
```
RuntimeError: Piper TTS not found
```

**Solution:**
```bash
pip install piper-tts
# Or check PATH
which piper
```

### Audio Quality Issues

**Problem:** Robotic or poor quality audio

**Solutions:**
- Use a better voice model (e.g., `-medium` or `-high`)
- Adjust speed (try 0.9x or 1.1x)
- Consider Coqui TTS for higher quality (slower)

### Memory Issues

**Problem:** Out of memory during batch synthesis

**Solutions:**
- Enable caching to avoid re-synthesis
- Reduce batch size
- Process questions in chunks

## Advanced Usage

### Multiple Languages

```yaml
# Synthesize different languages
audio_synthesis:
  enabled: true
  provider: piper
  voice: es_ES-mls_9972-low  # Spanish
  language: es
```

### Custom Speed

```yaml
# Faster speech (1.5x)
audio_synthesis:
  enabled: true
  speed: 1.5  # Useful for testing
```

```yaml
# Slower speech (0.8x)
audio_synthesis:
  enabled: true
  speed: 0.8  # Clearer for difficult medical terms
```

### Cloud TTS (gTTS)

```yaml
# Google TTS (requires internet)
audio_synthesis:
  enabled: true
  provider: gtts
  voice: en
  language: en
```

**Note:** gTTS is slower and requires internet but supports 100+ languages.

## Best Practices

1. **Enable caching** to speed up repeated evaluations
2. **Use Piper** for best speed/quality trade-off
3. **Test voice quality** before large-scale synthesis
4. **Pre-synthesize datasets** for production use
5. **Monitor cache size** to avoid disk space issues

## Examples

### Example 1: PubMed QA with TTS

```yaml
experiment_name: pubmed_qa_with_tts

model:
  pipeline_mode: asr_text_retrieval
  asr_model_type: whisper
  asr_model_name: openai/whisper-medium
  text_emb_model_type: jina_v4

data:
  dataset_type: text_query_retrieval
  questions_path: pubmed_qa/questions.json
  corpus_path: pubmed_qa/corpus.json

audio_synthesis:
  enabled: true
  provider: piper
  voice: en_US-lessac-medium
  cache_dir: .tts_cache
  output_dir: pubmed_qa/synthesized_audio
```

### Example 2: Medical Domain with Custom Voice

```yaml
audio_synthesis:
  enabled: true
  provider: piper
  voice: en_US-amy-medium  # Female voice
  speed: 0.9  # Slightly slower for medical terms
  cache_dir: .tts_cache_medical
```

## See Also

- [Dataset Types Guide](dataset-types.md)
- [Configuration Reference](../README.md#configuration)
- [Piper Documentation](https://github.com/rhasspy/piper)
