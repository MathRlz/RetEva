# Available Models

This guide covers all ASR, text embedding, and audio embedding models supported by the evaluator.

## ASR Models

### Whisper

OpenAI's Whisper model for automatic speech recognition.

**Type**: `whisper`

**Available Models**:

| Model Name | Parameters | Speed | Accuracy | VRAM |
|------------|-----------|-------|----------|------|
| `openai/whisper-tiny` | 39M | Fastest | Lower | ~1GB |
| `openai/whisper-base` | 74M | Fast | Good | ~1GB |
| `openai/whisper-small` | 244M | Medium | Better | ~2GB |
| `openai/whisper-medium` | 769M | Slow | Very Good | ~5GB |
| `openai/whisper-large` | 1.5B | Slowest | Best | ~10GB |
| `openai/whisper-large-v2` | 1.5B | Slowest | Best | ~10GB |
| `openai/whisper-large-v3` | 1.5B | Slowest | Best | ~10GB |

**Configuration**:

```yaml
model:
  asr_model_type: whisper
  asr_model_name: openai/whisper-base
  asr_device: cuda:0
  asr_batch_size: 16
```

**Example**:

```python
from evaluator.models import WhisperModel

model = WhisperModel(
    model_name="openai/whisper-base",
    device="cuda:0"
)

transcription = model.transcribe("audio.wav")
print(transcription)  # "What are the symptoms of diabetes?"
```

### Faster Whisper

CTranslate2-optimized Whisper for faster inference.

**Type**: `faster_whisper`

**Installation**: `pip install faster-whisper`

**Available Models**:

| Model Name | Same as Whisper | Speed Improvement |
|------------|-----------------|-------------------|
| `tiny` | whisper-tiny | 4-5x faster |
| `base` | whisper-base | 4-5x faster |
| `small` | whisper-small | 4-5x faster |
| `medium` | whisper-medium | 4-5x faster |
| `large-v2` | whisper-large-v2 | 4-5x faster |
| `large-v3` | whisper-large-v3 | 4-5x faster |

**Configuration**:

```yaml
model:
  asr_model_type: faster_whisper
  asr_model_name: large-v3
  asr_device: cuda:0
```

**Note**: Faster Whisper uses different model names (no `openai/` prefix).

### Wav2Vec2

Facebook's Wav2Vec2 model, good for specific languages.

**Type**: `wav2vec2`

**Available Models**:

| Model Name | Language | Use Case |
|------------|----------|----------|
| `facebook/wav2vec2-base-960h` | English | General English ASR |
| `facebook/wav2vec2-large-960h-lv60-self` | English | High accuracy English |
| `jonatasgrosman/wav2vec2-large-xlsr-53-polish` | Polish | Polish medical speech |
| `jonatasgrosman/wav2vec2-large-xlsr-53-english` | English | Multilingual base |

**Configuration**:

```yaml
model:
  asr_model_type: wav2vec2
  asr_model_name: facebook/wav2vec2-base-960h
  asr_device: cuda:0
```

## Text Embedding Models

### LaBSE

Language-Agnostic BERT Sentence Embedding.

**Type**: `labse`

**Details**:
- **Dimensions**: 768
- **Languages**: 109 languages
- **Best for**: Multilingual retrieval
- **Speed**: Fast

**Configuration**:

```yaml
model:
  text_emb_model_type: labse
  text_emb_model_name: sentence-transformers/LaBSE
  text_emb_device: cuda:1
  text_emb_batch_size: 32
```

### Jina Embeddings V4

Latest Jina embeddings with strong performance.

**Type**: `jina_v4`

**Details**:
- **Dimensions**: 1024
- **Max Sequence**: 8192 tokens
- **Best for**: Long documents, high accuracy
- **Speed**: Medium

**Configuration**:

```yaml
model:
  text_emb_model_type: jina_v4
  text_emb_model_name: jinaai/jina-embeddings-v4
  text_emb_device: cuda:1
```

### BGE-M3

BAAI General Embedding M3 (Multi-lingual, Multi-functionality, Multi-granularity).

**Type**: `bge_m3`

**Details**:
- **Dimensions**: 1024
- **Languages**: 100+ languages
- **Best for**: Multilingual, dense+sparse retrieval
- **Speed**: Medium

**Configuration**:

```yaml
model:
  text_emb_model_type: bge_m3
  text_emb_model_name: BAAI/bge-m3
  text_emb_device: cuda:1
```

### Nemotron

NVIDIA Nemotron embeddings (experimental).

**Type**: `nemotron`

**Details**:
- **Dimensions**: 4096
- **Best for**: High-capacity embeddings
- **Speed**: Slower
- **VRAM**: High (~8GB)

**Configuration**:

```yaml
model:
  text_emb_model_type: nemotron
  text_emb_device: cuda:1
  text_emb_batch_size: 8  # Lower batch size due to size
```

### CLIP

OpenAI CLIP text encoder (for audio-text alignment).

**Type**: `clip`

**Details**:
- **Dimensions**: 512
- **Best for**: Audio-text retrieval
- **Speed**: Fast

**Configuration**:

```yaml
model:
  text_emb_model_type: clip
  text_emb_model_name: openai/clip-vit-base-patch32
```

## Audio Embedding Models

### CLAP-Style

Contrastive Language-Audio Pre-training style embeddings.

**Type**: `clap_style`

**Details**:
- Direct audio→embedding without ASR
- Aligns audio and text in shared space
- Best for audio-based retrieval

**Configuration**:

```yaml
model:
  pipeline_mode: audio_emb_retrieval
  audio_emb_model_type: clap_style
  audio_emb_device: cuda:0
  text_emb_model_type: labse  # For corpus indexing
```

### Attention Pooled

Attention-pooled Wav2Vec2 features for audio embeddings.

**Type**: `attention_pool`

**Details**:
- Uses Wav2Vec2 as feature extractor
- Attention pooling over temporal features
- Configurable pooling strategy

**Configuration**:

```yaml
model:
  pipeline_mode: audio_emb_retrieval
  audio_emb_model_type: attention_pool
  audio_emb_device: cuda:0
```

## TTS Models (Audio Synthesis)

### Piper

Local/offline TTS backend.

**Provider**: `piper`

Use when you want fast local synthesis and already have Piper voice files.

### XTTS-v2

Coqui multilingual TTS with optional speaker voice cloning.

**Provider**: `xtts_v2`

Notes:
- Supports many languages via `audio_synthesis.language`
- If `audio_synthesis.voice` points to a WAV file, it is used as speaker reference

### MMS (Meta)

Hugging Face MMS multilingual TTS backend.

**Provider**: `mms`

Notes:
- Broad language coverage
- You can set `audio_synthesis.voice` to explicit HF model id (e.g. `facebook/mms-tts-pol`)
- If not set, model is picked from `audio_synthesis.language`

### TTS Configuration Example

```yaml
audio_synthesis:
  enabled: true
  provider: mms               # piper | xtts_v2 | mms
  language: pl
  voice: facebook/mms-tts-pol # optional for mms
  sample_rate: 16000
```

## Adding New Models

### Adding an ASR Model

1. Create a new model class in `evaluator/models/asr/`:

```python
from evaluator.models.base import BaseASRModel

class MyASRModel(BaseASRModel):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        # Initialize your model
        
    def transcribe(self, audio_path: str) -> str:
        # Implement transcription
        pass
        
    def transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        # Implement batch transcription
        pass
```

2. Register in `evaluator/models/__init__.py`:

```python
from evaluator.models.registry import register_model

@register_model("my_asr", model_class=MyASRModel, model_type="asr")
def create_my_asr_model(config):
    return MyASRModel(
        model_name=config.model_name,
        device=config.device
    )
```

3. Use in config:

```yaml
model:
  asr_model_type: my_asr
  asr_model_name: my-org/my-model
```

### Adding a Text Embedding Model

1. Create model class in `evaluator/models/embeddings/`:

```python
from evaluator.models.base import BaseEmbeddingModel

class MyEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        
    def encode(self, texts: list[str]) -> np.ndarray:
        # Return shape (len(texts), embedding_dim)
        pass
        
    @property
    def embedding_dim(self) -> int:
        return 768  # Your model's dimension
```

2. Register the model:

```python
@register_model("my_embedding", model_class=MyEmbeddingModel, model_type="embedding")
def create_my_embedding_model(config):
    return MyEmbeddingModel(
        model_name=config.model_name,
        device=config.device
    )
```

### Adding an Audio Embedding Model

Similar to text embeddings, but for direct audio→embedding:

```python
from evaluator.models.base import BaseAudioEmbeddingModel

class MyAudioEmbeddingModel(BaseAudioEmbeddingModel):
    def encode_audio(self, audio_paths: list[str]) -> np.ndarray:
        # Return shape (len(audio_paths), embedding_dim)
        pass
```

## Model Selection Guidelines

### For English Medical Speech

**Best Accuracy**:
- ASR: `whisper-large-v3` or `faster_whisper large-v3`
- Embedding: `jina_v4` or `bge_m3`

**Best Speed**:
- ASR: `faster_whisper base`
- Embedding: `labse`

**Balanced**:
- ASR: `whisper-base` or `faster_whisper medium`
- Embedding: `labse` or `jina_v4`

### For Multilingual

**Best Choice**:
- ASR: Language-specific Wav2Vec2 or `whisper-large-v3`
- Embedding: `labse` or `bge_m3`

### For Limited GPU Memory

**Configuration**:
- ASR: `whisper-tiny` or `whisper-base`
- Embedding: `labse` (768d) or `clip` (512d)
- Batch size: 8-16
- Use CPU for one component

### For Direct Audio Retrieval

**Configuration**:
- Audio Embedding: `clap_style` or `attention_pool`
- Text Embedding: `labse` (for corpus)
- No ASR needed

## Performance Comparison

Approximate throughput on NVIDIA A100 (40GB):

| Model | Type | Batch Size | Samples/sec | VRAM |
|-------|------|-----------|-------------|------|
| whisper-tiny | ASR | 16 | ~50 | 1GB |
| whisper-base | ASR | 16 | ~40 | 1.5GB |
| faster_whisper large-v3 | ASR | 16 | ~25 | 6GB |
| labse | Embedding | 32 | ~800 | 2GB |
| jina_v4 | Embedding | 32 | ~600 | 3GB |
| bge_m3 | Embedding | 32 | ~500 | 3GB |

## Next Steps

- Configure models in [Configuration](configuration.md)
- Understand [Pipeline Modes](pipelines.md)
- Set up [GPU Management](gpu_management.md) for multi-model systems
