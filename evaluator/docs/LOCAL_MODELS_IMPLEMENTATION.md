# Local Models WebUI Implementation Plan

> **Purpose**: Add WebUI pages to manage all local AI models (LLM, ASR, Text Embeddings, Audio Embeddings, Vector DBs) through the existing web interface.

## Overview

The model infrastructure already exists in `evaluator/models/`. We need to:
1. Move `llm_server/` into `evaluator/models/llm/` for consistency
2. Create WebUI pages exposing each model type
3. Add navigation and config export

## Existing Infrastructure

```
evaluator/models/
├── registry.py          # ModelRegistry class with plugin-style registration
├── factory.py           # create_asr_model(), create_text_embedding_model(), etc.
├── base.py              # BaseTextEmbeddingModel, BaseAudioEmbeddingModel
├── base_asr.py          # BaseASRModel
├── whisper.py           # Whisper ASR
├── faster_whisper.py    # Faster Whisper ASR  
├── wav2vec2.py          # Wav2Vec2 ASR
├── labse.py             # LaBSE text embeddings (768-dim)
├── jina.py              # JINA v4 text embeddings (1024-dim)
├── bgem3.py             # BGE-M3 text embeddings (1024-dim)
├── nemotron.py          # Nemotron text embeddings (1024-dim)
├── clip.py              # CLIP text embeddings (768-dim)
├── attention_pool.py    # Attention Pool audio embeddings
├── clap_style.py        # CLAP-style audio embeddings
├── hubert.py            # HuBERT audio embeddings
└── wavlm.py             # WavLM audio embeddings

evaluator/
├── storage/vector_store.py # VectorStore: InMemory, FAISS, ChromaDB, Qdrant
└── config.py               # ModelConfig, VectorDBConfig dataclasses

llm_server/              # TO BE MOVED → evaluator/models/llm/
├── ollama.py            # OllamaServer class
├── model_registry.py    # Ollama model catalog (30 verified models)
├── factory.py           # Server factory
└── base.py              # BaseLLMServer
```

---

## Tasks

### Task 1: Move llm_server into evaluator/models/llm/

**Goal**: Consolidate LLM server code into the evaluator/models structure.

**Steps**:
1. Create `evaluator/models/llm/` directory
2. Move files from `llm_server/` to `evaluator/models/llm/`:
   - `ollama.py` → `evaluator/models/llm/ollama.py`
   - `model_registry.py` → `evaluator/models/llm/registry.py`
   - `factory.py` → `evaluator/models/llm/factory.py`
   - `base.py` → `evaluator/models/llm/base.py`
3. Create `evaluator/models/llm/__init__.py` with exports
4. Update imports in `webui/app.py`:
   ```python
   # OLD:
   from llm_server.ollama import OllamaServer
   from llm_server.model_registry import ModelRegistry
   
   # NEW:
   from evaluator.models.llm.ollama import OllamaServer
   from evaluator.models.llm.registry import ModelRegistry
   ```
5. Update any relative imports within the moved files
6. Delete old `llm_server/` directory
7. Test: Start WebUI, verify LLM Server page still works

**Files to modify**:
- Create: `evaluator/models/llm/__init__.py`
- Move: `llm_server/*.py` → `evaluator/models/llm/`
- Edit: `webui/app.py` (update imports)
- Delete: `llm_server/` directory

---

### Task 2: Create unified LLM registry

**Goal**: Adapt LLM registry to follow evaluator/models pattern.

**Steps**:
1. Review `evaluator/models/registry.py` pattern (ModelRegistry class)
2. Update `evaluator/models/llm/registry.py` to use consistent patterns
3. Keep existing Ollama model catalog (30 verified models)
4. Add method to list models for WebUI

**No functional changes** - just code organization.

---

### Task 3: Create ASR Models page (/asr-models)

**Goal**: WebUI page to browse, test, and configure ASR models.

**Create files**:
- `webui/templates/asr-models.html`
- `webui/static/asr-models.js`
- `webui/static/asr-models.css`

**Add to webui/app.py**:
```python
@app.route('/asr-models')
def asr_models_page():
    return render_template('asr-models.html')

@app.route('/api/asr/list')
def list_asr_models():
    """Return list of available ASR models."""
    models = [
        {
            'id': 'whisper',
            'name': 'OpenAI Whisper',
            'variants': ['tiny', 'base', 'small', 'medium', 'large-v3'],
            'description': 'OpenAI Whisper speech recognition',
            'default': 'openai/whisper-base'
        },
        {
            'id': 'faster-whisper',
            'name': 'Faster Whisper',
            'variants': ['tiny', 'base', 'small', 'medium', 'large-v3'],
            'description': 'CTranslate2 optimized Whisper (4x faster)',
            'default': 'Systran/faster-whisper-base'
        },
        {
            'id': 'wav2vec2',
            'name': 'Wav2Vec2 XLSR',
            'variants': ['xlsr-53'],
            'description': 'Facebook Wav2Vec2 for multilingual ASR',
            'default': 'facebook/wav2vec2-large-xlsr-53'
        }
    ]
    return jsonify(models)

@app.route('/api/asr/test', methods=['POST'])
def test_asr_model():
    """Test ASR model with sample audio."""
    model_type = request.json.get('model_type')
    model_name = request.json.get('model_name')
    # Use evaluator/models/factory.py to create model
    # Transcribe test audio and return result
    pass

@app.route('/api/asr/config', methods=['POST'])
def generate_asr_config():
    """Generate ModelConfig YAML for selected ASR model."""
    pass
```

**UI Features**:
- Model cards showing Whisper, Faster-Whisper, Wav2Vec2
- Variant selector (tiny/base/small/medium/large)
- "Test" button → transcribe sample audio
- "Generate Config" → show YAML to copy

**Reference**: Follow `webui/templates/llm-server.html` pattern

---

### Task 4: Create Text Embeddings page (/text-embeddings)

**Goal**: WebUI page for text embedding models.

**Create files**:
- `webui/templates/text-embeddings.html`
- `webui/static/text-embeddings.js`
- `webui/static/text-embeddings.css`

**Models to list** (from evaluator/models/):
| Model | File | Dimensions | HuggingFace ID |
|-------|------|------------|----------------|
| LaBSE | labse.py | 768 | sentence-transformers/LaBSE |
| JINA v4 | jina.py | 1024 | jinaai/jina-embeddings-v3 |
| BGE-M3 | bgem3.py | 1024 | BAAI/bge-m3 |
| Nemotron | nemotron.py | 1024 | nvidia/NV-Embed-v2 |
| CLIP | clip.py | 768 | openai/clip-vit-base-patch32 |

**API endpoints**:
- `GET /api/text-embeddings/list` - List models with metadata
- `POST /api/text-embeddings/test` - Embed sample text, show dimensions
- `POST /api/text-embeddings/config` - Generate YAML config

**UI Features**:
- Model cards with dimension info
- Test: Enter two sentences → show cosine similarity
- Show embedding preview (first 10 values)

---

### Task 5: Create Audio Embeddings page (/audio-embeddings)

**Goal**: WebUI page for E2E audio embedding models.

**Create files**:
- `webui/templates/audio-embeddings.html`
- `webui/static/audio-embeddings.js`
- `webui/static/audio-embeddings.css`

**Models to list** (from evaluator/models/):
| Model | File | Description |
|-------|------|-------------|
| AttentionPool | attention_pool.py | Whisper encoder with attention pooling |
| CLAP-Style | clap_style.py | Contrastive audio-text model |
| HuBERT | hubert.py | Self-supervised audio representations |
| WavLM | wavlm.py | Microsoft WavLM audio encoder |

**API endpoints**:
- `GET /api/audio-embeddings/list` - List models
- `POST /api/audio-embeddings/test` - Encode sample audio
- `POST /api/audio-embeddings/config` - Generate YAML

---

### Task 6: Create Vector Databases page (/vector-databases)

**Goal**: WebUI page for vector store configuration.

**Create files**:
- `webui/templates/vector-databases.html`
- `webui/static/vector-databases.js`
- `webui/static/vector-databases.css`

**Backends to list** (from evaluator/storage/vector_store.py):
| Backend | Class | Description |
|---------|-------|-------------|
| InMemory | InMemoryVectorStore | Simple numpy-based, no dependencies |
| FAISS | FAISSVectorStore | Facebook AI Similarity Search |
| FAISS GPU | FAISSVectorStore | FAISS with GPU acceleration |
| ChromaDB | ChromaVectorStore | Embedded vector database |
| Qdrant | QdrantVectorStore | Production vector search engine |

**API endpoints**:
- `GET /api/vectordb/list` - List backends with requirements
- `POST /api/vectordb/test` - Build small test index + search
- `POST /api/vectordb/config` - Generate VectorDBConfig YAML

**UI Features**:
- Backend cards with dependency info
- Options per backend (e.g., FAISS index type, GPU flag)
- Test: Add 10 vectors → search → show results

---

### Task 7: Create Local Models overview page (/local-models)

**Goal**: Overview page linking all model pages + combined config export.

**Create files**:
- `webui/templates/local-models.html`
- `webui/static/local-models.js`

**Features**:
- Cards linking to each model page (LLM, ASR, Text Emb, Audio Emb, VectorDB)
- Show current selections (stored in session/localStorage)
- "Export Complete Config" → combined YAML for experiment

**Combined YAML output example**:
```yaml
# Generated by Medical Speech Retrieval Evaluator WebUI
# Copy this to your experiment config file

model:
  asr_type: "faster-whisper"
  asr_model: "Systran/faster-whisper-base"
  text_embedding_type: "labse"
  text_embedding_model: "sentence-transformers/LaBSE"
  audio_embedding_type: "attention-pool"
  audio_embedding_model: "openai/whisper-base"

vector_db:
  backend: "faiss"
  index_type: "IVF"
  use_gpu: false

llm:
  backend: "ollama"
  model: "meditron-7b"
  port: 11434
```

---

### Task 8: Add Local Models navigation menu

**Goal**: Add navigation to access all model pages.

**Modify**: `webui/templates/base.html` (or layout template)

**Add dropdown menu**:
```html
<nav>
  <a href="/">Home</a>
  <div class="dropdown">
    <a href="/local-models">Local Models ▾</a>
    <div class="dropdown-content">
      <a href="/llm-server">LLM Server</a>
      <a href="/asr-models">ASR Models</a>
      <a href="/text-embeddings">Text Embeddings</a>
      <a href="/audio-embeddings">Audio Embeddings</a>
      <a href="/vector-databases">Vector Databases</a>
    </div>
  </div>
  <a href="/experiments">Experiments</a>
</nav>
```

**Also update**: Home page with quick links

---

## Dependency Graph

```
Task 1 (Move llm_server)
    ↓
Task 2 (LLM registry cleanup)
    ↓
┌───┴───┬───────┬───────┐
↓       ↓       ↓       ↓
Task 3  Task 4  Task 5  Task 6
(ASR)   (Text)  (Audio) (VectorDB)
└───┬───┴───┬───┴───┬───┘
    ↓       ↓       ↓
    Task 7 (Overview)
         ↓
    Task 8 (Navigation)
```

---

## Testing Checklist

For each page, verify:
- [ ] Page loads without errors
- [ ] Model list displays correctly
- [ ] Model selection works (click to select)
- [ ] Test function works (transcribe/embed/search)
- [ ] Config YAML generates correctly
- [ ] Config can be copied to clipboard

---

## Notes for Implementation

1. **Use existing model factory**: Don't create new model loading code
   ```python
   from evaluator.models.factory import create_asr_model
   model = create_asr_model(model_type, model_name)
   ```

2. **HuggingFace cache**: Models download automatically on first use
   - Check `~/.cache/huggingface/hub/` for cached models
   - Show "Downloaded" vs "Not cached" status in UI

3. **Error handling**: Wrap model operations in try/catch
   - GPU out of memory
   - Network errors during download
   - Missing dependencies

4. **Session storage**: Use localStorage to remember selections
   ```javascript
   localStorage.setItem('selected_asr', JSON.stringify({type: 'whisper', model: 'base'}));
   ```

5. **CSS reuse**: Copy patterns from `llm-server.css`
   - `.model-card` styling
   - `.selected` state
   - `.result-panel` for test output
