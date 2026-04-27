# LLM Model Management Guide

Complete guide for managing local LLM models in the Web UI.

---

## Features

### 1. Quick Model Testing
**Location:** Local Models page → Quick Testing tab → LLM Test

**Purpose:** Test any installed model with a custom prompt instantly.

**How to use:**
1. Navigate to **Local Models** page
2. Click **Quick Testing** tab
3. Select a model from dropdown (shows only installed models)
4. Enter a test prompt
5. Click **Test LLM**

**What it shows:**
- Model response
- Response time
- Token count (if available)

**Example:**
```
Model: mistral:7b-instruct
Prompt: What is pneumonia?
Response: Pneumonia is an infection...
Time: 2.3s | Tokens: 127
```

---

### 2. LLM Server Management
**Location:** LLM Server page

**Purpose:** View, pull, and manage Ollama models.

#### View Installed Models
1. Go to **LLM Server** page
2. Select **Model Source: Installed Models**
3. See only models that are actually pulled and ready

**Shows:**
- Model name and version (e.g., `mistral:7b-instruct`)
- Size on disk (e.g., `4.1GB`)
- Parameters (e.g., `7.2B`)
- Quantization level (e.g., `Q4_K_M`)
- Model family

#### Pull New Models
1. Go to **LLM Server** page
2. Select **Model Source: Available to Pull**
3. Browse the registry (30 models available)
4. Filter by:
   - Medical Domain
   - General Purpose
   - Small (< 8B params)
5. Click a model card to select it
6. Click **📥 Pull Selected Model**
7. Watch the progress bar
8. Model becomes available when complete

**Registry includes:**
- **Medical Domain:**
  - BioMistral 7B
  - Meditron 7B
  - Clinical-NLP Llama 7B
  
- **General Purpose:**
  - Mistral 7B Instruct
  - Llama 3 (8B, 70B)
  - Qwen 2/2.5 (various sizes)
  - Phi-3 Mini
  - Gemma 2B/7B

- **Small/Fast:**
  - Qwen2 1.5B
  - Phi-3 Mini 3.8B
  - Gemma 2B
  - TinyLlama 1.1B

**Pull Progress:**
- Shows download percentage
- Status messages (pulling, verifying, complete)
- Auto-refreshes installed list when done
- Error messages if pull fails

**Visual Indicators:**
- Registry models show **✓ Installed** badge if already pulled
- Selected model has blue border
- Hover effects on all cards

---

## API Endpoints

### GET /api/llm-server/models
Returns only installed models.

**Response:**
```json
{
  "success": true,
  "models": [
    {
      "name": "mistral:7b-instruct",
      "display_name": "Mistral 7B-Instruct",
      "size": "4.1GB",
      "parameters": "7.2B",
      "quantization": "Q4_K_M",
      "family": "llama"
    }
  ]
}
```

### GET /api/llm-server/registry
Returns all available models from the registry.

**Response:**
```json
{
  "success": true,
  "models": [
    {
      "name": "biomistral-7b",
      "ollama_name": "biomistral:7b-instruct",
      "display_name": "BioMistral 7B Instruct",
      "domain": "medical",
      "parameters": "7B",
      "min_ram_gb": 8,
      "recommended_for": ["medical_qa", "clinical_notes"]
    }
  ]
}
```

### POST /api/llm-server/pull
Initiates model download.

**Request:**
```json
{
  "model": "mistral:7b-instruct"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Started pulling mistral:7b-instruct",
  "model": "mistral:7b-instruct"
}
```

### GET /api/llm-server/pull-progress
Tracks download progress.

**Request:**
```
GET /api/llm-server/pull-progress?model=mistral:7b-instruct
```

**Response:**
```json
{
  "status": "pulling",
  "progress": 45,
  "message": "Downloading: 45%"
}
```

**Status values:**
- `pulling` - Download in progress
- `complete` - Pull finished successfully
- `error` - Pull failed
- `unknown` - No pull in progress

---

## Troubleshooting

### "Model not found" when testing
**Problem:** Dropdown is empty or shows error.

**Solution:**
1. Ensure Ollama is running: `systemctl status ollama`
2. Check available models: `ollama list`
3. If no models, pull one first from LLM Server page

### Pull fails with 404
**Problem:** Model name not found in Ollama registry.

**Solution:**
- Only pull models from the "Available to Pull" list
- Check model name is correct (e.g., `mistral:7b-instruct` not `mistral`)
- Ensure internet connection is active

### Pull hangs at 0%
**Problem:** Progress bar stuck at 0%.

**Solution:**
1. Check Ollama logs: `journalctl -u ollama -f`
2. Verify disk space: `df -h`
3. Restart Ollama: `sudo systemctl restart ollama`
4. Try pull from CLI: `ollama pull model-name`

### Model shows in registry but won't start
**Problem:** Model appears installed but server won't start.

**Solution:**
1. Verify with: `ollama list`
2. Test from CLI: `ollama run model-name "test"`
3. Check model is fully downloaded (no corrupted pull)
4. Re-pull if necessary

---

## Best Practices

### Choosing Models

**For Medical/Clinical Tasks:**
- Primary: `biomistral:7b-instruct` (best medical accuracy)
- Alternative: `meditron:7b` (smaller, faster)
- Budget: `clinical-nlp-llama:7b`

**For General Query Optimization:**
- Primary: `mistral:7b-instruct` (best balance)
- Speed: `phi3:mini` (3.8B, very fast)
- Quality: `llama3:8b` (highest quality)

**For Low-Memory Systems:**
- `qwen2:1.5b` (2GB RAM)
- `tinyllama:1.1b` (1GB RAM)
- `gemma:2b` (3GB RAM)

### Disk Space Management

**Model Sizes (approximate):**
- 1B params: 0.8-1.5 GB
- 3B params: 2-3 GB
- 7B params: 4-5 GB
- 13B params: 7-9 GB
- 30B params: 18-22 GB
- 70B params: 40-50 GB

**Recommendations:**
- Keep 10GB free space for system
- Delete unused models: `ollama rm model-name`
- Use quantized versions (Q4_K_M is good balance)
- Monitor with: `du -sh ~/.ollama/models`

### Performance Tuning

**GPU Layers:**
- `-1` (All): Best performance, requires GPU
- `0` (CPU only): Slowest, works anywhere
- `20-40`: Hybrid, good for limited VRAM

**For 8GB VRAM:**
- 7B models: Use all GPU layers
- 13B models: Use 30-35 layers
- 30B models: Use 15-20 layers (or CPU only)

**For 24GB VRAM:**
- Up to 30B: Use all GPU layers
- 70B models: Use 35-40 layers

---

## Command Line Equivalents

All Web UI features can also be done via CLI:

```bash
# List installed models
ollama list

# Pull a model
ollama pull mistral:7b-instruct

# Test a model
ollama run mistral:7b-instruct "What is pneumonia?"

# Remove a model
ollama rm model-name

# Check Ollama status
systemctl status ollama

# View Ollama logs
journalctl -u ollama -f
```

---

## Integration with Evaluator

After pulling models via Web UI, use them in your config:

```yaml
llm_server:
  enabled: true
  backend: ollama
  model: mistral:7b-instruct
  host: localhost
  port: 11434

query_optimization:
  enabled: true
  use_local_server: true
  
judge:
  llm_as_judge:
    enabled: true
    use_local_server: true
```

The evaluator will automatically use the local model instead of OpenAI API.
