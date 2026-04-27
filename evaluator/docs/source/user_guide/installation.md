# Installation

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)
- 8GB+ RAM (16GB+ recommended for larger models)

## Quick Install

### Basic Installation

Install the full runtime package (all non-test dependencies):

```bash
pip install -e .
```

This installs runtime dependencies including:
- PyTorch
- Transformers
- Sentence-Transformers
- FAISS (CPU version)
- Faster-Whisper
- FastAPI + Uvicorn
- Jupyter/Notebook + visualization stack
- TQDM, PyYAML, Datasets
- Jiwer (for WER/CER metrics)

### Development Installation

Use `dev` extra for tests and developer tooling:

```bash
# Development tools (pytest, black, flake8, mypy)
pip install -e ".[dev]"
```

### Optional GPU FAISS

If you want FAISS GPU acceleration, install it manually in your environment:

```bash
pip install faiss-gpu
```

## Manual Installation

If you prefer to install dependencies manually:

```bash
# Core dependencies
pip install torch transformers sentence-transformers faiss-cpu tqdm pyyaml datasets jiwer numpy

# GPU-accelerated FAISS (replaces faiss-cpu)
pip install faiss-gpu

# Faster Whisper
pip install faster-whisper

# Web API + visualization stack (already included in default install)
pip install fastapi "uvicorn[standard]" jupyter notebook ipywidgets matplotlib seaborn plotly pandas scipy kaleido
```

## Environment Configuration

### Data Directory

By default, the evaluator looks for data in the current directory. You can set a custom data directory:

```bash
export EVALUATOR_DATA_DIR=/path/to/your/data
```

### CUDA Setup

For GPU acceleration, ensure CUDA is properly installed:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### Hugging Face Cache

Models are cached in `~/.cache/huggingface/`. To change the cache directory:

```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

## Verification

Verify your installation:

```bash
# Test basic import
python -c "from evaluator import evaluate_from_preset, list_presets; print(list_presets())"

# Run a simple evaluation (requires data)
python -c "from evaluator import list_presets; print('Available presets:', list_presets())"
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

1. Use CPU devices in config:
   ```yaml
   model:
     asr_device: cpu
     text_emb_device: cpu
   ```

2. Reduce batch size:
   ```yaml
   data:
     batch_size: 8
   ```

3. Enable caching to avoid recomputation:
   ```yaml
   cache:
     enabled: true
     cache_embeddings: true
   ```

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Check installed packages
pip list | grep -E "torch|transformers|sentence-transformers"

# Reinstall missing packages
pip install torch transformers sentence-transformers
```

### FAISS GPU Issues

If FAISS GPU fails to load:

```bash
# Fall back to CPU FAISS
pip uninstall faiss-gpu
pip install faiss-cpu

# Or in config:
vector_db:
  type: faiss  # instead of faiss_gpu
```

### PyTorch CUDA Mismatch

If CUDA versions don't match:

```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

- See [Quickstart](quickstart.md) for your first evaluation
- Read [Configuration](configuration.md) for detailed setup options
- Explore [Models](models.md) for available ASR and embedding models
