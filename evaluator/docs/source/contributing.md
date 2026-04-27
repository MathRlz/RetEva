Contributing
=============

We welcome contributions to the Medical Speech Retrieval Evaluator! This document provides guidelines for contributing to the project.

## Setting Up Development Environment

### Prerequisites
- Python 3.8 or higher
- Git
- pip and virtualenv (recommended)

### Installation for Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/evaluator.git
cd evaluator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install runtime package + development tools:
```bash
pip install -e ".[dev]"
```

4. Install additional development tools:
```bash
pip install sphinx sphinx_rtd_theme myst-parser sphinx-autodoc-typehints
```

## Running Tests

We use pytest for testing. Run tests with:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=evaluator --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::test_whisper_transcription
```

Tests are located in the `tests/` directory and should follow pytest conventions.

## Code Style Guidelines

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function arguments and return values
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Formatting
- Use `black` for code formatting:
```bash
black evaluator/ tests/
```

- Use `flake8` for linting:
```bash
flake8 evaluator/ tests/
```

- Use `mypy` for type checking:
```bash
mypy evaluator/
```

### Documentation
- Write docstrings using Google style format
- Include parameter descriptions, return types, and examples
- Use type hints in function signatures
- Document exceptions that can be raised

Example docstring:
```python
def evaluate_model(config: EvaluationConfig) -> EvaluationResults:
    """Evaluate a model using the provided configuration.
    
    Args:
        config: Evaluation configuration containing model, data, and pipeline settings.
        
    Returns:
        EvaluationResults containing MRR, MAP, NDCG, Recall, and WER metrics.
        
    Raises:
        ConfigurationError: If configuration is invalid.
        EvaluationError: If evaluation fails during processing.
    """
```

## Extensibility quick guides (core-first)

### Add new dataset source/type

1. Add/extend loader in `evaluator/datasets/loaders/`.
2. Register runtime spec in `evaluator/datasets/runtime.py` (`DatasetRuntimeSpec` list + resolver).
3. Ensure adapter output matches `QueryDataset` contract (`AudioSamplesQueryDataset` or dataset class with `__getitem__` keys).
4. If retrieval mode used, provide corpus path support (`doc_id` + `text`).
5. Add tests:
   - `tests/test_dataset_runtime_registry.py`
   - `tests/test_evaluation_service_dataset_loading.py`
   - WebAPI preflight path in `tests/test_webapi.py` (400 on missing required fields).

Smoke test:
```bash
pytest -q tests/test_dataset_runtime_registry.py tests/test_evaluation_service_dataset_loading.py
```

### Add new model family/model type

1. Register model class with registry decorator in `evaluator/models/registry.py` path usage.
2. If model requires custom initialization, add builder hook in `evaluator/models/factory.py` (audio uses registered builder hooks).
3. Ensure `ModelServiceProvider.list_available_models()` emits normalized metadata entry:
   - `type`, `name`, `capabilities`, `requires_path`, `default_device_hint`.
4. Add tests in `tests/test_model_service_provider.py`.

Smoke test:
```bash
pytest -q tests/test_model_service_provider.py
```

### Add new pipeline mode

1. Add mode in config enum (`evaluator/config/types.py`).
2. Add declarative mode spec in `evaluator/pipeline/stage_graph.py` (`PipelineModeSpec` registry).
3. Wire pipeline construction in `evaluator/pipeline/factory.py`.
4. Update validation/compatibility checks in services if required.
5. Add tests in:
   - `tests/test_stage_graph.py`
   - integration tests for execution path.

Smoke test:
```bash
pytest -q tests/test_stage_graph.py
```

## How to Add New Models

### Adding ASR Models

1. Create a new file in `evaluator/models/` (e.g., `new_asr_model.py`):

```python
from evaluator.models.base import ASRModel

class NewASRModel(ASRModel):
    """Your ASR model implementation.
    
    Args:
        model_name: Model identifier (e.g., 'new-asr-v1').
        device: Device for inference (e.g., 'cuda:0' or 'cpu').
    """
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        super().__init__(model_name, device)
        # Load your model here
        
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Transcribed text.
        """
        # Implementation here
        pass
        
    def estimate_memory_gb(self) -> float:
        """Estimate GPU memory required in GB."""
        return 4.0  # Example
```

2. Register the model in `evaluator/models/__init__.py`:

```python
from evaluator.models.new_asr_model import NewASRModel

__all__ = [
    # ... existing models ...
    'NewASRModel',
]
```

3. Add factory method in `evaluator/models/factory.py`:

```python
def create_asr_model(model_type: str, **kwargs) -> ASRModel:
    """Create ASR model instance."""
    if model_type == 'new_asr':
        return NewASRModel(**kwargs)
    # ... other models ...
```

### Adding Text Embedding Models

Follow similar steps in `evaluator/models/` but inherit from `TextEmbeddingModel`. Key methods to implement:

- `encode(texts: List[str]) -> np.ndarray`: Encode list of texts to embeddings
- `batch_encode(texts: List[str], batch_size: int) -> np.ndarray`: Batch encoding with size limit
- `estimate_memory_gb() -> float`: Memory requirement estimate

### Adding Audio Embedding Models

Inherit from `AudioEmbeddingModel` and implement:

- `encode_audio(audio_path: str) -> np.ndarray`: Encode single audio file
- `batch_encode_audio(audio_paths: List[str], batch_size: int) -> np.ndarray`: Batch encoding
- `estimate_memory_gb() -> float`: Memory requirement estimate

## How to Add New Pipeline Modes

Pipeline modes define the complete evaluation workflow. To add a new mode:

1. Create a new pipeline class in `evaluator/pipeline/`:

```python
from evaluator.pipeline.base import Pipeline

class CustomPipeline(Pipeline):
    """Custom evaluation pipeline."""
    
    def process(self, input_data):
        """Process input through the pipeline."""
        pass
```

2. Register in the pipeline factory in `evaluator/pipeline/__init__.py`

3. Add configuration option to `evaluator/config.py`

4. Update `create_pipeline_from_config()` in factory functions

## How to Add New Model Service / Provider Binding

Service-managed runtime now resolves models through `ModelServiceProvider`.
When adding new model families or providers, bind them in one place.

1. Add or update service adapter in:
   - `evaluator/services/model_services.py`
2. Add provider wiring in:
   - `evaluator/services/model_provider.py`
   - include:
     - key shape for reuse/cache identity
     - `get_*` constructor path
     - optional `move_*` and `release_*` controls
3. If model family should be discoverable, update:
   - `ModelServiceProvider.list_available_models()`
4. Add tests in:
   - `tests/test_model_service_provider.py`
   - include reuse + release (+ move if device-backed)

## Creating Tests

When adding new features, write tests following these guidelines:

```python
import pytest
from evaluator import create_asr_model

class TestNewModel:
    """Tests for NewModel."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return create_asr_model('new_asr')
    
    def test_model_initialization(self, model):
        """Test model can be initialized."""
        assert model is not None
        
    def test_transcription(self, model, sample_audio):
        """Test transcription functionality."""
        result = model.transcribe(sample_audio)
        assert isinstance(result, str)
        assert len(result) > 0
```

## Documentation

When adding features, please update documentation:

1. **API Documentation**: Update docstrings in your code
2. **User Guide**: Add/update relevant markdown files in `docs/source/user_guide/`
3. **API Reference**: The RST files in `docs/source/api/` are auto-generated from docstrings

Build documentation locally:
```bash
cd docs
make html
# View in browser: open build/html/index.html
```

## Pull Request Process

1. Create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and write tests

3. Run linting and tests:
```bash
black evaluator/ tests/
flake8 evaluator/ tests/
mypy evaluator/
pytest --cov=evaluator
```

4. Update documentation if needed

5. Commit with clear messages:
```bash
git commit -m "Add feature: description of what was added"
```

6. Include Co-authored-by trailer in commit message:
```
Add feature: description

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

7. Push to your fork and create a Pull Request

8. Ensure all CI/CD checks pass

9. Respond to review comments

10. Once approved, your PR will be merged!

## Reporting Issues

Found a bug? Please create an issue with:

- Clear title describing the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, GPU info)
- Error messages and stack traces if available

Example:
```
Title: Whisper transcription fails with non-English audio

Steps:
1. Load Spanish language audio file
2. Run evaluation with whisper model
3. See error in logs

Expected: Transcription in Spanish
Actual: RuntimeError about language detection

Environment:
- Python 3.10
- torch 2.0
- CUDA 11.8
```

## Code of Conduct

- Be respectful and inclusive
- No harassment or discrimination
- Welcome diverse perspectives
- Focus on the code, not the person
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to ask questions in issues or discussions. We're here to help!

Thank you for contributing to the Medical Speech Retrieval Evaluator! 🎉
