# Changelog

## [0.1.0] - 2024-03-31

### Added
- **Initial Release**: Complete evaluation framework for medical speech retrieval
- **Multi-Pipeline Support**: ASR→Text→Embedding, Audio→Embedding, and ASR-only modes
- **ASR Models**: Whisper, Wav2Vec2, Faster-Whisper support
- **Text Embedding Models**: LaBSE, Jina V4, Nemotron, BGE-M3, CLIP
- **Audio Embedding Models**: Attention pooling and CLAP-style embeddings
- **Vector Stores**: In-memory, FAISS (CPU/GPU), with customizable indices
- **Retrieval Strategies**: Dense, sparse, and hybrid retrieval with re-ranking
- **Metrics Calculation**: 
  - Information Retrieval: MRR, MAP, NDCG@k, Recall@k, Precision@k
  - Speech Recognition: WER (Word Error Rate), CER (Character Error Rate)
- **Result Caching**: Efficient caching of embeddings and transcriptions
- **Device Management**: GPU pooling with strategies (RoundRobin, MemoryAware, PackingStrategy)
- **MLflow Integration**: Experiment tracking and logging
- **Configuration System**: YAML-based configuration with presets
- **CLI Interface**: Command-line evaluation with parameter overrides
- **Presets**: Predefined configurations (whisper_labse, wav2vec_jina, audio_only, fast_dev)
- **Data Loaders**: PubMedQA and ADMED dataset support
- **Jupyter Notebooks**: Interactive analysis and visualization notebooks
- **Web UI**: Flask-based web interface for running evaluations
- **Ablation Studies**: Component analysis and impact measurement
- **Statistical Testing**: Bootstrap CI, paired t-tests, Wilcoxon tests
- **Benchmark Suite**: Performance benchmarking utilities
- **Documentation**: Comprehensive API reference and user guides
- **Tests**: Unit tests with pytest and coverage reporting

### Features by Module

#### Core Evaluation (`evaluator.py`)
- `evaluate()`: Main evaluation function
- `evaluate_from_config()`: Configuration-based evaluation
- `evaluate_from_preset()`: Preset-based quick evaluation
- `quick_evaluate()`: Simple one-liner evaluation
- `evaluate_phased()`: Step-by-step evaluation with intermediate results

#### Configuration (`evaluator/config.py`)
- Comprehensive configuration management
- Type-safe config classes
- YAML parsing and validation
- Environment variable support
- Device and GPU pooling configuration

#### Models (`evaluator/models/`)
- Factory functions for creating models
- Consistent model interface
- Memory estimation utilities
- Batch processing support

#### Pipelines (`evaluator/pipeline/`)
- ASR pipeline: Audio→Text
- Text embedding: Text→Vectors
- Audio embedding: Audio→Vectors
- Retrieval: Vector search and ranking
- Pipeline bundling for coordinated execution

#### Vector Stores (`evaluator/storage/backends/`)
- In-memory store for small datasets
- FAISS for efficient similarity search
- GPU-accelerated FAISS for large-scale retrieval
- Configurable index types (Flat, IVFFlat, HNSW)

#### Retrieval (`evaluator/retrieval/`)
- Dense retrieval with vector similarity
- Sparse retrieval with BM25
- Hybrid retrieval combining both approaches
- Re-ranking modules

#### Metrics (`evaluator/ir_metrics.py`, `evaluator/stt_metrics.py`)
- IR metrics: MRR, MAP, NDCG, Recall, Precision, F1
- STT metrics: WER, CER with reference transcriptions

#### Device Management (`evaluator/devices/`)
- GPU monitoring and pooling
- Multiple allocation strategies
- Memory awareness
- Device utilization tracking

#### Tracking (`evaluator/tracking/`)
- MLflow experiment tracking
- Custom tracker interface
- Result logging and persistence

#### Analysis (`evaluator/analysis/`)
- Result comparison utilities
- Statistical functions
- Performance analysis

#### Ablation (`evaluator/ablation/`)
- Component impact analysis
- Grid search ablation
- Report generation

#### Visualization (`evaluator/visualization/`)
- Plot generation utilities
- Statistical visualization
- Interactive visualizations

### Fixed
- N/A (initial release)

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Security
- Input validation in configuration parsing
- Safe YAML loading to prevent code injection

## Future Roadmap

### Planned Features
- Distributed evaluation across multiple GPUs/machines
- Additional LLM evaluation metrics
- Real-time streaming evaluation
- Model quantization support (INT8, FP16)
- Additional embedding models (OpenAI, Cohere, etc.)
- Advanced re-ranking algorithms
- Interactive result visualization dashboard
- Automated model selection based on corpus
- Cross-lingual evaluation support

### Known Limitations
- Single-machine GPU pooling only (distributed training planned)
- FAISS GPU support requires specific CUDA versions
- Some models may require significant GPU memory for batch processing

---

For detailed information, see the documentation at `docs/source/` or build HTML docs with `make html` in the docs directory.
