"""Cache key generation utilities.

Provides standardized key generation functions for various cache types.
All keys are MD5 hashes computed from deterministic input parameters.
"""
import json
import hashlib
from typing import Optional, Any, Dict

CACHE_SCHEMA_VERSION = "v1"


def _compute_hash(*args: Any) -> str:
    """
    Compute MD5 hash from arguments.
    
    Args:
        *args: Variable arguments to hash
        
    Returns:
        MD5 hash as hexadecimal string
    """
    content = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()


def _canonicalize(value: Any) -> Any:
    """Convert nested structures into deterministic JSON-serializable form."""
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted(_canonicalize(v) for v in value)
    return value


def manifest_fingerprint(manifest: Dict[str, Any], schema_version: str = CACHE_SCHEMA_VERSION) -> str:
    """Compute stable fingerprint for cache manifest payloads."""
    canonical_manifest = _canonicalize(manifest)
    return _compute_hash(schema_version, canonical_manifest)


def dataset_fingerprint(
    dataset_identity: str,
    *,
    trace_limit: Optional[int] = None,
    source: Optional[Dict[str, Any]] = None,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Fingerprint dataset identity and source-related settings."""
    return manifest_fingerprint(
        {
            "kind": "dataset",
            "dataset_identity": dataset_identity,
            "trace_limit": trace_limit,
            "source": source or {},
        },
        schema_version=schema_version,
    )


def model_fingerprint(
    model_name: str,
    *,
    model_type: Optional[str] = None,
    inference: Optional[Dict[str, Any]] = None,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Fingerprint model identity and inference-relevant options."""
    return manifest_fingerprint(
        {
            "kind": "model",
            "model_name": model_name,
            "model_type": model_type,
            "inference": inference or {},
        },
        schema_version=schema_version,
    )


def retrieval_fingerprint(
    *,
    vector_store_type: str,
    retrieval_strategy: Dict[str, Any],
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Fingerprint retrieval stack configuration."""
    return manifest_fingerprint(
        {
            "kind": "retrieval",
            "vector_store_type": vector_store_type,
            "strategy": retrieval_strategy,
        },
        schema_version=schema_version,
    )


def preprocessing_fingerprint(
    preprocessing: Dict[str, Any],
    *,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Fingerprint preprocessing stages that impact cached artifacts."""
    return manifest_fingerprint(
        {
            "kind": "preprocessing",
            "preprocessing": preprocessing,
        },
        schema_version=schema_version,
    )


def model_key(audio_hash: str, model_name: str) -> str:
    """
    Generate cache key for ASR model features.
    
    Args:
        audio_hash: Hash of the audio data
        model_name: Name/identifier of the ASR model
        
    Returns:
        Cache key string
        
    Example:
        >>> model_key("a1b2c3", "openai/whisper-small")
        '5d41402abc4b2a76b9719d911017c592'
    """
    return _compute_hash(audio_hash, model_name)


def embedding_key(text: str, model_name: str) -> str:
    """
    Generate cache key for text embeddings.
    
    Args:
        text: Text to embed
        model_name: Name/identifier of the embedding model
        
    Returns:
        Cache key string
        
    Example:
        >>> embedding_key("hello world", "sentence-transformers/labse")
        '7d793037a0760186574b0282f2f435e7'
    """
    return _compute_hash(text, model_name)


def transcription_key(audio_hash: str, model_name: str, language: Optional[str] = None) -> str:
    """
    Generate cache key for transcriptions.
    
    Args:
        audio_hash: Hash of the audio data
        model_name: Name/identifier of the ASR model
        language: Optional language code (for multilingual models)
        
    Returns:
        Cache key string
        
    Example:
        >>> transcription_key("a1b2c3", "openai/whisper-small", "en")
        '9bf31c7ff062936a96d3c8bd1f8f2ff3'
        >>> transcription_key("a1b2c3", "openai/whisper-small")
        '098f6bcd4621d373cade4e832627b4f6'
    """
    return _compute_hash(audio_hash, model_name, language)


def audio_embedding_key(audio_hash: str, model_name: str) -> str:
    """
    Generate cache key for audio embeddings.
    
    Args:
        audio_hash: Hash of the audio data
        model_name: Name/identifier of the audio embedding model
        
    Returns:
        Cache key string
        
    Example:
        >>> audio_embedding_key("a1b2c3", "clap-audio-encoder")
        'c4ca4238a0b923820dcc509a6f75849b'
    """
    return _compute_hash(audio_hash, model_name)


def vector_db_key(dataset_name: str, dataset_size: int, model_name: str) -> str:
    """
    Generate cache key for vector database.
    
    Args:
        dataset_name: Name of the dataset
        dataset_size: Number of documents in the dataset
        model_name: Name/identifier of the embedding model used
        
    Returns:
        Cache key string
        
    Example:
        >>> vector_db_key("pubmed", 1000, "labse")
        'c81e728d9d4c2f636f067f89cc14862c'
    """
    return _compute_hash(dataset_name, dataset_size, model_name)


def vector_db_manifest_key(
    *,
    dataset_fp: str,
    model_fp: str,
    retrieval_fp: str,
    preprocessing_fp: Optional[str] = None,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Generate canonical vector DB key from cache manifest fingerprints."""
    return _compute_hash(
        schema_version,
        dataset_fp,
        model_fp,
        retrieval_fp,
        preprocessing_fp or "",
    )


def unique_texts_key(dataset_name: str, dataset_size: int) -> str:
    """
    Generate cache key for unique texts list.
    
    Args:
        dataset_name: Name of the dataset
        dataset_size: Total number of samples in dataset
        
    Returns:
        Cache key string
        
    Example:
        >>> unique_texts_key("pubmed", 5000)
        'e4da3b7fbbce2345d7772b0674a318d5'
    """
    return _compute_hash(dataset_name, dataset_size)


def unique_texts_manifest_key(
    *,
    dataset_fp: str,
    preprocessing_fp: Optional[str] = None,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Generate canonical unique-text cache key from manifest fingerprints."""
    return _compute_hash(schema_version, dataset_fp, preprocessing_fp or "")
