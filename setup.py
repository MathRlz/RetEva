"""Setup file for evaluator package."""

from setuptools import setup, find_packages

setup(
    name="evaluator",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core ML
        "torch>=2.0,<3",
        "transformers>=4.30,<5",
        "sentence-transformers>=2.2,<4",
        "faiss-cpu>=1.7,<2",
        "faster-whisper>=0.9,<2",
        # Data & metrics
        "datasets>=2.14",
        "jiwer>=3.0",
        "numpy>=1.24",
        "scipy>=1.9",
        "scikit-learn>=1.2",
        "rank-bm25>=0.2",
        "pandas>=1.5",
        # Visualization
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "plotly>=5.14",
        "kaleido>=0.2",
        # Utilities
        "tqdm>=4.64",
        "pyyaml>=6.0",
    ],
    extras_require={
        "webapi": [
            "fastapi>=0.100",
            "uvicorn[standard]>=0.20",
        ],
        "notebooks": [
            "jupyter",
            "notebook",
            "ipywidgets",
        ],
        "chromadb": [
            "chromadb>=0.4",
        ],
        "qdrant": [
            "qdrant-client>=1.6",
        ],
        "backends": [
            "chromadb>=0.4",
            "qdrant-client>=1.6",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "all": [
            "fastapi>=0.100",
            "uvicorn[standard]>=0.20",
            "jupyter",
            "notebook",
            "ipywidgets",
            "chromadb>=0.4",
            "qdrant-client>=1.6",
        ],
    },
    description="Evaluation framework for medical speech retrieval",
    author="Krystian",
    license="MIT",
    entry_points={
        "console_scripts": [
            "evaluator-webapi=evaluator.webapi.__main__:main",
        ],
    },
)
