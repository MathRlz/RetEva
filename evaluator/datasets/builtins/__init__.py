"""Built-in datasets — one module per dataset, against the per-type ABC extension API.

These are the canonical worked examples for adding a dataset: pick the type base
(:class:`~evaluator.datasets.types.AudioTranscriptionDataset` / ``AudioRetrievalDataset`` /
``TextRetrievalDataset`` / ``MultimodalQADataset``), set the class attributes that differ from
the type's defaults, implement ``from_config``, and decorate with ``@register_eval_dataset``.
The :class:`DatasetDescriptor` is derived from the class — capability metadata has one author.

Importing this package registers every built-in (import side-effect); ``datasets/__init__`` does
so. **To add a dataset: drop a new module here and import it below.**
"""

from . import (  # noqa: F401 — import side-effect registers each dataset
    admed_voice,
    fleurs,
    hani_medical,
    huggingface_audio,
    local_audio,
    pubmed_qa,
)
