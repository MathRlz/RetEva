Models API Reference
====================

.. _api-models:

ASR Models
----------

.. automodule:: evaluator.models
   :members:
   :undoc-members:
   :show-inheritance:

Whisper (OpenAI)
~~~~~~~~~~~~~~~~~

.. autoclass:: evaluator.models.WhisperModel
   :members:
   :undoc-members:
   :show-inheritance:

Wav2Vec2 (Facebook)
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evaluator.models.Wav2Vec2Model
   :members:
   :undoc-members:
   :show-inheritance:

Text Embedding Models
---------------------

LaBSE
~~~~~~

.. autoclass:: evaluator.models.LabseModel
   :members:
   :undoc-members:
   :show-inheritance:

Jina V4
~~~~~~~~

.. autoclass:: evaluator.models.JinaV4Model
   :members:
   :undoc-members:
   :show-inheritance:

Nemotron
~~~~~~~~~

.. autoclass:: evaluator.models.NemotronModel
   :members:
   :undoc-members:
   :show-inheritance:

BGE-M3
~~~~~~~~

.. autoclass:: evaluator.models.BgeM3Model
   :members:
   :undoc-members:
   :show-inheritance:

CLIP
~~~~~~

.. autoclass:: evaluator.models.ClipModel
   :members:
   :undoc-members:
   :show-inheritance:

Audio Embedding Models
----------------------

Attention Pooling
~~~~~~~~~~~~~~~~~~~

.. autoclass:: evaluator.models.AttentionPoolAudioModel
   :members:
   :undoc-members:
   :show-inheritance:

Model Factory
--------------

.. autofunction:: evaluator.create_asr_model
.. autofunction:: evaluator.create_text_embedding_model
.. autofunction:: evaluator.create_audio_embedding_model
