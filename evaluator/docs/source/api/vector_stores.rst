Vector Stores API Reference
===========================

.. _api-vector-stores:

Vector Store Base
------------------

.. automodule:: evaluator.storage.backends
   :members:
   :undoc-members:
   :show-inheritance:

In-Memory Vector Store
-----------------------

.. autoclass:: evaluator.InMemoryVectorStore
   :members:
   :undoc-members:
   :show-inheritance:

FAISS Vector Store
-------------------

.. autoclass:: evaluator.FaissVectorStore
   :members:
   :undoc-members:
   :show-inheritance:

FAISS GPU Vector Store
-----------------------

.. autoclass:: evaluator.FaissGpuVectorStore
   :members:
   :undoc-members:
   :show-inheritance:

Vector Store Factory
---------------------

.. autofunction:: evaluator.create_vector_store_from_config
