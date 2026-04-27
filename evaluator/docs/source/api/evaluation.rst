Evaluation API Reference
========================

.. _api-evaluation:

Main Evaluation Functions
--------------------------

.. autofunction:: evaluator.evaluate
.. autofunction:: evaluator.evaluate_from_config
.. autofunction:: evaluator.evaluate_from_preset
.. autofunction:: evaluator.quick_evaluate
.. autofunction:: evaluator.evaluate_phased

Evaluation Results
-------------------

.. autoclass:: evaluator.EvaluationResults
   :members:
   :undoc-members:
   :show-inheritance:

Metrics Calculation
--------------------

Information Retrieval Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: evaluator.metrics.ir
   :members:
   :undoc-members:
   :show-inheritance:

Speech-to-Text Metrics
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: evaluator.metrics.stt
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Module
------------------

.. automodule:: evaluator.evaluation
   :members:
   :undoc-members:
   :show-inheritance:
