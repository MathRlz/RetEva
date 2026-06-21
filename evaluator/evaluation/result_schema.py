"""Typed result object returned by ``run_graph``.

``RunResults`` is a ``dict`` **subclass**: it behaves exactly like the old dict
(``results["MRR"]``, ``.get(...)``, ``.update(...)``, iteration, ``json.dumps`` all
work unchanged, so nothing downstream breaks and it is its own serialized form at the
edge). It carries the well-known metric keys below; consumers read them by key.

Key groups (all optional — present only for the relevant mode / enabled feature):
    meta        pipeline_mode, phased, oracle_mode, latency
    ASR         asr, WER, CER, TW_WER
    embedding   embedder, audio_embedder, text_embedder, embedding_alignment
    IR          MRR, MAP, Recall@{1,5,10}, NDCG@{1,5,10}, Precision@{1,5,10}
    diagnostics wer_recall5_correlation, first_relevant_rank_distribution,
                retrieval_failure_rate, retrieval_failure_analysis, per_speaker,
                MRR_ci, Recall@5_ci, NDCG@5_ci
    features    answer_generation, query_traces, llm_judge,
                judge_vs_Recall5_correlation, judge_vs_MRR_correlation
    oracle      oracle_MRR, oracle_Recall@5, oracle_NDCG@5, asr_degradation_factor

Boundary vs :class:`~evaluator.evaluation.results.EvaluationResults` (T1): ``RunResults`` is
the **internal** metrics payload the executor returns (a plain mapping at the engine edge);
``EvaluationResults`` is the **public** wrapper the API builds from it, adding the
``EvaluationConfig`` + run metadata + JSON file I/O. Kept separate by design (see that class).
"""


class RunResults(dict):
    """Dict of evaluation metrics — the executor's internal result payload (see module docstring
    for the key groups). Consumers read it by key, so it stays a plain ``dict`` subclass."""
