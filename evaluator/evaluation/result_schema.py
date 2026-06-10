"""Typed result object returned by ``run_graph``.

The evaluation result was historically a plain ``dict[str, Any]`` that accreted
~40 heterogeneous keys, so consumers had to guess names. ``RunResults`` is a
``dict`` **subclass**: it behaves exactly like the old dict (``results["MRR"]``,
``.get(...)``, ``.update(...)``, iteration, ``json.dumps`` all work unchanged, so
nothing downstream breaks and it is its own serialized form at the edge), but it
also exposes typed attribute/accessor methods for the well-known metrics so
callers get IDE/type-checker support instead of guessing string keys.

Dynamic, k-valued metrics (``Recall@1/5/10``, ``NDCG@k`` …) are reached via the
``recall_at`` / ``ndcg_at`` / ``precision_at`` helpers.

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

from typing import Any, Dict, List, Optional, Tuple


class RunResults(dict):
    """Dict of evaluation metrics with typed accessors for the common keys."""

    # --- meta ---
    @property
    def pipeline_mode(self) -> Optional[str]:
        return self.get("pipeline_mode")

    @property
    def oracle_mode(self) -> bool:
        return bool(self.get("oracle_mode", False))

    @property
    def latency(self) -> Dict[str, float]:
        return self.get("latency", {})

    # --- ASR ---
    @property
    def wer(self) -> Optional[float]:
        return self.get("WER")

    @property
    def cer(self) -> Optional[float]:
        return self.get("CER")

    @property
    def tw_wer(self) -> Optional[float]:
        return self.get("TW_WER")

    # --- IR ---
    @property
    def mrr(self) -> Optional[float]:
        return self.get("MRR")

    @property
    def map(self) -> Optional[float]:
        return self.get("MAP")

    def recall_at(self, k: int) -> Optional[float]:
        return self.get(f"Recall@{k}")

    def ndcg_at(self, k: int) -> Optional[float]:
        return self.get(f"NDCG@{k}")

    def precision_at(self, k: int) -> Optional[float]:
        return self.get(f"Precision@{k}")

    # --- diagnostics ---
    @property
    def retrieval_failure_rate(self) -> Optional[float]:
        return self.get("retrieval_failure_rate")

    @property
    def per_speaker(self) -> Optional[Dict[str, Dict[str, float]]]:
        return self.get("per_speaker")

    def ci(self, metric: str) -> Optional[Tuple[float, float]]:
        """Bootstrap confidence interval for a metric, e.g. ``ci("MRR")``."""
        return self.get(f"{metric}_ci")

    # --- optional feature outputs ---
    @property
    def query_traces(self) -> Optional[List[Dict[str, Any]]]:
        return self.get("query_traces")

    @property
    def llm_judge(self) -> Optional[Dict[str, Any]]:
        return self.get("llm_judge")

    @property
    def answer_generation(self) -> Optional[Dict[str, Any]]:
        return self.get("answer_generation")

    # --- oracle baseline ---
    @property
    def asr_degradation_factor(self) -> Optional[float]:
        return self.get("asr_degradation_factor")

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` copy (e.g. for storage that rejects subclasses)."""
        return dict(self)
