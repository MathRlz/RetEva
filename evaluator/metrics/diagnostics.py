"""Diagnostic retrieval/ASR metrics extracted from the evaluation loop.

These are the per-run analysis metrics. Each function here is pure: it takes already-collected
per-query data and returns a plain dict/scalar, so it can be unit-tested in
isolation and reused by any evaluation engine.

Functions:
    - first_relevant_rank_distribution: histogram of first-relevant rank + failure rate
    - wer_recall_correlation: Pearson r between per-query WER and Recall@5
    - categorize_failures: split retrieval misses into asr/embedding/near-miss buckets
    - embedding_alignment: mean/std cosine similarity between paired audio & text embeddings
    - per_speaker_breakdown: group WER / Recall@5 by speaker_id from traces
    - judge_calibration: correlate per-query judge score with Recall@5 / MRR
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def first_relevant_rank_distribution(
    all_retrieved: List[List[str]],
    all_relevant: List[Dict[str, int]],
) -> Tuple[Dict[str, int], float]:
    """Histogram of the rank of the first relevant doc, plus the failure rate.

    Returns:
        (distribution, failure_rate) where distribution has buckets
        "1", "2", "3-5", "6-10", "not_found" and failure_rate is the fraction
        of queries with no relevant doc retrieved.
    """
    rank_dist: Dict[str, int] = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, "not_found": 0}
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        first_rank = None
        for pos, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                first_rank = pos
                break
        if first_rank is None:
            rank_dist["not_found"] += 1
        elif first_rank == 1:
            rank_dist["1"] += 1
        elif first_rank == 2:
            rank_dist["2"] += 1
        elif first_rank <= 5:
            rank_dist["3-5"] += 1
        else:
            rank_dist["6-10"] += 1

    n_queries = len(all_retrieved)
    failure_rate = rank_dist["not_found"] / n_queries if n_queries > 0 else 0.0
    return rank_dist, failure_rate


def wer_recall_correlation(
    wer_scores: List[float],
    per_query_recall: List[float],
) -> Optional[float]:
    """Pearson correlation between per-query WER and Recall@5.

    Returns None when the inputs are misaligned, too few (<5), or either series
    has zero variance (correlation undefined).
    """
    if not wer_scores or len(wer_scores) != len(per_query_recall) or len(wer_scores) < 5:
        return None
    wer_arr = np.array(wer_scores)
    rec_arr = np.array(per_query_recall)
    if wer_arr.std() > 0 and rec_arr.std() > 0:
        return float(np.corrcoef(wer_arr, rec_arr)[0, 1])
    return None


def categorize_failures(
    wer_scores: List[float],
    per_query_recall: List[float],
    rank_dist: Dict[str, int],
    all_relevant: Optional[List[Dict[str, int]]] = None,
    corpus_doc_ids: Optional[set] = None,
) -> Dict[str, int]:
    """Bucket retrieval misses by likely cause.

    - corpus_gap: missed and the relevant doc is not in the corpus at all
    - asr_failure: missed and WER high (>0.3)
    - embedding_mismatch: missed despite near-perfect WER (<0.1)
    - near_miss: partial recall while some relevant docs landed at rank 2-5

    ``corpus_gap`` is only detectable when both ``all_relevant`` (per-query relevance
    maps) and ``corpus_doc_ids`` (the set of doc-ids actually in the corpus / vector
    store) are supplied; it is checked first so a genuine corpus gap is not
    misattributed to ASR/embedding. The key is always present (0 when uncheckable).
    """
    cats: Dict[str, int] = {
        "corpus_gap": 0, "asr_failure": 0, "embedding_mismatch": 0, "near_miss": 0,
    }
    near_miss_possible = rank_dist.get("2", 0) + rank_dist.get("3-5", 0) > 0
    can_check_corpus = corpus_doc_ids is not None and all_relevant is not None
    for i, (wer_v, rec_v) in enumerate(zip(wer_scores, per_query_recall)):
        if rec_v == 0.0 and can_check_corpus and i < len(all_relevant):
            relevant_keys = set(all_relevant[i].keys())
            if relevant_keys and relevant_keys.isdisjoint(corpus_doc_ids):
                cats["corpus_gap"] += 1
                continue
        if rec_v == 0.0 and wer_v > 0.3:
            cats["asr_failure"] += 1
        elif rec_v == 0.0 and wer_v < 0.1:
            cats["embedding_mismatch"] += 1
        elif 0.0 < rec_v < 1.0 and near_miss_possible:
            cats["near_miss"] += 1
    return cats


def embedding_alignment(
    audio_emb: Optional[np.ndarray],
    text_emb: Optional[np.ndarray],
) -> Optional[Dict[str, float]]:
    """Mean/std cosine similarity between paired audio and text embeddings.

    Returns None when either array is missing or empty. Low mean (e.g. < 0.3)
    signals the audio and text embedding spaces are poorly aligned.
    """
    if audio_emb is None or text_emb is None:
        return None
    n_pairs = min(len(audio_emb), len(text_emb))
    if n_pairs == 0:
        return None
    a = audio_emb[:n_pairs]
    t = text_emb[:n_pairs]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    t_norm = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    cos_sims = (a_norm * t_norm).sum(axis=1)
    return {
        "audio_text_cosine_mean": float(cos_sims.mean()),
        "audio_text_cosine_std": float(cos_sims.std()),
    }


def per_speaker_breakdown(traces: List[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, float]]]:
    """Group per-query WER and Recall@5 by speaker_id present in trace metadata.

    Returns None when no trace carries a speaker_id. Exposes accent/speaker bias
    invisible in aggregate metrics.
    """
    spk_wer: Dict[str, List[float]] = {}
    spk_rec: Dict[str, List[float]] = {}
    for trace in traces:
        meta = trace.get("metadata")
        spk = meta.get("speaker_id") if isinstance(meta, dict) else None
        if not spk:
            continue
        spk_wer.setdefault(spk, [])
        spk_rec.setdefault(spk, [])
        if "per_query_wer" in trace:
            spk_wer[spk].append(trace["per_query_wer"])
        if "recall_at_5" in trace:
            spk_rec[spk].append(trace["recall_at_5"])

    if not spk_wer and not spk_rec:
        return None

    all_spks = set(spk_wer) | set(spk_rec)
    return {
        spk: {
            "wer": sum(spk_wer.get(spk, [])) / max(len(spk_wer.get(spk, [])), 1),
            "recall_5": sum(spk_rec.get(spk, [])) / max(len(spk_rec.get(spk, [])), 1),
            "n_queries": max(len(spk_wer.get(spk, [])), len(spk_rec.get(spk, []))),
        }
        for spk in sorted(all_spks)
    }


def judge_calibration(
    judge_scores: List[float],
    per_query_recall: List[float],
    all_retrieved: List[List[str]],
    all_relevant: List[Dict[str, int]],
) -> Dict[str, float]:
    """Correlate per-query judge score with Recall@5 and MRR.

    Returns a dict that may contain "judge_vs_Recall5_correlation" and
    "judge_vs_MRR_correlation". Empty when fewer than 5 valid judge scores or
    judge scores have zero variance.
    """
    out: Dict[str, float] = {}
    scores_arr = np.array(judge_scores)
    valid_mask = ~np.isnan(scores_arr)
    if valid_mask.sum() < 5 or len(per_query_recall) < len(judge_scores):
        return out

    js_clean = scores_arr[valid_mask]
    if js_clean.std() <= 0:
        return out

    rec_for_judge = np.array(per_query_recall[: len(judge_scores)])[valid_mask]
    rr_for_judge = np.array([
        1.0 / (pos + 1) if pos >= 0 else 0.0
        for pos in (
            next((i for i, d in enumerate(all_retrieved[qi]) if d in all_relevant[qi]), -1)
            for qi in range(len(judge_scores))
        )
    ])[valid_mask]

    if rec_for_judge.std() > 0:
        out["judge_vs_Recall5_correlation"] = float(np.corrcoef(js_clean, rec_for_judge)[0, 1])
    if rr_for_judge.std() > 0:
        out["judge_vs_MRR_correlation"] = float(np.corrcoef(js_clean, rr_for_judge)[0, 1])
    return out
