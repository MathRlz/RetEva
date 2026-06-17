"""Answer generation (RAG Phase 4.5).

Two modes, selected by whether a context (retrieved docs) is present:

* **RAG / open-book** — a ``retrieved`` edge feeds the ``generate`` node; the model answers
  *grounded in* the retrieved passages (the default prompts say "use ONLY the context").
* **Closed-book** — no ``retrieved`` edge (no-corpus dataset, or a graph wired without a
  retrieval node); the context is empty, so the model answers from its own parametric
  knowledge. The prompts switch to a no-context variant (no dangling "Context:" block, no
  "use only the context" instruction), and the hallucination heuristic is N/A (there is no
  context to ground against). This quantifies how much retrieval *adds* over the model alone.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise medical research assistant. "
    "Answer based ONLY on the provided context. Be brief and direct."
)

DEFAULT_PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

_COT_SYSTEM_PROMPT = (
    "You are a medical research assistant. "
    "Think step by step using ONLY the provided context, then give a concise answer."
)

_COT_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step:\n1. What does the context say?\n2. What is the direct answer?\n\n"
    "Final answer:"
)

# Closed-book (no retrieved context): answer from the model's own knowledge. No "Context:"
# block, no "use only the context" instruction (which is contradictory with no context).
_CLOSED_BOOK_SYSTEM_PROMPT = (
    "You are a concise medical research assistant. "
    "Answer from your own knowledge. Be brief and direct."
)

_CLOSED_BOOK_PROMPT_TEMPLATE = "Question: {question}\n\nAnswer:"

_CLOSED_BOOK_COT_SYSTEM_PROMPT = (
    "You are a medical research assistant. "
    "Think step by step from your own knowledge, then give a concise answer."
)

_CLOSED_BOOK_COT_PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "Think step by step:\n1. What do you know?\n2. What is the direct answer?\n\n"
    "Final answer:"
)


def _prompt_defaults(method: str, closed_book: bool) -> Tuple[str, str]:
    """Default (system_prompt, prompt_template) for a method, in the right context mode.

    A user-set ``config.system_prompt`` / ``config.prompt_template`` still overrides these;
    the closed-book templates simply drop the ``{context}`` placeholder (``str.format``
    ignores the unused ``context`` kwarg)."""
    if method == "chain_of_thought":
        return (
            (_CLOSED_BOOK_COT_SYSTEM_PROMPT, _CLOSED_BOOK_COT_PROMPT_TEMPLATE)
            if closed_book
            else (_COT_SYSTEM_PROMPT, _COT_PROMPT_TEMPLATE)
        )
    return (
        (_CLOSED_BOOK_SYSTEM_PROMPT, _CLOSED_BOOK_PROMPT_TEMPLATE)
        if closed_book
        else (DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_TEMPLATE)
    )


def _build_context(
    retrieved_payloads: List[Dict[str, Any]],
    max_docs: int,
    max_chars: int,
) -> Tuple[str, List[str]]:
    """Build context string from top retrieved docs. Returns (context_text, doc_ids)."""
    parts, doc_ids = [], []
    for payload in retrieved_payloads[:max_docs]:
        doc_id = str(payload.get("doc_id", payload.get("id", "")))
        text = payload.get("text", payload.get("content", ""))
        if text:
            parts.append(text[:max_chars])
            doc_ids.append(doc_id)
    return "\n\n".join(parts), doc_ids


def generate_single_answer(
    question: str,
    retrieved_payloads: List[Dict[str, Any]],
    config,
    client: LLMClient,
) -> Dict[str, Any]:
    """Generate answer for one question using retrieved context.

    Returns dict with keys: generated_answer, method, context_doc_ids.
    """
    context, doc_ids = _build_context(
        retrieved_payloads, config.context_docs, config.context_max_chars
    )
    # No context ⇒ closed-book: switch to the no-context prompts (answer from own knowledge).
    closed_book = not context
    method = config.method

    if method == "simple":
        d_sys, d_tmpl = _prompt_defaults("simple", closed_book)
        sys_p = config.system_prompt or d_sys
        tmpl = config.prompt_template or d_tmpl
        msgs = [
            {"role": "system", "content": sys_p},
            {
                "role": "user",
                "content": tmpl.format(question=question, context=context),
            },
        ]
        answer = client.call(msgs)

    elif method == "chain_of_thought":
        d_sys, d_tmpl = _prompt_defaults("chain_of_thought", closed_book)
        sys_p = config.system_prompt or d_sys
        tmpl = config.prompt_template or d_tmpl
        msgs = [
            {"role": "system", "content": sys_p},
            {
                "role": "user",
                "content": tmpl.format(question=question, context=context),
            },
        ]
        answer = client.call(msgs)

    elif method == "multi_query":
        # Rephrase → answer each → synthesize
        rephrase_msgs = [
            {
                "role": "system",
                "content": (
                    "Rephrase the following medical question in 3 different ways. "
                    "Return only the rephrased questions, one per line, no numbering."
                ),
            },
            {"role": "user", "content": question},
        ]
        rephrases_raw = client.call(rephrase_msgs)
        rephrases = [q.strip() for q in rephrases_raw.splitlines() if q.strip()][:3]
        if not rephrases:
            rephrases = [question]

        d_sys, d_tmpl = _prompt_defaults("simple", closed_book)
        base_sys = config.system_prompt or d_sys
        tmpl = config.prompt_template or d_tmpl
        partials = []
        for q in rephrases:
            msgs = [
                {"role": "system", "content": base_sys},
                {"role": "user", "content": tmpl.format(question=q, context=context)},
            ]
            partials.append(client.call(msgs))

        synth_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. "
                    "Synthesize the partial answers into one concise final answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    "Partial answers:\n"
                    + "\n".join(f"- {a}" for a in partials)
                    + "\n\nFinal answer:"
                ),
            },
        ]
        answer = client.call(synth_msgs)

    else:
        raise ValueError(
            f"Unknown method: {method!r}. Options: simple, chain_of_thought, multi_query"
        )

    return {
        "generated_answer": answer,
        "method": method,
        "context_doc_ids": doc_ids,
        "closed_book": closed_book,
    }


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _compute_hallucination_rate(
    answer: str, retrieved_payloads: List[Dict[str, Any]]
) -> Optional[float]:
    """Fraction of answer tokens not present in retrieved context (simple token-overlap
    heuristic). Closed-book (no context docs at all) ⇒ ``None``: there is no context to ground
    against, so a grounding score is undefined (≠ a RAG run whose docs are present but empty)."""
    if not retrieved_payloads:
        return None
    context_text = " ".join(
        p.get("text", p.get("content", "")) for p in retrieved_payloads
    )
    context_tokens = _tokenize(context_text)
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    coverage = len(answer_tokens & context_tokens) / len(answer_tokens)
    return 1.0 - coverage


def _compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1/2/L F1 scores."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ROUGE computation. "
            "Install it: pip install rouge-score"
        )
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, hypothesis)
    return {
        "rouge1": s["rouge1"].fmeasure,
        "rouge2": s["rouge2"].fmeasure,
        "rougeL": s["rougeL"].fmeasure,
    }


def generate_answers(
    traces_data: Tuple[List, List, List],
    all_query_texts: List[str],
    corpus_lookup: Dict[str, Dict[str, Any]],
    config,
) -> Dict[str, Any]:
    """Generate answers for all (or max_cases) queries and optionally compute ROUGE.

    Args:
        traces_data: Tuple of (all_query_ids, all_relevant, all_results_with_scores).
        all_query_texts: Query text per index (ASR hypothesis or ground-truth question).
        corpus_lookup: doc_id → payload dict used to find reference answers.
        config: AnswerGenerationConfig instance.

    Returns:
        Dict with keys: model, method, cases, details, mean_rouge1, mean_rouge2, mean_rougeL.
        Each detail has: query_id, question, generated_answer, reference_answer,
        rouge1, rouge2, rougeL, context_doc_ids.
    """
    from tqdm import tqdm
    from ..utils.progress import progress_disabled

    client = LLMClient(config.to_llm_config(), component="answer_gen")

    all_query_ids, all_relevant, all_results_with_scores = traces_data
    n = len(all_query_texts)
    if config.max_cases > 0:
        n = min(n, config.max_cases)

    details: List[Dict[str, Any]] = []
    failed = 0

    for i in tqdm(range(n), desc="Generating answers", disable=progress_disabled()):
        query_id = all_query_ids[i] if i < len(all_query_ids) else str(i)
        question = all_query_texts[i] if i < len(all_query_texts) else ""
        retrieved_payloads = (
            [p for p, _ in all_results_with_scores[i]]
            if i < len(all_results_with_scores)
            else []
        )

        try:
            gen = generate_single_answer(question, retrieved_payloads, config, client)
        except Exception as exc:
            logger.warning("Answer generation failed for query %s: %s", query_id, exc)
            gen = {
                "generated_answer": "",
                "method": config.method,
                "context_doc_ids": [],
                "closed_book": not retrieved_payloads,
            }
            failed += 1

        # Generation only — the comparison metrics are the answer_metrics node's job.
        detail = {
            "query_id": query_id,
            "question": question,
            "generated_answer": gen["generated_answer"],
            "reference_answer": "",
            "context_doc_ids": gen["context_doc_ids"],
        }
        # Tag only closed-book details (additive); RAG details stay byte-identical (parity).
        if gen.get("closed_book"):
            detail["closed_book"] = True
        details.append(detail)

    if failed > 0:
        logger.warning("Answer generation: %d/%d queries failed", failed, n)
    return {
        "model": config.model,
        "method": config.method,
        "cases": len(details),
        "failed_cases": failed,
        "details": details,
    }


def score_answers(
    answer_results: Dict[str, Any],
    traces_data: Tuple[List, List, List],
    corpus_lookup: Dict[str, Dict[str, Any]],
    config,
) -> Dict[str, Any]:
    """answer_metrics comparison: score the generated answers vs their reference answers +
    retrieved context. Enriches each detail (rouge / hallucination / dose-safety /
    context-recall) IN PLACE and adds the ``mean_*`` aggregates to ``answer_results``."""
    all_query_ids, all_relevant, all_results_with_scores = traces_data
    rouge1_list: List[float] = []
    rouge2_list: List[float] = []
    rougeL_list: List[float] = []
    hallucination_list: List[float] = []
    dosage_safety_list: List[float] = []
    context_recall_list: List[float] = []

    for i, detail in enumerate(answer_results.get("details", [])):
        answer = detail.get("generated_answer", "")
        retrieved_payloads = (
            [p for p, _ in all_results_with_scores[i]]
            if i < len(all_results_with_scores)
            else []
        )
        hal_rate = _compute_hallucination_rate(answer, retrieved_payloads)
        detail["hallucination_rate"] = hal_rate  # None ⇒ closed-book (N/A)
        if hal_rate is not None:
            hallucination_list.append(hal_rate)
        detail.setdefault("rouge1", None)
        detail.setdefault("rouge2", None)
        detail.setdefault("rougeL", None)

        if config.compute_rouge and config.reference_metadata_field:
            ref_answer = _lookup_reference(
                detail.get("query_id"),
                all_relevant,
                i,
                corpus_lookup,
                config.reference_metadata_field,
            )
            if ref_answer and answer:
                detail["reference_answer"] = ref_answer
                try:
                    rouge = _compute_rouge(answer, ref_answer)
                    detail.update(rouge)
                    rouge1_list.append(rouge["rouge1"])
                    rouge2_list.append(rouge["rouge2"])
                    rougeL_list.append(rouge["rougeL"])
                except Exception as exc:
                    logger.warning(
                        "ROUGE computation failed for %s: %s",
                        detail.get("query_id"),
                        exc,
                    )
                from ..metrics.rag import context_recall, drug_dosage_safety

                contexts = [
                    p.get("text", p.get("content", "")) for p in retrieved_payloads
                ]
                safety = drug_dosage_safety(answer, ref_answer)
                crecall = context_recall(ref_answer, contexts)
                detail["drug_dosage_safety"] = safety
                detail["context_recall"] = crecall
                dosage_safety_list.append(safety)
                context_recall_list.append(crecall)

    def _mean(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    answer_results.update(
        {
            "mean_rouge1": _mean(rouge1_list),
            "mean_rouge2": _mean(rouge2_list),
            "mean_rougeL": _mean(rougeL_list),
            "mean_hallucination_rate": _mean(hallucination_list),
            "mean_drug_dosage_safety": _mean(dosage_safety_list),
            "mean_context_recall": _mean(context_recall_list),
        }
    )
    return answer_results


def _lookup_reference(
    query_id: Any,
    all_relevant: List[Dict],
    idx: int,
    corpus_lookup: Dict[str, Dict[str, Any]],
    field: str,
) -> str:
    """Find reference answer text for a query, trying relevant doc IDs first."""
    # Try relevant doc IDs (most reliable for datasets where query has matching doc)
    relevant_ids = list(all_relevant[idx].keys()) if idx < len(all_relevant) else []
    for rel_id in relevant_ids:
        doc = corpus_lookup.get(str(rel_id))
        if doc:
            val = doc.get(field) or doc.get("metadata", {}).get(field, "")
            if val:
                return str(val)
    # Fallback: query_id as doc key, stripping common "q_" prefix
    qid_str = str(query_id)
    for key in (qid_str, qid_str.lstrip("q_").lstrip("Q_")):
        doc = corpus_lookup.get(key)
        if doc:
            val = doc.get(field) or doc.get("metadata", {}).get(field, "")
            if val:
                return str(val)
    return ""
