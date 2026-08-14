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

from ..llm_client.client import LLMClient

logger = logging.getLogger(__name__)

# The answer FORMAT is part of the contract: a decision line scored against the dataset's
# yes/no/maybe label (PubMedQA `final_decision`) plus prose scored against the reference answer
# (`long_answer`). One call yields both; `parse_structured_answer` splits them back apart.
_FORMAT_INSTRUCTION = (
    "Reply in exactly this format, nothing before it:\n"
    "Short Answer: yes or no or maybe\n"
    "Long Answer: <your detailed answer>"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise medical research assistant. "
    "Answer based ONLY on the provided context. Be brief and direct.\n"
    + _FORMAT_INSTRUCTION
)

DEFAULT_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\nQuestion: {question}\n\n"
    "Short Answer:"
)

_COT_SYSTEM_PROMPT = (
    "You are a medical research assistant. "
    "Think step by step using ONLY the provided context, then give a concise answer.\n"
    + _FORMAT_INSTRUCTION
)

_COT_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step:\n1. What does the context say?\n2. What is the direct answer?\n\n"
    "Then give your final answer in the required format.\n\n"
    "Short Answer:"
)

# Closed-book (no retrieved context): answer from the model's own knowledge. No "Context:"
# block, no "use only the context" instruction (which is contradictory with no context).
_CLOSED_BOOK_SYSTEM_PROMPT = (
    "You are a concise medical research assistant. "
    "Answer from your own knowledge. Be brief and direct.\n"
    + _FORMAT_INSTRUCTION
)

_CLOSED_BOOK_PROMPT_TEMPLATE = "Question: {question}\n\nShort Answer:"

_CLOSED_BOOK_COT_SYSTEM_PROMPT = (
    "You are a medical research assistant. "
    "Think step by step from your own knowledge, then give a concise answer.\n"
    + _FORMAT_INSTRUCTION
)

_CLOSED_BOOK_COT_PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "Think step by step:\n1. What do you know?\n2. What is the direct answer?\n\n"
    "Then give your final answer in the required format.\n\n"
    "Short Answer:"
)

# yes/no/maybe decision labels (PubMedQA `final_decision`) + the "model did not say" sentinel.
DECISION_LABELS = ("yes", "no", "maybe")
DECISION_UNKNOWN = "unknown"

# `Short Answer: yes` — tolerant of markdown bold, a dash/colon separator and leading blanks.
_SHORT_RE = re.compile(
    r"short\s*answer\s*[:\-–]?\s*\**\s*(yes|no|maybe)\b", re.IGNORECASE
)
_LONG_RE = re.compile(r"long\s*answer\s*[:\-–]?\s*\**\s*", re.IGNORECASE)
# A bare leading decision: the prompt ends with "Short Answer:", so a model that completes it
# literally replies "yes\nLong Answer: ..." with no label of its own. The token must END the
# clause — otherwise prose like "no idea really" reads as a `no` decision.
_BARE_DECISION_RE = re.compile(
    r"^\W*\**\s*(yes|no|maybe)\b\s*(?=[.,;:!)\]\n]|\*|$)", re.IGNORECASE
)


def _normalize_decision(value: Any) -> Optional[str]:
    """A dataset label → one of :data:`DECISION_LABELS`, or None when it is not a decision."""
    token = str(value or "").strip().strip(".!,").lower()
    return token if token in DECISION_LABELS else None


def parse_structured_answer(text: str) -> Tuple[str, str]:
    """Split a reply into ``(decision, long_answer)``.

    ``decision`` is one of :data:`DECISION_LABELS` or :data:`DECISION_UNKNOWN` when the model
    ignored the format. ``long_answer`` is the ``Long Answer:`` section, falling back to the
    whole reply (minus a leading decision line) — so a format miss degrades the decision metric
    only, never silently zeroes ROUGE.
    """
    raw = (text or "").strip()
    if not raw:
        return DECISION_UNKNOWN, ""

    m = _SHORT_RE.search(raw) or _BARE_DECISION_RE.match(raw)
    decision = m.group(1).lower() if m else DECISION_UNKNOWN

    long_m = _LONG_RE.search(raw)
    if long_m:
        return decision, raw[long_m.end():].strip()
    # No "Long Answer:" label: drop the decision line (if that is all it was) and keep the rest.
    if m is not None:
        rest = raw[m.end():].lstrip(" \t:.-–\n")
        if rest:
            return decision, rest
    return decision, raw


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
    *,
    question: str = "",
    config: Any = None,
    corpus_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    embedder: Any = None,
) -> Tuple[str, List[str]]:
    """Build context string from top retrieved docs. Returns (context_text, doc_ids).

    With ``config.context_source == "full_text"`` each doc contributes the chunks of its full
    article closest to ``question`` instead of the indexed passage; a doc without full text
    falls back to that passage."""
    full_text_mode = (
        getattr(config, "context_source", "retrieved_text") == "full_text"
        and corpus_lookup is not None
    )
    parts, doc_ids = [], []
    for payload in retrieved_payloads[:max_docs]:
        doc_id = str(payload.get("doc_id", payload.get("id", "")))
        text = ""
        if full_text_mode:
            text = select_full_text_context(
                question, payload, corpus_lookup, config, embedder
            ) or ""
        if not text:
            text = (payload.get("text", payload.get("content", "")) or "")[:max_chars]
        if text:
            parts.append(text)
            doc_ids.append(doc_id)
    return "\n\n".join(parts), doc_ids


def chunk_text(text: str, size: int, overlap: int = 0) -> List[str]:
    """Split ``text`` into ~``size``-char chunks, preferring paragraph then sentence breaks.

    A full article is far longer than any prompt budget, so it is chunked and filtered rather
    than truncated (the head of a paper is Introduction/Methods — rarely the answer).
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            # Prefer a paragraph break, else a sentence end, else a hard cut.
            cut = max(window.rfind("\n\n"), window.rfind(". "))
            if cut > size // 2:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, end) if overlap <= 0 else end - overlap
    return [c for c in chunks if c]


def _cosine_rank(query_vec, chunk_vecs) -> List[int]:
    """Chunk indices ordered by cosine similarity to the query (ties → chunk order)."""
    import numpy as np

    q = np.asarray(query_vec, dtype="float32").ravel()
    m = np.asarray(chunk_vecs, dtype="float32")
    qn = float(np.linalg.norm(q)) or 1.0
    mn = np.linalg.norm(m, axis=1)
    mn[mn == 0] = 1.0
    sims = (m @ q) / (mn * qn)
    # negate for descending, index as the tie-break → deterministic
    return sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))


def select_full_text_context(
    question: str,
    payload: Dict[str, Any],
    corpus_lookup: Dict[str, Dict[str, Any]],
    config,
    embedder: Any,
) -> Optional[str]:
    """The chunks of a doc's FULL article closest to ``question``, or None to fall back.

    Returns None when the doc has no ``metadata.full_text``, when no embedder is available, or
    when embedding fails — the caller then uses the retrieved passage exactly as before.
    """
    doc_id = str(payload.get("doc_id", payload.get("id", "")))
    doc = corpus_lookup.get(doc_id) or {}
    full = (doc.get("metadata") or {}).get("full_text") or payload.get("full_text")
    if not full:
        return None
    chunks = chunk_text(full, config.context_chunk_chars)
    if not chunks:
        return None
    keep = max(1, int(config.context_chunks))
    if embedder is None or len(chunks) <= keep:
        picked = chunks[:keep]
    else:
        try:
            vecs = embedder.process_batch(chunks)
            qvec = embedder.process_batch([question])[0]
            order = _cosine_rank(qvec, vecs)[:keep]
            picked = [chunks[i] for i in sorted(order)]  # keep document order in the prompt
        except Exception as exc:  # noqa: BLE001 - context selection must not fail a run
            logger.warning(
                "full-text chunk selection failed for doc %s (%s); using the first %d chunks",
                doc_id, exc, keep,
            )
            picked = chunks[:keep]
    return "\n\n".join(picked)[:config.context_max_chars]


def generate_single_answer(
    question: str,
    retrieved_payloads: List[Dict[str, Any]],
    config,
    client: LLMClient,
    corpus_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    embedder: Any = None,
) -> Dict[str, Any]:
    """Generate answer for one question using retrieved context.

    Returns dict with keys: generated_answer, method, context_doc_ids.
    """
    context, doc_ids = _build_context(
        retrieved_payloads, config.context_docs, config.context_max_chars,
        question=question, config=config, corpus_lookup=corpus_lookup, embedder=embedder,
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
    embedder: Any = None,
) -> Dict[str, Any]:
    """Generate answers for all (or max_cases) queries and optionally compute ROUGE.

    Args:
        traces_data: Tuple of (all_query_ids, all_relevant, all_results_with_scores).
        all_query_texts: Query text per index (ASR hypothesis or ground-truth question).
        corpus_lookup: doc_id → payload dict used to find reference answers (and, with
            ``context_source: full_text``, each doc's full article).
        config: AnswerGenerationConfig instance.
        embedder: the run's text-embedding pipeline, used to rank full-article chunks against
            the question. None → the first chunks are used instead of the closest ones.

    Returns:
        Dict with keys: model, method, cases, details, mean_rouge1, mean_rouge2, mean_rougeL.
        Each detail has: query_id, question, generated_answer, reference_answer,
        rouge1, rouge2, rougeL, context_doc_ids.
    """
    from ..llm_client.parallel import map_completions

    client = LLMClient(config.to_llm_config(), component="answer_gen")

    all_query_ids, all_relevant, all_results_with_scores = traces_data
    n = len(all_query_texts)
    if config.max_cases > 0:
        n = min(n, config.max_cases)

    failures: List[str] = []

    def _one(i: int) -> Dict[str, Any]:
        query_id = all_query_ids[i] if i < len(all_query_ids) else str(i)
        question = all_query_texts[i] if i < len(all_query_texts) else ""
        retrieved_payloads = (
            [p for p, _ in all_results_with_scores[i]]
            if i < len(all_results_with_scores)
            else []
        )

        try:
            gen = generate_single_answer(
                question, retrieved_payloads, config, client,
                corpus_lookup=corpus_lookup, embedder=embedder,
            )
        except Exception as exc:
            logger.warning("Answer generation failed for query %s: %s", query_id, exc)
            gen = {
                "generated_answer": "",
                "method": config.method,
                "context_doc_ids": [],
                "closed_book": not retrieved_payloads,
            }
            failures.append(str(query_id))

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
        return detail

    # Ordered + error-isolated: concurrency buys wall clock, never a different report.
    details = map_completions(
        range(n), _one,
        workers=getattr(config, "concurrency", 1),
        desc="Generating answers", unit="query",
    )
    failed = len(failures)

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
    decision_gt: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """answer_metrics comparison: score the generated answers vs their reference answers +
    retrieved context. Enriches each detail (decision / rouge / hallucination / dose-safety /
    context-recall) IN PLACE and adds the ``mean_*`` aggregates to ``answer_results``.

    ``decision_gt`` maps query_id → the dataset's yes/no/maybe label (PubMedQA
    ``final_decision``, published as ``short_answers``); when given, the reply's
    ``Short Answer:`` line is exact-matched against it. ROUGE and the reference-based metrics
    score the ``Long Answer:`` prose, not the decision line."""
    all_query_ids, all_relevant, all_results_with_scores = traces_data
    rouge1_list: List[float] = []
    rouge2_list: List[float] = []
    rougeL_list: List[float] = []
    hallucination_list: List[float] = []
    dosage_safety_list: List[float] = []
    context_recall_list: List[float] = []

    decision_hits: List[float] = []
    decision_unknown = 0

    for i, detail in enumerate(answer_results.get("details", [])):
        reply = detail.get("generated_answer", "")
        # The reply carries BOTH fields; the decision is scored on its own and the prose is what
        # the reference-based metrics compare (scoring the whole reply would count the decision
        # line as answer text).
        decision, answer = parse_structured_answer(reply)
        detail["decision_pred"] = decision
        if decision == DECISION_UNKNOWN:
            decision_unknown += 1
        if decision_gt:
            gt = _normalize_decision(decision_gt.get(str(detail.get("query_id"))))
            if gt:
                detail["decision_gt"] = gt
                detail["decision_correct"] = bool(decision == gt)
                decision_hits.append(1.0 if decision == gt else 0.0)
        retrieved_payloads = (
            [p for p, _ in all_results_with_scores[i]]
            if i < len(all_results_with_scores)
            else []
        )
        hal_rate = _compute_hallucination_rate(answer or reply, retrieved_payloads)
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
            # Exact match on the yes/no/maybe decision; a reply that ignored the format counts
            # as wrong, and `decision_unknown_rate` says how often that happened (so a format
            # regression is visible instead of looking like a wrong answer).
            "mean_decision_accuracy": _mean(decision_hits),
            "decision_unknown_rate": (
                decision_unknown / len(answer_results["details"])
                if answer_results.get("details") else None
            ),
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
