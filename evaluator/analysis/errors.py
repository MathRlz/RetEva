"""Error analysis tools for ASR and retrieval evaluation.

Provides functions for analyzing error patterns in speech recognition
and information retrieval systems, including word error breakdowns,
retrieval failure analysis, and error categorization.
"""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from jiwer import (
    process_words,
    Compose,
    ToLowerCase,
    RemovePunctuation,
    RemoveMultipleSpaces,
)

# Text normalization pipeline matching stt_metrics.py
_normalize = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces()])


def analyze_asr_errors(results: dict) -> dict:
    """Analyze ASR (Automatic Speech Recognition) error patterns.

    Examines transcription results to identify common error types including
    substitutions, insertions, and deletions, as well as patterns related
    to word length and frequently misrecognized words.

    Args:
        results: Results dictionary containing ASR evaluation data.
            Expected format with 'details' or 'per_sample' containing
            'reference' and 'hypothesis' pairs, or 'asr_details' list.

    Returns:
        Dictionary containing:
        - error_counts: {substitutions, insertions, deletions, total_errors}
        - error_rates: {substitution_rate, insertion_rate, deletion_rate}
        - by_word_length: {length: {count, errors, error_rate}}
        - common_substitutions: List of (ref_word, hyp_word, count) tuples
        - common_insertions: List of (inserted_word, count) tuples
        - common_deletions: List of (deleted_word, count) tuples
        - misrecognized_words: List of (word, error_count) tuples

    Example:
        >>> results = {
        ...     "details": [
        ...         {"reference": "the quick brown fox", "hypothesis": "the quik brown box"},
        ...         {"reference": "hello world", "hypothesis": "hello word"}
        ...     ]
        ... }
        >>> analysis = analyze_asr_errors(results)
        >>> print(analysis["error_counts"]["substitutions"])
        3
    """
    pairs = _extract_asr_pairs(results)
    if not pairs:
        return _empty_asr_analysis()
    acc = _accumulate_asr_errors(pairs)  # parse + count
    return _build_asr_analysis(acc)      # aggregate


class _AsrErrorTally:
    """Accumulators for the ASR alignment pass (F12: separates count from aggregate)."""

    def __init__(self) -> None:
        self.substitutions: List[Tuple[str, str]] = []
        self.insertions: List[str] = []
        self.deletions: List[str] = []
        self.word_stats: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"count": 0, "errors": 0}
        )
        self.word_errors: Counter = Counter()

    def _record_error_word(self, word: str) -> None:
        self.word_errors[word] += 1
        self.word_stats[len(word)]["errors"] += 1


def _tally_alignment(chunks, ref_words: List[str], hyp_words: List[str],
                     tally: "_AsrErrorTally") -> None:
    """Fold one pair's jiwer alignment chunks into the tally (the inner, deepest loop)."""
    for chunk in chunks:
        if chunk.type == "substitute":
            for i, j in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                tally.substitutions.append((ref_words[i], hyp_words[j]))
                tally._record_error_word(ref_words[i])
        elif chunk.type == "insert":
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                tally.insertions.append(hyp_words[i])
        elif chunk.type == "delete":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                tally.deletions.append(ref_words[i])
                tally._record_error_word(ref_words[i])
        # chunk.type == "equal": correct match, nothing to record


def _accumulate_asr_errors(pairs: List[Tuple[str, str]]) -> "_AsrErrorTally":
    """Normalize + align each (ref, hyp) pair and tally substitutions/insertions/deletions."""
    tally = _AsrErrorTally()
    for ref, hyp in pairs:
        ref_norm, hyp_norm = _normalize(ref), _normalize(hyp)
        output = process_words(ref_norm, hyp_norm)
        ref_words, hyp_words = ref_norm.split(), hyp_norm.split()
        for word in ref_words:
            tally.word_stats[len(word)]["count"] += 1
        _tally_alignment(output.alignments[0], ref_words, hyp_words, tally)
    return tally


def _build_asr_analysis(tally: "_AsrErrorTally") -> dict:
    """Assemble the public analysis dict from the accumulated tally (counts → rates/top-N)."""
    n_sub, n_ins, n_del = (
        len(tally.substitutions), len(tally.insertions), len(tally.deletions)
    )
    total_errors = n_sub + n_ins + n_del
    total_ref_words = sum(stats["count"] for stats in tally.word_stats.values())

    def _rate(n: int) -> float:
        return n / total_ref_words if total_ref_words else 0.0

    by_word_length = {
        length: {
            "count": stats["count"],
            "errors": stats["errors"],
            "error_rate": stats["errors"] / stats["count"] if stats["count"] else 0.0,
        }
        for length, stats in sorted(tally.word_stats.items())
    }
    return {
        "error_counts": {
            "substitutions": n_sub,
            "insertions": n_ins,
            "deletions": n_del,
            "total_errors": total_errors,
        },
        "error_rates": {
            "substitution_rate": _rate(n_sub),
            "insertion_rate": _rate(n_ins),
            "deletion_rate": _rate(n_del),
        },
        "by_word_length": by_word_length,
        "common_substitutions": [
            (ref_w, hyp_w, count)
            for (ref_w, hyp_w), count in Counter(tally.substitutions).most_common(20)
        ],
        "common_insertions": Counter(tally.insertions).most_common(20),
        "common_deletions": Counter(tally.deletions).most_common(20),
        "misrecognized_words": tally.word_errors.most_common(20),
    }


def analyze_retrieval_failures(results: dict, top_k: int = 10) -> dict:
    """Analyze information retrieval failure patterns.

    Examines retrieval results to identify queries that failed to retrieve
    relevant documents, near misses where relevant documents appeared in
    top-k but not top-1, and query characteristics that correlate with failure.

    Args:
        results: Results dictionary containing retrieval evaluation data.
            Expected format with 'details' containing query information
            with 'query', 'retrieved', and 'relevant' fields.
        top_k: Number of top results to consider for near-miss analysis.

    Returns:
        Dictionary containing:
        - failed_queries: List of queries with no relevant doc in top-k
        - near_misses: List of queries with relevant in top-k but not top-1
        - failure_rate: Proportion of queries that failed
        - near_miss_rate: Proportion of queries that were near misses
        - query_characteristics: Statistics about failing queries
        - rank_distribution: Distribution of relevant doc ranks

    Example:
        >>> results = {
        ...     "details": [
        ...         {"query": "what is diabetes", "retrieved": ["d1", "d2"],
        ...          "relevant": {"d3": 1}},
        ...         {"query": "flu symptoms", "retrieved": ["d4", "d5"],
        ...          "relevant": {"d5": 1}}
        ...     ]
        ... }
        >>> analysis = analyze_retrieval_failures(results)
        >>> len(analysis["failed_queries"])
        1
    """
    details = results.get("details", [])

    if not details:
        return _empty_retrieval_analysis()

    failed_queries: List[Dict[str, Any]] = []
    near_misses: List[Dict[str, Any]] = []
    successful_queries: List[Dict[str, Any]] = []
    rank_distribution: Counter = Counter()

    for item in details:
        query = item.get("query", "")
        retrieved = item.get("retrieved", [])
        relevant = item.get("relevant", {})

        if not relevant:
            continue

        # Find rank of first relevant document
        first_relevant_rank = None
        for rank, doc_id in enumerate(retrieved[:top_k], 1):
            if relevant.get(doc_id, 0) > 0:
                first_relevant_rank = rank
                break

        query_info = {
            "query": query,
            "query_length": len(query.split()),
            "retrieved_count": len(retrieved),
            "relevant_count": sum(1 for v in relevant.values() if v > 0),
        }

        if first_relevant_rank is None:
            # Complete failure - no relevant in top-k
            failed_queries.append(query_info)
        elif first_relevant_rank > 1:
            # Near miss - relevant in top-k but not top-1
            query_info["first_relevant_rank"] = first_relevant_rank
            near_misses.append(query_info)
            rank_distribution[first_relevant_rank] += 1
        else:
            # Success - relevant in top-1
            successful_queries.append(query_info)
            rank_distribution[1] += 1

    total_queries = len(failed_queries) + len(near_misses) + len(successful_queries)

    # Compute query characteristics that correlate with failure
    query_characteristics = _compute_query_characteristics(
        failed_queries, near_misses, successful_queries
    )

    return {
        "failed_queries": failed_queries,
        "near_misses": near_misses,
        "failure_rate": len(failed_queries) / total_queries if total_queries else 0.0,
        "near_miss_rate": len(near_misses) / total_queries if total_queries else 0.0,
        "success_rate": (
            len(successful_queries) / total_queries if total_queries else 0.0
        ),
        "total_queries": total_queries,
        "query_characteristics": query_characteristics,
        "rank_distribution": dict(rank_distribution),
    }


def categorize_errors(results: dict, categories: dict) -> dict:
    """Categorize errors by predefined types.

    Analyzes errors and categorizes them based on user-defined categories,
    such as medical terms, numbers, abbreviations, etc.

    Args:
        results: Results dictionary containing evaluation data with error details.
        categories: Dictionary mapping category names to patterns or word lists.
            Each value can be:
            - A list of words to match exactly
            - A regex pattern string (prefixed with 'regex:')
            - A callable taking a word and returning bool

    Returns:
        Dictionary containing:
        - category_counts: {category_name: count}
        - category_errors: {category_name: [list of errors]}
        - category_rates: {category_name: error_rate_for_category}
        - uncategorized: List of errors not matching any category

    Example:
        >>> results = {
        ...     "details": [
        ...         {"reference": "take 500mg daily", "hypothesis": "take 500 mg daily"},
        ...         {"reference": "Dr Smith", "hypothesis": "doctor Smith"}
        ...     ]
        ... }
        >>> categories = {
        ...     "numbers": ["regex:[0-9]+"],
        ...     "abbreviations": ["mg", "dr", "ml"]
        ... }
        >>> analysis = categorize_errors(results, categories)
    """
    # First, get ASR error analysis
    asr_analysis = analyze_asr_errors(results)

    # Collect all error words
    all_errors: List[Tuple[str, str]] = []  # (error_type, word)

    for ref_word, hyp_word, count in asr_analysis.get("common_substitutions", []):
        for _ in range(count):
            all_errors.append(("substitution", ref_word))

    for word, count in asr_analysis.get("common_deletions", []):
        for _ in range(count):
            all_errors.append(("deletion", word))

    for word, count in asr_analysis.get("common_insertions", []):
        for _ in range(count):
            all_errors.append(("insertion", word))

    # Categorize errors
    category_counts: Dict[str, int] = defaultdict(int)
    category_errors: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    categorized_indices: Set[int] = set()

    # Build matchers for each category
    matchers = _build_category_matchers(categories)

    for idx, (error_type, word) in enumerate(all_errors):
        for cat_name, matcher in matchers.items():
            if matcher(word):
                category_counts[cat_name] += 1
                category_errors[cat_name].append((error_type, word))
                categorized_indices.add(idx)
                break  # Each error belongs to first matching category

    # Uncategorized errors
    uncategorized = [
        (error_type, word)
        for idx, (error_type, word) in enumerate(all_errors)
        if idx not in categorized_indices
    ]

    # Compute category rates (relative to total errors)
    total_errors = len(all_errors)
    category_rates = {
        cat: count / total_errors if total_errors else 0.0
        for cat, count in category_counts.items()
    }

    return {
        "category_counts": dict(category_counts),
        "category_errors": {k: list(v) for k, v in category_errors.items()},
        "category_rates": category_rates,
        "uncategorized": uncategorized,
        "total_errors": total_errors,
        "uncategorized_rate": (
            len(uncategorized) / total_errors if total_errors else 0.0
        ),
    }


# --- Helper functions ---


def _extract_asr_pairs(results: dict) -> List[Tuple[str, str]]:
    """Extract reference/hypothesis pairs from results dictionary."""
    pairs: List[Tuple[str, str]] = []

    # Check 'details' list
    details = results.get("details", [])
    for item in details:
        if isinstance(item, dict):
            ref = item.get("reference") or item.get("ref") or item.get("ground_truth")
            hyp = item.get("hypothesis") or item.get("hyp") or item.get("transcription")
            if ref and hyp:
                pairs.append((str(ref), str(hyp)))

    # Check 'asr_details' list
    asr_details = results.get("asr_details", [])
    for item in asr_details:
        if isinstance(item, dict):
            ref = item.get("reference") or item.get("ref")
            hyp = item.get("hypothesis") or item.get("hyp")
            if ref and hyp:
                pairs.append((str(ref), str(hyp)))

    # Check per_sample format
    per_sample = results.get("per_sample", {})
    refs = per_sample.get("reference", per_sample.get("references", []))
    hyps = per_sample.get("hypothesis", per_sample.get("hypotheses", []))
    if refs and hyps and len(refs) == len(hyps):
        for ref, hyp in zip(refs, hyps):
            pairs.append((str(ref), str(hyp)))

    return pairs


def _empty_asr_analysis() -> dict:
    """Return empty ASR analysis structure."""
    return {
        "error_counts": {
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "total_errors": 0,
        },
        "error_rates": {
            "substitution_rate": 0.0,
            "insertion_rate": 0.0,
            "deletion_rate": 0.0,
        },
        "by_word_length": {},
        "common_substitutions": [],
        "common_insertions": [],
        "common_deletions": [],
        "misrecognized_words": [],
    }


def _empty_retrieval_analysis() -> dict:
    """Return empty retrieval analysis structure."""
    return {
        "failed_queries": [],
        "near_misses": [],
        "failure_rate": 0.0,
        "near_miss_rate": 0.0,
        "success_rate": 0.0,
        "total_queries": 0,
        "query_characteristics": {},
        "rank_distribution": {},
    }


def _compute_query_characteristics(
    failed: List[Dict[str, Any]],
    near_misses: List[Dict[str, Any]],
    successful: List[Dict[str, Any]],
) -> dict:
    """Compute statistics about query characteristics for different outcomes."""
    characteristics = {}

    # Average query length by outcome
    if failed:
        characteristics["avg_length_failed"] = sum(
            q["query_length"] for q in failed
        ) / len(failed)
    else:
        characteristics["avg_length_failed"] = 0.0

    if near_misses:
        characteristics["avg_length_near_miss"] = sum(
            q["query_length"] for q in near_misses
        ) / len(near_misses)
    else:
        characteristics["avg_length_near_miss"] = 0.0

    if successful:
        characteristics["avg_length_success"] = sum(
            q["query_length"] for q in successful
        ) / len(successful)
    else:
        characteristics["avg_length_success"] = 0.0

    # Count queries by length buckets
    def _length_bucket(length: int) -> str:
        if length <= 2:
            return "short (1-2)"
        elif length <= 5:
            return "medium (3-5)"
        else:
            return "long (6+)"

    length_buckets_failed: Counter = Counter()
    length_buckets_success: Counter = Counter()

    for q in failed:
        length_buckets_failed[_length_bucket(q["query_length"])] += 1

    for q in successful:
        length_buckets_success[_length_bucket(q["query_length"])] += 1

    characteristics["length_distribution_failed"] = dict(length_buckets_failed)
    characteristics["length_distribution_success"] = dict(length_buckets_success)

    return characteristics


def _build_category_matchers(categories: dict) -> Dict[str, callable]:
    """Build matcher functions for each category."""
    matchers = {}

    for cat_name, pattern in categories.items():
        if callable(pattern):
            # Already a callable
            matchers[cat_name] = pattern
        elif isinstance(pattern, list):
            # List of words or patterns
            word_set = set()
            regex_patterns = []

            for item in pattern:
                if isinstance(item, str):
                    if item.startswith("regex:"):
                        regex_patterns.append(re.compile(item[6:], re.IGNORECASE))
                    else:
                        word_set.add(item.lower())

            def make_matcher(ws: Set[str], rp: List) -> callable:
                def matcher(word: str) -> bool:
                    w_lower = word.lower()
                    if w_lower in ws:
                        return True
                    for regex in rp:
                        if regex.search(word):
                            return True
                    return False

                return matcher

            matchers[cat_name] = make_matcher(word_set, regex_patterns)
        elif isinstance(pattern, str):
            # Single pattern (regex or word)
            if pattern.startswith("regex:"):
                regex = re.compile(pattern[6:], re.IGNORECASE)
                matchers[cat_name] = lambda w, r=regex: bool(r.search(w))
            else:
                p_lower = pattern.lower()
                matchers[cat_name] = lambda w, p=p_lower: w.lower() == p

    return matchers


# Report builders live in error_report.py; re-export to preserve the public API.
from .error_report import generate_error_report  # noqa: E402,F401
