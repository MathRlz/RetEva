"""Error analysis tools for ASR and retrieval evaluation.

Provides functions for analyzing error patterns in speech recognition
and information retrieval systems, including word error breakdowns,
retrieval failure analysis, and error categorization.
"""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from jiwer import process_words, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces


# Text normalization pipeline matching stt_metrics.py
_normalize = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces()
])


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
    # Extract reference/hypothesis pairs
    pairs = _extract_asr_pairs(results)
    
    if not pairs:
        return _empty_asr_analysis()
    
    # Track error statistics
    substitutions: List[Tuple[str, str]] = []
    insertions: List[str] = []
    deletions: List[str] = []
    word_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {"count": 0, "errors": 0})
    word_errors: Counter = Counter()
    
    for ref, hyp in pairs:
        ref_norm = _normalize(ref)
        hyp_norm = _normalize(hyp)
        
        # Use jiwer to get alignment
        output = process_words(ref_norm, hyp_norm)
        
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        
        # Track word lengths
        for word in ref_words:
            word_len = len(word)
            word_stats[word_len]["count"] += 1
        
        # Process alignments to extract error details
        for chunk in output.alignments[0]:
            if chunk.type == "equal":
                pass  # Correct match
            elif chunk.type == "substitute":
                for i, j in zip(
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx)
                ):
                    ref_word = ref_words[i]
                    hyp_word = hyp_words[j]
                    substitutions.append((ref_word, hyp_word))
                    word_errors[ref_word] += 1
                    word_stats[len(ref_word)]["errors"] += 1
            elif chunk.type == "insert":
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    insertions.append(hyp_words[i])
            elif chunk.type == "delete":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    ref_word = ref_words[i]
                    deletions.append(ref_word)
                    word_errors[ref_word] += 1
                    word_stats[len(ref_word)]["errors"] += 1
    
    # Build analysis results
    total_errors = len(substitutions) + len(insertions) + len(deletions)
    total_ref_words = sum(stats["count"] for stats in word_stats.values())
    
    # Error counts
    error_counts = {
        "substitutions": len(substitutions),
        "insertions": len(insertions),
        "deletions": len(deletions),
        "total_errors": total_errors,
    }
    
    # Error rates (relative to total reference words)
    error_rates = {
        "substitution_rate": len(substitutions) / total_ref_words if total_ref_words else 0.0,
        "insertion_rate": len(insertions) / total_ref_words if total_ref_words else 0.0,
        "deletion_rate": len(deletions) / total_ref_words if total_ref_words else 0.0,
    }
    
    # Error rate by word length
    by_word_length = {}
    for length, stats in sorted(word_stats.items()):
        by_word_length[length] = {
            "count": stats["count"],
            "errors": stats["errors"],
            "error_rate": stats["errors"] / stats["count"] if stats["count"] else 0.0,
        }
    
    # Common substitutions (ref_word -> hyp_word)
    sub_counter = Counter(substitutions)
    common_substitutions = [
        (ref_w, hyp_w, count)
        for (ref_w, hyp_w), count in sub_counter.most_common(20)
    ]
    
    # Common insertions
    ins_counter = Counter(insertions)
    common_insertions = [
        (word, count)
        for word, count in ins_counter.most_common(20)
    ]
    
    # Common deletions
    del_counter = Counter(deletions)
    common_deletions = [
        (word, count)
        for word, count in del_counter.most_common(20)
    ]
    
    # Most frequently misrecognized words
    misrecognized_words = word_errors.most_common(20)
    
    return {
        "error_counts": error_counts,
        "error_rates": error_rates,
        "by_word_length": by_word_length,
        "common_substitutions": common_substitutions,
        "common_insertions": common_insertions,
        "common_deletions": common_deletions,
        "misrecognized_words": misrecognized_words,
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
        "success_rate": len(successful_queries) / total_queries if total_queries else 0.0,
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
        "uncategorized_rate": len(uncategorized) / total_errors if total_errors else 0.0,
    }


def generate_error_report(results: dict, categories: Optional[dict] = None) -> str:
    """Generate a human-readable summary of error analysis.
    
    Produces a formatted text report summarizing ASR errors, retrieval
    failures, and optionally categorized errors.
    
    Args:
        results: Results dictionary containing evaluation data.
        categories: Optional category definitions for error categorization.
            
    Returns:
        Formatted string report suitable for display or logging.
        
    Example:
        >>> results = {
        ...     "details": [
        ...         {"reference": "hello world", "hypothesis": "hello word",
        ...          "query": "greeting", "retrieved": ["d1"], "relevant": {"d1": 1}}
        ...     ]
        ... }
        >>> print(generate_error_report(results))
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ERROR ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # ASR Error Analysis
    asr_analysis = analyze_asr_errors(results)
    if asr_analysis["error_counts"]["total_errors"] > 0:
        lines.append("-" * 70)
        lines.append("ASR ERROR ANALYSIS")
        lines.append("-" * 70)
        lines.append("")
        
        # Error counts
        ec = asr_analysis["error_counts"]
        lines.append(f"Total Errors: {ec['total_errors']}")
        lines.append(f"  Substitutions: {ec['substitutions']}")
        lines.append(f"  Insertions: {ec['insertions']}")
        lines.append(f"  Deletions: {ec['deletions']}")
        lines.append("")
        
        # Error rates
        er = asr_analysis["error_rates"]
        lines.append("Error Rates (per reference word):")
        lines.append(f"  Substitution Rate: {er['substitution_rate']:.4f}")
        lines.append(f"  Insertion Rate: {er['insertion_rate']:.4f}")
        lines.append(f"  Deletion Rate: {er['deletion_rate']:.4f}")
        lines.append("")
        
        # Common substitutions
        if asr_analysis["common_substitutions"]:
            lines.append("Top 10 Substitutions (reference -> hypothesis):")
            for ref_w, hyp_w, count in asr_analysis["common_substitutions"][:10]:
                lines.append(f"  '{ref_w}' -> '{hyp_w}': {count} times")
            lines.append("")
        
        # Common deletions
        if asr_analysis["common_deletions"]:
            lines.append("Top 10 Deleted Words:")
            for word, count in asr_analysis["common_deletions"][:10]:
                lines.append(f"  '{word}': {count} times")
            lines.append("")
        
        # Common insertions
        if asr_analysis["common_insertions"]:
            lines.append("Top 10 Inserted Words:")
            for word, count in asr_analysis["common_insertions"][:10]:
                lines.append(f"  '{word}': {count} times")
            lines.append("")
        
        # Error by word length
        if asr_analysis["by_word_length"]:
            lines.append("Error Rate by Word Length:")
            for length, stats in list(asr_analysis["by_word_length"].items())[:10]:
                lines.append(
                    f"  {length} chars: {stats['errors']}/{stats['count']} "
                    f"({stats['error_rate']:.2%})"
                )
            lines.append("")
    
    # Retrieval Failure Analysis
    retrieval_analysis = analyze_retrieval_failures(results)
    if retrieval_analysis["total_queries"] > 0:
        lines.append("-" * 70)
        lines.append("RETRIEVAL FAILURE ANALYSIS")
        lines.append("-" * 70)
        lines.append("")
        
        lines.append(f"Total Queries: {retrieval_analysis['total_queries']}")
        lines.append(f"Failure Rate: {retrieval_analysis['failure_rate']:.2%}")
        lines.append(f"Near Miss Rate: {retrieval_analysis['near_miss_rate']:.2%}")
        lines.append(f"Success Rate: {retrieval_analysis['success_rate']:.2%}")
        lines.append("")
        
        # Failed queries
        if retrieval_analysis["failed_queries"]:
            lines.append(f"Failed Queries ({len(retrieval_analysis['failed_queries'])}):")
            for q in retrieval_analysis["failed_queries"][:5]:
                lines.append(f"  - \"{q['query']}\" (length: {q['query_length']} words)")
            if len(retrieval_analysis["failed_queries"]) > 5:
                lines.append(f"  ... and {len(retrieval_analysis['failed_queries']) - 5} more")
            lines.append("")
        
        # Near misses
        if retrieval_analysis["near_misses"]:
            lines.append(f"Near Misses ({len(retrieval_analysis['near_misses'])}):")
            for q in retrieval_analysis["near_misses"][:5]:
                lines.append(
                    f"  - \"{q['query']}\" (relevant at rank {q['first_relevant_rank']})"
                )
            if len(retrieval_analysis["near_misses"]) > 5:
                lines.append(f"  ... and {len(retrieval_analysis['near_misses']) - 5} more")
            lines.append("")
        
        # Query characteristics
        qc = retrieval_analysis["query_characteristics"]
        if qc:
            lines.append("Query Characteristics:")
            lines.append(f"  Avg Query Length (failed): {qc.get('avg_length_failed', 0):.1f} words")
            lines.append(f"  Avg Query Length (success): {qc.get('avg_length_success', 0):.1f} words")
            lines.append("")
    
    # Category Analysis
    if categories:
        cat_analysis = categorize_errors(results, categories)
        if cat_analysis["total_errors"] > 0:
            lines.append("-" * 70)
            lines.append("ERROR CATEGORIZATION")
            lines.append("-" * 70)
            lines.append("")
            
            lines.append(f"Total Categorized Errors: {cat_analysis['total_errors']}")
            lines.append("")
            
            for cat_name, count in cat_analysis["category_counts"].items():
                rate = cat_analysis["category_rates"].get(cat_name, 0)
                lines.append(f"  {cat_name}: {count} ({rate:.1%})")
            
            uncategorized_count = len(cat_analysis["uncategorized"])
            lines.append(
                f"  Uncategorized: {uncategorized_count} "
                f"({cat_analysis['uncategorized_rate']:.1%})"
            )
            lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


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
    successful: List[Dict[str, Any]]
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
