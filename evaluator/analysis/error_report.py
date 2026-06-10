"""Human-readable report builders for error analysis.

Formats the output of the error/failure analysis functions in
``analysis/errors.py`` into text reports suitable for display or logging.
"""

from typing import Optional

from .errors import (
    analyze_asr_errors,
    analyze_retrieval_failures,
    categorize_errors,
)


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
            lines.append(
                f"Failed Queries ({len(retrieval_analysis['failed_queries'])}):"
            )
            for q in retrieval_analysis["failed_queries"][:5]:
                lines.append(
                    f"  - \"{q['query']}\" (length: {q['query_length']} words)"
                )
            if len(retrieval_analysis["failed_queries"]) > 5:
                lines.append(
                    f"  ... and {len(retrieval_analysis['failed_queries']) - 5} more"
                )
            lines.append("")

        # Near misses
        if retrieval_analysis["near_misses"]:
            lines.append(f"Near Misses ({len(retrieval_analysis['near_misses'])}):")
            for q in retrieval_analysis["near_misses"][:5]:
                lines.append(
                    f"  - \"{q['query']}\" (relevant at rank {q['first_relevant_rank']})"
                )
            if len(retrieval_analysis["near_misses"]) > 5:
                lines.append(
                    f"  ... and {len(retrieval_analysis['near_misses']) - 5} more"
                )
            lines.append("")

        # Query characteristics
        qc = retrieval_analysis["query_characteristics"]
        if qc:
            lines.append("Query Characteristics:")
            lines.append(
                f"  Avg Query Length (failed): {qc.get('avg_length_failed', 0):.1f} words"
            )
            lines.append(
                f"  Avg Query Length (success): {qc.get('avg_length_success', 0):.1f} words"
            )
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
