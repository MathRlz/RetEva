from typing import Dict, Optional
from jiwer import cer, wer, RemovePunctuation, ToLowerCase, RemoveMultipleSpaces, Compose

normalize = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces()
])

def word_error_rate(ref: str, hyp: str) -> float:
    ref = normalize(ref)
    hyp = normalize(hyp)
    return wer(ref, hyp)

def character_error_rate(ref: str, hyp: str) -> float:
    ref = normalize(ref)
    hyp = normalize(hyp)
    return cer(ref, hyp)


def term_weighted_wer(
    ref: str,
    hyp: str,
    term_weights: Optional[Dict[str, float]] = None,
    default_weight: float = 1.0,
) -> float:
    """Word error rate where domain terms carry higher penalty weight.

    Each word in *ref* contributes ``term_weights.get(word, default_weight)``
    to both the denominator (total weight) and the numerator (weight of
    incorrectly matched words).  Substitutions/deletions on high-weight terms
    increase the score more than errors on common words.

    Args:
        ref: Reference transcription.
        hyp: Hypothesis transcription.
        term_weights: Mapping of (lowercase) term → weight multiplier.
            Terms not in the mapping use *default_weight*.
        default_weight: Weight for words not listed in term_weights. Default 1.0.

    Returns:
        Weighted error rate in [0, ∞). Returns 0.0 for empty reference.
    """
    if term_weights is None:
        term_weights = {}

    ref_norm = normalize(ref)
    hyp_norm = normalize(hyp)

    ref_words = ref_norm.split() if ref_norm.strip() else []
    hyp_words = hyp_norm.split() if hyp_norm.strip() else []

    if not ref_words:
        return 0.0

    # Build per-word weight list for reference
    weights = [term_weights.get(w, default_weight) for w in ref_words]
    total_weight = sum(weights)
    if total_weight == 0.0:
        return 0.0

    # Weighted edit distance via dynamic programming
    n, m = len(ref_words), len(hyp_words)
    # dp[i][j] = weighted cost to align ref[:i] with hyp[:j]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + weights[i - 1]  # deletion cost
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + default_weight   # insertion cost (unweighted)

    for i in range(1, n + 1):
        w = weights[i - 1]
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + w,              # deletion
                    dp[i][j - 1] + default_weight,  # insertion
                    dp[i - 1][j - 1] + w,           # substitution
                )

    return dp[n][m] / total_weight

