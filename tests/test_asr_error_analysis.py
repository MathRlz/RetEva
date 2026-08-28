"""E3/F12: analyze_asr_errors decomposed into accumulate/tally/build — behaviour pinned."""

from evaluator.analysis.errors import analyze_asr_errors


def test_empty_results_returns_zeroed_structure():
    out = analyze_asr_errors({})
    assert out["error_counts"]["total_errors"] == 0
    assert out["common_substitutions"] == []
    assert out["by_word_length"] == {}


def test_substitution_insertion_deletion_counts():
    results = {
        "details": [
            # one substitution: quick->quik, one substitution box->fox  (ref vs hyp)
            {"reference": "the quick brown fox", "hypothesis": "the quik brown box"},
            # one deletion (world dropped) -> actually a substitution world->word
            {"reference": "hello world", "hypothesis": "hello word"},
        ]
    }
    out = analyze_asr_errors(results)
    ec = out["error_counts"]
    # 3 substitutions total (quik, box, word), no insertions/deletions
    assert ec["substitutions"] == 3
    assert ec["insertions"] == 0
    assert ec["deletions"] == 0
    assert ec["total_errors"] == 3
    # substitution rate = 3 errors / 6 reference words
    assert abs(out["error_rates"]["substitution_rate"] - 3 / 6) < 1e-9
    # common substitutions carry (ref, hyp, count) triples
    subs = {(r, h): c for r, h, c in out["common_substitutions"]}
    assert subs[("quick", "quik")] == 1
    assert subs[("fox", "box")] == 1
    assert subs[("world", "word")] == 1


def test_pure_insertion_and_deletion():
    out_ins = analyze_asr_errors(
        {"details": [{"reference": "alpha beta", "hypothesis": "alpha beta gamma"}]}
    )
    assert out_ins["error_counts"]["insertions"] == 1
    assert out_ins["error_counts"]["substitutions"] == 0

    out_del = analyze_asr_errors(
        {"details": [{"reference": "alpha beta gamma", "hypothesis": "alpha beta"}]}
    )
    assert out_del["error_counts"]["deletions"] == 1
    assert ("gamma", 1) in out_del["common_deletions"]
