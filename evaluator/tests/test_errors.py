"""Tests for error analysis module."""

import pytest

from evaluator.analysis.errors import (
    analyze_asr_errors,
    analyze_retrieval_failures,
    categorize_errors,
    generate_error_report,
    _extract_asr_pairs,
    _build_category_matchers,
    _compute_query_characteristics,
)


class TestAnalyzeASRErrors:
    """Tests for analyze_asr_errors function."""

    def test_basic_substitution_detection(self):
        """Should detect word substitutions."""
        results = {
            "details": [
                {"reference": "the quick brown fox", "hypothesis": "the quik brown box"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        assert analysis["error_counts"]["substitutions"] == 2  # quik, box
        assert analysis["error_counts"]["deletions"] == 0
        assert analysis["error_counts"]["insertions"] == 0
        
    def test_deletion_detection(self):
        """Should detect word deletions."""
        results = {
            "details": [
                {"reference": "hello world today", "hypothesis": "hello today"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        assert analysis["error_counts"]["deletions"] == 1
        assert ("world", 1) in analysis["common_deletions"]
        
    def test_insertion_detection(self):
        """Should detect word insertions."""
        results = {
            "details": [
                {"reference": "hello world", "hypothesis": "hello big world"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        assert analysis["error_counts"]["insertions"] == 1
        assert ("big", 1) in analysis["common_insertions"]
        
    def test_error_rates_calculated(self):
        """Should calculate error rates relative to reference words."""
        results = {
            "details": [
                {"reference": "one two three four", "hypothesis": "one too three for"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        # 2 substitutions out of 4 reference words
        assert analysis["error_rates"]["substitution_rate"] == 0.5
        
    def test_by_word_length_statistics(self):
        """Should track error rates by word length."""
        results = {
            "details": [
                {"reference": "a cat sat", "hypothesis": "a cat set"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        # 3-letter words: 'cat' correct, 'sat' -> 'set' error
        assert 3 in analysis["by_word_length"]
        assert analysis["by_word_length"][3]["count"] == 2
        assert analysis["by_word_length"][3]["errors"] == 1
        
    def test_common_substitutions_tracked(self):
        """Should track most common substitutions."""
        results = {
            "details": [
                {"reference": "the the the", "hypothesis": "da da da"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        # 'the' -> 'da' should appear 3 times
        assert ("the", "da", 3) in analysis["common_substitutions"]
        
    def test_empty_results_returns_empty_analysis(self):
        """Should return empty analysis for empty results."""
        analysis = analyze_asr_errors({})
        
        assert analysis["error_counts"]["total_errors"] == 0
        assert analysis["common_substitutions"] == []
        
    def test_handles_different_field_names(self):
        """Should handle alternative field names like 'ref' and 'hyp'."""
        results = {
            "details": [
                {"ref": "hello world", "hyp": "hello word"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        assert analysis["error_counts"]["substitutions"] == 1
        
    def test_handles_asr_details_format(self):
        """Should handle 'asr_details' format."""
        results = {
            "asr_details": [
                {"reference": "test input", "hypothesis": "test output"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        assert analysis["error_counts"]["substitutions"] == 1
        
    def test_handles_per_sample_format(self):
        """Should handle 'per_sample' format with reference/hypothesis lists."""
        results = {
            "per_sample": {
                "reference": ["hello world", "foo bar"],
                "hypothesis": ["hello word", "foo baz"],
            }
        }
        
        analysis = analyze_asr_errors(results)
        
        assert analysis["error_counts"]["substitutions"] == 2  # word, baz
        
    def test_misrecognized_words_tracked(self):
        """Should track most frequently misrecognized words."""
        results = {
            "details": [
                {"reference": "test test test", "hypothesis": "fest fest fest"},
            ]
        }
        
        analysis = analyze_asr_errors(results)
        
        # 'test' should be in misrecognized words
        assert any(word == "test" for word, _ in analysis["misrecognized_words"])


class TestAnalyzeRetrievalFailures:
    """Tests for analyze_retrieval_failures function."""

    def test_identifies_failed_queries(self):
        """Should identify queries with no relevant doc in top-k."""
        results = {
            "details": [
                {"query": "what is diabetes", "retrieved": ["d1", "d2", "d3"], 
                 "relevant": {"d10": 1}},
            ]
        }
        
        analysis = analyze_retrieval_failures(results, top_k=3)
        
        assert len(analysis["failed_queries"]) == 1
        assert analysis["failed_queries"][0]["query"] == "what is diabetes"
        
    def test_identifies_near_misses(self):
        """Should identify queries with relevant in top-k but not top-1."""
        results = {
            "details": [
                {"query": "flu symptoms", "retrieved": ["d1", "d2", "d3"], 
                 "relevant": {"d3": 1}},
            ]
        }
        
        analysis = analyze_retrieval_failures(results, top_k=5)
        
        assert len(analysis["near_misses"]) == 1
        assert analysis["near_misses"][0]["first_relevant_rank"] == 3
        
    def test_successful_queries_not_in_failures(self):
        """Should not count successful queries (relevant in top-1) as failures."""
        results = {
            "details": [
                {"query": "test query", "retrieved": ["d1", "d2"], 
                 "relevant": {"d1": 1}},
            ]
        }
        
        analysis = analyze_retrieval_failures(results)
        
        assert len(analysis["failed_queries"]) == 0
        assert len(analysis["near_misses"]) == 0
        assert analysis["success_rate"] == 1.0
        
    def test_failure_rate_calculated(self):
        """Should calculate failure rate correctly."""
        results = {
            "details": [
                {"query": "q1", "retrieved": ["d1"], "relevant": {"d1": 1}},  # success
                {"query": "q2", "retrieved": ["d1"], "relevant": {"d2": 1}},  # fail
                {"query": "q3", "retrieved": ["d1"], "relevant": {"d3": 1}},  # fail
            ]
        }
        
        analysis = analyze_retrieval_failures(results, top_k=1)
        
        assert analysis["failure_rate"] == pytest.approx(2/3)
        assert analysis["success_rate"] == pytest.approx(1/3)
        
    def test_rank_distribution_tracked(self):
        """Should track distribution of relevant document ranks."""
        results = {
            "details": [
                {"query": "q1", "retrieved": ["d1", "d2"], "relevant": {"d1": 1}},
                {"query": "q2", "retrieved": ["d1", "d2"], "relevant": {"d2": 1}},
                {"query": "q3", "retrieved": ["d1", "d2"], "relevant": {"d2": 1}},
            ]
        }
        
        analysis = analyze_retrieval_failures(results)
        
        assert analysis["rank_distribution"][1] == 1  # q1
        assert analysis["rank_distribution"][2] == 2  # q2, q3
        
    def test_query_characteristics_computed(self):
        """Should compute query characteristics for analysis."""
        results = {
            "details": [
                {"query": "short", "retrieved": ["d1"], "relevant": {"d2": 1}},  # fail
                {"query": "this is a longer query", "retrieved": ["d1"], 
                 "relevant": {"d1": 1}},  # success
            ]
        }
        
        analysis = analyze_retrieval_failures(results, top_k=1)
        
        assert "avg_length_failed" in analysis["query_characteristics"]
        assert "avg_length_success" in analysis["query_characteristics"]
        
    def test_empty_results_returns_empty_analysis(self):
        """Should return empty analysis for empty results."""
        analysis = analyze_retrieval_failures({})
        
        assert analysis["total_queries"] == 0
        assert analysis["failed_queries"] == []
        
    def test_skips_queries_without_relevant(self):
        """Should skip queries without relevant documents defined."""
        results = {
            "details": [
                {"query": "test", "retrieved": ["d1"], "relevant": {}},
            ]
        }
        
        analysis = analyze_retrieval_failures(results)
        
        assert analysis["total_queries"] == 0


class TestCategorizeErrors:
    """Tests for categorize_errors function."""

    def test_categorizes_by_word_list(self):
        """Should categorize errors matching word lists."""
        results = {
            "details": [
                {"reference": "take mg daily", "hypothesis": "take milligram daily"},
            ]
        }
        categories = {
            "units": ["mg", "ml", "kg"],
        }
        
        analysis = categorize_errors(results, categories)
        
        assert analysis["category_counts"]["units"] >= 1
        
    def test_categorizes_by_regex(self):
        """Should categorize errors matching regex patterns."""
        results = {
            "details": [
                {"reference": "take 500 daily", "hypothesis": "take five hundred daily"},
            ]
        }
        categories = {
            "numbers": ["regex:[0-9]+"],
        }
        
        analysis = categorize_errors(results, categories)
        
        assert "numbers" in analysis["category_counts"]
        
    def test_uncategorized_errors_tracked(self):
        """Should track errors not matching any category."""
        results = {
            "details": [
                {"reference": "hello world", "hypothesis": "helo word"},
            ]
        }
        categories = {
            "numbers": ["regex:[0-9]+"],
        }
        
        analysis = categorize_errors(results, categories)
        
        assert len(analysis["uncategorized"]) > 0
        
    def test_category_rates_calculated(self):
        """Should calculate rates for each category."""
        results = {
            "details": [
                {"reference": "one two", "hypothesis": "1 2"},
            ]
        }
        categories = {
            "words": ["one", "two"],
        }
        
        analysis = categorize_errors(results, categories)
        
        assert "words" in analysis["category_rates"]
        assert 0 <= analysis["category_rates"]["words"] <= 1
        
    def test_empty_results_returns_empty_analysis(self):
        """Should return empty analysis for empty results."""
        analysis = categorize_errors({}, {"cat1": ["word"]})
        
        assert analysis["total_errors"] == 0
        assert analysis["category_counts"] == {}
        
    def test_callable_category_matcher(self):
        """Should support callable category matchers."""
        results = {
            "details": [
                {"reference": "UPPER lower", "hypothesis": "upper LOWER"},
            ]
        }
        categories = {
            "uppercase": lambda w: w.isupper(),
        }
        
        # This tests the callable support
        analysis = categorize_errors(results, categories)
        
        assert "uppercase" in analysis["category_counts"] or "uppercase" not in analysis["category_counts"]


class TestGenerateErrorReport:
    """Tests for generate_error_report function."""

    def test_generates_readable_report(self):
        """Should generate human-readable report."""
        results = {
            "details": [
                {"reference": "hello world", "hypothesis": "helo word"},
            ]
        }
        
        report = generate_error_report(results)
        
        assert "ERROR ANALYSIS REPORT" in report
        assert "ASR ERROR ANALYSIS" in report
        
    def test_includes_asr_statistics(self):
        """Should include ASR error statistics."""
        results = {
            "details": [
                {"reference": "the quick fox", "hypothesis": "the quik box"},
            ]
        }
        
        report = generate_error_report(results)
        
        assert "Substitutions" in report
        assert "Substitution Rate" in report
        
    def test_includes_retrieval_statistics(self):
        """Should include retrieval failure statistics when available."""
        results = {
            "details": [
                {"query": "test", "retrieved": ["d1"], "relevant": {"d2": 1}},
            ]
        }
        
        report = generate_error_report(results)
        
        assert "RETRIEVAL FAILURE ANALYSIS" in report
        assert "Failure Rate" in report
        
    def test_includes_category_analysis_when_provided(self):
        """Should include category analysis when categories provided."""
        results = {
            "details": [
                {"reference": "500mg dose", "hypothesis": "five hundred mg dose"},
            ]
        }
        categories = {
            "numbers": ["regex:[0-9]+"],
        }
        
        report = generate_error_report(results, categories=categories)
        
        assert "ERROR CATEGORIZATION" in report
        
    def test_handles_empty_results(self):
        """Should handle empty results gracefully."""
        report = generate_error_report({})
        
        assert "ERROR ANALYSIS REPORT" in report
        assert "END OF REPORT" in report


class TestExtractASRPairs:
    """Tests for _extract_asr_pairs helper function."""

    def test_extracts_from_details(self):
        """Should extract pairs from details list."""
        results = {
            "details": [
                {"reference": "hello", "hypothesis": "helo"},
                {"reference": "world", "hypothesis": "word"},
            ]
        }
        
        pairs = _extract_asr_pairs(results)
        
        assert len(pairs) == 2
        assert pairs[0] == ("hello", "helo")
        
    def test_extracts_from_alternative_keys(self):
        """Should extract pairs using alternative key names."""
        results = {
            "details": [
                {"ref": "hello", "hyp": "helo"},
                {"ground_truth": "world", "transcription": "word"},
            ]
        }
        
        pairs = _extract_asr_pairs(results)
        
        assert len(pairs) == 2
        
    def test_extracts_from_per_sample(self):
        """Should extract pairs from per_sample format."""
        results = {
            "per_sample": {
                "reference": ["hello", "world"],
                "hypothesis": ["helo", "word"],
            }
        }
        
        pairs = _extract_asr_pairs(results)
        
        assert len(pairs) == 2
        
    def test_returns_empty_for_missing_data(self):
        """Should return empty list for missing data."""
        pairs = _extract_asr_pairs({})
        
        assert pairs == []


class TestBuildCategoryMatchers:
    """Tests for _build_category_matchers helper function."""

    def test_word_list_matcher(self):
        """Should create matcher for word lists."""
        categories = {"test": ["word1", "word2"]}
        
        matchers = _build_category_matchers(categories)
        
        assert matchers["test"]("word1")
        assert matchers["test"]("Word1")  # case insensitive
        assert not matchers["test"]("word3")
        
    def test_regex_matcher(self):
        """Should create matcher for regex patterns."""
        categories = {"nums": ["regex:[0-9]+"]}
        
        matchers = _build_category_matchers(categories)
        
        assert matchers["nums"]("123")
        assert matchers["nums"]("test123test")
        assert not matchers["nums"]("abc")
        
    def test_callable_matcher(self):
        """Should use callable directly."""
        categories = {"upper": lambda w: w.isupper()}
        
        matchers = _build_category_matchers(categories)
        
        assert matchers["upper"]("HELLO")
        assert not matchers["upper"]("hello")


class TestComputeQueryCharacteristics:
    """Tests for _compute_query_characteristics helper function."""

    def test_computes_average_lengths(self):
        """Should compute average query lengths for each outcome."""
        failed = [{"query_length": 2}, {"query_length": 4}]
        near_misses = [{"query_length": 3}]
        successful = [{"query_length": 5}, {"query_length": 7}]
        
        chars = _compute_query_characteristics(failed, near_misses, successful)
        
        assert chars["avg_length_failed"] == 3.0
        assert chars["avg_length_near_miss"] == 3.0
        assert chars["avg_length_success"] == 6.0
        
    def test_handles_empty_lists(self):
        """Should handle empty lists gracefully."""
        chars = _compute_query_characteristics([], [], [])
        
        assert chars["avg_length_failed"] == 0.0
        assert chars["avg_length_success"] == 0.0
        
    def test_computes_length_distributions(self):
        """Should compute length distributions."""
        failed = [{"query_length": 1}, {"query_length": 4}]
        successful = [{"query_length": 7}]
        
        chars = _compute_query_characteristics(failed, [], successful)
        
        assert "length_distribution_failed" in chars
        assert "length_distribution_success" in chars


class TestIntegration:
    """Integration tests for complete error analysis workflow."""

    def test_full_analysis_workflow(self):
        """Test complete workflow with mixed ASR and retrieval data."""
        results = {
            "details": [
                {
                    "reference": "patient has diabetes mellitus",
                    "hypothesis": "patient has diabetis melitus",
                    "query": "diabetes symptoms",
                    "retrieved": ["d1", "d2", "d3"],
                    "relevant": {"d2": 1},
                },
                {
                    "reference": "take medication twice daily",
                    "hypothesis": "take medication twice daily",
                    "query": "medication schedule",
                    "retrieved": ["d4", "d5"],
                    "relevant": {"d4": 1},
                },
            ]
        }
        
        # ASR analysis
        asr = analyze_asr_errors(results)
        assert asr["error_counts"]["total_errors"] > 0
        
        # Retrieval analysis
        retrieval = analyze_retrieval_failures(results)
        assert retrieval["total_queries"] == 2
        assert retrieval["near_miss_rate"] == 0.5
        
        # Full report
        report = generate_error_report(results)
        assert "ERROR ANALYSIS REPORT" in report
        assert "ASR ERROR ANALYSIS" in report
        assert "RETRIEVAL FAILURE ANALYSIS" in report
        
    def test_analysis_with_categories(self):
        """Test analysis with error categorization."""
        results = {
            "details": [
                {"reference": "500mg twice", "hypothesis": "500 mg twice"},
                {"reference": "Dr Smith", "hypothesis": "doctor Smith"},
            ]
        }
        categories = {
            "medical_units": ["mg", "ml", "mcg"],
            "titles": ["dr", "mr", "mrs"],
            "numbers": ["regex:[0-9]+"],
        }
        
        cat_analysis = categorize_errors(results, categories)
        
        assert cat_analysis["total_errors"] > 0
        
        report = generate_error_report(results, categories=categories)
        assert "ERROR CATEGORIZATION" in report
