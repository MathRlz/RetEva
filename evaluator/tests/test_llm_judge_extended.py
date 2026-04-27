"""Tests for extended LLM judge functionality."""

import pytest
from unittest.mock import Mock, patch
from evaluator.config import JudgeConfig
from evaluator.judge.calibration import (
    JudgeCalibrator,
    detect_score_bias,
    compute_inter_rater_agreement,
    normalize_scores_min_max,
    apply_temperature_scaling,
    ensemble_judge_scores,
    ScoreTracker,
)
from evaluator.judge.prompts import (
    get_judge_prompt,
    get_multi_aspect_prompt,
    get_structured_judge_prompt,
    get_few_shot_examples,
    ASPECT_PROMPTS,
)
from evaluator.judge.core import run_llm_judging


class TestJudgeConfig:
    """Tests for JudgeConfig validation."""
    
    def test_valid_basic_config(self):
        """Test creating valid basic config."""
        config = JudgeConfig(
            enabled=True,
            model="gpt-4",
            max_cases=100
        )
        
        assert config.enabled is True
        assert config.judge_aspects == ["relevance"]
    
    def test_valid_multi_aspect_config(self):
        """Test multi-aspect configuration."""
        config = JudgeConfig(
            enabled=True,
            judge_aspects=["relevance", "accuracy", "completeness"],
            score_aggregation="average"
        )
        
        assert len(config.judge_aspects) == 3
    
    def test_weighted_aggregation_valid(self):
        """Test weighted aggregation with proper weights."""
        config = JudgeConfig(
            enabled=True,
            judge_aspects=["relevance", "accuracy"],
            score_aggregation="weighted",
            aspect_weights={"relevance": 0.7, "accuracy": 0.3}
        )
        
        assert config.score_aggregation == "weighted"
    
    def test_weighted_aggregation_missing_weights(self):
        """Test error when weights are missing for weighted aggregation."""
        with pytest.raises(ValueError, match="aspect_weights must be provided"):
            JudgeConfig(
                judge_aspects=["relevance", "accuracy"],
                score_aggregation="weighted"
            )
    
    def test_weighted_aggregation_invalid_sum(self):
        """Test error when weights don't sum to 1.0."""
        with pytest.raises(ValueError, match="aspect_weights must sum to 1.0"):
            JudgeConfig(
                judge_aspects=["relevance", "accuracy"],
                score_aggregation="weighted",
                aspect_weights={"relevance": 0.6, "accuracy": 0.6}
            )
    
    def test_invalid_aspect(self):
        """Test error on invalid aspect."""
        with pytest.raises(ValueError, match="Invalid judge aspect"):
            JudgeConfig(judge_aspects=["invalid_aspect"])
    
    def test_invalid_aggregation(self):
        """Test error on invalid aggregation method."""
        with pytest.raises(ValueError, match="score_aggregation must be one of"):
            JudgeConfig(score_aggregation="invalid")
    
    def test_invalid_output_format(self):
        """Test error on invalid output format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            JudgeConfig(output_format="invalid")


class TestJudgeExecutionLimits:
    """Tests for run_llm_judging max_cases behavior."""

    @patch("evaluator.judge.core.judge_trace_with_openai_compatible")
    def test_max_cases_zero_means_all(self, mock_judge):
        traces = [{"query_id": "q1"}, {"query_id": "q2"}, {"query_id": "q3"}]
        mock_judge.return_value = {"score": 4.0, "verdict": "good", "reason": ""}

        out = run_llm_judging(
            traces,
            api_base="http://localhost:11434/v1/chat/completions",
            model="test-model",
            api_key_env="OPENAI_API_KEY",
            temperature=0.0,
            max_cases=0,
            timeout_s=30,
        )

        assert out["cases"] == 3
        assert len(out["details"]) == 3
        assert mock_judge.call_count == 3

    def test_negative_max_cases_raises(self):
        with pytest.raises(RuntimeError, match="max_cases must be >= 0"):
            run_llm_judging(
                [],
                api_base="http://localhost:11434/v1/chat/completions",
                model="test-model",
                api_key_env="OPENAI_API_KEY",
                temperature=0.0,
                max_cases=-1,
                timeout_s=30,
            )


class TestJudgePrompts:
    """Tests for judge prompt generation."""
    
    def test_get_judge_prompt_basic(self):
        """Test basic judge prompt generation."""
        system, user = get_judge_prompt(
            "What causes diabetes?",
            "Diabetes is caused by...",
            aspect="relevance"
        )
        
        assert "medical" in system.lower()
        assert "diabetes" in user.lower()
        assert "1-5" in user
    
    def test_get_judge_prompt_with_cot(self):
        """Test prompt with chain-of-thought."""
        system, user = get_judge_prompt(
            "Test query",
            "Test document",
            aspect="relevance",
            chain_of_thought=True
        )
        
        assert "step by step" in user.lower()
        assert "reasoning" in user.lower()
    
    def test_get_judge_prompt_with_examples(self):
        """Test prompt with few-shot examples."""
        examples = [
            {
                "query": "Example query",
                "document": "Example doc",
                "score": 5,
                "reasoning": "Perfect match"
            }
        ]
        
        system, user = get_judge_prompt(
            "Test query",
            "Test document",
            few_shot_examples=examples
        )
        
        assert "Example query" in system
        assert "Perfect match" in system
    
    def test_get_judge_prompt_invalid_aspect(self):
        """Test error on invalid aspect."""
        with pytest.raises(ValueError, match="Unknown aspect"):
            get_judge_prompt("query", "doc", aspect="invalid")
    
    def test_get_multi_aspect_prompt(self):
        """Test multi-aspect prompt generation."""
        system, user = get_multi_aspect_prompt(
            "Test query",
            "Test document",
            aspects=["relevance", "accuracy"]
        )
        
        assert "relevance" in user.lower()
        assert "accuracy" in user.lower()
    
    def test_get_structured_judge_prompt(self):
        """Test structured prompt generation."""
        system, user = get_structured_judge_prompt(
            "Test query",
            "Test document",
            reference_answer="Reference answer"
        )
        
        assert "JSON" in system
        assert "Reference answer" in user
    
    def test_aspect_prompts_exist(self):
        """Test all aspects have prompt templates."""
        required_aspects = ["relevance", "accuracy", "completeness", "clarity", "factuality"]
        
        for aspect in required_aspects:
            assert aspect in ASPECT_PROMPTS
            assert "name" in ASPECT_PROMPTS[aspect]
            assert "criteria" in ASPECT_PROMPTS[aspect]
    
    def test_get_few_shot_examples(self):
        """Test few-shot example retrieval."""
        examples = get_few_shot_examples(n=2)
        
        assert len(examples) == 2
        assert "query" in examples[0]
        assert "document" in examples[0]
        assert "score" in examples[0]


class TestJudgeCalibrator:
    """Tests for judge calibration."""
    
    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        calibrator = JudgeCalibrator()
        
        assert not calibrator.is_fitted
        assert len(calibrator.score_history) == 0
    
    def test_add_scores(self):
        """Test adding scores to calibrator."""
        calibrator = JudgeCalibrator()
        
        calibrator.add_score(4.0, "relevance")
        calibrator.add_score(3.5, "accuracy")
        
        assert len(calibrator.score_history) == 2
        assert len(calibrator.aspect_scores["relevance"]) == 1
    
    def test_fit_insufficient_samples(self):
        """Test fitting with insufficient samples."""
        calibrator = JudgeCalibrator()
        
        for _ in range(5):
            calibrator.add_score(3.0)
        
        result = calibrator.fit(min_samples=10)
        
        assert result is False
        assert not calibrator.is_fitted
    
    def test_fit_sufficient_samples(self):
        """Test fitting with sufficient samples."""
        calibrator = JudgeCalibrator()
        
        scores = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 3.0, 3.5, 4.0, 4.5]
        for score in scores:
            calibrator.add_score(score)
        
        result = calibrator.fit(min_samples=10)
        
        assert result is True
        assert calibrator.is_fitted
        assert calibrator.mean_score > 0
    
    def test_calibrate_score_not_fitted(self):
        """Test calibration without fitting."""
        calibrator = JudgeCalibrator()
        
        calibrated = calibrator.calibrate_score(4.0)
        
        assert calibrated == 4.0  # Returns original
    
    def test_calibrate_score_fitted(self):
        """Test score calibration after fitting."""
        calibrator = JudgeCalibrator()
        
        # Add biased scores (all high)
        scores = [4.0, 4.5, 5.0, 4.5, 4.0, 5.0, 4.5, 4.0, 4.5, 5.0]
        for score in scores:
            calibrator.add_score(score)
        
        calibrator.fit(min_samples=10)
        
        # Calibrate should bring scores closer to target mean (3.0)
        calibrated = calibrator.calibrate_score(5.0, target_mean=3.0)
        
        assert 1.0 <= calibrated <= 5.0
        assert calibrated < 5.0  # Should be adjusted down
    
    def test_get_statistics(self):
        """Test getting calibration statistics."""
        calibrator = JudgeCalibrator()
        
        calibrator.add_score(3.0, "relevance")
        calibrator.add_score(4.0, "accuracy")
        
        stats = calibrator.get_statistics()
        
        assert "overall" in stats
        assert "relevance" in stats
        assert "accuracy" in stats


class TestScoreBias:
    """Tests for score bias detection."""
    
    def test_no_bias(self):
        """Test no bias detection."""
        scores = [2.5, 3.0, 3.5, 3.0, 2.8, 3.2]
        
        is_biased, bias = detect_score_bias(scores, expected_mean=3.0)
        
        assert not is_biased
        assert abs(bias) < 0.5
    
    def test_positive_bias(self):
        """Test positive bias detection."""
        scores = [4.5, 5.0, 4.8, 4.7, 5.0, 4.9]
        
        is_biased, bias = detect_score_bias(scores, expected_mean=3.0, threshold=0.5)
        
        assert is_biased
        assert bias > 0.5
    
    def test_negative_bias(self):
        """Test negative bias detection."""
        scores = [1.5, 2.0, 1.8, 2.2, 1.9, 2.1]
        
        is_biased, bias = detect_score_bias(scores, expected_mean=3.0, threshold=0.5)
        
        assert is_biased
        assert bias < -0.5


class TestInterRaterAgreement:
    """Tests for inter-rater agreement."""
    
    def test_perfect_agreement(self):
        """Test perfect agreement."""
        scores_a = [3.0, 4.0, 5.0, 2.0, 3.5]
        scores_b = [3.0, 4.0, 5.0, 2.0, 3.5]
        
        metrics = compute_inter_rater_agreement(scores_a, scores_b)
        
        assert metrics["correlation"] == pytest.approx(1.0)
        assert metrics["mae"] == 0.0
        assert metrics["exact_agreement"] == 1.0
    
    def test_partial_agreement(self):
        """Test partial agreement."""
        scores_a = [3.0, 4.0, 5.0, 2.0]
        scores_b = [3.5, 4.0, 4.5, 2.5]
        
        metrics = compute_inter_rater_agreement(scores_a, scores_b)
        
        assert 0.0 < metrics["correlation"] < 1.0
        assert metrics["mae"] > 0.0
        assert 0.0 < metrics["exact_agreement"] < 1.0
    
    def test_mismatched_lengths(self):
        """Test error on mismatched lengths."""
        scores_a = [3.0, 4.0]
        scores_b = [3.0, 4.0, 5.0]
        
        with pytest.raises(ValueError, match="same length"):
            compute_inter_rater_agreement(scores_a, scores_b)


class TestScoreNormalization:
    """Tests for score normalization."""
    
    def test_min_max_normalization(self):
        """Test min-max normalization."""
        scores = [2.0, 3.0, 4.0, 5.0]
        
        normalized = normalize_scores_min_max(scores, target_min=1.0, target_max=5.0)
        
        assert min(normalized) == 1.0
        assert max(normalized) == 5.0
    
    def test_min_max_all_same(self):
        """Test normalization when all scores are same."""
        scores = [3.0, 3.0, 3.0]
        
        normalized = normalize_scores_min_max(scores)
        
        assert all(s == 3.0 for s in normalized)
    
    def test_temperature_scaling(self):
        """Test temperature scaling."""
        scores = [2.0, 3.0, 4.0, 5.0]
        
        # High temperature = softer distribution
        soft = apply_temperature_scaling(scores, temperature=2.0)
        
        # Should still be in valid range
        assert all(1.0 <= s <= 5.0 for s in soft)
    
    def test_temperature_scaling_identity(self):
        """Test temperature=1.0 is identity."""
        scores = [2.0, 3.0, 4.0, 5.0]
        
        scaled = apply_temperature_scaling(scores, temperature=1.0)
        
        # Should be very close to original
        for original, scaled_val in zip(scores, scaled):
            assert abs(original - scaled_val) < 0.1


class TestEnsembleScores:
    """Tests for ensemble scoring."""
    
    def test_mean_ensemble(self):
        """Test mean ensemble."""
        scores1 = [3.0, 4.0, 5.0]
        scores2 = [3.5, 4.5, 4.5]
        
        ensemble = ensemble_judge_scores([scores1, scores2], method="mean")
        
        assert len(ensemble) == 3
        assert ensemble[0] == 3.25
    
    def test_median_ensemble(self):
        """Test median ensemble."""
        scores1 = [3.0, 4.0, 5.0]
        scores2 = [3.5, 4.5, 4.5]
        scores3 = [3.2, 4.2, 4.8]
        
        ensemble = ensemble_judge_scores([scores1, scores2, scores3], method="median")
        
        assert len(ensemble) == 3
        assert ensemble[0] == 3.2
    
    def test_weighted_ensemble(self):
        """Test weighted ensemble."""
        scores1 = [3.0, 4.0, 5.0]
        scores2 = [5.0, 5.0, 5.0]
        
        ensemble = ensemble_judge_scores(
            [scores1, scores2],
            method="weighted",
            weights=[0.7, 0.3]
        )
        
        assert len(ensemble) == 3
        # Weighted avg of [3.0, 5.0] with [0.7, 0.3]
        assert ensemble[0] == pytest.approx(3.6)
    
    def test_weighted_without_weights(self):
        """Test error when weights missing for weighted ensemble."""
        scores1 = [3.0, 4.0]
        scores2 = [4.0, 5.0]
        
        with pytest.raises(ValueError, match="weights must be provided"):
            ensemble_judge_scores([scores1, scores2], method="weighted")


class TestScoreTracker:
    """Tests for score tracking."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ScoreTracker()
        
        assert len(tracker.scores) == 0
    
    def test_add_scores(self):
        """Test adding scores."""
        tracker = ScoreTracker()
        
        tracker.add(4.0, aspect="relevance", query="test query")
        tracker.add(3.5, aspect="accuracy", query="test query 2")
        
        assert len(tracker.scores) == 2
        assert tracker.aspects[0] == "relevance"
    
    def test_get_summary(self):
        """Test summary statistics."""
        tracker = ScoreTracker()
        
        tracker.add(3.0, aspect="relevance")
        tracker.add(4.0, aspect="relevance")
        tracker.add(5.0, aspect="accuracy")
        
        summary = tracker.get_summary()
        
        assert "mean" in summary
        assert "std" in summary
        assert "aspects" in summary
        assert "relevance" in summary["aspects"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
