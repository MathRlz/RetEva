"""Unit tests for evaluation module functions and logic."""
import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import numpy as np

from evaluator.metrics.ir import (
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision,
)
from evaluator.metrics.stt import word_error_rate, character_error_rate


class TestReciprocalRank:
    """Tests for reciprocal_rank function."""

    def test_first_relevant_at_position_1(self):
        """First result is relevant, should return 1.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1": 1}
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_first_relevant_at_position_2(self):
        """First relevant at position 2, should return 0.5."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2": 1}
        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_first_relevant_at_position_3(self):
        """First relevant at position 3, should return 1/3."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3": 1}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_documents(self):
        """No relevant documents found, should return 0.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4": 1}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_retrieved_list(self):
        """Empty retrieved list, should return 0.0."""
        retrieved = []
        relevant = {"doc1": 1}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_relevant_dict(self):
        """Empty relevant dict, should return 0.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_multiple_relevant_returns_first_match(self):
        """Multiple relevant docs, should return rank of first match."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2": 1, "doc3": 1}
        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_graded_relevance_zero_not_relevant(self):
        """Relevance grade 0 should not be considered relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1": 0, "doc2": 1}
        assert reciprocal_rank(retrieved, relevant) == 0.5


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_all_relevant(self):
        """All k results are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1": 1, "doc2": 1, "doc3": 1}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_half_relevant(self):
        """Half of k results are relevant."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1": 1, "doc3": 1}
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_none_relevant(self):
        """No relevant results in top k."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4": 1}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_k_zero(self):
        """k=0 should return 0.0."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1": 1}
        assert precision_at_k(retrieved, relevant, 0) == 0.0

    def test_k_larger_than_retrieved(self):
        """k larger than retrieved list uses k in denominator."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1": 1, "doc2": 1}
        # Precision@10 = 2 relevant / 10 = 0.2
        assert precision_at_k(retrieved, relevant, 10) == pytest.approx(0.2)


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_all_relevant_retrieved(self):
        """All relevant documents retrieved in top k."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1": 1, "doc2": 1}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self):
        """Only some relevant documents retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1": 1, "doc4": 1}
        assert recall_at_k(retrieved, relevant, 3) == 0.5

    def test_no_recall(self):
        """No relevant documents in top k."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4": 1, "doc5": 1}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_empty_relevant(self):
        """No relevant documents should return 0.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_zero_relevance_grade_not_counted(self):
        """Relevance grade 0 should not be counted as relevant."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1": 1, "doc2": 0}  # Only doc1 is relevant
        assert recall_at_k(retrieved, relevant, 2) == 1.0

    def test_recall_increases_with_k(self):
        """Recall should increase or stay same as k increases."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1": 1, "doc3": 1}
        r1 = recall_at_k(retrieved, relevant, 1)
        r2 = recall_at_k(retrieved, relevant, 2)
        r3 = recall_at_k(retrieved, relevant, 3)
        assert r1 <= r2 <= r3


class TestDCGAtK:
    """Tests for dcg_at_k function."""

    def test_dcg_single_relevant(self):
        """DCG with single relevant document at position 1."""
        retrieved = ["doc1"]
        relevant = {"doc1": 1}
        # DCG = (2^1 - 1) / log2(2) = 1 / 1 = 1.0
        assert dcg_at_k(retrieved, relevant, 1) == pytest.approx(1.0)

    def test_dcg_graded_relevance(self):
        """DCG with graded relevance scores."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1": 2, "doc2": 1}
        # DCG = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3) = 3/1 + 1/1.585 = 3.631
        expected = (2**2 - 1) / np.log2(2) + (2**1 - 1) / np.log2(3)
        assert dcg_at_k(retrieved, relevant, 2) == pytest.approx(expected)

    def test_dcg_no_relevant(self):
        """DCG with no relevant documents returns 0."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc3": 1}
        assert dcg_at_k(retrieved, relevant, 2) == pytest.approx(0.0)


class TestNDCGAtK:
    """Tests for ndcg_at_k function."""

    def test_perfect_ranking(self):
        """NDCG should be 1.0 for perfect ranking."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1": 2, "doc2": 1}
        assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """NDCG for reversed optimal order."""
        retrieved = ["doc2", "doc1"]
        relevant = {"doc1": 2, "doc2": 1}
        # Not 0 because there are still relevant docs, just not optimally ordered
        result = ndcg_at_k(retrieved, relevant, 2)
        assert 0 < result < 1

    def test_no_relevant_documents(self):
        """NDCG with no relevant returns 0.0."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc3": 1}
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0

    def test_ndcg_bounded_zero_to_one(self):
        """NDCG should always be between 0 and 1."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"c": 2, "a": 1}
        result = ndcg_at_k(retrieved, relevant, 5)
        assert 0 <= result <= 1


class TestAveragePrecision:
    """Tests for average_precision function."""

    def test_perfect_average_precision(self):
        """AP = 1.0 when all relevant docs are retrieved first."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1": 1, "doc2": 1, "doc3": 1}
        # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert average_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_single_relevant_at_start(self):
        """Single relevant doc at position 1."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1": 1}
        # AP = 1/1 / 1 = 1.0
        assert average_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_single_relevant_at_end(self):
        """Single relevant doc at last position."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3": 1}
        # AP = (1/3) / 1 = 0.333
        assert average_precision(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_documents(self):
        """AP = 0 when no relevant documents."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {}
        assert average_precision(retrieved, relevant) == 0.0

    def test_relevant_but_not_retrieved(self):
        """Relevant document not in retrieved list."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc3": 1}
        assert average_precision(retrieved, relevant) == 0.0

    def test_interleaved_relevant(self):
        """Relevant docs at positions 1 and 3."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1": 1, "doc3": 1}
        # AP = ((1/1) + (2/3)) / 2 = (1 + 0.667) / 2 = 0.833
        expected = (1 / 1 + 2 / 3) / 2
        assert average_precision(retrieved, relevant) == pytest.approx(expected)


class TestWordErrorRate:
    """Tests for word_error_rate function."""

    def test_identical_strings(self):
        """WER should be 0 for identical strings."""
        ref = "hello world"
        hyp = "hello world"
        assert word_error_rate(ref, hyp) == 0.0

    def test_one_word_different(self):
        """WER with one word substitution."""
        ref = "hello world"
        hyp = "hello there"
        # 1 substitution / 2 words = 0.5
        assert word_error_rate(ref, hyp) == 0.5

    def test_case_insensitive(self):
        """WER should be case insensitive."""
        ref = "Hello World"
        hyp = "hello world"
        assert word_error_rate(ref, hyp) == 0.0

    def test_punctuation_ignored(self):
        """WER should ignore punctuation."""
        ref = "Hello, world!"
        hyp = "hello world"
        assert word_error_rate(ref, hyp) == 0.0

    def test_extra_spaces_ignored(self):
        """WER should ignore multiple spaces."""
        ref = "hello   world"
        hyp = "hello world"
        assert word_error_rate(ref, hyp) == 0.0


class TestCharacterErrorRate:
    """Tests for character_error_rate function."""

    def test_identical_strings(self):
        """CER should be 0 for identical strings."""
        ref = "hello"
        hyp = "hello"
        assert character_error_rate(ref, hyp) == 0.0

    def test_one_char_different(self):
        """CER with one character substitution."""
        ref = "hello"
        hyp = "hallo"
        # 1 substitution / 5 chars = 0.2
        assert character_error_rate(ref, hyp) == pytest.approx(0.2)

    def test_case_insensitive(self):
        """CER should be case insensitive."""
        ref = "Hello"
        hyp = "hello"
        assert character_error_rate(ref, hyp) == 0.0


class TestEvaluationModeDetection:
    """Tests for evaluation mode detection logic."""

    def test_audio_text_retrieval_mode(self):
        """Detects audio_text_retrieval mode when both pipelines present."""
        from evaluator.evaluation import evaluate_with_pipeline

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_dataset.configure_mock(**{'__iter__': MagicMock(return_value=iter([]))})

        mock_audio_emb = MagicMock()
        mock_text_emb = MagicMock()
        mock_retrieval = MagicMock()

        with patch("evaluator.evaluation.sample_wise.DataLoader") as mock_loader:
            mock_loader.return_value = iter([])
            with patch("evaluator.evaluation.sample_wise.logger"):
                result = evaluate_with_pipeline(
                    dataset=mock_dataset,
                    retrieval_pipeline=mock_retrieval,
                    audio_embedding_pipeline=mock_audio_emb,
                    text_embedding_pipeline=mock_text_emb,
                    k=5,
                )
        assert result["pipeline_mode"] == "audio_text_retrieval"

    def test_audio_emb_retrieval_mode(self):
        """Detects audio_emb_retrieval mode with only audio pipeline."""
        from evaluator.evaluation import evaluate_with_pipeline

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)

        mock_audio_emb = MagicMock()
        mock_retrieval = MagicMock()

        with patch("evaluator.evaluation.sample_wise.DataLoader") as mock_loader:
            mock_loader.return_value = iter([])
            with patch("evaluator.evaluation.sample_wise.logger"):
                result = evaluate_with_pipeline(
                    dataset=mock_dataset,
                    retrieval_pipeline=mock_retrieval,
                    audio_embedding_pipeline=mock_audio_emb,
                    k=5,
                )
        assert result["pipeline_mode"] == "audio_emb_retrieval"

    def test_asr_text_retrieval_mode(self):
        """Detects asr_text_retrieval mode with ASR and text embedding."""
        from evaluator.evaluation import evaluate_with_pipeline

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)

        mock_asr = MagicMock()
        mock_text_emb = MagicMock()
        mock_retrieval = MagicMock()

        with patch("evaluator.evaluation.sample_wise.DataLoader") as mock_loader:
            mock_loader.return_value = iter([])
            with patch("evaluator.evaluation.sample_wise.logger"):
                result = evaluate_with_pipeline(
                    dataset=mock_dataset,
                    retrieval_pipeline=mock_retrieval,
                    asr_pipeline=mock_asr,
                    text_embedding_pipeline=mock_text_emb,
                    k=5,
                )
        assert result["pipeline_mode"] == "asr_text_retrieval"

    def test_asr_only_mode(self):
        """Detects asr_only mode with ASR but no retrieval."""
        from evaluator.evaluation import evaluate_with_pipeline

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)

        mock_asr = MagicMock()
        mock_text_emb = MagicMock()

        with patch("evaluator.evaluation.sample_wise.DataLoader") as mock_loader:
            mock_loader.return_value = iter([])
            with patch("evaluator.evaluation.sample_wise.logger"):
                result = evaluate_with_pipeline(
                    dataset=mock_dataset,
                    asr_pipeline=mock_asr,
                    text_embedding_pipeline=mock_text_emb,
                    k=5,
                )
        assert result["pipeline_mode"] == "asr_only"

    def test_invalid_mode_raises_error(self):
        """Raises ValueError when no pipelines provided."""
        from evaluator.evaluation import evaluate_with_pipeline

        mock_dataset = MagicMock()

        with pytest.raises(ValueError, match="Must provide either"):
            evaluate_with_pipeline(dataset=mock_dataset, k=5)


class TestResultAggregation:
    """Tests for result aggregation in evaluation."""

    def test_mrr_computation(self):
        """Test that MRR is computed correctly."""
        # Simulate retrieved and relevant lists
        all_retrieved = [
            ["doc1", "doc2", "doc3"],  # RR = 1.0 (first is relevant)
            ["doc2", "doc1", "doc3"],  # RR = 0.5 (second is relevant)
        ]
        all_relevant = [{"doc1": 1}, {"doc1": 1}]

        rr_scores = [
            reciprocal_rank(ret, rel)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        mrr = sum(rr_scores) / len(rr_scores)
        assert mrr == pytest.approx(0.75)

    def test_map_computation(self):
        """Test that MAP is computed correctly."""
        all_retrieved = [
            ["doc1", "doc2"],  # AP = 1.0
            ["doc2", "doc1"],  # AP = (1/2) / 1 = 0.5
        ]
        all_relevant = [{"doc1": 1}, {"doc1": 1}]

        ap_scores = [
            average_precision(ret, rel)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        map_score = sum(ap_scores) / len(ap_scores)
        assert map_score == pytest.approx(0.75)

    def test_recall_at_k_aggregation(self):
        """Test recall@k aggregation across queries."""
        all_retrieved = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
        ]
        all_relevant = [
            {"doc1": 1, "doc2": 1},  # R@3 = 1.0
            {"doc4": 1, "doc7": 1},  # R@3 = 0.5
        ]

        recall_scores = [
            recall_at_k(ret, rel, 3)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        avg_recall = sum(recall_scores) / len(recall_scores)
        assert avg_recall == pytest.approx(0.75)


class TestBuildRelevantFromItem:
    """Tests for _build_relevant_from_item helper."""

    def test_with_relevance_grades(self):
        """Uses relevance_grades when available."""
        from evaluator.evaluation import _build_relevant_from_item

        item = {"relevance_grades": {"doc1": 2, "doc2": 1}, "transcription": "text"}
        result = _build_relevant_from_item(item)
        assert result == {"doc1": 2, "doc2": 1}

    def test_with_groundtruth_doc_ids(self):
        """Uses groundtruth_doc_ids when relevance_grades not available."""
        from evaluator.evaluation import _build_relevant_from_item

        item = {"groundtruth_doc_ids": ["doc1", "doc2"], "transcription": "text"}
        result = _build_relevant_from_item(item)
        assert result == {"doc1": 1, "doc2": 1}

    def test_fallback_to_transcription(self):
        """Falls back to transcription when no other relevance info."""
        from evaluator.evaluation import _build_relevant_from_item

        item = {"transcription": "the query text"}
        result = _build_relevant_from_item(item)
        assert result == {"the query text": 1}

    def test_empty_relevance_grades_uses_fallback(self):
        """Empty relevance_grades triggers fallback."""
        from evaluator.evaluation import _build_relevant_from_item

        item = {
            "relevance_grades": {},
            "groundtruth_doc_ids": ["doc1"],
            "transcription": "text",
        }
        result = _build_relevant_from_item(item)
        assert result == {"doc1": 1}


class TestPayloadToKey:
    """Tests for _payload_to_key helper."""

    def test_dict_with_doc_id(self):
        """Extracts doc_id from dict payload."""
        from evaluator.evaluation import _payload_to_key

        payload = {"doc_id": "123", "text": "some text"}
        assert _payload_to_key(payload) == "123"

    def test_dict_with_text_only(self):
        """Extracts text when doc_id not present."""
        from evaluator.evaluation import _payload_to_key

        payload = {"text": "some text"}
        assert _payload_to_key(payload) == "some text"

    def test_string_payload(self):
        """Handles string payload directly."""
        from evaluator.evaluation import _payload_to_key

        payload = "just a string"
        assert _payload_to_key(payload) == "just a string"

    def test_numeric_payload(self):
        """Handles numeric payload by converting to string."""
        from evaluator.evaluation import _payload_to_key

        payload = 42
        assert _payload_to_key(payload) == "42"

    def test_search_results_to_keys_handles_mixed_entry_shapes(self):
        """Converts tuple and dataclass retrieval outputs through one adapter."""
        from evaluator.evaluation import _search_results_to_keys
        from evaluator.models.retrieval.contracts import ScoredRetrievalResult

        results = [
            ({"doc_id": "doc-a", "text": "A"}, 0.9),
            ScoredRetrievalResult(payload={"text": "fallback text"}, score=0.8),
            ("raw-string", 0.7),
        ]
        assert _search_results_to_keys(results) == ["doc-a", "fallback text", "raw-string"]


class TestTTSIntegration:
    """Integration tests for TTS-enabled evaluation."""
    
    def test_tts_config_creation(self):
        """Test that TTS configuration can be created."""
        from evaluator.config import AudioSynthesisConfig
        
        # Create TTS config
        config = AudioSynthesisConfig(
            provider="piper",
            voice="en_US-lessac-medium",
            cache_dir="/tmp/tts_cache"
        )
        
        # Verify config is properly set
        assert config.provider == "piper"
        assert config.voice == "en_US-lessac-medium"
        assert config.cache_dir == "/tmp/tts_cache"
    
    def test_tts_cache_is_used_across_queries(self):
        """Test that TTS cache prevents redundant synthesis of repeated texts."""
        from evaluator.pipeline.audio.synthesis import AudioSynthesizer
        from evaluator.config import AudioSynthesisConfig
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AudioSynthesisConfig(
                provider="piper",
                voice="en_US-lessac-medium",
                cache_dir=f"{temp_dir}/cache"
            )
            
            with patch('evaluator.models.tts.piper_tts.PiperTTS') as mock_provider_class:
                mock_provider = MagicMock()
                mock_provider.synthesize.return_value = np.random.randn(16000).astype(np.float32)
                mock_provider_class.return_value = mock_provider
                
                synth = AudioSynthesizer(config)
                
                # Synthesize same text twice
                text = "The patient presents with acute symptoms."
                audio1 = synth.synthesize(text)
                audio2 = synth.synthesize(text)
                
                # Provider should only be called once (second uses cache)
                assert mock_provider.synthesize.call_count == 1
                
                # Audio should be identical
                np.testing.assert_array_equal(audio1, audio2)
    
    def test_tts_handles_medical_terminology(self):
        """Test that TTS can handle complex medical terminology."""
        from evaluator.pipeline.audio.synthesis import AudioSynthesizer
        from evaluator.config import AudioSynthesisConfig
        import tempfile
        
        medical_texts = [
            "Hypertension management requires pharmacological intervention.",
            "Acute myocardial infarction presents with chest pain.",
            "Differential diagnosis includes pneumonia and bronchitis."
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AudioSynthesisConfig(
                provider="piper",
                voice="en_US-lessac-medium",
                cache_dir=f"{temp_dir}/cache"
            )
            
            with patch('evaluator.models.tts.piper_tts.PiperTTS') as mock_provider_class:
                mock_provider = MagicMock()
                mock_provider.synthesize.return_value = np.random.randn(22050).astype(np.float32)
                mock_provider_class.return_value = mock_provider
                
                synth = AudioSynthesizer(config)
                
                # Synthesize all medical texts
                for text in medical_texts:
                    audio = synth.synthesize(text)
                    assert isinstance(audio, np.ndarray)
                    assert audio.dtype == np.float32
                    assert len(audio) > 0
                
                # All texts should have been synthesized
                assert mock_provider.synthesize.call_count == len(medical_texts)
