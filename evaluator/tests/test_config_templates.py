"""Tests for config templates."""

from evaluator.config import ConfigTemplates


class TestConfigTemplates:
    """Template helpers should produce usable baseline configs."""

    def test_minimal_template(self):
        config = ConfigTemplates.minimal()
        assert config.experiment_name == "minimal_eval"
        assert config.model.pipeline_mode == "asr_text_retrieval"
        assert config.model.asr_model_type == "whisper"
        assert config.model.text_emb_model_type == "labse"
        assert config.vector_db.k == 5

    def test_custom_dataset_template(self):
        config = ConfigTemplates.custom_dataset("q.json", "c.json")
        assert config.experiment_name == "custom_dataset_eval"
        assert config.data.questions_path == "q.json"
        assert config.data.corpus_path == "c.json"

    def test_audio_embedding_only_template(self):
        config = ConfigTemplates.audio_embedding_only()
        assert config.model.pipeline_mode == "audio_emb_retrieval"
        assert config.model.audio_emb_model_type == "attention_pool"
        assert config.model.audio_emb_dim == 768
        assert config.vector_db.k == 5

