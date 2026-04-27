"""Reusable configuration templates for common evaluation scenarios."""

from .evaluation import EvaluationConfig


class ConfigTemplates:
    """Factory helpers returning preconfigured EvaluationConfig instances."""

    @staticmethod
    def minimal() -> EvaluationConfig:
        """Minimal setup for first run or smoke test."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.experiment_name = "minimal_eval"
        config.model.pipeline_mode = "asr_text_retrieval"
        config.model.asr_model_type = "whisper"
        config.model.text_emb_model_type = "labse"
        config.data.batch_size = 16
        config.vector_db.k = 5
        return config

    @staticmethod
    def custom_dataset(
        questions_path: str,
        corpus_path: str,
    ) -> EvaluationConfig:
        """Template for evaluating user-provided dataset files."""
        config = ConfigTemplates.minimal()
        config.experiment_name = "custom_dataset_eval"
        config.data.questions_path = questions_path
        config.data.corpus_path = corpus_path
        return config

    @staticmethod
    def audio_embedding_only() -> EvaluationConfig:
        """Template for direct audio-embedding retrieval path."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.experiment_name = "audio_embedding_eval"
        config.model.pipeline_mode = "audio_emb_retrieval"
        config.model.audio_emb_model_type = "attention_pool"
        config.model.text_emb_model_type = "labse"
        config.model.audio_emb_dim = 768
        config.data.batch_size = 16
        config.vector_db.k = 5
        return config

