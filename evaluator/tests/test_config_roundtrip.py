"""Tests for config serialization round-trip consistency."""
import unittest
from dataclasses import fields

from evaluator.config import EvaluationConfig


class TestConfigRoundTrip(unittest.TestCase):
    """Verify from_dict → to_dict → from_dict produces identical configs."""

    def test_default_config_round_trip(self):
        """Default config survives serialize → deserialize."""
        original = EvaluationConfig()
        d = original.to_dict(include_config=True)
        restored = EvaluationConfig.from_dict(d)
        d2 = restored.to_dict(include_config=True)
        self.assertEqual(d, d2)

    def test_preset_round_trip(self):
        """Preset configs survive round-trip."""
        for preset in ("fast_dev", "whisper_labse", "wav2vec_jina", "audio_only"):
            with self.subTest(preset=preset):
                original = EvaluationConfig.from_preset(preset, validate=False)
                d = original.to_dict(include_config=True)
                restored = EvaluationConfig.from_dict(d, validate=False)
                d2 = restored.to_dict(include_config=True)
                self.assertEqual(d, d2, f"Round-trip failed for preset '{preset}'")

    def test_nested_config_round_trip(self):
        """Nested dict format survives round-trip via _nested_config."""
        from evaluator.webapi.app import _nested_config
        original = EvaluationConfig()
        nested = _nested_config(original)
        restored = EvaluationConfig.from_dict(nested)
        nested2 = _nested_config(restored)
        self.assertEqual(nested, nested2)

    def test_modified_config_round_trip(self):
        """Config with non-default values survives round-trip."""
        config = EvaluationConfig()
        config.experiment_name = "roundtrip_test"
        config.model.asr_model_type = "whisper"
        config.model.asr_size = "large-v3"
        config.model.pipeline_mode = "asr_only"
        config.data.batch_size = 64
        config.data.trace_limit = 100
        config.vector_db.k = 20
        config.vector_db.retrieval_mode = "hybrid"
        config.cache.enabled = False
        config.checkpoint_enabled = False

        d = config.to_dict(include_config=True)
        restored = EvaluationConfig.from_dict(d)

        self.assertEqual(restored.experiment_name, "roundtrip_test")
        self.assertEqual(restored.model.asr_model_type, "whisper")
        self.assertEqual(restored.model.asr_size, "large-v3")
        self.assertEqual(str(restored.model.pipeline_mode), "asr_only")
        self.assertEqual(restored.data.batch_size, 64)
        self.assertEqual(restored.vector_db.k, 20)
        self.assertEqual(restored.vector_db.retrieval_mode, "hybrid")
        self.assertFalse(restored.cache.enabled)
        self.assertFalse(restored.checkpoint_enabled)


class TestConfigFieldCoverage(unittest.TestCase):
    """Ensure auto-serializer covers all EvaluationConfig fields."""

    def test_all_fields_classified(self):
        """Every EvaluationConfig field must be tracked by the serializer."""
        classified = (
            EvaluationConfig._RUNTIME_FIELDS
            | EvaluationConfig._RUNTIME_SCALARS
            | EvaluationConfig._EXPERIMENT_SUBCONFIGS
            | {"experiment_name", "output_dir"}  # top-level experiment scalars
        )
        all_field_names = {f.name for f in fields(EvaluationConfig)}
        internal = {f.name for f in fields(EvaluationConfig)
                    if f.name.startswith("_")}

        for field_name in all_field_names - internal:
            self.assertIn(
                field_name,
                classified,
                f"Field '{field_name}' not classified — "
                f"add it to _RUNTIME_FIELDS, _RUNTIME_SCALARS, or _EXPERIMENT_SUBCONFIGS",
            )

    def test_runtime_dict_has_new_model_fields(self):
        """Verify auto-serializer includes recently added model fields."""
        config = EvaluationConfig()
        config.model.asr_size = "base"
        config.model.asr_params = {"beam_size": 5}
        runtime = config.to_runtime_dict()

        self.assertEqual(runtime["model"]["asr_size"], "base")
        self.assertEqual(runtime["model"]["asr_params"], {"beam_size": 5})
        self.assertIn("text_emb_size", runtime["model"])
        self.assertIn("audio_emb_size", runtime["model"])
        self.assertIn("text_emb_params", runtime["model"])
        self.assertIn("audio_emb_params", runtime["model"])


if __name__ == "__main__":
    unittest.main()
