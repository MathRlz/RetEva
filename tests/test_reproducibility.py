"""Roadmap 1 — reproducibility surfacing: content fingerprint (R1a), HF revision pin (R1b),
resolved-config sidecar (R1c)."""

import sys
import types

from evaluator.evaluation.provenance import build_provenance, dataset_content_fingerprint


class _DS:
    def __len__(self):
        return 3

    def get_corpus(self):
        return [{"doc_id": "d1", "text": "hello"}, {"doc_id": "d2", "text": "world"}]


def test_fingerprint_counts_and_is_deterministic():
    fp = dataset_content_fingerprint(_DS())
    assert fp["corpus_docs"] == 2 and fp["questions"] == 3
    assert "corpus_sha256" in fp
    assert dataset_content_fingerprint(_DS()) == fp  # deterministic


def test_fingerprint_is_content_sensitive():
    class _Changed(_DS):
        def get_corpus(self):
            return [{"doc_id": "d1", "text": "CHANGED"}, {"doc_id": "d2", "text": "world"}]

    assert (
        dataset_content_fingerprint(_Changed())["corpus_sha256"]
        != dataset_content_fingerprint(_DS())["corpus_sha256"]
    )


def test_fingerprint_lands_in_provenance():
    prov = build_provenance(None, dataset=dataset_content_fingerprint(_DS()))
    assert prov["dataset"]["corpus_docs"] == 2
    # absent when no dataset supplied
    assert "dataset" not in build_provenance(None)


def test_offload_block_omitted_when_empty():
    # 2c: the soft-offload block must stay absent unless a model was parked warm — so the
    # default (full-free) policy keeps the provenance byte-identical to before.
    assert "offload" not in build_provenance(None)
    assert "offload" not in build_provenance(None, offload=None)
    prov = build_provenance(None, offload={"soft_offloads": 3, "evictions": 1})
    assert prov["offload"] == {"soft_offloads": 3, "evictions": 1}


def test_hf_loader_parses_repo_at_revision(monkeypatch):
    from evaluator.datasets.loaders.huggingface import HuggingFaceDatasetLoader

    captured = {}

    def fake_load_dataset(**kwargs):
        captured.update(kwargs)
        return {"train": []}  # minimal

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    loader = HuggingFaceDatasetLoader(dataset_name="org/ds@abc123", split="train")
    loader._load_hf_dataset()
    assert captured["path"] == "org/ds"  # @revision stripped from the path
    assert captured["revision"] == "abc123"


def test_resolved_config_sidecar_written(tmp_path):
    # Graph-first Phase 5: the resolved-config sidecar is the executed DAG as node-centric YAML
    # (no pipeline_mode), not a flat JSON.
    import yaml

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from evaluator.evaluation.handlers.sinks import _write_resolved_config
    from tests.graph_test_helpers import explicit_graph

    cfg = EvaluationConfig.from_dict(
        build_evaluation_config_kwargs({
            "experiment": {"name": "t"},
            "dataset": {"id": "pubmed_qa",
                        "questions": "examples/data/pubmed_qa_small/questions.json",
                        "corpus": "examples/data/pubmed_qa_small/corpus.json"},
            "graph": explicit_graph([
                {"id": "dataset_source", "type": "dataset_source"},
                {"id": "asr", "type": "asr", "params": {"model": "whisper"}},
                {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
                {"id": "corpus_embedding", "type": "corpus_embedding"},
                {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
                {"id": "retrieval", "type": "retrieval", "params": {"k": 5}},
            ]),
        }),
        validate=False,
    )
    state = types.SimpleNamespace(config=cfg)
    _write_resolved_config(state, str(tmp_path), "myrun")

    out = tmp_path / "config_resolved_myrun.yaml"
    assert out.exists()
    doc = yaml.safe_load(out.read_text())
    kinds = {n["type"] for n in doc["graph"]["nodes"]}
    assert {"asr", "text_embedding", "retrieval"} <= kinds   # the executed DAG, not a flat dict
    assert "mode" not in doc["graph"]                        # graph-first: no pipeline_mode
    assert doc["experiment"]["name"] == "t"
