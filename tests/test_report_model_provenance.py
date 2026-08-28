"""A per-node model override (`params.model`/`params.name`) only lives on
``s.asr_pipeline`` / ``s.text_embedding_pipeline`` for the duration of that node's own
handler — ``_node_pipeline`` reverts it to the shared/global pipeline on exit (see
``executor/node_pipeline.py``). The report/provenance assembler (``metrics_provenance.py``)
runs once per branch, after that branch's asr/text_embedding/retrieval nodes, so reading
those attributes directly there always saw whichever pipeline was last restored — the same
value for every branch, regardless of which model that branch actually used.

These tests pin the fix: `_node_pipeline` publishes the resolved per-branch identity as a
node artifact while its override is live, `retrieval` forwards the embedder's identity
onto its own bus entry, and `_record_model_info` / `_build_provenance` recover it via
`sibling_artifact` — falling back to the old (correct, never-swapped) pipeline read when a
branch used the shared default and never published an override.
"""

from evaluator.evaluation.handlers.metrics_provenance import (
    _build_provenance,
    _record_model_info,
)
from evaluator.pipeline.graph.registry import StageNode

from tests.graph_test_helpers import make_state


class _FakeModel:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _FakePipeline:
    def __init__(self, name):
        self.model = _FakeModel(name)


class _FakeModelConfig:
    """Stands in for `config.model`: the run's flat global defaults (what every branch
    without an override actually uses, and what the old buggy code always reported)."""

    asr_model_type = "wav2vec2"
    asr_size = None
    asr_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
    asr_adapter_path = None
    asr_params = None
    text_emb_model_type = "labse"
    text_emb_size = None
    text_emb_model_name = "sentence-transformers/LaBSE"
    text_emb_adapter_path = None
    text_emb_model_path = None
    text_emb_embedding_space = None
    text_emb_params = None


class _FakeConfig:
    model = _FakeModelConfig()
    vector_db = None


def _branch_state():
    default_asr = _FakePipeline("Wav2Vec2Model - wav2vec2-large-xlsr-53-polish")
    default_text_emb = _FakePipeline("LaBseModel - LaBSE")
    return make_state(
        asr_pipeline=default_asr,
        text_embedding_pipeline=default_text_emb,
        retrieval_pipeline=object(),  # only needs to be non-None (is_asr_text_retrieval)
        config=_FakeConfig(),
    ), default_asr, default_text_emb


def test_asr_report_field_uses_the_branch_override_not_the_reverted_default():
    s, default_asr, _ = _branch_state()

    # asr_a's node ran with a whisper override; _node_pipeline published the resolved
    # identity while it was live, then (like the real revert) put the default back.
    s.current_node = StageNode(id="asr_a", stage="asr", bindings=())
    s.put_artifact("query_text", ["hyp a"])
    s.put_artifact(
        "asr_model_provenance",
        {"type": "whisper", "name": "openai/whisper-large-v3", "resolved": "Whisper - large-v3"},
    )
    s.asr_pipeline = default_asr  # the transient swap already reverted by report time

    # asr_b's node overrode to a different model.
    s.current_node = StageNode(id="asr_b", stage="asr", bindings=())
    s.put_artifact("query_text", ["hyp b"])
    s.put_artifact(
        "asr_model_provenance",
        {"type": "m4t", "name": "facebook/seamless-m4t-v2-large", "resolved": "SeamlessM4T"},
    )
    s.asr_pipeline = default_asr

    # asr_c's node used the shared default (no override → nothing published).
    s.current_node = StageNode(id="asr_c", stage="asr", bindings=())
    s.put_artifact("query_text", ["hyp c"])

    for asr_node, expected in (("asr_a", "Whisper - large-v3"), ("asr_b", "SeamlessM4T")):
        s.current_node = StageNode(
            id=f"metrics_{asr_node}", stage="measure", params={"family": "report"},
            bindings=(("query_text", asr_node),),
        )
        results = {}
        _record_model_info(results, s)
        assert results["asr"] == expected, asr_node

        prov = _build_provenance(s)
        assert prov["asr"]["resolved"] == expected, asr_node

    # No override on this branch: falls back to the shared pipeline, still correct.
    s.current_node = StageNode(
        id="metrics_asr_c", stage="measure", params={"family": "report"},
        bindings=(("query_text", "asr_c"),),
    )
    results = {}
    _record_model_info(results, s)
    assert results["asr"] == "Wav2Vec2Model - wav2vec2-large-xlsr-53-polish"
    assert _build_provenance(s)["asr"]["resolved"] == (
        "Wav2Vec2Model - wav2vec2-large-xlsr-53-polish"
    )


def test_embedder_report_field_uses_the_branch_override_via_retrieval_forwarding():
    s, _, default_text_emb = _branch_state()

    # text_embedding_a overrode to jina; retrieval_a is the node that actually reads its
    # query vectors (recorded via s.input, exactly like the real retrieval handler).
    s.current_node = StageNode(id="text_embedding_a", stage="embed", bindings=())
    s.put_artifact("text_query_vectors", [[0.1, 0.2]])
    s.put_artifact(
        "text_embedding_model_provenance",
        {"type": "jina", "resolved": "JinaModel - jina-embeddings-v4"},
    )

    s.current_node = StageNode(
        id="retrieval_a", stage="search",
        bindings=(("text_query_vectors", "text_embedding_a"),),
        input_aliases=(("query_vectors", ("text_query_vectors",)),),
    )
    s.input("query_vectors")  # records data_flow, as the real retrieval handler does
    s.put_artifact("retrieved", [[("doc1", 0.9)]])
    # Forwarding step the fix adds to retrieval.py: republish the resolved producer's
    # embedder identity on retrieval's own bus entry.
    s.put_artifact(
        "query_text_embedder_model_provenance",
        s.ctx.get("text_embedding_a", "text_embedding_model_provenance"),
    )

    # A second branch used the shared default embedder (no override).
    s.current_node = StageNode(id="text_embedding_b", stage="embed", bindings=())
    s.put_artifact("text_query_vectors", [[0.3, 0.4]])

    s.current_node = StageNode(
        id="retrieval_b", stage="search",
        bindings=(("text_query_vectors", "text_embedding_b"),),
        input_aliases=(("query_vectors", ("text_query_vectors",)),),
    )
    s.input("query_vectors")
    s.put_artifact("retrieved", [[("doc2", 0.8)]])

    s.current_node = StageNode(
        id="metrics_a", stage="measure", params={"family": "report"},
        bindings=(("retrieved", "retrieval_a"),),
    )
    results = {}
    _record_model_info(results, s)
    assert results["embedder"] == "JinaModel - jina-embeddings-v4"
    assert _build_provenance(s)["text_emb"]["resolved"] == "JinaModel - jina-embeddings-v4"

    s.current_node = StageNode(
        id="metrics_b", stage="measure", params={"family": "report"},
        bindings=(("retrieved", "retrieval_b"),),
    )
    results = {}
    _record_model_info(results, s)
    assert results["embedder"] == default_text_emb.model.name()
    assert _build_provenance(s)["text_emb"]["resolved"] == default_text_emb.model.name()
