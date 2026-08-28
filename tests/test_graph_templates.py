"""Mode-removal Phase 4: the 4 former pipeline modes survive as named graph *templates* — a
config-creation skeleton (no runtime role). A template serializes to an embeddable {nodes, edges}.
"""

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs
from evaluator.pipeline.graph.modes import build_graph_for_config
from evaluator.pipeline.graph.templates import (
    GRAPH_TEMPLATES,
    graph_template,
    list_graph_templates,
    template_graph_spec,
)


def test_list_graph_templates():
    names = {t["name"] for t in list_graph_templates()}
    assert names == set(GRAPH_TEMPLATES)
    assert {"asr_text_retrieval", "audio_emb_retrieval", "asr_only"} <= names


def test_graph_template_builds_expected_skeleton():
    g = graph_template("asr_text_retrieval")
    stages = {n.stage for n in g.nodes}
    assert {"asr", "embed", "search"} <= stages   # asr + embed + retrieval


def test_template_graph_spec_is_embeddable_and_round_trips():
    spec = template_graph_spec("asr_text_retrieval")
    kinds = {n["type"] for n in spec["nodes"]}
    assert {"asr", "text_embedding", "retrieval"} <= kinds
    assert spec["edges"]  # carries dependency edges

    # Drop the template spec under a config's graph: → it builds the SAME node set as the
    # template itself (the skeleton round-trips through the loader).
    cfg = EvaluationConfig.from_dict(
        build_evaluation_config_kwargs({
            "dataset": {"id": "pubmed_qa",
                        "questions": "examples/data/pubmed_qa_small/questions.json",
                        "corpus": "examples/data/pubmed_qa_small/corpus.json"},
            "graph": {"nodes": spec["nodes"], "edges": spec["edges"]},
        }),
        validate=False,
    )
    assert cfg.graph_template == "asr_text_retrieval"   # label derived from the seeded nodes
    built = {n.id for n in build_graph_for_config(cfg).nodes}
    template_ids = {n["id"] for n in spec["nodes"]}
    assert built == template_ids


def test_audio_text_template_has_fusion():
    # audio_text_retrieval's distinguishing feature is embedding fusion → the template enables it.
    stages = [n.stage for n in graph_template("audio_text_retrieval").nodes]
    assert "combine" in stages   # the fusion (combine) node
