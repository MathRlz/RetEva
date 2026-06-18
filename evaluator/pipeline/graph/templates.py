"""Named graph *templates* — the user-facing starting points the web UI / CLI expand into an
explicit DAG.

These are the former pipeline *modes*, relocated to a config-creation concern: the runtime no
longer knows them (it just runs the explicit graph a config carries); a template is only a
convenient skeleton the builder seeds the canvas with, then the user fills in models/params via the
registry-driven forms. ``template_graph_spec`` emits the self-contained ``{nodes, edges}`` block a
config's ``graph:`` accepts, so "pick a template" and "hand-wire a DAG" produce the same shape.
"""

from typing import Any, Dict, List

from .modes import build_stage_graph
from .operators import node_kind
from .registry import StageGraph

# template name -> human label (the former pipeline-mode labels, owned here now).
GRAPH_TEMPLATES: Dict[str, str] = {
    "asr_text_retrieval": "ASR + Text Retrieval",
    "audio_emb_retrieval": "Audio Embedding Retrieval",
    "audio_text_retrieval": "Audio-to-Text Retrieval",
    "asr_only": "ASR Only",
}


def list_graph_templates() -> List[Dict[str, str]]:
    """The available templates as ``[{name, label}]`` for a UI picker."""
    return [{"name": name, "label": label} for name, label in GRAPH_TEMPLATES.items()]


def graph_template(name: str, **features: Any) -> StageGraph:
    """The wired DAG for a template ``name`` (+ optional feature toggles). Thin wrapper over
    :func:`build_stage_graph`; ``audio_text_retrieval`` defaults to embedding fusion (its
    distinguishing feature) unless overridden."""
    if name == "audio_text_retrieval":
        features.setdefault("embedding_fusion_enabled", True)
    return build_stage_graph(name, **features)


def template_graph_spec(name: str, **features: Any) -> Dict[str, Any]:
    """A template as an explicit, embeddable ``{nodes, edges}`` block — node *kinds* (the
    structural skeleton, no models attached) + the dependency edges. Drop it under a config's
    ``graph:`` and the loader round-trips it; the user fills models via the node forms."""
    g = graph_template(name, **features)
    nodes = [{"id": n.id, "type": node_kind(n.stage, n.params)} for n in g.nodes]
    edges = [{"from": dep, "to": n.id} for n in g.nodes for dep in n.depends_on]
    return {"nodes": nodes, "edges": edges}
