"""Publication export of a built StageGraph: Graphviz DOT.

``evaluator graph --config x.yaml --format dot -o fig.dot`` then
``dot -Tpdf fig.dot -o fig.pdf`` (or -Tsvg) gives a vector pipeline figure for a
paper/thesis. Colors follow the taxonomy ``category`` axis (same grouping the web DAG
uses); data-flow edges are labeled with the artifact they carry, ordering-only
dependencies are dashed and unlabeled.
"""

from typing import Optional

from .display import display_label
from .registry import StageGraph, _resolve, get_stage_node_def

# Print-friendly pastel fills per taxonomy category (dark text stays readable).
_CATEGORY_FILL = {
    "source": "#dbeafe",     # blue
    "model": "#fee2e2",      # red
    "transform": "#f1f5f9",  # slate
    "metric": "#dcfce7",     # green
    "sink": "#fef9c3",       # yellow
}


def _node_category(node) -> str:
    try:
        return str(_resolve(get_stage_node_def(node.stage).category, node.params))
    except Exception:  # noqa: BLE001 - unregistered stage in a hand-built graph
        return "transform"


def _quote(s: str) -> str:
    return '"' + str(s).replace('"', r"\"") + '"'


def graph_to_dot(graph: StageGraph, *, title: Optional[str] = None) -> str:
    """The graph as Graphviz DOT (left→right, category-colored, artifact-labeled edges)."""
    lines = [
        "digraph evaluator_dag {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9, color="#475569"];',
    ]
    if title:
        lines += [f"  label={_quote(title)};", "  labelloc=t;", "  fontsize=14;"]

    for node in graph.nodes:
        fill = _CATEGORY_FILL.get(_node_category(node), "#f1f5f9")
        label = display_label(node.stage, node.params)
        if node.id != node.stage:
            label = f"{label}\\n({node.id})"
        lines.append(
            f"  {_quote(node.id)} [label={_quote(label)}, fillcolor={_quote(fill)}];"
        )

    # Data-flow edges from the DECLARED bindings (artifact, producer) — the same wiring
    # the run reads. Ordering-only deps (depends_on with no binding) are dashed.
    for node in graph.nodes:
        bound_producers = set()
        seen = set()
        for artifact, producer in node.bindings:
            bound_producers.add(producer)
            key = (producer, artifact)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"  {_quote(producer)} -> {_quote(node.id)} [label={_quote(artifact)}];"
            )
        for dep in node.depends_on:
            if dep not in bound_producers:
                lines.append(f"  {_quote(dep)} -> {_quote(node.id)} [style=dashed];")

    lines.append("}")
    return "\n".join(lines) + "\n"
