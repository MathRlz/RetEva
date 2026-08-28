"""B5 pins (CRITIQUE.md): `evaluator graph --emit-edges --write` rewrites in place.

Text-level rewrite: comments survive, stale port-level entries are replaced, hand-added
ordering-only ``{from, to}`` edges are kept, and a config without an ``edges:`` block gains
one right after ``graph.nodes``.
"""

import yaml

from evaluator.cli.main import write_edges_block

# Same-name edges are written in the shorthand form (no `output:`) — see
# tests/test_edge_shorthand.py for the rename spelling.
NEW_EDGES = [
    {"from": "dataset_source", "to": "asr", "input": "query_audio"},
]

WITH_EDGES = """\
# top comment
experiment: {name: t}
graph:
  # the spine
  nodes:
  - dataset_source
  - asr        # transcribe
  edges:
  - {from: dataset_source, output: stale_artifact, to: asr, input: stale_artifact}
  - {from: metrics, to: answer_gen}   # ordering-only, hand-added
nodes:
  asr: {model: whisper}
"""


def test_replace_keeps_comments_and_ordering_edges(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(WITH_EDGES)
    assert write_edges_block(str(cfg), NEW_EDGES) == 1
    text = cfg.read_text()
    assert "# top comment" in text and "# the spine" in text and "# transcribe" in text
    assert "stale_artifact" not in text
    loaded = yaml.safe_load(text)
    assert loaded["graph"]["edges"] == [
        {"from": "dataset_source", "to": "asr", "input": "query_audio"},
        {"from": "metrics", "to": "answer_gen"},
    ]
    assert loaded["nodes"] == {"asr": {"model": "whisper"}}  # top-level block untouched


def test_insert_after_nodes_when_no_edges_block(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(WITH_EDGES.replace(
        "  edges:\n"
        "  - {from: dataset_source, output: stale_artifact, to: asr, input: stale_artifact}\n"
        "  - {from: metrics, to: answer_gen}   # ordering-only, hand-added\n",
        "",
    ))
    write_edges_block(str(cfg), NEW_EDGES)
    loaded = yaml.safe_load(cfg.read_text())
    assert loaded["graph"]["edges"] == NEW_EDGES
    assert loaded["graph"]["nodes"] == ["dataset_source", "asr"]
