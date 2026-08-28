"""Phase 1 — save / load builder graphs. The GraphStore (directory of JSON) + the /api/graphs
endpoints: a named canvas spec persists and reopens as a render payload for the canvas.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from evaluator.webapi.app import create_app  # noqa: E402
from evaluator.webapi.graph_store import GraphStore  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]

_SPEC = {
    "mode": "asr_text_retrieval",
    "nodes": [
        {"id": "dataset_source", "type": "source", "params": {
            "dataset": "pubmed_qa",
            "questions": str(_ROOT / "examples/data/pubmed_qa_small/questions.json"),
            "corpus": str(_ROOT / "examples/data/pubmed_qa_small/corpus.json")}},
        {"id": "asr", "type": "convert", "params": {"model": "whisper"}},
        {"id": "text_embedding", "type": "embed", "params": {"model": "labse"}},
        {"id": "corpus_embedding", "type": "embed", "params": {"axis": "corpus"}},
        {"id": "vector_db", "type": "index", "params": {}},
        {"id": "retrieval", "type": "search", "params": {}},
        {"id": "metrics", "type": "measure", "params": {}},
        {"id": "finalize", "type": "sink", "params": {}},
    ],
    "edges": [
        {"from": "dataset_source", "to": "asr"}, {"from": "asr", "to": "text_embedding"},
        {"from": "dataset_source", "to": "corpus_embedding"},
        {"from": "corpus_embedding", "to": "vector_db"},
        {"from": "text_embedding", "to": "retrieval"},
        {"from": "vector_db", "to": "retrieval"},
        {"from": "retrieval", "to": "metrics"}, {"from": "metrics", "to": "finalize"},
    ],
}


# ── store unit ──────────────────────────────────────────────────────────────────


def test_store_save_list_get_delete(tmp_path):
    store = GraphStore(tmp_path / "graphs")
    assert store.list() == []  # missing dir → empty, not an error
    slug = store.save("My Graph!", _SPEC)
    assert slug == "My Graph"  # sanitised (no '!')
    index = store.list()
    assert index == [{"name": "My Graph", "mode": "asr_text_retrieval", "n_nodes": 8}]
    assert store.get("My Graph")["mode"] == "asr_text_retrieval"
    store.delete("My Graph")
    assert store.list() == []


def test_store_missing_raises_keyerror(tmp_path):
    store = GraphStore(tmp_path / "graphs")
    with pytest.raises(KeyError):
        store.get("nope")
    with pytest.raises(KeyError):
        store.delete("nope")


def test_store_rejects_empty_name(tmp_path):
    store = GraphStore(tmp_path / "graphs")
    with pytest.raises(ValueError):
        store.save("!!!", _SPEC)


def test_store_construction_does_not_touch_fs(tmp_path):
    target = tmp_path / "lazy"
    GraphStore(target)
    assert not target.exists()  # created lazily on first save, not at construction


# ── endpoints ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(graph_store=GraphStore(tmp_path / "graphs")))


def test_endpoints_round_trip(client):
    assert client.get("/api/graphs").json() == {"graphs": []}
    saved = client.post("/api/graphs", json={"name": "demo", "spec": _SPEC})
    assert saved.status_code == 200 and saved.json()["name"] == "demo"
    listed = client.get("/api/graphs").json()["graphs"]
    assert listed[0]["name"] == "demo" and listed[0]["n_nodes"] == 8

    got = client.get("/api/graphs/demo").json()
    # the render payload redraws the canvas, with the saved params preserved
    assert got["render"]["mode"] == "asr_text_retrieval"
    asr = next(n for n in got["render"]["nodes"] if n["id"] == "asr")
    assert asr["params"]["model"] == "whisper"

    assert client.delete("/api/graphs/demo").json() == {"deleted": "demo"}
    assert client.get("/api/graphs/demo").status_code == 404


def test_save_requires_name_and_spec(client):
    assert client.post("/api/graphs", json={"name": "x"}).status_code == 400
    assert client.post("/api/graphs", json={"spec": _SPEC}).status_code == 400


def test_builder_page_has_save_load_ui(client):
    t = client.get("/ui/builder").text
    assert 'id="btn-save-graph"' in t
    assert 'id="saved-list"' in t


def test_saved_graph_round_trips_branches_and_llm(client):
    """A saved graph's variant set + global LLM survive load (the render payload carries them
    so the builder re-hydrates the Branches / LLM panels)."""
    spec = dict(_SPEC)
    spec["branches"] = [{"id": "labse", "text_embedding": {"model": "labse"}},
                        {"id": "jina", "text_embedding": {"model": "jina_v4"}}]
    spec["llm"] = {"model": "llama3", "use_local_server": True, "temperature": 0.3}
    client.post("/api/graphs", json={"name": "variants", "spec": spec})
    render = client.get("/api/graphs/variants").json()["render"]
    assert render["branches"] == spec["branches"]
    assert render["llm"] == spec["llm"]
