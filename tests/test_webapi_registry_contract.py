"""The registry <-> webapi contract (see evaluator-architecture.md §12.2, TODO.md "Registry <->
webapi contract"). Some webapi lookup tables read the node registry generically (a new node just
works); a few are hand-maintained dicts that silently go stale when a node is added -- exactly
the bug class that produced 3 of the 7 "verify all configs work with UI" bugs (corpus_embedding
inspect fields, retrieval-only fork detection, a hardcoded node-kind list). Each test here pins
one such contract point: it must either cover every relevant registry entry, or name an explicit
exemption with a reason -- never silently miss one.
"""

from evaluator import EvaluationConfig, get_preset, list_presets
from evaluator.pipeline.graph import build as graph_build
from evaluator.pipeline.graph.registry import model_node_kinds
from evaluator.webapi.form_builder import node_catalogue


def build_spec_js_equivalent(data):
    """1:1 translation of the FIXED `_config_run.html` buildSpec() (see
    test_introspection_schema.py for the end-to-end version against a real payload) -- kept
    standalone here since this test only needs the port-resolution logic in isolation, not a
    full canvas payload."""
    nodes = [{"id": n["id"], "type": n["type"], "params": n["params"]} for n in data["nodes"]]
    edges = []
    for n in data["nodes"]:
        ports = n.get("input_ports") or []
        for art, prod in n.get("bindings") or []:
            port = next((p for p in ports if art in (p.get("names") or [])), None)
            edges.append({
                "from": prod, "output": art, "to": n["id"],
                "input": port["label"] if port else art,
            })
    return {"mode": data["mode"], "nodes": nodes, "edges": edges}


def test_buildspec_resolves_every_multi_producer_port_label():
    """buildSpec() reconstructs `{input: <port label>}` by matching a bound artifact name
    against the node's own `input_ports`. This session's bug fixed exactly one case
    (retrieval.query_vectors) -- pin ALL of them, so a future OneOf-collapsed port doesn't
    silently regress the same way."""
    cat = node_catalogue()
    checked = 0
    for n in cat["nodes"]:
        for port in n.get("input_ports") or []:
            names = port.get("names") or []
            if len(names) <= 1:
                continue
            for art in names:
                node_payload = {
                    "id": n["kind"], "type": n["kind"], "params": {},
                    "input_ports": n["input_ports"],
                    "bindings": [[art, "producer"]],
                }
                spec = build_spec_js_equivalent({"mode": "t", "nodes": [node_payload]})
                edge = spec["edges"][0]
                assert edge["input"] == port["label"], (
                    f"{n['kind']}.{art} resolved input={edge['input']!r}, "
                    f"expected the port's own label {port['label']!r}"
                )
                checked += 1
    assert checked >= 5, "expected several multi-producer ports in the registry to check"


def test_node_inspect_covers_every_model_node_kind():
    """`registry.py:model_node_kinds()` is the authoritative "which kinds are models" set
    (already used by the device-placement self-check). `form_builder.py:_node_inspect` /
    `MODEL_NODE_FIELDS` is a separate hand-written dict -- this is exactly how
    `corpus_embedding` silently returned `{}` (fixed by aliasing it onto `text_embedding`'s
    fields). Every model kind must resolve non-empty inspect fields."""
    from types import SimpleNamespace

    from evaluator.webapi.form_builder import _node_inspect

    preset = next((p for p in list_presets() if "small" in p), list_presets()[0])
    config = EvaluationConfig.from_dict(get_preset(preset))

    exempt = {}  # no known exemptions today -- every model kind resolves via MODEL_NODE_FIELDS
    # or an explicit alias (corpus_embedding -> text_embedding) in _node_inspect.
    for kind in sorted(model_node_kinds()):
        if kind in exempt:
            continue
        node = SimpleNamespace(stage=kind, params={})
        info = _node_inspect(config, node)
        assert info, (
            f"{kind!r} (in model_node_kinds()) resolved NO inspect fields -- either add it to "
            "MODEL_NODE_FIELDS / alias it in _node_inspect (form_builder.py), or add it to the "
            "`exempt` dict above with a reason."
        )


def test_variant_root_kinds_guard_no_undeclared_candidate():
    """The structural-plumbing fork detection only forks on `graph_build.VARIANT_ROOT_KINDS`
    (`{"retrieval", "answer_gen"}` today). A non-structural kind whose OWN output artifact is
    directly consumed by a structural (measure/sink) node is a PLAUSIBLE new fork root --
    comparing N of them needs its own metrics/finalize chain. Must be in VARIANT_ROOT_KINDS or on
    the exemption list below with a reason, never silently missed just because nothing today
    happens to exercise it."""
    cat = node_catalogue()
    structural_inputs = {
        name
        for n in cat["nodes"] if n["structural"]
        for p in n.get("input_ports") or []
        for name in p.get("names") or []
    }
    exempt = {
        "dataset_source": (
            "the shared root every variant reads from -- never itself a compared variant"
        ),
        "rerank": (
            "downstream refinement of a retrieval root within the SAME chain; ownership "
            "already follows the retrieval root by forward-reachability, not by rerank's own "
            "kind -- no config forks on rerank alone today. Revisit if one ever compares N "
            "rerank strategies on ONE shared retrieval."
        ),
        "asr": (
            "today's multi-ASR configs each pair an ASR variant with its own downstream "
            "retrieval node (already covered via the `retrieval` root). No config compares ASR "
            "variants under asr_only (no downstream retrieval) -- revisit if one appears."
        ),
    }
    for n in cat["nodes"]:
        if n["structural"]:
            continue
        kind = n["kind"]
        if kind in graph_build.VARIANT_ROOT_KINDS or kind in exempt:
            continue
        outs = set(n.get("outputs") or [])
        hit = outs & structural_inputs
        assert not hit, (
            f"{kind!r} produces {sorted(hit)}, consumed directly by a structural node -- a "
            "plausible new variant-fork root. Add it to graph_build.VARIANT_ROOT_KINDS, or extend the "
            "`exempt` dict above with a reason it's deliberately not one."
        )
