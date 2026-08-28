"""The registry <-> webapi contract (see evaluator-architecture.md §12.2, TODO.md "Registry <->
webapi contract"). Some webapi lookup tables read the node registry generically (a new node just
works); a few are hand-maintained dicts that silently go stale when a node is added -- exactly
the bug class that produced 3 of the 7 "verify all configs work with UI" bugs (corpus_embedding
inspect fields, retrieval-only fork detection, _infer_features's hardcoded kind list). Each test
here pins one such contract point: it must either cover every relevant registry entry, or name an
explicit exemption with a reason -- never silently miss one.
"""

import dataclasses
import inspect

from evaluator import EvaluationConfig, get_preset, list_presets
from evaluator.pipeline.graph.assembly import FeatureSet
from evaluator.pipeline.graph import modes as graph_modes
from evaluator.pipeline.graph.modes import GRAPH_TEMPLATE_SPECS
from evaluator.pipeline.graph.assembly import assemble_specs
from evaluator.pipeline.graph.operators import node_kind
from evaluator.pipeline.graph.registry import model_node_kinds
from evaluator.webapi.form_builder import node_catalogue


def _spec_type_params(spec):
    if isinstance(spec, str):
        return spec, {}
    return spec.get("type", spec.get("stage")), spec.get("params", {})


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


# ---- #3: port/edge reconstruction must handle EVERY multi-producer (OneOf) port ----

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


# ---- #4: canvas inspect-panel model-field resolution must cover every model node kind ----

def test_node_inspect_covers_every_model_node_kind():
    """`registry.py:model_node_kinds()` is the authoritative "which kinds are models" set
    (already used by the device-placement self-check). `form_builder.py:_node_inspect` /
    `_MODEL_NODE_FIELDS` is a separate hand-written dict -- this is exactly how
    `corpus_embedding` silently returned `{}` (fixed by aliasing it onto `text_embedding`'s
    fields). Every model kind must resolve non-empty inspect fields."""
    from types import SimpleNamespace

    from evaluator.webapi.form_builder import _node_inspect

    preset = next((p for p in list_presets() if "small" in p), list_presets()[0])
    config = EvaluationConfig.from_dict(get_preset(preset))

    exempt = {}  # no known exemptions today -- every model kind resolves via _MODEL_NODE_FIELDS
    # or an explicit alias (corpus_embedding -> text_embedding) in _node_inspect.
    for kind in sorted(model_node_kinds()):
        if kind in exempt:
            continue
        node = SimpleNamespace(stage=kind, params={})
        info = _node_inspect(config, node)
        assert info, (
            f"{kind!r} (in model_node_kinds()) resolved NO inspect fields -- either add it to "
            "_MODEL_NODE_FIELDS / alias it in _node_inspect (form_builder.py), or add it to the "
            "`exempt` dict above with a reason."
        )


# ---- #5: _infer_features must account for every FeatureSet flag ----

def test_infer_features_covers_every_feature_flag():
    """`_infer_features`'s own docstring: "a future structural node gated on a NEW flag must be
    added here." Every `FeatureSet` field must either be set by `_infer_features` itself, or be
    on one of its two named exemption sets (`_INFER_FEATURES_STRUCTURAL_GATING_EXEMPT`,
    `_INFER_FEATURES_SHAPE_ONLY_EXEMPT`) -- never silently ungoverned."""
    all_fields = {f.name for f in dataclasses.fields(FeatureSet)}
    src = inspect.getsource(graph_modes._infer_features)
    inferred = {name for name in all_fields if f"{name}=" in src}
    exempt = (
        graph_modes._INFER_FEATURES_STRUCTURAL_GATING_EXEMPT
        | graph_modes._INFER_FEATURES_SHAPE_ONLY_EXEMPT
    )
    ungoverned = all_fields - inferred - exempt
    assert not ungoverned, (
        f"FeatureSet field(s) {sorted(ungoverned)} are neither inferred by _infer_features nor "
        "on one of its exemption sets -- add an inference line or an exemption with a reason."
    )
    # the exemption sets themselves shouldn't carry stale/typo'd names
    assert exempt <= all_fields, sorted(exempt - all_fields)


# ---- #6: every non-structural registry kind must be reachable from some named template ----

def test_every_meaningful_kind_reachable_from_some_template():
    """A registered, user-authorable ("meaningful") node kind that no named template ever
    includes can be placed on the canvas but never composed via `evaluator graph --template`
    or a template-driven config -- builder/explicit-graph-only by construction. Confirmed live:
    true today only for `augmenter`. Must be an explicit, named exemption, not a silent gap."""
    template_kinds = set()
    for name in GRAPH_TEMPLATE_SPECS:
        for spec in assemble_specs(name, FeatureSet.maximal()):
            t, p = _spec_type_params(spec)
            template_kinds.add(node_kind(t, p))

    exempt = {
        "augmenter": (
            "opt-in robustness/noise testing, not a modality/pipeline-shape choice -- "
            "explicit-graph or builder-canvas only by design, never template-composed"
        ),
    }
    cat = node_catalogue()
    for n in cat["nodes"]:
        if n["structural"]:
            continue
        kind = n["kind"]
        if kind in template_kinds or kind in exempt:
            continue
        raise AssertionError(
            f"{kind!r} is a meaningful registry kind reachable from no named template's "
            "maximal assembly, and isn't on the `exempt` dict above with a reason -- either "
            "it's missing from every template's FeatureSet-gated assembly (a real gap), or "
            "it's deliberately explicit-graph-only and needs a documented exemption."
        )


# ---- #7: multi-variant fork detection must not silently miss a plausible new root kind ----

def test_variant_root_kinds_guard_no_undeclared_candidate():
    """`_complete_with_plumbing`'s fork detection only forks on `graph_modes.VARIANT_ROOT_KINDS`
    (`{"retrieval", "answer_gen"}` today). A non-structural kind whose OWN output artifact is
    directly consumed by a structural (measure/sink) node is a PLAUSIBLE new fork root --
    comparing N of them needs its own metrics/finalize chain, the exact bug class R5 fixed for
    `retrieval`. Must be in VARIANT_ROOT_KINDS or on the exemption list below with a reason,
    never silently missed just because nothing today happens to exercise it."""
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
        if kind in graph_modes.VARIANT_ROOT_KINDS or kind in exempt:
            continue
        outs = set(n.get("outputs") or [])
        hit = outs & structural_inputs
        assert not hit, (
            f"{kind!r} produces {sorted(hit)}, consumed directly by a structural node -- a "
            "plausible new variant-fork root. Add it to graph_modes.VARIANT_ROOT_KINDS, or extend the "
            "`exempt` dict above with a reason it's deliberately not one."
        )
