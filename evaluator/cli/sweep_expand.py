"""Expand a base graph config + an axes spec into an explicit multi-variant graph config.

There is no ``graph.branches`` any more (see ``config/graph_config.py``): a config is one
explicit graph, and comparing N variants of a node is just N distinctly-named node entries,
wired from whatever they share via ordinary edges. Hand-duplicating those entries for every
node param combination is exactly the tedious part a sweep tool should do for you.

Given a base config's ``graph.nodes``/``graph.edges`` and an axes spec naming which node/param
to vary and over which values, this expands the cartesian product of the axes into that same
shape: every node topologically downstream of a varied node (inclusive) gets one distinctly-
named copy per combination (``<node_id>_<combo_label>``); everything upstream/unrelated is left
as a single shared node referenced by every variant's edges — matching how these configs are
hand-authored (see e.g. ``configs/pubmed_qa_fulltext_big_rag_cmp.yaml``).

CLI: ``evaluator sweep --base <config.yaml> --axes <axes.yaml> --out <config.yaml>``
"""

from __future__ import annotations

import copy
import itertools
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..errors import ConfigurationError


class SweepExpandError(ConfigurationError):
    """Raised when an axes spec or base config can't be expanded."""


# ---------------------------------------------------------------------------
# axes spec
# ---------------------------------------------------------------------------


def load_axes_spec(path: str) -> Dict[str, Any]:
    """Load + validate an axes spec (YAML/JSON): ``{base?, name?, axes: [{node, param, values}]}``."""
    import json

    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    _validate_axes_spec(spec)
    return spec


def _validate_axes_spec(spec: Any) -> None:
    if not isinstance(spec, dict) or "axes" not in spec:
        raise SweepExpandError("axes spec must be a mapping with an 'axes' key")
    axes = spec["axes"]
    if not isinstance(axes, list) or not axes:
        raise SweepExpandError("axes spec 'axes' must be a non-empty list")
    for axis in axes:
        if not isinstance(axis, dict) or not {"node", "param", "values"} <= set(axis):
            raise SweepExpandError(
                "each axis needs 'node', 'param', and 'values' keys: "
                f"{axis!r}"
            )
        if not isinstance(axis["values"], list) or not axis["values"]:
            raise SweepExpandError(
                f"axis '{axis['node']}.{axis['param']}' needs a non-empty 'values' list"
            )


# ---------------------------------------------------------------------------
# base graph normalization
# ---------------------------------------------------------------------------


def _normalize_node(entry: Any) -> Tuple[str, str, Dict[str, Any]]:
    """A graph.nodes entry is a bare type string or a {id, type, params} dict."""
    if isinstance(entry, str):
        return entry, entry, {}
    if isinstance(entry, dict) and "type" in entry:
        node_id = str(entry.get("id") or entry["type"])
        return node_id, str(entry["type"]), dict(entry.get("params") or {})
    raise SweepExpandError(f"unrecognized graph.nodes entry: {entry!r}")


def _set_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    """Deep-set ``d[a][b]...[z] = value`` for a dot-separated path, creating dicts as needed."""
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _sanitize_label(s: str) -> str:
    return _LABEL_UNSAFE.sub("-", str(s)).strip("-")


def _combo_label(combo: Sequence[Tuple[str, str, Any]]) -> str:
    """Compact id suffix for one axis combination — the values joined, e.g. ``jina_v4_5``
    for a ``[text_embedding.model=jina_v4, retrieval.k=5]`` combo (matches the hand-authored
    convention, e.g. ``retrieval_dense``/``retrieval_sparse``). Values must be unique enough
    across combos to not collide once sanitized — ``expand_axes`` raises if two do."""
    return "_".join(_sanitize_label(value) for _node, _param, value in combo)


# ---------------------------------------------------------------------------
# affected-set (forward reachability from varied nodes)
# ---------------------------------------------------------------------------


def _forward_adjacency(node_ids: Sequence[str], edges: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if src in adj and dst in adj:
            adj[src].add(dst)
    return adj


def _affected_set(varied_ids: Set[str], adjacency: Dict[str, Set[str]]) -> Set[str]:
    """Every varied node plus everything topologically downstream of one (inclusive)."""
    affected: Set[str] = set()
    stack = list(varied_ids)
    while stack:
        nid = stack.pop()
        if nid in affected:
            continue
        affected.add(nid)
        stack.extend(adjacency.get(nid, ()))
    return affected


# ---------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------


def expand_axes(
    base_nodes: Sequence[Any], base_edges: Sequence[Dict[str, Any]], axes_spec: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Expand ``axes_spec['axes']`` over the base graph.

    Returns ``(new_nodes, new_edges, combos)`` — ``new_nodes``/``new_edges`` are ready to drop
    into ``graph.nodes``/``graph.edges``; ``combos`` is ``[{label, overrides}, ...]`` for preview.
    """
    axes = axes_spec["axes"]
    normalized = [_normalize_node(n) for n in base_nodes]
    ids = [nid for nid, _t, _p in normalized]
    by_id = {nid: (nid, t, p) for nid, t, p in normalized}

    varied_ids = {str(a["node"]) for a in axes}
    unknown = varied_ids - set(ids)
    if unknown:
        raise SweepExpandError(
            f"axes reference node id(s) not in base graph.nodes: {sorted(unknown)} "
            f"(known: {sorted(ids)})"
        )

    adjacency = _forward_adjacency(ids, base_edges)
    affected = _affected_set(varied_ids, adjacency)

    # Shared nodes: never touched, appear once, verbatim.
    new_nodes: List[Dict[str, Any]] = []
    for entry, (nid, _t, _p) in zip(base_nodes, normalized):
        if nid not in affected:
            new_nodes.append(entry if isinstance(entry, dict) else entry)

    shared_edges = [
        dict(e) for e in base_edges if e.get("from") not in affected and e.get("to") not in affected
    ]
    variant_edge_template = [
        e for e in base_edges if e.get("from") in affected or e.get("to") in affected
    ]

    combo_value_lists = [[(str(a["node"]), str(a["param"]), v) for v in a["values"]] for a in axes]

    new_edges: List[Dict[str, Any]] = list(shared_edges)
    combos: List[Dict[str, Any]] = []
    seen_labels: Set[str] = set()

    for combo in itertools.product(*combo_value_lists):
        label = _combo_label(combo)
        if label in seen_labels:
            raise SweepExpandError(
                f"two combinations sanitize to the same id suffix '{label}' — "
                "use more distinct axis values"
            )
        seen_labels.add(label)

        overrides: Dict[str, Dict[str, Any]] = {}
        for node, param, value in combo:
            _set_path(overrides.setdefault(node, {}), param, value)

        idmap = {nid: (f"{nid}_{label}" if nid in affected else nid) for nid in ids}

        for nid in ids:
            if nid not in affected:
                continue
            _orig_id, ntype, params = by_id[nid]
            merged = _deep_merge(params, overrides.get(nid, {}))
            node_out: Dict[str, Any] = {"id": idmap[nid], "type": ntype}
            if merged:
                node_out["params"] = merged
            new_nodes.append(node_out)

        for e in variant_edge_template:
            e2 = dict(e)
            e2["from"] = idmap.get(e2["from"], e2["from"])
            e2["to"] = idmap.get(e2["to"], e2["to"])
            new_edges.append(e2)

        combos.append({"label": label, "overrides": overrides})

    return new_nodes, new_edges, combos


def _require_explicit_graph(graph: Dict[str, Any]) -> None:
    if "nodes" not in graph:
        raise SweepExpandError("base config has no graph.nodes to expand")
    if "edges" not in graph:
        raise SweepExpandError(
            "base config has no explicit graph.edges — run "
            "`evaluator graph --config <base> --emit-edges --write` first"
        )


def expand_config(base_config: Dict[str, Any], axes_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of ``base_config`` with ``graph.nodes``/``graph.edges`` expanded."""
    graph = base_config.get("graph") or {}
    _require_explicit_graph(graph)
    new_nodes, new_edges, _combos = expand_axes(graph["nodes"], graph["edges"], axes_spec)
    out = copy.deepcopy(base_config)
    out["graph"]["nodes"] = new_nodes
    out["graph"]["edges"] = new_edges
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys

    import yaml

    parser = argparse.ArgumentParser(prog="evaluator sweep")
    parser.add_argument("--base", help="base graph config YAML (or set 'base:' in --axes)")
    parser.add_argument("--axes", required=True, help="axes spec (YAML/JSON)")
    parser.add_argument("--out", help="write the expanded config here (default: print to stdout)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the combination list + shared/varied node split, write nothing",
    )
    args = parser.parse_args(argv)

    axes_spec = load_axes_spec(args.axes)
    base_path = args.base or axes_spec.get("base")
    if not base_path:
        parser.error("--base is required (or set 'base:' in the axes spec)")

    with open(base_path, "r", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    graph = base_config.get("graph") or {}
    _require_explicit_graph(graph)
    new_nodes, new_edges, combos = expand_axes(graph["nodes"], graph["edges"], axes_spec)
    name = axes_spec.get("name", "sweep")
    print(f"sweep '{name}': {len(combos)} combinations over {len(axes_spec['axes'])} axis(es)")
    for c in combos:
        overrides_flat = {
            f"{node}.{k}" if not isinstance(v, dict) else node: v
            for node, params in c["overrides"].items()
            for k, v in params.items()
        }
        print(f"  {c['label']:<40} {overrides_flat}")
    base_node_count = len(graph["nodes"])
    print(
        f"\n{len(new_nodes)} nodes total ({base_node_count} in the base graph, "
        f"{len(combos)} combos), {len(new_edges)} edges"
    )

    if args.dry_run:
        return 0

    out_config = copy.deepcopy(base_config)
    out_config["graph"]["nodes"] = new_nodes
    out_config["graph"]["edges"] = new_edges
    dumped = yaml.safe_dump(out_config, sort_keys=False, allow_unicode=True)

    # Sanity-check: does the expanded config actually build? Node-centric configs are
    # translated by build_evaluation_config_kwargs, which from_yaml runs on load — so write
    # it out (a temp file if --out wasn't given) and load it back through the real chokepoint,
    # same path a run would take.
    import tempfile
    from pathlib import Path

    from ..config.evaluation import EvaluationConfig

    write_path = args.out
    tmp_path: Optional[str] = None
    if not write_path:
        fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="sweep_expand_")
        import os

        os.close(fd)
        write_path = tmp_path
    Path(write_path).write_text(dumped, encoding="utf-8")

    ok = True
    try:
        EvaluationConfig.from_yaml(write_path)
        print("\nvalidation: OK — expanded config loads via EvaluationConfig.from_yaml")
    except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
        ok = False
        print(f"\nvalidation: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.out:
        print(f"\nwrote {args.out}")
    else:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        print("\n" + dumped)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
