"""Multi-objective Pareto frontier over evaluation runs (Roadmap 4a).

The cross-run trade-off view a sweep wants: run A *dominates* run B when A is at-least-as-good
on every objective and strictly better on at least one. The frontier is the non-dominated set —
the runs where you can't improve one metric without giving up another (e.g. recall vs latency).
Pure functions over ``{"metrics": {...}}`` row dicts so they test without a DB or web layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

Objective = Tuple[str, str]  # (metric_name, "max" | "min")


def parse_objectives(spec: str) -> List[Objective]:
    """Parse ``"MRR:max,latency_ms:min"`` → ``[("MRR","max"), ("latency_ms","min")]``.

    A bare ``"MRR"`` defaults to ``max`` (higher is better — the common case)."""
    objectives: List[Objective] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, direction = part.partition(":")
        direction = (direction or "max").strip().lower()
        if direction not in ("max", "min"):
            raise ValueError(f"objective direction must be 'max' or 'min', got {direction!r}")
        objectives.append((name.strip(), direction))
    if not objectives:
        raise ValueError("no objectives parsed from spec")
    return objectives


def dominates(a: Dict[str, Any], b: Dict[str, Any], objectives: Sequence[Objective]) -> bool:
    """True if metrics ``a`` dominate metrics ``b``: ≥ on every objective, > on at least one.

    A missing metric on either side makes the pair incomparable on that objective, so ``a`` does
    not dominate (a run can't dominate on a metric it didn't measure)."""
    strict = False
    for name, direction in objectives:
        av, bv = a.get(name), b.get(name)
        if av is None or bv is None:
            return False
        if direction == "max":
            if av < bv:
                return False
            if av > bv:
                strict = True
        else:
            if av > bv:
                return False
            if av < bv:
                strict = True
    return strict


def pareto_frontier(
    rows: Sequence[Dict[str, Any]],
    objectives: Sequence[Objective],
    *,
    key: str = "metrics",
) -> List[Dict[str, Any]]:
    """The non-dominated subset of ``rows`` (each row's metrics under ``row[key]``), in input
    order. Rows missing any objective metric are excluded — they can't be placed on the
    frontier."""
    objectives = list(objectives)
    names = [n for n, _ in objectives]
    candidates = [
        r for r in rows if all(r.get(key, {}).get(n) is not None for n in names)
    ]
    frontier: List[Dict[str, Any]] = []
    for r in candidates:
        rm = r[key]
        if not any(dominates(o[key], rm, objectives) for o in candidates if o is not r):
            frontier.append(r)
    return frontier


def annotate_pareto(
    rows: Sequence[Dict[str, Any]],
    objectives: Sequence[Objective],
    *,
    key: str = "metrics",
) -> List[Dict[str, Any]]:
    """Return ``rows`` (those with all objective metrics) each tagged ``on_frontier: bool`` — the
    table view: every comparable run, with the non-dominated ones flagged."""
    frontier = {id(r) for r in pareto_frontier(rows, objectives, key=key)}
    names = [n for n, _ in objectives]
    out: List[Dict[str, Any]] = []
    for r in rows:
        if all(r.get(key, {}).get(n) is not None for n in names):
            out.append({**r, "on_frontier": id(r) in frontier})
    return out
