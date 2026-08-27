"""Cross-variant / cross-run comparison (the "unified comparison tool" roadmap item).

Consolidates the input side of "compare two things" onto one path: a `variants/<id>/`
directory (`metrics.json` + `answers.jsonl`, written by
`evaluation/results_io.py:save_run_artifacts`) or a whole run `output_dir` (auto-resolves to
its single variant when there's exactly one — the common case; a genuinely multi-variant
`output_dir` needs a specific `variants/<id>` path since there's no single baseline to pick).
The actual statistics are `analysis/significance.py:compare_experiments`, unchanged — this
module only locates + shapes its inputs, and adds the resolved-config + answer-level diffs it
doesn't do.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .significance import compare_experiments, format_comparison_report


class VariantCompareError(Exception):
    """A path couldn't be resolved to a comparable variant (ambiguous or empty)."""


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _variant_entries(path: Path) -> Optional[List[Path]]:
    """The variant leaf dirs found under ``path`` (a ``variants/`` dir, or a run
    ``output_dir`` containing one) — ``None`` when ``path`` isn't that kind of container at
    all (it's a leaf itself, or nothing)."""
    if path.name == "variants" and path.is_dir():
        variants_dir = path
    elif (path / "variants").is_dir():
        variants_dir = path / "variants"
    else:
        return None
    return sorted(p for p in variants_dir.iterdir() if (p / "metrics.json").exists())


def resolve_variant_dir(path: Union[str, Path]) -> Path:
    """A path the user pointed at → the actual `variants/<id>/` directory to read.

    Accepts: a variant leaf itself (has ``metrics.json``), a whole run ``output_dir`` (its
    ``variants/`` has exactly one entry — the common single-variant case), or
    ``output_dir/variants`` itself. Raises when ambiguous (more than one variant — point at a
    specific one, or pass this ONE path alone to `compare_paths` to compare all of them) or
    when nothing resolves.
    """
    path = Path(path)
    if (path / "metrics.json").exists():
        return path
    entries = _variant_entries(path)
    if entries:
        if len(entries) == 1:
            return entries[0]
        names = ", ".join(str(p) for p in entries)
        raise VariantCompareError(
            f"{path} has {len(entries)} variants ({names}) — point at a specific one, "
            "or pass this path alone (with no other paths) to compare all of them"
        )
    raise VariantCompareError(f"no metrics.json found under {path}")


def list_variant_dirs(path: Union[str, Path]) -> List[Path]:
    """Every variant leaf dir reachable from ``path`` — itself (as a 1-element list) when
    it's already a leaf or resolves to exactly one, or ALL of them (sorted) when ``path`` is
    a multi-variant run's ``output_dir``/``variants/`` dir. This is what makes "one run, N
    compared paths in the graph" (e.g. testing several ASR models or audio encoders in one
    run) usable as a single argument, instead of enumerating every variant id by hand."""
    path = Path(path)
    if (path / "metrics.json").exists():
        return [path]
    entries = _variant_entries(path)
    if entries:
        return entries
    raise VariantCompareError(f"no metrics.json found under {path}")


def load_variant_results(path: Union[str, Path]) -> Dict[str, Any]:
    """A variant/run directory's ``metrics.json`` + ``answers.jsonl``, shaped for
    ``compare_experiments`` (``details`` = the per-query records
    ``_extract_per_sample_scores`` infers numeric per-sample metrics from — per_query_wer/cer,
    recall_at_5, ...)."""
    variant_dir = resolve_variant_dir(path)
    with open(variant_dir / "metrics.json", "r", encoding="utf-8") as fh:
        results = json.load(fh)
    answers = _read_jsonl(variant_dir / "answers.jsonl")
    if answers:
        results = dict(results)
        results["details"] = answers
    return results


def _resolved_config_path(path: Union[str, Path]) -> Optional[Path]:
    """The ``*.config_resolved.yaml`` sidecar for a run — walks up from a variant leaf
    (``<output_dir>/variants/<id>/``) to its ``output_dir`` root, where
    ``save_run_artifacts`` writes it alongside the report JSON."""
    path = Path(path)
    for root in (path, path.parent, path.parent.parent):
        if not root.is_dir():
            continue
        matches = sorted(root.glob("*.config_resolved.yaml"))
        if matches:
            return matches[-1]
    return None


def config_diff(path_a: Union[str, Path], path_b: Union[str, Path]) -> List[str]:
    """Unified diff of the two runs' resolved configs — empty when either side has none, or
    both resolve to the SAME file (two variants inside one run share one config; there's
    nothing to diff)."""
    resolved_a, resolved_b = _resolved_config_path(path_a), _resolved_config_path(path_b)
    if resolved_a is None or resolved_b is None or resolved_a == resolved_b:
        return []
    return list(difflib.unified_diff(
        resolved_a.read_text(encoding="utf-8").splitlines(),
        resolved_b.read_text(encoding="utf-8").splitlines(),
        fromfile=str(resolved_a), tofile=str(resolved_b), lineterm="",
    ))


def answer_diffs(
    path_a: Union[str, Path], path_b: Union[str, Path], *, max_examples: int = 20,
) -> List[Dict[str, Any]]:
    """Per-query ``generated_answer`` differences between two variants, id-joined — the
    "what actually changed" companion to the metric deltas. Skips ids absent on either side
    or with identical text; caps at ``max_examples``."""
    by_id_a = {
        r.get("query_id"): r
        for r in _read_jsonl(resolve_variant_dir(path_a) / "answers.jsonl")
    }
    by_id_b = {
        r.get("query_id"): r
        for r in _read_jsonl(resolve_variant_dir(path_b) / "answers.jsonl")
    }
    out: List[Dict[str, Any]] = []
    for qid in sorted(set(by_id_a) & set(by_id_b), key=str):
        ans_a = (by_id_a[qid].get("generated_answer") or "").strip()
        ans_b = (by_id_b[qid].get("generated_answer") or "").strip()
        if ans_a != ans_b:
            out.append({"query_id": qid, "a": ans_a, "b": ans_b})
        if len(out) >= max_examples:
            break
    return out


def compare_paths(
    paths: List[Union[str, Path]], metric_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Baseline-vs-each comparison across N variant/run paths (``paths[0]`` = baseline) —
    metric deltas+significance, a resolved-config diff, and per-query answer diffs, for each
    of ``paths[1:]`` against the baseline.

    A SINGLE path that is itself multi-variant (a run with several compared paths in its
    graph — several ASR models, audio encoders, retrieval strategies, ...) auto-expands to
    all of its own variants (baseline = the first, alphabetically) — so pointing this at one
    such run's ``output_dir`` compares everything in it, with no need to enumerate every
    variant id by hand."""
    if len(paths) == 1:
        paths = list_variant_dirs(paths[0])
    if len(paths) < 2:
        raise VariantCompareError("need at least 2 variants/paths to compare")
    baseline = paths[0]
    baseline_results = load_variant_results(baseline)
    comparisons = []
    for other in paths[1:]:
        other_results = load_variant_results(other)
        comparisons.append({
            "baseline": str(baseline),
            "path": str(other),
            "metrics": compare_experiments(baseline_results, other_results, metric_names),
            "config_diff": config_diff(baseline, other),
            "answer_diffs": answer_diffs(baseline, other),
        })
    return {"baseline": str(baseline), "comparisons": comparisons}


def format_variant_comparison_report(bundle: Dict[str, Any]) -> str:
    """Human-readable text report for ``compare_paths``' output — one section per
    baseline-vs-variant comparison, reusing ``format_comparison_report``'s metric-panel
    formatting plus the config/answer diffs it doesn't cover."""
    lines = [f"Baseline: {bundle['baseline']}", ""]
    for cmp in bundle["comparisons"]:
        lines.append(format_comparison_report({
            "experiment_a": {"path": cmp["baseline"]},
            "experiment_b": {"path": cmp["path"]},
            "metrics": cmp["metrics"],
        }))
        if cmp["config_diff"]:
            lines.append("-" * 70)
            lines.append("CONFIG DIFF")
            lines.append("-" * 70)
            lines.extend(cmp["config_diff"])
            lines.append("")
        if cmp["answer_diffs"]:
            lines.append("-" * 70)
            lines.append(f"ANSWER DIFFS ({len(cmp['answer_diffs'])} changed)")
            lines.append("-" * 70)
            for d in cmp["answer_diffs"]:
                lines.append(f"[{d['query_id']}]")
                lines.append(f"  A: {d['a']}")
                lines.append(f"  B: {d['b']}")
            lines.append("")
    return "\n".join(lines)
