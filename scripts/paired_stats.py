"""Paired significance between systems from two (or the same) result JSONs.

For each pair (A, B) over the shared query ids:
  * Recall@1 and MRR of each side with a 95% percentile-bootstrap CI,
  * paired-bootstrap 95% CI of the difference B − A (resample queries, keep pairs),
  * exact McNemar test on Recall@1 hits,
  * Holm correction of the McNemar p-values across all pairs given in one call.

Needs result files written after per-query ``report.item_scores`` were added.

A system spec is ``PATH[@VARIANT[/BRANCH]]``; VARIANT/BRANCH may be omitted when the file
has exactly one. Example:

    python scripts/paired_stats.py \\
      --pair "Whisper+Jina vs SONAR" \\
             "res_asr.json@finalize_piper_whisper_jina" "res_emb.json@finalize_piper_sonar" \\
      --csv stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Dict, List, Tuple

from evaluator.analysis.significance import bootstrap_confidence_interval, holm, mcnemar_exact

METRICS = ("recall@1", "mrr")


def _pick(d: dict, key: str | None, what: str, where: str) -> Tuple[str, dict]:
    if key is None:
        if len(d) != 1:
            sys.exit(f"{where}: pick a {what}, one of: {sorted(d)}")
        key = next(iter(d))
    if key not in d:
        sys.exit(f"{where}: no {what} {key!r}, have: {sorted(d)}")
    return key, d[key]


def load_scores(spec: str) -> Dict[str, Dict[str, float]]:
    """``{metric: {query_id: score}}`` for one system spec."""
    path, _, rest = spec.partition("@")
    variant, _, branch = rest.partition("/")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    if "variants" in res:
        _, res = _pick(res["variants"], variant or None, "variant", path)
    item_scores = (res.get("report") or {}).get("item_scores")
    if not item_scores:
        sys.exit(f"{spec}: no report.item_scores (file predates per-query persistence; re-run)")
    _, scores = _pick(item_scores, branch or None, "branch", spec)
    return scores


def _ci(values: List[float], n_boot: int) -> Tuple[float, float]:
    return bootstrap_confidence_interval(values, n_bootstrap=n_boot, random_state=0)


def compare(a: dict, b: dict, n_boot: int) -> dict:
    ids = sorted(set(a["recall@1"]) & set(b["recall@1"]))
    dropped = len(set(a["recall@1"]) ^ set(b["recall@1"]))
    if dropped:
        print(f"warning: {dropped} query id(s) present on one side only, dropped", file=sys.stderr)
    row = {"n": len(ids)}
    for m in METRICS:
        va = [a[m][i] for i in ids]
        vb = [b[m][i] for i in ids]
        diffs = [y - x for x, y in zip(va, vb)]
        row[f"A_{m}"] = sum(va) / len(va)
        row[f"A_{m}_ci"] = _ci(va, n_boot)
        row[f"B_{m}"] = sum(vb) / len(vb)
        row[f"B_{m}_ci"] = _ci(vb, n_boot)
        row[f"delta_{m}"] = sum(diffs) / len(diffs)
        row[f"delta_{m}_ci"] = _ci(diffs, n_boot)  # paired: each resample keeps (A, B) together
    hits_a = [a["recall@1"][i] > 0 for i in ids]
    hits_b = [b["recall@1"][i] > 0 for i in ids]
    row["b_A_only"], row["c_B_only"], row["p_mcnemar"] = mcnemar_exact(hits_a, hits_b)
    return row


def _fmt(x: float, ci: Tuple[float, float]) -> str:
    return f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("LABEL", "A", "B"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--csv", help="also write the table as CSV here")
    args = ap.parse_args()

    rows = []
    for label, sa, sb in args.pair:
        row = compare(load_scores(sa), load_scores(sb), args.n_boot)
        rows.append({"pair": label, **row})
    for row, ph in zip(rows, holm([r["p_mcnemar"] for r in rows])):
        row["p_holm"] = ph

    print("| Pair | n | R@1 A | R@1 B | ΔR@1 (B−A) | b / c | McNemar p | Holm p | MRR A | MRR B | ΔMRR (B−A) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['pair']} | {r['n']} "
            f"| {_fmt(r['A_recall@1'], r['A_recall@1_ci'])} | {_fmt(r['B_recall@1'], r['B_recall@1_ci'])} "
            f"| {_fmt(r['delta_recall@1'], r['delta_recall@1_ci'])} | {r['b_A_only']} / {r['c_B_only']} "
            f"| {_p(r['p_mcnemar'])} | {_p(r['p_holm'])} "
            f"| {_fmt(r['A_mrr'], r['A_mrr_ci'])} | {_fmt(r['B_mrr'], r['B_mrr_ci'])} "
            f"| {_fmt(r['delta_mrr'], r['delta_mrr_ci'])} |"
        )

    if args.csv:
        flat = [
            {k: (f"{v[0]:.6f};{v[1]:.6f}" if isinstance(v, tuple) else v) for k, v in r.items()}
            for r in rows
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)


if __name__ == "__main__":
    main()
