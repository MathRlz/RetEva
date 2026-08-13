"""How much of a PubMedQA split has a downloadable full article (PMC open access)?

PubMedQA gives you a PMID per question; only the PMC open-access subset can be fetched. This
scans a split, resolves PMID→PMCID in batches, fetches the articles in batches, and reports how
many actually carry a body — then writes the qualifying PMIDs so a full-text-backed question set
can be built from them (``build_pubmed_campaign.py --pmids <file>``).

    python3 scripts/scan_pubmed_fulltext.py --config pqa_labeled
    python3 scripts/scan_pubmed_fulltext.py --config pqa_artificial --pool 4000

Cheap to re-run: article XML is cached under ``.cache/pmc`` by the fetcher.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_pubmed_fulltext import (  # noqa: E402
    fetch_articles,
    has_full_text,
    pmid_to_pmcid,
)


def scan(config: str, pool: int, cache_dir: Path, out_path: Path, email) -> int:
    from datasets import load_dataset

    rows = load_dataset("qiaojin/PubMedQA", config, split="train")
    n = min(pool, len(rows)) if pool else len(rows)
    pmids = [str(rows[i]["pubid"]) for i in range(n)]
    print(f"{config}: scanning {n} of {len(rows)} questions")

    pmcids = pmid_to_pmcid(pmids, email)
    print(f"  PMCID assigned : {len(pmcids)}/{n} ({100.0 * len(pmcids) / n:.1f}%)")

    xmls = fetch_articles(sorted(set(pmcids.values())), cache_dir)
    print(f"  articles served: {len(xmls)}/{len(pmcids)}")

    usable = {pmid: pmc for pmid, pmc in pmcids.items()
              if pmc in xmls and has_full_text(xmls[pmc])}
    pct = 100.0 * len(usable) / n if n else 0.0
    print(f"  FULL TEXT      : {len(usable)}/{n} ({pct:.1f}%) — open access with a body")

    out_path.write_text(json.dumps(
        {"config": config, "scanned": n, "with_pmcid": len(pmcids),
         "with_full_text": len(usable), "coverage": round(pct / 100.0, 4),
         "pmids": sorted(usable)}, indent=1))
    print(f"\nqualifying PMIDs → {out_path}")
    print(f"build a set from them: python3 scripts/build_pubmed_campaign.py "
          f"--config {config} --pmids {out_path} --full-text -n {len(usable)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pqa_labeled",
                    help="PubMedQA config: pqa_labeled (1k, expert yes/no/maybe) or "
                         "pqa_artificial (211k, auto-labelled)")
    ap.add_argument("--pool", type=int, default=0, help="scan only the first N (0 = all)")
    ap.add_argument("--cache-dir", default=".cache/pmc")
    ap.add_argument("--out", default=None, help="default: pubmed_fulltext_<config>.json")
    ap.add_argument("--email", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"pubmed_fulltext_{args.config}.json")
    return scan(args.config, args.pool, Path(args.cache_dir), out, args.email)


if __name__ == "__main__":
    raise SystemExit(main())
