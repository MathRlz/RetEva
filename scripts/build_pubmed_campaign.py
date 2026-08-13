"""Build the campaign-size PubMedQA question/corpus set from the HF `pqa_labeled` split.

The bundled `examples/data/pubmed_qa_small/` holds 20 questions — below the framework's
CI/power floor (n < 20 emits no bootstrap CI, and a Wilcoxon signed-rank test on n=5
cannot reach p<0.05 at all). This writes a larger, deterministic slice in exactly the same
JSON shape so the campaign configs get real statistical power.

    python3 scripts/build_pubmed_campaign.py [-n 200] [--out examples/data/pubmed_qa_campaign]

Deterministic: a seeded shuffle over the split, so re-running reproduces the same set.
Run it where the HF cache lives (the container).
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional


def _load_pmid_filter(path: Path) -> set:
    """PMIDs from a scan file (``scan_pubmed_fulltext.py`` output) or a plain id list."""
    data = json.loads(path.read_text())
    pmids = data.get("pmids", data) if isinstance(data, dict) else data
    return {str(p) for p in pmids}


def build(n: int, out_dir: Path, seed: int, *, config: str = "pqa_labeled",
          keep_pmids: Optional[set] = None, full_text: bool = False) -> None:
    from datasets import load_dataset

    rows = load_dataset("qiaojin/PubMedQA", config, split="train")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    if keep_pmids:
        # Restrict to a scanned id list (e.g. the questions whose article IS downloadable),
        # keeping the seeded order so the slice stays reproducible.
        order = [i for i in order if str(rows[i]["pubid"]) in keep_pmids]
        print(f"pmid filter: {len(order)} of {len(rows)} rows qualify")

    articles = {}
    if full_text:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fetch_pubmed_fulltext import fetch_articles, jats_to_text, pmid_to_pmcid

        pmids = [str(rows[i]["pubid"]) for i in order[:n]]
        pmcids = pmid_to_pmcid(pmids, None)
        xmls = fetch_articles(sorted(set(pmcids.values())), Path(".cache/pmc"))
        for pmid, pmcid in pmcids.items():
            body = jats_to_text(xmls.get(pmcid, ""))
            if body:
                articles[pmid] = (pmcid, body)
        print(f"full text: {len(articles)}/{len(pmids)} articles fetched from PMC")

    questions, corpus = [], []
    for idx in order[:n]:
        row = rows[idx]
        doc_id = str(row["pubid"])
        # one doc per question: its own contexts concatenated (self-retrieval GT, the
        # same contract pubmed_qa_small uses — the relevant doc is the question's source)
        text = " ".join(row["context"]["contexts"])
        questions.append({
            "question_id": f"q_{doc_id}",
            "question_text": row["question"],
            "groundtruth_doc_ids": [doc_id],
            "relevance_grades": {doc_id: 1},
            "language": "en",
            "metadata": {"pubid": row["pubid"]},
            "short_answer": row["final_decision"],
        })
        # The answer GT rides the DOC metadata, because that is where answer scoring looks for
        # it (`answer_generation.reference_metadata_field`, default "long_answer" →
        # evaluation/answer_gen.py:_lookup_reference reads the relevant doc, not the question).
        # Without this, ROUGE-L is reported as n/a. `text` is unchanged, so retrieval and the
        # corpus embeddings are byte-identical to a set built before this field existed.
        meta = {
            "long_answer": row["long_answer"],
            "final_decision": row["final_decision"],
        }
        if doc_id in articles:
            pmcid, body = articles[doc_id]
            # Generation-only: `text` (the abstract passages) still drives retrieval.
            meta.update({"full_text": body, "full_text_source": "pmc", "pmcid": pmcid})
        elif full_text:
            meta["full_text_source"] = "abstract"
        corpus.append({
            "doc_id": doc_id,
            "title": row["question"],
            "text": text,
            "metadata": meta,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.json").write_text(json.dumps(questions, indent=1))
    (out_dir / "corpus.json").write_text(json.dumps(corpus, indent=1))
    print(f"wrote {len(questions)} questions + {len(corpus)} docs to {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=200, help="number of questions (default 200)")
    ap.add_argument("--out", default="examples/data/pubmed_qa_campaign")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="pqa_labeled",
                    help="PubMedQA config: pqa_labeled (1k expert) | pqa_artificial (211k)")
    ap.add_argument("--pmids", default=None,
                    help="scan file (scan_pubmed_fulltext.py) or id list: keep only these PMIDs")
    ap.add_argument("--full-text", action="store_true",
                    help="also fetch each article from PMC into corpus metadata.full_text")
    args = ap.parse_args()
    keep = _load_pmid_filter(Path(args.pmids)) if args.pmids else None
    build(args.n, Path(args.out), args.seed, config=args.config,
          keep_pmids=keep, full_text=args.full_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
