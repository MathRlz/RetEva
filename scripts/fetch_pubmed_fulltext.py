"""Enrich a corpus JSON with PMC full-text articles (for generation, not retrieval).

PubMedQA ships only the abstract passages, which are the right size for retrieval but thin for
answering. This resolves each doc's PMID to a PMCID and fetches the article XML from PMC,
writing the body text into ``metadata.full_text``. ``text`` is NEVER modified, so the retrieval
index and the corpus embeddings stay byte-identical — a full-text run and an abstract run differ
only in what the generator reads.

Only the PMC **open-access** subset is fetchable, so expect partial coverage: every miss is
reported (per-PMID warnings, a summary line, and a ``*.fulltext_coverage.json`` sidecar) and its
doc keeps ``full_text_source: abstract``.

    python3 scripts/fetch_pubmed_fulltext.py examples/data/pubmed_qa_campaign/corpus.json

Options: ``--cache-dir`` (default ``.cache/pmc``, raw XML — re-runs are offline), ``--email``,
``--limit``. Set ``NCBI_API_KEY`` for 10 req/s instead of 3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_TOOL = "retrieval-evaluator"
# NCBI: 3 requests/second without a key, 10 with one.
_SLEEP_NO_KEY = 0.34
_SLEEP_WITH_KEY = 0.11
# Sections that never answer a question (and bloat the prompt).
_SKIP_SECTIONS = ("reference", "acknowledg", "author contribution", "competing interest",
                  "funding", "supplementary", "conflict")


def _sleep_s() -> float:
    return _SLEEP_WITH_KEY if os.environ.get("NCBI_API_KEY") else _SLEEP_NO_KEY


def _get(url: str, params: dict, *, tries: int = 4) -> Optional[str]:
    """GET with backoff on 429/5xx. Returns the body, or None when it stays unavailable."""
    import requests

    if os.environ.get("NCBI_API_KEY"):
        params = {**params, "api_key": os.environ["NCBI_API_KEY"]}
    for attempt in range(tries):
        try:
            resp = requests.get(url, params={**params, "tool": _TOOL}, timeout=60)
        except Exception as exc:  # noqa: BLE001 - network flake: retry, then give up
            print(f"  ! request failed ({exc}); retrying", file=sys.stderr)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.text
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        return None
    return None


def pmid_to_pmcid(pmids: List[str], email: Optional[str]) -> Dict[str, str]:
    """Map PMID → PMCID via the ID Converter (batched; missing ids are simply absent)."""
    out: Dict[str, str] = {}
    for start in range(0, len(pmids), 200):
        batch = pmids[start:start + 200]
        params = {"ids": ",".join(batch), "format": "json"}
        if email:
            params["email"] = email
        body = _get(_IDCONV, params)
        time.sleep(_sleep_s())
        if not body:
            print(f"  ! id conversion failed for {len(batch)} ids", file=sys.stderr)
            continue
        try:
            records = json.loads(body).get("records", [])
        except json.JSONDecodeError:
            print("  ! id conversion returned non-JSON", file=sys.stderr)
            continue
        for rec in records:
            if rec.get("pmcid") and rec.get("pmid"):
                out[str(rec["pmid"])] = str(rec["pmcid"])
    return out


def jats_to_text(xml_text: str) -> str:
    """JATS article XML → readable body text: section titles + paragraphs.

    Drops references / acknowledgements / funding / supplementary sections and every float
    (tables, figures) — none of them answer a question, all of them cost prompt budget.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    body = root.find(".//body")
    if body is None:
        return ""
    parts: List[str] = []

    def walk(node: ET.Element) -> None:
        for child in node:
            tag = child.tag.lower()
            if tag in ("table-wrap", "fig", "graphic", "media", "supplementary-material"):
                continue
            if tag == "sec":
                title_el = child.find("title")
                title = "".join(title_el.itertext()).strip() if title_el is not None else ""
                if any(s in title.lower() for s in _SKIP_SECTIONS):
                    continue
                if title:
                    parts.append(f"## {title}")
                walk(child)
            elif tag == "p":
                text = " ".join("".join(child.itertext()).split())
                if text:
                    parts.append(text)
            else:
                walk(child)

    walk(body)
    return "\n\n".join(parts).strip()


def fetch_articles(pmcids: List[str], cache_dir: Path, batch: int = 40) -> Dict[str, str]:
    """Article XML per PMCID, batched (efetch takes a comma-separated id list).

    One request per ``batch`` articles instead of one per article — the difference between
    seconds and minutes when scanning a whole dataset. Cached ids are served from disk and
    never requested; newly fetched ones are split back out and cached individually.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}
    missing: List[str] = []
    for pmcid in pmcids:
        path = cache_dir / f"{pmcid}.xml"
        if path.exists():
            out[pmcid] = path.read_text(errors="replace")
        else:
            missing.append(pmcid)

    for start in range(0, len(missing), batch):
        chunk = missing[start:start + batch]
        body = _get(_EFETCH, {"db": "pmc", "id": ",".join(chunk), "retmode": "xml"})
        time.sleep(_sleep_s())
        if not body:
            continue
        for pmcid, xml_text in _split_articles(body, chunk).items():
            (cache_dir / f"{pmcid}.xml").write_text(xml_text)
            out[pmcid] = xml_text
    return out


def _split_articles(bundle_xml: str, expected: List[str]) -> Dict[str, str]:
    """Split an efetch multi-article response into ``{PMCID: article xml}``.

    The PMCID comes from the article's own ``<article-id pub-id-type="pmc">`` — position in
    the response is NOT reliable (PMC drops ids it will not serve, which is exactly the
    not-open-access case we are measuring).
    """
    try:
        root = ET.fromstring(bundle_xml)
    except ET.ParseError:
        return {}
    articles = root.findall(".//article") or ([root] if root.tag == "article" else [])
    out: Dict[str, str] = {}
    for art in articles:
        pmcid = None
        for aid in art.findall(".//article-id"):
            # PMC labels it `pmcid` ("PMC1828111"); `pmc` appears in other JATS sources.
            if aid.get("pub-id-type") in ("pmcid", "pmc") and (aid.text or "").strip():
                digits = (aid.text or "").strip().removeprefix("PMC")
                pmcid = f"PMC{digits}"
                break
        if pmcid is None and len(expected) == 1:
            pmcid = expected[0]
        if pmcid:
            out[pmcid] = ET.tostring(art, encoding="unicode")
    return out


def has_full_text(xml_text: str) -> bool:
    """True when the article XML actually carries a body (i.e. PMC will serve the full text)."""
    return bool(jats_to_text(xml_text))


def fetch_article(pmcid: str, cache_dir: Path) -> Tuple[str, bool]:
    """Article XML for a PMCID, from the on-disk cache when present. Returns (xml, from_cache)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pmcid}.xml"
    if path.exists():
        return path.read_text(errors="replace"), True
    body = _get(_EFETCH, {"db": "pmc", "id": pmcid, "retmode": "xml"})
    time.sleep(_sleep_s())
    if not body:
        return "", False
    path.write_text(body)
    return body, False


def enrich(corpus_path: Path, cache_dir: Path, email: Optional[str],
           limit: Optional[int]) -> int:
    docs = json.loads(corpus_path.read_text())
    if limit:
        docs = docs[:limit]
    pmids = [str(d.get("doc_id", "")) for d in docs if d.get("doc_id")]
    print(f"resolving {len(pmids)} PMIDs → PMCIDs …")
    pmcids = pmid_to_pmcid(pmids, email)
    print(f"  {len(pmcids)}/{len(pmids)} have a PMCID")

    hits, misses, empty = 0, [], []
    for i, doc in enumerate(docs, 1):
        pmid = str(doc.get("doc_id", ""))
        meta = dict(doc.get("metadata") or {})
        pmcid = pmcids.get(pmid)
        if not pmcid:
            misses.append(pmid)
            meta["full_text_source"] = "abstract"
            doc["metadata"] = meta
            continue
        xml_text, cached = fetch_article(pmcid, cache_dir)
        text = jats_to_text(xml_text) if xml_text else ""
        if text:
            meta.update({"full_text": text, "full_text_source": "pmc", "pmcid": pmcid})
            hits += 1
        else:
            # A PMCID that is not in the open-access subset returns metadata without a body.
            empty.append(pmid)
            meta["full_text_source"] = "abstract"
            meta["pmcid"] = pmcid
        doc["metadata"] = meta
        if i % 25 == 0:
            print(f"  … {i}/{len(docs)} docs ({hits} with full text)")

    corpus_path.write_text(json.dumps(docs, indent=1))

    not_found = misses + empty
    for pmid in not_found[:10]:
        why = "no PMCID" if pmid in misses else "not in the PMC open-access subset"
        print(f"  ! PMID {pmid}: full article NOT found ({why}) — falling back to the abstract",
              file=sys.stderr)
    if len(not_found) > 10:
        print(f"  ! … and {len(not_found) - 10} more without full text", file=sys.stderr)

    pct = 100.0 * hits / len(docs) if docs else 0.0
    print(f"\nfull text: {hits}/{len(docs)} ({pct:.0f}%) from PMC; "
          f"{len(not_found)} abstract-only ({len(misses)} no PMCID, {len(empty)} not OA)")
    print(f"wrote {corpus_path} (text unchanged — retrieval and embeddings are unaffected)")

    sidecar = corpus_path.with_suffix(".fulltext_coverage.json")
    sidecar.write_text(json.dumps({
        "docs": len(docs), "full_text": hits, "coverage": round(pct / 100.0, 4),
        "no_pmcid": misses, "not_open_access": empty,
    }, indent=1))
    print(f"coverage detail → {sidecar}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="corpus.json to enrich in place")
    ap.add_argument("--cache-dir", default=".cache/pmc", help="raw XML cache (default .cache/pmc)")
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL"),
                    help="contact address for NCBI (courtesy; also $NCBI_EMAIL)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N docs (smoke test)")
    args = ap.parse_args()
    path = Path(args.corpus)
    if not path.exists():
        print(f"corpus not found: {path}", file=sys.stderr)
        return 2
    return enrich(path, Path(args.cache_dir), args.email, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
