"""JATS → text extraction for the PMC full-text enricher (no network).

The fetcher's only non-trivial logic is the XML→prose step: keep section titles and paragraphs,
drop the sections and floats that cost prompt budget without answering anything.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_pubmed_fulltext.py"
_spec = importlib.util.spec_from_file_location("fetch_pubmed_fulltext", _SCRIPT)
ftm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ftm)


_ARTICLE = """<?xml version="1.0"?>
<article>
  <front><article-meta><title-group><article-title>T</article-title></title-group></article-meta></front>
  <body>
    <sec><title>Introduction</title>
      <p>Aspirin has been studied <italic>extensively</italic> in this cohort.</p>
    </sec>
    <sec><title>Results</title>
      <p>The event rate fell by 12%.</p>
      <table-wrap><caption><p>Table 1 caption noise</p></caption></table-wrap>
      <fig><caption><p>Figure noise</p></caption></fig>
    </sec>
    <sec><title>Acknowledgements</title><p>We thank everyone.</p></sec>
    <sec><title>References</title><p>[1] Someone et al.</p></sec>
  </body>
</article>
"""


def test_keeps_body_sections_and_drops_noise():
    text = ftm.jats_to_text(_ARTICLE)
    assert "## Introduction" in text and "## Results" in text
    assert "Aspirin has been studied extensively" in text   # inline markup flattened
    assert "The event rate fell by 12%." in text
    for dropped in ("Table 1 caption", "Figure noise", "We thank everyone", "Someone et al"):
        assert dropped not in text


def test_missing_body_and_malformed_xml_are_empty_not_fatal():
    assert ftm.jats_to_text("<article><front/></article>") == ""
    assert ftm.jats_to_text("not xml at all") == ""


def test_split_articles_keys_by_the_articles_own_pmcid():
    """PMC labels the id `pmcid`, and drops ids it will not serve — so position in the
    response is not an id. Getting this wrong silently reported zero coverage."""
    bundle = (
        '<pmc-articleset>'
        '<article><front><article-meta>'
        '<article-id pub-id-type="pmcid">PMC111</article-id>'
        '<article-id pub-id-type="pmid">999</article-id>'
        '</article-meta></front><body><sec><title>R</title><p>text one</p></sec></body></article>'
        '<article><front><article-meta>'
        '<article-id pub-id-type="pmc">222</article-id>'   # bare digits, other JATS sources
        '</article-meta></front><body><p>text two</p></body></article>'
        '</pmc-articleset>'
    )
    split = ftm._split_articles(bundle, ["PMC111", "PMC222", "PMC333"])
    assert set(split) == {"PMC111", "PMC222"}          # PMC333 was not served
    assert ftm.has_full_text(split["PMC111"]) is True


def test_has_full_text_is_false_without_a_body():
    # PMC serves metadata for articles whose publisher forbids full-text download.
    assert ftm.has_full_text(
        '<article><front><article-meta/></front></article>'
    ) is False


def test_fetch_article_uses_the_cache(tmp_path):
    # A cached PMCID must not hit the network: no requests import needed for the hit path.
    (tmp_path / "PMC123.xml").write_text(_ARTICLE)
    xml, cached = ftm.fetch_article("PMC123", tmp_path)
    assert cached is True
    assert "Results" in ftm.jats_to_text(xml)
