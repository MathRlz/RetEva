"""B1/F14: the analysis exporters share ONE `_escape_latex` / `load_results` (no divergent
copies). Guards against a future re-fork of these helpers.
"""

import json

from evaluator.analysis import _common
from evaluator.analysis import branch_report, export, significance


def test_escape_latex_is_single_shared_impl():
    # branch_report + export re-export the same canonical function object
    assert branch_report._escape_latex is _common._escape_latex
    assert export._escape_latex is _common._escape_latex


def test_load_results_is_single_shared_impl():
    # significance + export (+ the cli importer) all bind the canonical loader
    assert significance.load_results is _common.load_results
    assert export.load_results is _common.load_results
    from evaluator.cli.export import load_results as cli_load_results

    assert cli_load_results is _common.load_results


def test_escape_latex_handles_special_chars():
    out = _common._escape_latex(r"a_b & 50% $x #1 {c} ~d ^e \z @f")
    for raw in ("&", "%", "$", "#", "~", "^"):
        assert raw not in out.replace(r"\&", "").replace(r"\%", "").replace(
            r"\$", ""
        ).replace(r"\#", "")
    assert r"\_" in out
    # backslash is replaced first; its braces are then re-escaped (matches both legacy copies)
    assert r"\textbackslash" in out
    assert "{@}" in out


def test_load_results_roundtrip(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"MRR": 0.5}), encoding="utf-8")
    assert _common.load_results(p) == {"MRR": 0.5}
    assert _common.load_results(str(p)) == {"MRR": 0.5}
