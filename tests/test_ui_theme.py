"""V2 pins: the light/dark theme contract.

Charts read their colors from CSS custom properties at plot time, so the tokens and the
no-flash stamp are load-bearing: if the stamp moves after the stylesheet the first paint
flashes the wrong theme, and if a template re-introduces a hex literal that chart stops
following the theme.
"""

import re
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "evaluator/webapi/static"
_TEMPLATES = Path(__file__).resolve().parents[1] / "evaluator/webapi/templates"


def test_theme_stamp_runs_before_the_stylesheet(client):
    """Anti-FOUC: the attribute must be set before app.css loads, or the first paint
    renders in the wrong theme and visibly flips."""
    html = client.get("/ui/results").text
    assert 0 < html.find("evaluator-theme") < html.find("app.css")
    assert 'id="theme-toggle"' in html


def test_both_token_sets_ship_and_differ():
    css = (_STATIC / "app.css").read_text()
    # dark is the base, light overrides it under BOTH scopes (OS + explicit toggle)
    assert "--chart-cat-1: #3987e5" in css          # dark, validated on #111827
    assert "--chart-cat-1: #2a78d6" in css          # light, validated on #fcfcfb
    assert '[data-theme="light"]' in css
    assert "prefers-color-scheme: light" in css
    # an explicit stamp must be able to beat the OS preference in both directions
    assert ':root:where(:not([data-theme="dark"]))' in css


def test_chart_templates_carry_no_color_literals():
    """Colors belong to the token layer; a hex here would freeze that chart in one theme."""
    for name in ("_leaderboard.html", "_pareto.html", "_run_detail.html"):
        body = (_TEMPLATES / name).read_text()
        assert not re.findall(r"color:\s*'#[0-9a-fA-F]{6}'", body), name


def test_charts_js_resolves_tokens_and_rerenders():
    js = (_STATIC / "charts.js").read_text()
    assert "getPropertyValue('--chart-cat-'" in js or "--chart-cat-" in js
    # a theme switch must re-read tokens and redraw with react (keeps zoom/pan)
    assert "evaluator:theme" in js and "Plotly.react" in js
    # and must not leak the previous plot on an htmx swap
    assert "Plotly.purge" in js
