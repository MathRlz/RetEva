"""Graph-parity golden lock (Phase 2b): every config builds the exact StageGraph frozen in
tests/data/graph_golden.json. The golden was captured from the mode+features configs, so the
explicit-graph migration is parity-safe iff this stays green — a migrated config must build the
identical graph (node ids, kinds, deps, params, bindings) its template form did.
"""
import glob
import json
import os

import pytest

from tests.graph_golden_util import graph_signature

_ROOT = os.path.dirname(os.path.dirname(__file__))
_GOLDEN = json.load(open(os.path.join(_ROOT, "tests/data/graph_golden.json")))
# every repo config (top-level + apm_tests + examples + campaign); user_uploads are runtime artifacts.
_CONFIGS = sorted(
    glob.glob(os.path.join(_ROOT, "configs/*.yaml"))
    + glob.glob(os.path.join(_ROOT, "configs/apm_tests/**/*.yaml"), recursive=True)
    + glob.glob(os.path.join(_ROOT, "configs/examples/*.yaml"))
    + glob.glob(os.path.join(_ROOT, "configs/campaign/*.yaml"))
)


def test_golden_covers_every_config():
    assert {os.path.basename(f) for f in _CONFIGS} == set(_GOLDEN)


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: os.path.basename(p))
def test_config_builds_golden_graph(path):
    assert graph_signature(path) == _GOLDEN[os.path.basename(path)]


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: os.path.basename(p))
def test_config_builds_golden_run_order(path):
    """E1 (explicit-edges migration): the RUN graph's binding ORDER + OneOf aliases are frozen —
    the runtime reads newest-producer-first and by alias priority, so order is semantics the
    sorted golden above cannot see. A wrong edge→binding reconstruction fails HERE."""
    _ORDER = json.load(open(os.path.join(_ROOT, "tests/data/graph_order_golden.json")))
    from tests.graph_golden_util import run_order_signature

    assert run_order_signature(path) == _ORDER[os.path.basename(path)]
