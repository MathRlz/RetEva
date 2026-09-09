"""Regenerate the graph-parity locks:

- tests/data/graph_golden.json — order-insensitive preview-graph signatures
- tests/data/graph_order_golden.json — order-SENSITIVE run-graph signatures
  (unsorted bindings + input_aliases; binding order is runtime semantics)

Run this ONLY after an INTENTIONAL graph change, review the diff (`git diff tests/data/`),
and commit it with the change that caused it — an unexplained diff means the change altered
what runs.

    python3 scripts/regen_graph_golden.py
"""
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tests.graph_golden_util import graph_signature, run_order_signature  # noqa: E402


def config_paths():
    return sorted(
        glob.glob(os.path.join(_ROOT, "configs/*.yaml"))
        + glob.glob(os.path.join(_ROOT, "configs/apm_tests/**/*.yaml"), recursive=True)
        + glob.glob(os.path.join(_ROOT, "configs/examples/*.yaml"))
        + glob.glob(os.path.join(_ROOT, "tests/data/configs/*.yaml"))
    )


def main():
    configs = config_paths()
    for name, fn in (("graph_golden.json", graph_signature),
                     ("graph_order_golden.json", run_order_signature)):
        golden = {os.path.basename(f): fn(f) for f in configs}
        out = os.path.join(_ROOT, "tests/data", name)
        json.dump(golden, open(out, "w"), indent=1, sort_keys=True)
        print(f"wrote {len(golden)} configs "
              f"({sum(len(v) for v in golden.values())} nodes) -> {out}")
    print("review with: git diff tests/data/")


if __name__ == "__main__":
    main()
