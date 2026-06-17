"""``evaluator replay`` — re-run a single query id through the full graph with a per-node trace.

Item-level diagnostics (Roadmap 2d): per-item seeding already makes one item deterministic,
so replaying one query id reproduces exactly what it did in a full run. This command keeps the
corpus whole, slices the query side to the one id, dumps every node's artifacts (the existing
``EVALUATOR_DUMP_ARTIFACTS`` hook) with debug logging on, and prints the per-node trace.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional


def parse_replay_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluator replay",
        description="Re-run one query id through the full graph and print its per-node trace.",
    )
    parser.add_argument("--config", required=True, help="Path to the evaluation config YAML")
    parser.add_argument(
        "--query-id", "--query_id", dest="query_id", required=True,
        help="The query/sample id to replay (sliced from the dataset; corpus stays whole)",
    )
    parser.add_argument(
        "--dump-dir", "--dump_dir", dest="dump_dir", default=None,
        help="Directory for per-node artifact dumps (default: replay_dumps/<query-id>)",
    )
    return parser.parse_args(args)


def _print_trace(dump_dir: Path, node_order: List[str], query_id: str) -> None:
    """Print the dumped node artifacts grouped by node, in execution order."""
    print(f"\n=== Replay trace for query '{query_id}' ===")
    seen = set()
    for node_id in node_order:
        files = sorted(dump_dir.glob(f"{node_id}.*.jsonl"))
        if not files:
            continue
        seen.add(node_id)
        print(f"\n[{node_id}]")
        for path in files:
            artifact = path.name[len(node_id) + 1: -len(".jsonl")]
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            value = rows[0].get("value") if len(rows) == 1 else [r.get("value") for r in rows]
            print(f"  {artifact}: {value!r}")
    if not seen:
        print("  (no artifacts dumped — the graph produced nothing for this id)")


def run_replay(args: argparse.Namespace) -> int:
    from evaluator.config import EvaluationConfig
    from evaluator.evaluation.runner import run_evaluation_from_config
    from evaluator.logging_config import setup_logging
    from evaluator.pipeline import build_graph_for_config
    from evaluator.storage.cache import CacheManager

    if not Path(args.config).exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        return 1

    config = EvaluationConfig.from_yaml(args.config, validate=True)
    setup_logging(
        experiment_name=f"replay_{args.query_id}",
        log_dir=config.logging.log_dir,
        console_level=logging.DEBUG,  # replay is a debug tool — show every node
        file_level=logging.DEBUG,
        verbosity=config.logging.verbosity,
    )

    graph = build_graph_for_config(config)
    node_order = [n.id for level in graph.topological_levels() for n in level]
    dump_dir = Path(args.dump_dir or os.path.join("replay_dumps", str(args.query_id)))

    # Drive the existing artifact-dump hook over every node, for this run only.
    prev_targets = os.environ.get("EVALUATOR_DUMP_ARTIFACTS")
    prev_dir = os.environ.get("EVALUATOR_DUMP_DIR")
    os.environ["EVALUATOR_DUMP_ARTIFACTS"] = ",".join(node_order)
    os.environ["EVALUATOR_DUMP_DIR"] = str(dump_dir)
    cache_manager = CacheManager(
        cache_dir=config.cache.cache_dir, enabled=config.cache.enabled
    )
    try:
        run_evaluation_from_config(
            config, cache_manager=cache_manager, query_ids={args.query_id}
        )
    except Exception as exc:  # surface the failure but still show whatever was dumped
        print(f"Error during replay: {exc}", file=sys.stderr)
        _print_trace(dump_dir, node_order, args.query_id)
        return 1
    finally:
        _restore_env("EVALUATOR_DUMP_ARTIFACTS", prev_targets)
        _restore_env("EVALUATOR_DUMP_DIR", prev_dir)

    _print_trace(dump_dir, node_order, args.query_id)
    print(f"\nArtifact dumps written to: {dump_dir}")
    return 0


def _restore_env(key: str, prev: Optional[str]) -> None:
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev


def main(args: Optional[List[str]] = None) -> int:
    return run_replay(parse_replay_args(args))


if __name__ == "__main__":
    sys.exit(main())
