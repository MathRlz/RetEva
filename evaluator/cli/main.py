"""Unified ``evaluator`` CLI — one front door for the scattered entry points.

Subcommands (each delegates to the existing implementation; no logic lives here):

- ``run``           — run an evaluation (same flags as ``python -m evaluator.cli``)
- ``presets``       — list presets, or ``presets show <name>`` to dump one as YAML
- ``graph``         — print the execution DAG for ``--config``/``--preset``
- ``datasets``      — list registered dataset descriptors
- ``cache``         — ``status`` / ``clear [--type T]`` for the artifact cache
- ``leaderboard``   — query the leaderboard SQLite (top runs by metric)
- ``replay``        — re-run one query id through the full graph + per-node trace (cli/replay.py)
- ``compare``       — compare result JSON files, or 2+ variant/run dirs (cli/compare.py)
- ``export``        — export results to CSV/Excel/LaTeX (cli/export.py)
- ``branch-report`` — thesis artifacts from a branched run (analysis/branch_report.py)
- ``benchmark``     — model micro-benchmarks (cli/benchmark.py)
- ``gpu``           — GPU status

Installed as the ``evaluator`` console script (see pyproject.toml).
"""

from __future__ import annotations

import sys
from typing import List, Optional

_USAGE = """\
usage: evaluator <command> [args]

commands:
  run            run an evaluation (evaluator run --config configs/foo.yaml)
  presets        list config presets; `presets show <name>` dumps the YAML
  graph          print the execution DAG (--config <path> | --preset <name>)
  datasets       list registered dataset descriptors
  cache          cache status|clear [--cache-dir DIR] [--type TYPE]
  leaderboard    top runs by metric [--metric M] [--limit N] [--output-dir DIR]
  replay         re-run one query id with a per-node artifact trace (--config --query-id)
  compare        compare result JSON files, or 2+ variant/run dirs (baseline-vs-each)
  export         export results (csv/excel/latex)
  branch-report  thesis artifacts from a branched run's results JSON
  benchmark      model micro-benchmarks
  gpu            GPU status

`evaluator <command> --help` shows the command's own flags.
"""


def _cmd_run(rest: List[str]) -> int:
    from evaluator.cli import parse_args, run_evaluation

    run_evaluation(parse_args(rest))
    return 0


def _cmd_presets(rest: List[str]) -> int:
    from evaluator.config.model_presets import get_preset, list_presets

    if rest and rest[0] == "show":
        if len(rest) < 2:
            print("usage: evaluator presets show <name>", file=sys.stderr)
            return 2
        import yaml

        print(yaml.safe_dump(get_preset(rest[1], auto_devices=False), sort_keys=False))
        return 0
    for name in list_presets():
        print(name)
    return 0


def _edge_line(e: dict) -> str:
    # `output` is omitted when it equals `input` (the same-name shorthand).
    out = f"output: {e['output']}, " if e.get("output", e["input"]) != e["input"] else ""
    return f"- {{from: {e['from']}, {out}to: {e['to']}, input: {e['input']}}}"


def write_edges_block(path: str, edges: List[dict]) -> int:
    """Rewrite the config's ``graph.edges:`` block in place (B5) — text-level so every
    comment survives (no yaml.dump round-trip). Replaces an existing ``edges:`` child (its
    port-level entries; hand-added ordering-only ``{from, to}`` entries are kept and
    re-appended) or inserts one after the ``nodes:`` child. The rewritten file must still
    parse as YAML or the original text is restored."""
    import re
    from pathlib import Path

    import yaml

    p = Path(path)
    original = p.read_text()
    lines = original.splitlines()
    g = next(
        (i for i, ln in enumerate(lines) if re.match(r"graph:\s*(#.*)?$", ln)), None
    )
    if g is None:
        raise SystemExit(f"{path}: no top-level graph: block")

    def child_span(start_idx: int, indent: str):
        """(end_idx, kept_ordering_lines) of the child block starting after start_idx."""
        j, kept = start_idx + 1, []
        while j < len(lines):
            ln = lines[j]
            stripped = ln.strip()
            if not stripped:
                break
            ind = len(ln) - len(ln.lstrip())
            in_block = ind > len(indent) or (
                ind >= len(indent) and (stripped.startswith("- ") or stripped.startswith("#"))
            )
            if not in_block:
                break
            if stripped.startswith("- ") and "output:" not in stripped:
                kept.append(ln)  # ordering-only edge — not derivable, keep it
            j += 1
        return j, kept

    edges_at = nodes_at = None
    for i in range(g + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln.startswith(" "):
            break  # left the graph block
        if re.match(r"(\s+)edges:\s*(#.*)?$", ln):
            edges_at = i
        elif re.match(r"(\s+)nodes:\s*(#.*)?$", ln):
            nodes_at = i
    if edges_at is not None:
        indent = lines[edges_at][: len(lines[edges_at]) - len(lines[edges_at].lstrip())]
        end, kept = child_span(edges_at, indent)
        block = [f"{indent}edges:"] + [f"{indent}{_edge_line(e)}" for e in edges] + kept
        lines[edges_at:end] = block
    elif nodes_at is not None:
        indent = lines[nodes_at][: len(lines[nodes_at]) - len(lines[nodes_at].lstrip())]
        end, _ = child_span(nodes_at, indent)
        block = [f"{indent}edges:"] + [f"{indent}{_edge_line(e)}" for e in edges]
        lines[end:end] = block
    else:
        raise SystemExit(f"{path}: graph: block has no nodes: child")

    p.write_text("\n".join(lines) + "\n")
    try:
        yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        p.write_text(original)
        raise SystemExit(f"{path}: rewrite produced invalid YAML ({exc}); restored")
    return len(edges)


def _cmd_graph(rest: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="evaluator graph")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="config YAML path")
    group.add_argument("--preset", help="preset name (configs/<name>.yaml)")
    parser.add_argument(
        "--emit-edges", action="store_true",
        help="print the explicit port-level `edges:` block for the config's graph "
             "(paste it under graph: — the authoring assistant for explicit wiring)",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="with --emit-edges --config: rewrite the file's graph.edges block in place "
             "(comment-preserving; keeps ordering-only edges; inserts after graph.nodes "
             "when absent)",
    )
    parser.add_argument(
        "--emit-metrics", action="store_true",
        help="print every metric the config's graph can satisfy as a ready-to-paste "
             "top-level `metrics:` allowlist block (opt-in metrics commented) — the "
             "authoring assistant for metric preregistration",
    )
    parser.add_argument(
        "--format", choices=["text", "dot"], default="text",
        help="dot: emit the DAG as Graphviz DOT (render with `dot -Tpdf`/-Tsvg for a "
             "publication figure); default: the textual level/port listing",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="with --format dot: write to this file instead of stdout",
    )
    args = parser.parse_args(rest)
    if args.write and not (args.emit_edges and args.config):
        parser.error("--write requires --emit-edges and --config")

    from evaluator.cli.commands import _print_graph
    from evaluator.config.evaluation import EvaluationConfig

    if args.config:
        config = EvaluationConfig.from_yaml(args.config, validate=not args.emit_edges)
    else:
        from evaluator.config.model_presets import get_preset

        config = EvaluationConfig.from_dict(get_preset(args.preset, auto_devices=False))
    if args.emit_edges:
        from evaluator.pipeline.graph.wiring import emit_edges

        nodes = (config.graph_override or {}).get("nodes") or []
        if not nodes:
            print("config has no graph.nodes — nothing to wire")
            return 1
        edges = emit_edges(nodes)
        if args.write:
            n = write_edges_block(args.config, edges)
            print(f"wrote {n} port-level edges to {args.config}")
            return 0
        print("  edges:")
        for e in edges:
            print(f"  {_edge_line(e)}")
        return 0
    if args.emit_metrics:
        from evaluator.evaluation.metric_registry import applicable_metrics
        from evaluator.pipeline.graph.modes import build_graph_for_config
        from evaluator.pipeline.introspection import effective_outputs

        graph = build_graph_for_config(config)
        produced: set = set()
        for n in graph.nodes:
            produced.update(effective_outputs(n.stage, n.params))
        print("metrics:")
        for spec in applicable_metrics(sorted(produced), include_opt_in=True):
            if spec.opt_in:
                print(f"# - {spec.name}   # opt-in: list it here or set "
                      f"query_correction.corrected_metrics")
            else:
                print(f"- {spec.name}")
        return 0
    if args.format == "dot":
        from evaluator.pipeline.graph.export import graph_to_dot
        from evaluator.pipeline.graph.modes import build_graph_for_config

        dot = graph_to_dot(
            build_graph_for_config(config), title=config.experiment_name
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(dot)
            print(f"wrote {args.output} (render: dot -Tpdf {args.output} -o fig.pdf)")
        else:
            print(dot, end="")
        return 0
    _print_graph(config)
    return 0


def _cmd_datasets(rest: List[str]) -> int:
    from evaluator.datasets.descriptor import get_descriptor, list_registered_datasets

    for dataset_id in list_registered_datasets():
        desc = get_descriptor(dataset_id)
        if desc is None:
            continue
        domain = getattr(desc, "domain", "general")
        print(f"{dataset_id}\t{desc.dataset_type}\t{domain}\t{desc.description}")
    return 0


def _cmd_cache(rest: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="evaluator cache")
    parser.add_argument("action", choices=["status", "clear", "prune"])
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--type", dest="cache_type", default=None,
                        help="clear only this cache type (e.g. embeddings)")
    parser.add_argument("--max-size-gb", type=float, default=0,
                        help="prune: evict least-recently-used entries until under this size")
    args = parser.parse_args(rest)

    from evaluator.storage.cache import CacheManager

    manager = CacheManager(cache_dir=args.cache_dir, enabled=True)
    if args.action == "status":
        stats = manager.get_cache_stats()
        print(f"cache dir: {args.cache_dir}")
        print(f"total: {stats.get('total_size_human', stats.get('total_size_bytes'))}")
        for category, info in sorted((stats.get("by_category") or {}).items()):
            print(f"  {category}: {info.get('size_human')} ({info.get('file_count')} files)")
        return 0
    if args.action == "prune":
        # L4: bound unbounded cache growth — drop orphaned manifest rows, then (if a budget
        # is given) LRU-evict until under --max-size-gb. Uses the existing eviction logic.
        orphaned = manager.compact_manifest()
        print(f"removed {orphaned} orphaned manifest entries")
        if args.max_size_gb and args.max_size_gb > 0:
            before = manager.get_cache_size_bytes()
            manager.max_size_gb = args.max_size_gb
            manager._enforce_size_limit()
            after = manager.get_cache_size_bytes()
            print(
                f"size eviction: {manager._human_readable_size(before)} -> "
                f"{manager._human_readable_size(after)} (limit {args.max_size_gb} GB)"
            )
        return 0
    if args.cache_type:
        manager.clear_type(args.cache_type)
        print(f"cleared cache type: {args.cache_type}")
    else:
        manager.clear_all()
        print(f"cleared all caches under {args.cache_dir}")
    return 0


def _cmd_leaderboard(rest: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="evaluator leaderboard")
    parser.add_argument("--metric", default="MRR")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--pipeline-mode", default=None)
    parser.add_argument("--name", default=None, help="experiment name filter")
    args = parser.parse_args(rest)

    from evaluator.analysis.leaderboard_views import leaderboard_rows

    rows = leaderboard_rows(
        metric=args.metric,
        limit=args.limit,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        pipeline_mode=args.pipeline_mode,
        name=args.name,
    )
    if not rows:
        print(f"no runs with metric {args.metric!r} in {args.output_dir}")
        return 0
    print(f"{'run':>4}  {'experiment':<32} {'dataset':<16} {'mode':<22} {args.metric}")
    for r in rows:
        value = r.get("metric_value")
        value_s = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        print(
            f"{r.get('run_id'):>4}  {str(r.get('experiment_name')):<32} "
            f"{str(r.get('dataset_name')):<16} {str(r.get('pipeline_mode')):<22} {value_s}"
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    command, rest = args[0], args[1:]
    try:
        if command == "run":
            return _cmd_run(rest)
        if command == "presets":
            return _cmd_presets(rest)
        if command == "graph":
            return _cmd_graph(rest)
        if command == "datasets":
            return _cmd_datasets(rest)
        if command == "cache":
            return _cmd_cache(rest)
        if command == "leaderboard":
            return _cmd_leaderboard(rest)
        if command == "replay":
            from evaluator.cli.replay import main as replay_main

            return replay_main(rest)
        if command == "compare":
            from evaluator.cli.compare import main as compare_main

            return compare_main(rest)
        if command == "export":
            from evaluator.cli.export import main as export_main

            return export_main(rest)
        if command == "branch-report":
            from evaluator.analysis.branch_report import main as report_main

            return report_main(rest)
        if command == "benchmark":
            from evaluator.cli.benchmark import main as benchmark_main

            benchmark_main(rest)
            return 0
        if command == "gpu":
            from evaluator.cli.gpu_status import show_gpu_status

            show_gpu_status()
            return 0
        print(f"unknown command: {command}\n\n{_USAGE}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — CLI surface: report and exit non-zero
        # The headline first (what most failures need), then the traceback — a bare `str(e)`
        # makes a one-line TypeError ("object of type 'NoneType' has no len()") unlocatable.
        # Also logged, so a run's log file keeps the frames when the console is piped away.
        import traceback

        from evaluator.logging_config import get_logger

        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        get_logger("evaluator").error("command %r failed", command, exc_info=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
