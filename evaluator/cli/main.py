"""Unified ``evaluator`` CLI — one front door for the scattered entry points.

Subcommands (each delegates to the existing implementation; no logic lives here):

- ``run``           — run an evaluation (same flags as ``python -m evaluator.cli``)
- ``presets``       — list presets, or ``presets show <name>`` to dump one as YAML
- ``graph``         — print the execution DAG for ``--config``/``--preset``
- ``datasets``      — list registered dataset descriptors
- ``cache``         — ``status`` / ``clear [--type T]`` for the artifact cache
- ``leaderboard``   — query the leaderboard SQLite (top runs by metric)
- ``replay``        — re-run one query id through the full graph + per-node trace (cli/replay.py)
- ``compare``       — compare result JSON files (cli/compare.py)
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
  sweep          expand+run a parameter sweep (evaluator sweep --config sweep.yaml [--run])
  replay         re-run one query id with a per-node artifact trace (--config --query-id)
  compare        compare result JSON files
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


def _cmd_graph(rest: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="evaluator graph")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="config YAML path")
    group.add_argument("--preset", help="preset name (configs/<name>.yaml)")
    args = parser.parse_args(rest)

    from evaluator.cli.commands import _print_graph
    from evaluator.config.evaluation import EvaluationConfig

    if args.config:
        config = EvaluationConfig.from_yaml(args.config)
    else:
        from evaluator.config.model_presets import get_preset

        config = EvaluationConfig.from_dict(get_preset(args.preset, auto_devices=False))
    _print_graph(config)
    return 0


def _cmd_datasets(rest: List[str]) -> int:
    from evaluator.datasets.descriptor import get_descriptor, list_registered_datasets

    for dataset_id in list_registered_datasets():
        desc = get_descriptor(dataset_id)
        if desc is None:
            continue
        modes = ",".join(desc.compatible_pipeline_modes or [])
        domain = getattr(desc, "domain", "general")
        print(f"{dataset_id}\t{desc.dataset_type}\t{domain}\t{modes}\t{desc.description}")
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

    from evaluator.webapi.routers.leaderboard import leaderboard_rows

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


def _cmd_sweep(rest: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="evaluator sweep")
    parser.add_argument("--config", required=True, help="sweep spec (YAML/JSON)")
    parser.add_argument(
        "--run", action="store_true",
        help="run each expanded config (heavy); default just lists them (dry run)",
    )
    args = parser.parse_args(rest)

    from evaluator.analysis.sweep import combo_label, expand_sweep, load_sweep_spec

    spec = load_sweep_spec(args.config)
    points = expand_sweep(spec)
    name = spec.get("name", "sweep")
    print(f"sweep '{name}': {len(points)} configs (experiment_group={name!r})")
    for combo, cfg in points:
        print(f"  {combo_label(combo):<44} mode={cfg.model.pipeline_mode}  → {cfg.experiment_name}")
    if not args.run:
        print("\n(dry run — pass --run to execute each via the normal evaluation path)")
        return 0

    from evaluator.services.evaluation_service import run_evaluation

    for i, (_, cfg) in enumerate(points, 1):
        print(f"\n=== [{i}/{len(points)}] running {cfg.experiment_name} ===")
        run_evaluation(cfg)
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
        if command == "sweep":
            return _cmd_sweep(rest)
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
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
