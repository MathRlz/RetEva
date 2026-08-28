"""CLI operational-flag application + error-tail head+tail excerpt + cache prune."""

from evaluator.cli.parser import apply_args_to_config, parse_args
from evaluator.config.evaluation import EvaluationConfig
from evaluator.webapi.jobs import _head_tail_excerpt


def test_cli_streaming_and_cpu_executor_flags():
    args = parse_args([
        "--streaming_window_size", "64",
        "--cpu_stage_executor", "process", "--cpu_stage_workers", "4",
    ])
    cfg = EvaluationConfig()
    apply_args_to_config(args, cfg)
    assert cfg.streaming.window_size == 64
    assert cfg.cpu_stage_executor == "process" and cfg.cpu_stage_workers == 4


def test_cli_streaming_and_cpu_executor_defaults_unset():
    cfg = EvaluationConfig()
    apply_args_to_config(parse_args([]), cfg)
    assert cfg.streaming.window_size is None       # default whole-dataset run
    assert cfg.cpu_stage_executor == "sync"        # default serial CPU stages


def test_error_excerpt_keeps_head_and_tail():
    lines = [f"line{i}" for i in range(500)]
    out = _head_tail_excerpt(lines, 50)
    assert "line0" in out  # head preserved (e.g. the OOM root cause)
    assert "line499" in out  # tail preserved
    assert "omitted" in out


def test_error_excerpt_short_log_verbatim():
    lines = ["a", "b", "c"]
    assert _head_tail_excerpt(lines, 50) == "a\nb\nc"


def test_cache_prune_command(tmp_path):
    # L4: `evaluator cache prune` runs (compact orphans + optional LRU size eviction).
    from evaluator.cli.main import _cmd_cache

    rc = _cmd_cache(["prune", "--cache-dir", str(tmp_path / "c"), "--max-size-gb", "1"])
    assert rc == 0
