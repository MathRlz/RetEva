"""Logging verbosity profiles + category gating (audit Batch F)."""

import logging

from evaluator.logging_config import (
    VERBOSITY_PROFILES,
    _category_of,
    _ConsoleCategoryFilter,
    setup_logging,
)


def _record(name, level):
    return logging.LogRecord(name, level, "f", 1, "msg", None, None)


def test_category_of_resolves_child_loggers_only():
    assert _category_of("evaluator.node") == "node"
    assert _category_of("evaluator.cache") == "cache"
    # arbitrary module loggers are lifecycle (not matched by a prefix collision)
    assert _category_of("evaluator.evaluation.executor.engine") == "lifecycle"
    assert _category_of("evaluator.storage.cache.manager") == "lifecycle"


def test_default_profile_hides_categories_but_shows_lifecycle_and_warnings():
    allowed = VERBOSITY_PROFILES["default"][0]
    f = _ConsoleCategoryFilter(allowed)
    assert f.filter(_record("evaluator", logging.INFO))  # lifecycle shows
    assert not f.filter(_record("evaluator.node", logging.INFO))  # node hidden
    assert not f.filter(_record("evaluator.cache", logging.INFO))
    assert f.filter(_record("evaluator.node", logging.WARNING))  # warnings always pass


def test_verbose_profile_shows_node_timing_not_diag():
    allowed = VERBOSITY_PROFILES["verbose"][0]
    f = _ConsoleCategoryFilter(allowed)
    assert f.filter(_record("evaluator.node", logging.INFO))
    assert f.filter(_record("evaluator.timing", logging.INFO))
    assert f.filter(_record("evaluator.cache", logging.INFO))
    assert not f.filter(_record("evaluator.diag", logging.INFO))  # diag is debug-only


def test_debug_profile_shows_every_category():
    allowed = VERBOSITY_PROFILES["debug"][0]
    f = _ConsoleCategoryFilter(allowed)
    for cat in ("node", "timing", "runtime", "cache", "diag"):
        assert f.filter(_record(f"evaluator.{cat}", logging.INFO))


def test_setup_logging_applies_profile_to_console_and_third_party(tmp_path):
    setup_logging(log_dir=str(tmp_path), verbosity="debug")
    root = logging.getLogger("evaluator")
    console = next(
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    assert console.level == logging.DEBUG
    assert logging.getLogger("transformers").level == logging.INFO  # debug → 3rd-party INFO

    # re-applying a quieter profile updates the live handler + third-party level
    setup_logging(log_dir=str(tmp_path), verbosity="default")
    assert console.level == logging.INFO
    assert logging.getLogger("transformers").level == logging.WARNING
    assert any(isinstance(flt, _ConsoleCategoryFilter) for flt in console.filters)


def test_cli_flags_resolve_verbosity(monkeypatch):
    from evaluator.cli.parser import apply_args_to_config, parse_args
    from evaluator.config.evaluation import EvaluationConfig

    monkeypatch.delenv("EVALUATOR_VERBOSITY", raising=False)
    cases = {tuple(): "default", ("-v",): "verbose", ("-vv",): "debug",
             ("--verbosity", "debug"): "debug"}
    for argv, expected in cases.items():
        cfg = EvaluationConfig()
        apply_args_to_config(parse_args(list(argv)), cfg)
        assert cfg.logging.verbosity == expected, argv

    # -vv overrides --verbosity; env is the fallback when no flag is given
    cfg = EvaluationConfig()
    apply_args_to_config(parse_args(["-vv", "--verbosity", "verbose"]), cfg)
    assert cfg.logging.verbosity == "debug"
    monkeypatch.setenv("EVALUATOR_VERBOSITY", "verbose")
    cfg = EvaluationConfig()
    apply_args_to_config(parse_args([]), cfg)
    assert cfg.logging.verbosity == "verbose"
