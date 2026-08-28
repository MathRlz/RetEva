"""The unified variant/run comparison tool: locating variant/run directories, and
baseline-vs-each metric deltas+significance / config diff / answer diffs across them
(`analysis/variant_compare.py`). Fixtures are built with the real `save_run_artifacts`
(Phase B) so this test exercises the actual on-disk shape, not a hand-guessed one.
"""

import json

import pytest

from evaluator.analysis.variant_compare import (
    VariantCompareError,
    answer_diffs,
    compare_paths,
    config_diff,
    format_variant_comparison_report,
    list_variant_dirs,
    load_variant_results,
    resolve_variant_dir,
)
from evaluator.config.evaluation import EvaluationConfig
from evaluator.evaluation.results_io import save_run_artifacts


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _make_run(tmp_path, name, mrr, generated_answer="iron deficiency"):
    """A real run directory (report/resolved-config/variants/run/...) via save_run_artifacts."""
    from tests.graph_test_helpers import explicit_graph

    out_dir = tmp_path / name
    cfg = EvaluationConfig()
    cfg.graph_override = explicit_graph([
        {"id": "dataset_source", "type": "dataset_source",
         "params": {"dataset": name}},  # differs per run → resolved config differs too
    ])
    cfg.output_dir = str(out_dir)
    cfg.experiment_name = name
    results = {
        "MRR": mrr,
        "report": {"traces": {"query_traces": [
            {
                "query_id": "q1", "generated_answer": generated_answer,
                "per_query_recall5": mrr,
            },
            {
                "query_id": "q2", "generated_answer": "same for both",
                "per_query_recall5": mrr,
            },
        ]}},
    }
    save_run_artifacts(results, cfg, _Logger())
    return out_dir


def _make_multi_variant_run(tmp_path, name, variant_mrrs):
    """One run whose graph compared several paths (e.g. multiple ASR models/audio encoders)
    — the real `{"variants": {node_id: report, ...}}` shape `run_graph` returns for that case
    (see `executor/run.py:_collect_final_result`), saved via the real `save_run_artifacts`."""
    from tests.graph_test_helpers import explicit_graph

    out_dir = tmp_path / name
    cfg = EvaluationConfig()
    cfg.graph_override = explicit_graph([
        {"id": "dataset_source", "type": "dataset_source", "params": {"dataset": name}},
    ])
    cfg.output_dir = str(out_dir)
    cfg.experiment_name = name
    results = {
        "variants": {
            vid: {
                "MRR": mrr,
                "report": {"traces": {"query_traces": [
                    {"query_id": "q1", "generated_answer": vid, "per_query_recall5": mrr},
                ]}},
            }
            for vid, mrr in variant_mrrs.items()
        },
    }
    save_run_artifacts(results, cfg, _Logger())
    return out_dir


def test_list_variant_dirs_finds_every_variant_in_a_multi_variant_run(tmp_path):
    run = _make_multi_variant_run(
        tmp_path, "asr_cmp", {"whisper": 0.5, "wav2vec2": 0.6, "conformer": 0.4}
    )
    dirs = list_variant_dirs(run)
    assert [d.name for d in dirs] == ["conformer", "wav2vec2", "whisper"]  # sorted
    # a leaf itself, and output_dir/variants, resolve the same way
    assert list_variant_dirs(run / "variants") == dirs
    assert list_variant_dirs(dirs[0]) == [dirs[0]]


def test_compare_single_multi_variant_run_auto_expands_to_all_variants(tmp_path):
    # The exact ask this is for: "I tested multiple ASR models / audio encoders in one run —
    # does compare work with that?" — pointing at the run alone should compare all of them.
    run = _make_multi_variant_run(
        tmp_path, "asr_cmp", {"whisper": 0.5, "wav2vec2": 0.6, "conformer": 0.4}
    )

    bundle = compare_paths([run])

    assert bundle["baseline"] == str(run / "variants" / "conformer")  # first, alphabetically
    compared = {c["path"] for c in bundle["comparisons"]}
    assert compared == {str(run / "variants" / "wav2vec2"), str(run / "variants" / "whisper")}
    whisper_cmp = next(c for c in bundle["comparisons"] if c["path"].endswith("whisper"))
    assert whisper_cmp["metrics"]["MRR"]["mean_a"] == 0.4  # conformer (baseline)
    assert whisper_cmp["metrics"]["MRR"]["mean_b"] == 0.5  # whisper
    # variants inside the SAME run share one resolved config — nothing to diff
    assert whisper_cmp["config_diff"] == []


def test_cli_compare_one_multi_variant_run_argument(tmp_path):
    from evaluator.cli.compare import parse_compare_args, run_compare

    run = _make_multi_variant_run(tmp_path, "asr_cmp", {"whisper": 0.5, "wav2vec2": 0.6})
    out = tmp_path / "out.json"

    args = parse_compare_args([str(run), "--format", "json", "-o", str(out)])
    assert run_compare(args) == 0
    written = json.loads(out.read_text())
    assert len(written["comparisons"]) == 1  # 2 variants: baseline + 1 comparison


def test_resolve_variant_dir_from_a_single_variant_output_dir(tmp_path):
    run = _make_run(tmp_path, "run_a", 0.5)
    assert resolve_variant_dir(run) == run / "variants" / "run"
    assert resolve_variant_dir(run / "variants" / "run") == run / "variants" / "run"


def test_resolve_variant_dir_ambiguous_multi_variant_raises(tmp_path):
    out_dir = tmp_path / "multi"
    for vid in ("dense", "sparse"):
        d = out_dir / "variants" / vid
        d.mkdir(parents=True)
        (d / "metrics.json").write_text("{}")
    with pytest.raises(VariantCompareError, match="dense"):
        resolve_variant_dir(out_dir)


def test_resolve_variant_dir_raises_when_nothing_found(tmp_path):
    with pytest.raises(VariantCompareError):
        resolve_variant_dir(tmp_path / "nope")


def test_load_variant_results_merges_metrics_and_answers(tmp_path):
    run = _make_run(tmp_path, "run_a", 0.7)
    results = load_variant_results(run)
    assert results["MRR"] == 0.7
    assert len(results["details"]) == 2
    assert results["details"][0]["query_id"] == "q1"


def test_config_diff_differs_across_runs_empty_within_one(tmp_path):
    run_a = _make_run(tmp_path, "run_a", 0.5)
    run_b = _make_run(tmp_path, "run_b", 0.6)
    diff = config_diff(run_a, run_b)
    assert diff  # the two runs used different text_emb_model_type
    assert any("run_a" in line for line in diff)
    assert config_diff(run_a, run_a) == []  # same run, same config: nothing to diff


def test_answer_diffs_only_reports_changed_ids(tmp_path):
    run_a = _make_run(tmp_path, "run_a", 0.5, generated_answer="iron deficiency")
    run_b = _make_run(tmp_path, "run_b", 0.6, generated_answer="anemia")
    diffs = answer_diffs(run_a, run_b)
    assert len(diffs) == 1  # q1 differs, q2 ("same for both") doesn't
    assert diffs[0]["query_id"] == "q1"
    assert diffs[0]["a"] == "iron deficiency" and diffs[0]["b"] == "anemia"


def test_compare_paths_baseline_vs_each(tmp_path):
    run_a = _make_run(tmp_path, "run_a", 0.5)
    run_b = _make_run(tmp_path, "run_b", 0.9)
    run_c = _make_run(tmp_path, "run_c", 0.3)

    bundle = compare_paths([run_a, run_b, run_c])

    assert bundle["baseline"] == str(run_a)
    assert [c["path"] for c in bundle["comparisons"]] == [str(run_b), str(run_c)]
    b_vs_a = bundle["comparisons"][0]["metrics"]["MRR"]
    assert b_vs_a["mean_a"] == 0.5 and b_vs_a["mean_b"] == 0.9
    assert b_vs_a["diff"] == pytest.approx(0.4)
    assert bundle["comparisons"][0]["config_diff"]
    report = format_variant_comparison_report(bundle)
    assert "run_a" in report and "run_b" in report and "run_c" in report


def test_compare_paths_needs_at_least_two():
    with pytest.raises(VariantCompareError):
        compare_paths(["only_one"])


# ── CLI dispatch: legacy 2-file mode vs. new directory mode ──

def test_cli_compare_dispatches_to_legacy_mode_for_two_json_files(tmp_path):
    from evaluator.cli.compare import parse_compare_args, run_compare

    file_a = tmp_path / "a.json"
    file_b = tmp_path / "b.json"
    file_a.write_text(json.dumps({"MRR": 0.5}))
    file_b.write_text(json.dumps({"MRR": 0.6}))
    out = tmp_path / "out.json"

    args = parse_compare_args([str(file_a), str(file_b), "--format", "json", "-o", str(out)])
    assert run_compare(args) == 0
    written = json.loads(out.read_text())
    assert "experiment_a" in written and "experiment_b" in written  # legacy shape


def test_cli_compare_dispatches_to_variant_mode_for_directories(tmp_path):
    from evaluator.cli.compare import parse_compare_args, run_compare

    run_a = _make_run(tmp_path, "run_a", 0.5)
    run_b = _make_run(tmp_path, "run_b", 0.9)
    out = tmp_path / "out.json"

    args = parse_compare_args([str(run_a), str(run_b), "--format", "json", "-o", str(out)])
    assert run_compare(args) == 0
    written = json.loads(out.read_text())
    assert "comparisons" in written  # new bundle shape, not the legacy one
