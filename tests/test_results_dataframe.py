"""C2 pin (CRITIQUE.md): EvaluationResults.to_dataframe gives the tidy metric table."""

from evaluator import EvaluationConfig, EvaluationResults


def _results():
    report = {
        "branches": {
            "asr": {"recall@5": {"mean": 0.8, "n": 10, "ci": [0.7, 0.9]}},
            "ref": {"recall@5": {"mean": 0.9, "n": 10}},
        }
    }
    return EvaluationResults(
        metrics={"MRR": 0.75, "report": report}, config=EvaluationConfig()
    )


def test_to_dataframe_one_row_per_branch_metric():
    df = _results().to_dataframe()
    assert list(df.columns) == ["branch", "metric", "mean", "ci_lower", "ci_upper", "n"]
    assert len(df) == 2
    by_branch = df.set_index("branch")
    assert by_branch.loc["asr", "mean"] == 0.8
    assert by_branch.loc["asr", "ci_lower"] == 0.7
    assert by_branch.loc["ref", "mean"] == 0.9


def test_to_dataframe_empty_without_report():
    df = EvaluationResults(metrics={"MRR": 0.5}, config=EvaluationConfig()).to_dataframe()
    assert len(df) == 0
