"""4b: query correction's per-item repair runs through `parallel_map`, so the
`cpu_stage_executor` knob (sync/thread/process) applies — and every backend gives a byte-identical,
order-preserving result (sync default == the batch `correct_query_texts` call). Uses the `rule`
method (deterministic, client-free → picklable for the process backend)."""
import functools

from evaluator.evaluation.executor.cpu_parallel import parallel_map
from evaluator.evaluation.query_correction import (
    correct_one_text,
    correct_query_texts,
    resolve_correction_client,
)
from evaluator.config.query_correction import QueryCorrectionConfig


def _cfg():
    # Rule method with an explicit replacement so correction visibly changes some items.
    return QueryCorrectionConfig(
        enabled=True, method="rule",
        replacements={"met foreman": "metformin"}, use_default_rules=False,
    )


def _correct_all(backend):
    cfg = _cfg()
    client = resolve_correction_client(cfg)  # None for the rule method
    texts = [
        f"patient {i} takes met foreman daily" if i % 2 else f"clean query {i}"
        for i in range(12)
    ]
    fn = functools.partial(correct_one_text, config=cfg, client=client)
    return parallel_map(fn, texts, backend=backend, workers=2)


def test_correction_backends_are_byte_identical():
    base = _correct_all("sync")
    assert _correct_all("thread") == base
    assert _correct_all("process") == base       # picklable (rule method, no live client)


def test_correction_matches_the_batch_call():
    cfg = _cfg()
    texts = [
        f"patient {i} takes met foreman daily" if i % 2 else f"clean query {i}"
        for i in range(12)
    ]
    # The per-item map (sync) is byte-identical to the original batch path.
    assert _correct_all("sync") == correct_query_texts(texts, cfg)
    # And it actually corrected the targeted items (order preserved).
    assert all("metformin" in t for i, t in enumerate(_correct_all("sync")) if i % 2)
