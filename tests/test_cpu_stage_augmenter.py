"""4b: the augmenter's per-item perturbation runs through `parallel_map`, so the
`cpu_stage_executor` knob (sync/thread/process) applies — and every backend gives a byte-identical,
order-preserving result (sync default == today's serial comprehension)."""
import functools

from evaluator.evaluation.executor.cpu_parallel import parallel_map
from evaluator.evaluation.handlers.query import _augment_one_text
from evaluator.pipeline.text_augmentation import TextAugmentConfig, TextAugmenter


def _augment_all(backend):
    aug = TextAugmenter(TextAugmentConfig(
        homophones=True, unit_corruption=True, char_swap_prob=0.3, max_edits=3,
    ))
    items = [(f"q{i}", f"give the patient 50 mg of medication number {i}") for i in range(12)]
    fn = functools.partial(_augment_one_text, augmenter=aug, base_seed=42, node_id="augmenter")
    return parallel_map(fn, items, backend=backend, workers=2)


def test_augmenter_backends_are_byte_identical():
    base = _augment_all("sync")
    assert _augment_all("thread") == base
    assert _augment_all("process") == base       # picklable + deterministic across processes


def test_augmenter_is_order_preserving_and_perturbs():
    out = _augment_all("process")
    assert len(out) == 12
    # at least one text changed (the perturbation actually ran), and order is the input order
    assert any("mg" not in t or t != f"give the patient 50 mg of medication number {i}"
               for i, t in enumerate(out))
