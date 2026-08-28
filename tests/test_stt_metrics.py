"""WER / CER transcription metrics."""

import pytest

from evaluator.metrics.stt import character_error_rate, word_error_rate


def test_wer_perfect_and_total():
    assert word_error_rate("the cat sat", "the cat sat") == pytest.approx(0.0)
    # one substitution out of three words
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_wer_insertion_and_deletion():
    assert word_error_rate("the cat", "the cat sat") == pytest.approx(0.5)  # 1 insertion / 2
    assert word_error_rate("the cat sat", "the cat") == pytest.approx(1 / 3)  # 1 deletion / 3


def test_cer_character_level():
    assert character_error_rate("abc", "abc") == pytest.approx(0.0)
    assert character_error_rate("abc", "abd") == pytest.approx(1 / 3)
