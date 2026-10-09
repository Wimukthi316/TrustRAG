"""Network-free tests for the phase-3b leakage controls in synth_medical_spans.

The injector round-trip is covered by test_synth_medical_spans.py. This file
covers the two additions that keep the synthetic TRAIN set disjoint from the
MedHallu example-level eval: the answer-text normaliser used for exclusion, and
the deterministic row hash used for the train/test carve. Neither touches the
network, so these run in CI with no dataset download.
"""

from __future__ import annotations

from src.c1_detector.synth_medical_spans import _norm, _row_bucket


def test_norm_collapses_case_and_whitespace():
    assert _norm("  Aspirin   REDUCED\tmortality\n") == "aspirin reduced mortality"
    # Two spellings of the same answer collapse to one key, so exclusion by set
    # membership catches a reformatted duplicate.
    assert _norm("The dose was 5 mg.") == _norm("the  dose   was 5 mg.")


def test_norm_distinguishes_different_answers():
    assert _norm("Metformin lowers glucose.") != _norm("Metformin raises glucose.")


def test_row_bucket_is_deterministic():
    a = _row_bucket("pqa_artificial", "42", 0)
    b = _row_bucket("pqa_artificial", "42", 0)
    assert a == b


def test_row_bucket_in_range():
    for rid in ("0", "1", "999", "abc", "pubmed:12345"):
        assert 0 <= _row_bucket("pqa_artificial", rid, 0) < 1000


def test_row_bucket_depends_on_config_seed_and_id():
    base = _row_bucket("pqa_artificial", "42", 0)
    # Changing any component should (almost always) move the bucket; assert the
    # inputs actually feed the hash by requiring at least one of them to differ.
    variants = {
        _row_bucket("pqa_labeled", "42", 0),
        _row_bucket("pqa_artificial", "43", 0),
        _row_bucket("pqa_artificial", "42", 1),
    }
    assert variants != {base}


def test_row_split_fraction_is_approximately_honoured():
    # Over many ids the fraction under the cutoff should track test_fraction,
    # which is what makes the 12% synthetic test slice the size we claim.
    cutoff = 120  # test_fraction 0.12 -> bucket < 120
    ids = [f"row{i}" for i in range(6000)]
    in_test = sum(1 for i in ids if _row_bucket("pqa_artificial", i, 0) < cutoff)
    assert 0.09 < in_test / len(ids) < 0.15


def test_train_and_test_slices_are_disjoint_by_construction():
    # A row is in exactly one slice: the carve is a single threshold on one hash,
    # so no id can be both under and not-under the cutoff.
    cutoff = 300
    ids = [f"row{i}" for i in range(2000)]
    train = {i for i in ids if not _row_bucket("pqa_artificial", i, 0) < cutoff}
    test = {i for i in ids if _row_bucket("pqa_artificial", i, 0) < cutoff}
    assert train.isdisjoint(test)
    assert train | test == set(ids)
