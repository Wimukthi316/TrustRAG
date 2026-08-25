"""Layer 2's protocol: both conventions, the theta boundary, and the exclusions."""

from __future__ import annotations


from src.c4_attribution.dataset import ResponseRecord, SpanRecord
from src.c4_attribution.evidence_labels import EvidenceLabel
from src.c4_attribution import matching as m


def _row(context: str, meta: str, span_text: str = "the span") -> SpanRecord:
    answer = f"Something about {span_text} here."
    start = answer.index(span_text)
    response = ResponseRecord(
        response_id="r1",
        source_id="s1",
        split="test",
        task="qa",
        model="test",
        question="q",
        context=context,
        answer=answer,
    )
    label = EvidenceLabel(
        response_id="r1",
        start=start,
        end=start + len(span_text),
        span_text=span_text,
        label_type="Evident Conflict",
        raw_meta=meta,
        implicit_true=False,
        due_to_null=False,
        has_original="Original:" in meta,
        snippet=(
            meta.split("Original:", 1)[1].strip() if "Original:" in meta else None
        ),
        no_evidence="not mentioned" in meta,
        closing_marker=None,
    )
    response.labels = [label]
    return SpanRecord(response=response, label=label)


def test_theta_is_fixed_at_the_declared_operating_point():
    assert m.THETA == 0.9
    assert 0.9 in m.THETA_SWEEP


def test_verbatim_inside_exactly_one_sentence_is_unambiguous():
    row = _row("Alpha runs fast. Beta walks slowly.", "Original: Beta walks slowly")
    match = m.match(row)
    assert match.kind == m.KIND_VERBATIM_UNIQUE
    assert match.gold_index == 1
    assert match.convention_a == 1
    assert match.convention_b == 1


def test_verbatim_in_several_sentences_ties_to_the_lowest_index():
    row = _row("Beta walks. Alpha runs. Beta walks.", "Original: Beta walks")
    match = m.match(row)
    assert match.kind == m.KIND_VERBATIM_MULTI
    assert match.gold_index == 0
    assert match.convention_b is None  # B refuses to guess between them


def test_a_paraphrase_is_convention_A_only():
    row = _row(
        "The building opened in nineteen ninety eight. Nothing else happened.",
        "Original: building opened nineteen ninety eight",
    )
    match = m.match(row)
    assert match.kind == m.KIND_CONTAINMENT
    assert match.convention_a == 0
    assert match.convention_b is None


def test_below_theta_is_unmatched_and_excluded_rather_than_guessed():
    row = _row("Totally unrelated prose here. And more of it.",
               "Original: quantum chromodynamics lagrangian")
    match = m.match(row)
    assert match.kind == m.KIND_UNMATCHED
    assert match.convention_a is None
    assert match.scorable is False


def test_theta_moves_the_boundary_in_the_expected_direction():
    row = _row(
        "Alpha beta gamma delta. Nothing here at all.",
        "Original: alpha beta gamma epsilon",
    )
    strict = m.match(row, theta=1.0)
    loose = m.match(row, theta=0.5)
    assert strict.kind == m.KIND_UNMATCHED
    assert loose.kind == m.KIND_CONTAINMENT


def test_no_evidence_and_no_label_are_different_kinds():
    denial = _row("Alpha runs.", "Original: the date is not mentioned")
    silent = _row("Alpha runs.", "EVIDENT CONFLICT")
    assert m.match(denial).kind == m.KIND_NO_EVIDENCE
    assert m.match(silent).kind == m.KIND_NO_SUPERVISION
    # Neither is scorable, and pooling them would inflate Layer 7.
    assert m.match(denial).scorable is False
    assert m.match(silent).scorable is False


def test_kind_counts_are_exhaustive():
    rows = [
        _row("Alpha runs. Beta walks.", "Original: Beta walks"),
        _row("Alpha runs.", "Original: not mentioned"),
        _row("Alpha runs.", "EVIDENT CONFLICT"),
    ]
    counts = m.kind_counts(m.match_all(rows))
    assert counts["n"] == 3
    assert sum(counts["counts"].values()) == 3


def test_theta_sweep_is_monotone_in_the_matched_rate():
    rows = [
        _row("Alpha beta gamma delta epsilon. Nothing.", "Original: alpha beta gamma"),
        _row("Alpha runs. Beta walks.", "Original: Beta walks"),
    ]
    sweep = m.theta_sweep(rows)
    rates = [r["convention_a_rate"] for r in sweep]
    assert rates == sorted(rates, reverse=True)
