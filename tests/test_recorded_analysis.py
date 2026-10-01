"""The offline demo payloads must be rebuilt from artefacts, never hand-written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.recorded_analysis import (
    C2_ARTIFACT,
    CALIB_DUMP,
    CASES,
    DEFAULT_ALPHA,
    DIAL_ALPHAS,
    OUT,
    RAW_DIR,
    TEST_DUMP,
    RecordedRuns,
    alpha_key,
    build_cases,
)
from src.common.schema import AnalysisResult, ConformalDecision

needs_artefacts = pytest.mark.skipif(
    not all(
        p.exists()
        for p in (TEST_DUMP, CALIB_DUMP, C2_ARTIFACT, RAW_DIR / "response.jsonl")
    ),
    reason="C1's dumps or C2's artefact are not in this working copy",
)


@pytest.fixture(scope="module")
def runs() -> RecordedRuns:
    return RecordedRuns()


@needs_artefacts
def test_every_shipped_case_still_builds(runs: RecordedRuns):
    for case in CASES:
        result = runs.build(case["id"])
        assert isinstance(result, AnalysisResult)
        assert result.answer
        assert result.context_sentences


@needs_artefacts
def test_a_rebuilt_span_slices_its_own_text_out_of_the_answer(runs: RecordedRuns):
    # The schema validator enforces this, so a failure here means the dump and
    # the answer have drifted apart rather than that the check is missing.
    for case in CASES:
        result = runs.build(case["id"])
        for span in result.spans:
            assert result.answer[span.start : span.end] == span.text


@needs_artefacts
def test_pass_spans_are_dropped_exactly_as_the_serving_path_drops_them(runs: RecordedRuns):
    for case in CASES:
        result = runs.build(case["id"])
        assert all(
            span.conformal_decision is not ConformalDecision.PASS
            for span in result.spans
        )


@needs_artefacts
def test_the_calibrated_score_is_c2s_own_calibrator_not_a_reimplementation(
    runs: RecordedRuns,
):
    from src.c2_calibration.calibration import calibrator_from_dict

    artifact = json.loads(C2_ARTIFACT.read_text(encoding="utf-8"))
    calibrator = calibrator_from_dict(artifact["calibrator"])
    result = runs.build(CASES[0]["id"])
    record = runs.by_id[CASES[0]["id"]]
    raw = {
        (int(s["start"]), int(s["end"])): float(s["mean_prob"])
        for s in record["pred_spans"]
    }
    for span in result.spans:
        expected = float(calibrator.transform([raw[(span.start, span.end)]])[0])
        assert span.calibrated_score == pytest.approx(expected, abs=5e-5)


@needs_artefacts
def test_the_conformal_decision_changes_with_alpha_in_the_expected_direction(
    runs: RecordedRuns,
):
    # A weaker promise decides more spans, so the number of surviving spans can
    # only fall as alpha rises: more of them come back PASS and get dropped.
    counts = [len(runs.build(CASES[0]["id"], alpha).spans) for alpha in (0.05, 0.2, 0.4)]
    assert counts == sorted(counts, reverse=True)


@needs_artefacts
def test_the_withdrawn_case_really_is_withdrawn(runs: RecordedRuns):
    # The third case exists to show the promise being withdrawn while the marks
    # stay. If the reference or the input ever changes so that it no longer is,
    # the case has to be re-picked rather than quietly shown as a normal one.
    withdrawn = next(c for c in CASES if c["slug"] == "promise-withdrawn")
    result = runs.build(withdrawn["id"])
    assert result.guarantee_applies is False
    assert result.distribution_check is not None
    assert result.distribution_check.checked is True
    assert result.spans, "the marks must stay when the promise is withdrawn"


@needs_artefacts
def test_the_shipped_json_matches_what_the_code_produces_now():
    """A stale committed payload is a screen full of numbers nobody can reproduce."""
    if not OUT.exists():
        pytest.skip("recorded-runs.json has not been generated in this copy")
    shipped = json.loads(Path(OUT).read_text(encoding="utf-8"))
    from src.common.recorded_analysis import build_cases

    rebuilt = build_cases()
    assert [c["slug"] for c in shipped["cases"]] == [c["slug"] for c in rebuilt]
    for a, b in zip(shipped["cases"], rebuilt):
        # `timestamp` is stamped at build time and is the one field allowed to move.
        left = dict(a["result"])
        right = dict(b["result"])
        left.pop("timestamp", None)
        right.pop("timestamp", None)
        assert left == right, (
            f"{a['slug']} on disk differs from what the code produces now. "
            "Regenerate with: python -m src.common.recorded_analysis"
        )


@needs_artefacts
def test_the_cases_cover_the_states_they_were_chosen_for(runs: RecordedRuns):
    results = {c["slug"]: runs.build(c["id"]) for c in CASES}
    all_spans = [s for r in results.values() for s in r.spans]
    decisions = {s.conformal_decision for s in all_spans}
    assert ConformalDecision.FLAG in decisions
    assert ConformalDecision.ABSTAIN in decisions
    assert any(s.evidence_sentence for s in all_spans), "no case shows an evidence hit"
    assert any(not s.evidence_sentence for s in all_spans), "no case shows an abstention"
    assert any(not r.guarantee_applies for r in results.values())


@needs_artefacts
def test_every_dial_setting_is_built(runs: RecordedRuns):
    """The dial selects between builds; it must never interpolate between them.

    A setting with no build falls back to the default one, and the reader then
    sees a result that belongs to a different setting under a label that says
    otherwise. Nothing on the screen would look wrong.
    """
    for case in build_cases(runs):
        built = {alpha_key(DEFAULT_ALPHA), *case["variants"]}
        missing = [alpha_key(a) for a in DIAL_ALPHAS if alpha_key(a) not in built]
        assert not missing, f"{case['slug']} has no build at alpha {missing}"


@needs_artefacts
def test_loosening_the_promise_leaves_fewer_passages_undecided(runs: RecordedRuns):
    """A dial that moves and changes nothing demonstrates the opposite of its point.

    A larger alpha is a weaker promise, so it must not leave MORE passages
    undecided than a smaller one. This is the direction the whole trade rests
    on and it is asserted rather than assumed, because a regeneration against a
    stale artefact could produce six identical results and the page would look
    exactly the same.
    """
    cases = build_cases(runs)

    def undecided(key: str) -> int:
        total = 0
        for case in cases:
            builds = {alpha_key(DEFAULT_ALPHA): case["result"], **case["variants"]}
            total += sum(
                1
                for span in builds[key]["spans"]
                if span["conformal_decision"] == "abstain"
            )
        return total

    strict = undecided(alpha_key(min(DIAL_ALPHAS)))
    loose = undecided(alpha_key(max(DIAL_ALPHAS)))
    assert strict >= loose, (
        f"loosening the promise left more passages undecided ({loose}) than "
        f"tightening it ({strict}), which is backwards"
    )


@needs_artefacts
def test_the_prediction_set_travels_with_the_decision(runs: RecordedRuns):
    """ABSTAIN is two opposite states and only the set separates them.

    Recorded runs are what the interface shows when there is no backend, so a
    field missing here is a field missing in front of a panel.
    """
    for case in CASES:
        for span in runs.build(case["id"]).spans:
            assert span.prediction_set is not None, (
                f"{case['slug']}: a recorded span carries no prediction set"
            )
            assert set(span.prediction_set) <= {0, 1}
            if span.conformal_decision is ConformalDecision.FLAG:
                assert span.prediction_set == [1]
            if span.conformal_decision is ConformalDecision.PASS:
                assert span.prediction_set == [0]
