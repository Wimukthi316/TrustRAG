"""Layer 12: served scores equal offline scores, and the evidence string claims nothing."""

from __future__ import annotations

import json

import pytest

from src.c4_attribution import serving
from src.c4_attribution.lexical import idf_overlap_scores

SENTENCES = [
    "The museum opened in 1998 after a long restoration.",
    "Admission is free on the first Sunday of each month.",
    "The cafe on the ground floor closes at four.",
]


def _artifact(threshold=None):
    return serving.build_artifact(
        backend="lexical",
        abstention_threshold=threshold,
        lac_thresholds={"0.1": 0.85},
        offline_backend="nli_signed|sentence",
        notes=["test"],
    )


def test_served_scores_equal_offline_scores_to_1e_9():
    attributor = serving.C4Attributor(_artifact())
    for span in ("opened in 1998", "admission is free", "closes at four", "unrelated"):
        served = attributor.scores(span, SENTENCES)
        offline = idf_overlap_scores(span, SENTENCES)
        assert len(served) == len(offline)
        for a, b in zip(served, offline):
            assert a == pytest.approx(b, abs=1e-9)


def test_the_artifact_is_json_and_round_trips(tmp_path):
    path = tmp_path / "c4_artifact.json"
    serving.write_artifact(_artifact(0.25), path)
    loaded = serving.load_artifact(path)
    assert loaded == json.loads(path.read_text(encoding="utf-8"))
    assert loaded["served_backend"] == "lexical"
    assert loaded["abstention_threshold"] == 0.25


def test_a_missing_artifact_leaves_the_layer_unavailable_rather_than_crashing(tmp_path):
    attributor = serving.C4Attributor.load(tmp_path / "nothing.json")
    assert attributor.available is False
    result = attributor.annotate("anything", SENTENCES)
    assert result["evidence_sentence"] is None


def test_the_evidence_string_never_claims_a_guarantee():
    attributor = serving.C4Attributor(_artifact())
    result = attributor.annotate("opened in 1998", SENTENCES)
    assert serving.evidence_is_safe(result["evidence_sentence"] or "")
    for word in serving.FORBIDDEN_IN_EVIDENCE:
        assert word not in (result["evidence_sentence"] or "").lower()


def test_the_safety_check_actually_rejects_the_words_it_names():
    assert serving.evidence_is_safe("this sentence supports the claim") is True
    assert serving.evidence_is_safe("at 90% coverage this is the evidence") is False
    assert serving.evidence_is_safe("we guarantee this sentence") is False


def test_abstaining_returns_all_three_fields_as_none():
    attributor = serving.C4Attributor(_artifact(threshold=0.99))
    result = attributor.annotate("completely unrelated tokens", SENTENCES)
    assert result["evidence_sentence"] is None
    assert result["evidence_index"] is None
    assert result["entailment_score"] is None
    assert result["abstained"] is True


def test_the_returned_index_points_at_the_returned_sentence():
    attributor = serving.C4Attributor(_artifact())
    result = attributor.annotate("admission is free", SENTENCES)
    assert SENTENCES[result["evidence_index"]] == result["evidence_sentence"]


def test_the_score_fits_the_frozen_contracts_bounds():
    from src.common.schema import Span

    attributor = serving.C4Attributor(_artifact())
    result = attributor.annotate("opened in 1998", SENTENCES)
    span = Span(
        start=0,
        end=4,
        text="Some",
        evidence_sentence=result["evidence_sentence"],
        evidence_index=result["evidence_index"],
        entailment_score=result["entailment_score"],
    )
    assert 0.0 <= span.entailment_score <= 1.0
    assert span.evidence_index >= 0


def test_an_empty_context_abstains_rather_than_indexing_into_nothing():
    attributor = serving.C4Attributor(_artifact())
    result = attributor.annotate("anything", [])
    assert result["evidence_index"] is None


def test_the_detector_does_not_append_c4_to_model_version():
    """The badge names what produced the span scores, and C4 produces none.

    C3 tried appending and broke four pinned assertions in test_serving.py.
    This reads the property's own source rather than the whole file, so the
    comment explaining the rule cannot satisfy the rule.
    """
    import inspect

    from backend.app.services.lettucedetect_detector import LettuceDetectDetector

    source = inspect.getsource(LettuceDetectDetector.model_version.fget)
    assert "c4" not in source.lower()
    assert "+c2" in source
