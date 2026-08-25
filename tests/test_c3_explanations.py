"""Tests for C3's template explanations and the way they are scored.

The explanation is the only part of C3 a non-technical reader will ever look at,
so the tests are about what it is allowed to claim. It must not commit to a
severity the conformal set did not commit to, it must not quote a sentence that
is not in the source, and it must never carry a coverage number -- that decision
belongs to the caller, which knows whether the guarantee applies.
"""

from __future__ import annotations

import json

import pytest

from src.c3_explanation.explanation_eval import (
    CONSTANT_EXPLANATION,
    build_rows,
    lcs_length,
    lookup_meta,
    main,
    meta_by_span,
    normalise_for_substring,
    quoted_is_verbatim,
    rouge_l,
    score_rows,
    snippet_containment,
)
from src.c3_explanation.explanations import explain, explain_many
from src.c3_explanation.grounding import split_sentences

CONTEXT = (
    "Anne Frank died in Bergen-Belsen in 1945. "
    "The Red Cross archives were re-examined by researchers. "
    "Witnesses reported symptoms before February 7."
)
SENTENCES = split_sentences(CONTEXT)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def test_a_conflict_quotes_the_sentence_it_contradicts():
    out = explain("died in 2022", SENTENCES, error_type="evident_conflict")
    assert "The source says" in out["explanation"]
    assert out["quoted_sentence"] in SENTENCES
    assert out["committed_to"] == "conflict"


def test_a_baseless_span_says_nothing_supports_it():
    out = explain(
        "the museum issued a commemorative postage stamp",
        SENTENCES,
        error_type="evident_baseless_info",
    )
    assert out["explanation"].startswith("No sentence in the source supports")
    assert out["committed_to"] == "baseless"


def test_a_subtle_type_hedges_and_an_evident_one_does_not():
    subtle = explain("died in 1944", SENTENCES, error_type="subtle_conflict")
    evident = explain("died in 1944", SENTENCES, error_type="evident_conflict")
    assert "small one" in subtle["explanation"]
    assert "small one" not in evident["explanation"]


def test_a_set_sharing_a_mechanism_commits_to_it_and_not_to_severity():
    """The Layer 3 finding, said out loud instead of guessed at."""
    out = explain(
        "died in 2022",
        SENTENCES,
        type_set=["evident_conflict", "subtle_conflict"],
    )
    assert out["committed_to"] == "conflict"
    assert out["mechanism"] == "conflict"
    assert out["severity"] is None
    assert "severe" in out["explanation"]


def test_a_set_spanning_both_mechanisms_refuses_to_pick_one():
    out = explain(
        "died in 2022",
        SENTENCES,
        type_set=["evident_conflict", "evident_baseless_info"],
    )
    assert out["committed_to"] == "unsupported"
    assert "not decidable" in out["explanation"]


def test_the_set_takes_precedence_over_a_single_type():
    out = explain(
        "died in 2022",
        SENTENCES,
        error_type="evident_conflict",
        type_set=["evident_conflict", "evident_baseless_info"],
    )
    assert out["committed_to"] == "unsupported"


def test_an_explanation_never_states_a_coverage_number():
    """Whether the guarantee applies is the caller's call, not this module's."""
    for kwargs in (
        {"error_type": "evident_conflict"},
        {"error_type": "subtle_baseless_info"},
        {"type_set": ["evident_conflict", "subtle_conflict"]},
        {"type_set": ["evident_conflict", "evident_baseless_info"]},
        {},
    ):
        text = explain("some span", SENTENCES, **kwargs)["explanation"].lower()
        for banned in ("coverage", "%", "confidence level of", "guarantee"):
            assert banned not in text


def test_no_type_at_all_says_so_rather_than_inventing_one():
    out = explain("some span", SENTENCES)
    assert out["committed_to"] == "none"
    assert "no error type was assigned" in out["explanation"]


def test_an_empty_context_does_not_crash_or_quote():
    out = explain("some span", [], error_type="evident_conflict")
    assert out["quoted_sentence"] is None
    assert out["quoted_sentence_index"] is None
    assert "could not be split into sentences" in out["explanation"]


def test_a_long_sentence_is_clipped_for_a_tooltip():
    long_sentence = "word " * 400
    out = explain("word word", [long_sentence.strip()], error_type="evident_conflict")
    assert "…" in out["explanation"]
    assert len(out["explanation"]) < 600


def test_explain_many_preserves_order():
    spans = [
        {"text": "died in 2022", "error_type": "evident_conflict"},
        {"text": "a postage stamp", "error_type": "evident_baseless_info"},
    ]
    out = explain_many(spans, SENTENCES)
    assert [o["committed_to"] for o in out] == ["conflict", "baseless"]


def test_explanations_are_deterministic():
    first = explain("died in 2022", SENTENCES, error_type="evident_conflict")
    second = explain("died in 2022", SENTENCES, error_type="evident_conflict")
    assert first == second


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_lcs_and_rouge_are_hand_computable():
    assert lcs_length(["a", "b", "c"], ["a", "c"]) == 2
    assert lcs_length([], ["a"]) == 0
    scores = rouge_l("a b c", "a c")
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["f1"] == pytest.approx(2 * (2 / 3) / (2 / 3 + 1))


def test_snippet_containment_returns_none_when_there_is_no_reference():
    """A span the annotator quoted nothing for is excluded, not scored zero."""
    assert snippet_containment("", "any sentence") is None
    assert snippet_containment("the red cross", None) == pytest.approx(0.0)
    assert snippet_containment(
        "red cross archives", "The Red Cross archives were re-examined."
    ) == pytest.approx(1.0)
    assert snippet_containment("red cross zebra", "The Red Cross archives.") == (
        pytest.approx(2 / 3)
    )


def test_self_faithfulness_catches_a_quote_that_is_not_in_the_context():
    assert quoted_is_verbatim(
        "The Red Cross archives were re-examined by researchers.", CONTEXT
    )
    assert not quoted_is_verbatim("The Red Cross burned the archives.", CONTEXT)
    assert quoted_is_verbatim(None, CONTEXT) is None


def test_normalise_for_substring_ignores_whitespace_and_case():
    assert normalise_for_substring("  The   RED\ncross ") == "the red cross"


def test_lookup_meta_falls_back_to_the_overlapping_annotation():
    table = {(0, 10): "first", (20, 30): "second"}
    assert lookup_meta(table, 0, 10) == ("first", "exact")
    assert lookup_meta(table, 18, 32) == ("second", "overlap")
    assert lookup_meta(table, 100, 110) == ("", "none")


def test_meta_by_span_indexes_by_response_and_offsets():
    raw = [
        {
            "id": "7",
            "labels": [
                {"start": 0, "end": 5, "meta": "EVIDENT CONFLICT\nOriginal: x"},
            ],
        }
    ]
    table = meta_by_span(raw)
    assert table["7"][(0, 5)].startswith("EVIDENT CONFLICT")


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

PROCESSED = [
    {
        "id": "1",
        "task_type": "qa",
        "context": CONTEXT,
        "answer": "Anne Frank died in 2022 according to the museum.",
        "spans": [
            {
                "start": 0,
                "end": 23,
                "text": "Anne Frank died in 2022",
                "error_types": ["evident_conflict"],
            }
        ],
    }
]

RAW = [
    {
        "id": "1",
        "labels": [
            {
                "start": 0,
                "end": 23,
                "meta": (
                    "EVIDENT CONFLICT\n"
                    "Original: Anne Frank died in Bergen-Belsen in 1945\n"
                    "Generated: Anne Frank died in 2022"
                ),
            }
        ],
    }
]


def test_the_constant_floor_scores_zero_on_grounding():
    rows = build_rows(PROCESSED, meta_by_span(RAW), constant=True)
    block = score_rows(rows)
    assert rows[0]["explanation"] == CONSTANT_EXPLANATION
    assert block["snippet"]["mean_containment"] == pytest.approx(0.0)
    assert block["self_faithfulness"]["quote_rate"] == pytest.approx(0.0)


def test_the_template_beats_the_floor_on_the_grounding_metric():
    metas = meta_by_span(RAW)
    floor = score_rows(build_rows(PROCESSED, metas, constant=True))
    template = score_rows(build_rows(PROCESSED, metas, "gold"))
    assert (
        template["snippet"]["mean_containment"] > floor["snippet"]["mean_containment"]
    )
    assert template["self_faithfulness"]["verbatim_rate"] == pytest.approx(1.0)


def test_main_writes_the_artefact_and_records_the_unrun_llm_row(tmp_path):
    processed = tmp_path / "processed.jsonl"
    processed.write_text(
        "\n".join(json.dumps(r) for r in PROCESSED) + "\n", encoding="utf-8"
    )
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in RAW) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "--processed",
                str(processed),
                "--raw",
                str(raw),
                "--baseline-probs",
                "",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    written = json.loads(
        (out_dir / "explanation_metrics.json").read_text(encoding="utf-8")
    )
    assert written["llm_comparison"]["n"] == 0
    assert "not run" in written["llm_comparison"]["status"]
    assert written["join"]["exact"] == 1
    assert "constant_floor" in written["systems"]
