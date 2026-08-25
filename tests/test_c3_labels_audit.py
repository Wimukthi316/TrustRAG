"""Tests for C3 Layer 1, the RAGTruth error-type label audit.

The audit's job is to state what the label is before anything models it, so the
tests here check the two things that would quietly corrupt that statement: a
prefix vocabulary that swallows strings it does not really recognise, and a
cross-tabulation whose denominators do not reconcile with the span count.

Every expected value below is hand-computed from the fixture, not copied from a
run of the code.
"""

from __future__ import annotations

import json

import pytest

from src.c1_detector.ragtruth_labels import assert_label_coverage
from src.c3_explanation.labels_audit import (
    MECHANISM_BASELESS,
    MECHANISM_CONFLICT,
    SEVERITY_EVIDENT,
    SEVERITY_SUBTLE,
    INDEPENDENT_SEVERITY_WORDS,
    axes_of_error_type,
    axes_of_meta_prefix,
    build_report,
    constant_predictor_floor,
    format_report,
    main,
    meta_prefix,
    meta_structure,
    original_snippet,
    severity_word,
    split_by_response_id,
    task_type_by_source_id,
)
from src.common.schema import ErrorType

# --------------------------------------------------------------------------
# Fixture: seven annotated spans over four responses.
# --------------------------------------------------------------------------
#
#   A  resp 1  train  Evident Conflict      prefix EVIDENT CONFLICT        agrees
#   B  resp 1  train  Evident Baseless Info prefix HIGH INTRO OF NEW INFO  agrees
#   G  resp 1  train  Subtle Conflict       prefix SUBTLE CONFLICT         agrees
#   C  resp 2  train  Subtle Baseless Info  prefix LOW INTRO OF NEW INFO   agrees
#   F  resp 2  train  Subtle Baseless Info  prefix LOW INTRO OF NEW INFO   agrees
#   D  resp 3  test   Subtle Baseless Info  prefix HIGH INTRO OF NEW INFO  severity disagrees
#   E  resp 3  test   Evident Baseless Info prefix '"OutdoorSeating"'      unmapped

RESPONSES = [
    {
        "id": "1",
        "source_id": "s1",
        "model": "gpt-4-0613",
        "split": "train",
        "quality": "good",
        "response": "irrelevant",
        "labels": [
            {
                "start": 0,
                "end": 3,
                "text": "aaa",
                "meta": "EVIDENT CONFLICT\nOriginal: the sky is blue\nGenerated: the sky is green",
                "label_type": "Evident Conflict",
                "implicit_true": False,
                "due_to_null": False,
            },
            {
                "start": 4,
                "end": 7,
                "text": "bbb",
                "meta": "HIGH INTRO OF NEW INFO\nIt is not mentioned in the source content.",
                "label_type": "Evident Baseless Info",
                "implicit_true": True,
                "due_to_null": False,
            },
            {
                "start": 8,
                "end": 11,
                "text": "ggg",
                "meta": "SUBTLE CONFLICT:\nOriginal: tubers (like bulbs)\n",
                "label_type": "Subtle Conflict",
                "implicit_true": False,
                "due_to_null": False,
            },
        ],
    },
    {
        "id": "2",
        "source_id": "s2",
        "model": "llama-2-7b-chat",
        "split": "train",
        "quality": "truncated",
        "response": "irrelevant",
        "labels": [
            {
                "start": 0,
                "end": 3,
                "text": "ccc",
                "meta": "LOW INTRO OF NEW INFO\nminor addition",
                "label_type": "Subtle Baseless Info",
                "implicit_true": False,
                "due_to_null": True,
            },
            {
                "start": 4,
                "end": 7,
                "text": "fff",
                "meta": "LOW INTRO OF NEW INFO\nminor addition",
                "label_type": "Subtle Baseless Info",
                "implicit_true": False,
                "due_to_null": False,
            },
        ],
    },
    {
        "id": "3",
        "source_id": "s1",
        "model": "gpt-4-0613",
        "split": "test",
        "quality": "good",
        "response": "irrelevant",
        "labels": [
            {
                "start": 0,
                "end": 3,
                "text": "ddd",
                "meta": "HIGH INTRO OF NEW INFO\nadded detail",
                "label_type": "Subtle Baseless Info",
                "implicit_true": False,
                "due_to_null": False,
            },
            {
                "start": 4,
                "end": 7,
                "text": "eee",
                "meta": '"OutdoorSeating": true',
                "label_type": "Evident Baseless Info",
                "implicit_true": False,
                "due_to_null": False,
            },
        ],
    },
    {
        "id": "4",
        "source_id": "s2",
        "model": "llama-2-7b-chat",
        "split": "test",
        "quality": "good",
        "response": "irrelevant",
        "labels": [],
    },
]

TASK_BY_SOURCE = {"s1": "qa", "s2": "summarization"}


@pytest.fixture()
def report():
    return build_report(RESPONSES, task_by_source=TASK_BY_SOURCE)


# --------------------------------------------------------------------------
# Prefix extraction and the annotator vocabulary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "meta,expected",
    [
        ("EVIDENT CONFLICT\nOriginal: x", "EVIDENT CONFLICT"),
        ("SUBTLE CONFLICT:\nAIGC: y", "SUBTLE CONFLICT"),
        ("  HIGH INTRO OF NEW INFO  \nz", "HIGH INTRO OF NEW INFO"),
        ("Original: February 7 (1945)", "Original"),
        ("", ""),
        (None, ""),
        ("   ", ""),
    ],
)
def test_meta_prefix(meta, expected):
    assert meta_prefix(meta) == expected


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("EVIDENT CONFLICT", (MECHANISM_CONFLICT, SEVERITY_EVIDENT)),
        ("SUBTLE CONFLICT", (MECHANISM_CONFLICT, SEVERITY_SUBTLE)),
        ("Conflict", (MECHANISM_CONFLICT, None)),
        ("HIGH INTRO OF NEW INFO", (MECHANISM_BASELESS, SEVERITY_EVIDENT)),
        ("HIGH INTRODUCTION OF NEW INFORMATION", (MECHANISM_BASELESS, SEVERITY_EVIDENT)),
        ("LOW INTRODUCTION OF NEW INFORMATION", (MECHANISM_BASELESS, SEVERITY_SUBTLE)),
        ("HIGH INTRODUCTION OF INFORMATION", (MECHANISM_BASELESS, SEVERITY_EVIDENT)),
        ("Introduction of new information", (MECHANISM_BASELESS, None)),
    ],
)
def test_axes_of_meta_prefix_recognises_the_observed_vocabulary(prefix, expected):
    assert axes_of_meta_prefix(prefix) == expected


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "Original",
        "AIGC",
        '"OutdoorSeating"',
        "This was not mentioned in Source Content.",
        "HIGH",
        "SUBTLE",
    ],
)
def test_unrecognised_prefixes_are_not_folded_into_a_category(prefix):
    """A prefix that names no mechanism must stay unmapped.

    Widening this is how an agreement rate becomes meaningless: every unmatched
    string would be scored against whatever category happened to be nearby.
    """
    assert axes_of_meta_prefix(prefix) == (None, None)


def test_axes_of_error_type_covers_every_enum_member():
    assert axes_of_error_type(ErrorType.EVIDENT_CONFLICT) == (
        MECHANISM_CONFLICT,
        SEVERITY_EVIDENT,
    )
    assert axes_of_error_type(ErrorType.SUBTLE_CONFLICT) == (
        MECHANISM_CONFLICT,
        SEVERITY_SUBTLE,
    )
    assert axes_of_error_type(ErrorType.EVIDENT_BASELESS) == (
        MECHANISM_BASELESS,
        SEVERITY_EVIDENT,
    )
    assert axes_of_error_type(ErrorType.SUBTLE_BASELESS) == (
        MECHANISM_BASELESS,
        SEVERITY_SUBTLE,
    )
    assert axes_of_error_type(ErrorType.UNKNOWN) == (None, None)


def test_meta_structure_and_original_snippet():
    meta = "EVIDENT CONFLICT\nOriginal: February 7 (1945)\nGenerated: February 7, 2022"
    assert meta_structure(meta) == {
        "original": True,
        "generated": True,
        "aigc": False,
    }
    assert original_snippet(meta) == "February 7 (1945)"
    assert original_snippet("HIGH INTRO OF NEW INFO\nnothing quoted") is None
    assert original_snippet(None) is None
    assert meta_structure(None) == {
        "original": False,
        "generated": False,
        "aigc": False,
    }


# --------------------------------------------------------------------------
# L1.1
# --------------------------------------------------------------------------


def test_assert_label_coverage_accepts_the_four_published_strings():
    assert_label_coverage(
        [
            "Evident Conflict",
            "Subtle Conflict",
            "Evident Baseless Info",
            "Subtle Baseless Info",
        ]
    )


def test_assert_label_coverage_raises_on_an_unseen_string():
    with pytest.raises(ValueError) as excinfo:
        assert_label_coverage(["Evident Conflict", "Wildly New Category"])
    assert "wildly new category" in str(excinfo.value)


def test_l1_1_records_the_distinct_strings(report):
    block = report["l1_1_label_coverage"]
    assert block["assert_label_coverage_passed"] is True
    assert block["unmapped_raw_strings"] == []
    assert block["distinct_raw_label_types"] == {
        "Evident Baseless Info": 2,
        "Evident Conflict": 1,
        "Subtle Baseless Info": 3,
        "Subtle Conflict": 1,
    }


def test_l1_1_reports_a_failure_instead_of_raising():
    broken = [dict(RESPONSES[0])]
    broken[0] = json.loads(json.dumps(RESPONSES[0]))
    broken[0]["labels"][0]["label_type"] = "Brand New Category"
    block = build_report(broken)["l1_1_label_coverage"]
    assert block["assert_label_coverage_passed"] is False
    assert block["unmapped_raw_strings"] == ["brand new category"]
    assert block["error"]


# --------------------------------------------------------------------------
# L1.2
# --------------------------------------------------------------------------


def test_l1_2_denominators_reconcile_with_the_span_count(report):
    block = report["l1_2_meta_agreement"]
    assert report["n_spans"] == 7
    assert block["n_prefix_mapped"] + block["n_prefix_unmapped"] == report["n_spans"]
    assert block["n_prefix_mapped"] == 6
    assert block["n_prefix_unmapped"] == 1


def test_l1_2_agreement_rates_are_hand_computed(report):
    block = report["l1_2_meta_agreement"]
    assert block["mechanism"]["n_comparable"] == 6
    assert block["mechanism"]["n_agree"] == 6
    assert block["mechanism"]["agreement"] == pytest.approx(1.0)
    # Span D is the only disagreement: the annotator wrote HIGH, the label says Subtle.
    assert block["severity"]["n_comparable"] == 6
    assert block["severity"]["n_agree"] == 5
    assert block["severity"]["agreement"] == pytest.approx(5 / 6)
    assert block["joint"]["n_agree"] == 5


def test_l1_2_severity_table_names_the_disagreeing_cell(report):
    table = report["l1_2_meta_agreement"]["severity"]["table"]
    assert table["evident"] == {"evident": 2, "subtle": 1}
    assert table["subtle"] == {"subtle": 3}


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("EVIDENT CONFLICT", "evident"),
        ("SUBTLE CONFLICT", "subtle"),
        ("HIGH INTRO OF NEW INFO", "high"),
        ("LOW INTRODUCTION OF NEW INFORMATION", "low"),
        ("Introduction of new information", None),
        ("", None),
    ],
)
def test_severity_word_reports_what_the_annotator_typed(prefix, expected):
    assert severity_word(prefix) == expected


def test_severity_agreement_separates_the_independent_vocabulary(report):
    """`high`/`low` are evidence about the label; `evident`/`subtle` are not.

    The published label already contains the words evident and subtle, so a span
    whose comment opens with EVIDENT CONFLICT agrees with `label_type` by
    construction. Pooling the two would let that carry the half that is genuinely
    independent, which is the only half worth quoting.
    """
    block = report["l1_2_meta_agreement"]["severity_by_annotator_word"]
    assert set(block["independent_words"]) == set(INDEPENDENT_SEVERITY_WORDS)
    # Fixture: HIGH on B (evident, agrees) and on D (subtle, disagrees); LOW on
    # C and F (both subtle, agree). So three of four.
    assert block["independent"]["n_comparable"] == 4
    assert block["independent"]["n_agree"] == 3
    assert block["independent"]["agreement"] == pytest.approx(0.75)
    # A and G used the label's own words and both agree, which proves nothing.
    assert block["shares_label_vocabulary"]["n_comparable"] == 2
    assert block["shares_label_vocabulary"]["n_agree"] == 2
    assert block["table"]["high"] == {"evident": 1, "subtle": 1}
    assert block["table"]["low"] == {"subtle": 2}


def test_l1_2_counts_the_quoting_conventions(report):
    structure = report["l1_2_meta_agreement"]["structure"]
    assert structure["counts"] == {"original": 2, "generated": 1, "aigc": 0}
    assert structure["rates"]["original"] == pytest.approx(2 / 7)


def test_l1_2_keeps_the_unmapped_prefixes_visible(report):
    unmapped = report["l1_2_meta_agreement"]["top_unmapped_prefixes"]
    assert {row["prefix"] for row in unmapped} == {'"OutdoorSeating"'}


# --------------------------------------------------------------------------
# L1.3
# --------------------------------------------------------------------------


def test_l1_3_census(report):
    census = report["l1_3_census"]
    assert census["implicit_true"]["n"] == 1
    assert census["implicit_true"]["rate"] == pytest.approx(1 / 7)
    assert census["implicit_true"]["by_error_type"]["evident_baseless_info"]["n"] == 1
    assert census["due_to_null"]["n"] == 1
    assert census["quality"]["responses"]["counts"] == {"good": 3, "truncated": 1}
    assert census["generator_model"]["spans_by_model_and_type"]["llama-2-7b-chat"] == {
        "subtle_baseless_info": 2
    }


def test_l1_3_task_type_is_carried_through(report):
    census = report["l1_3_census"]
    assert census["due_to_null"]["by_task"]["summarization"]["n"] == 1
    assert census["due_to_null"]["by_task"]["qa"]["n"] == 0


# --------------------------------------------------------------------------
# L1.4
# --------------------------------------------------------------------------


def test_constant_predictor_floor_is_hand_computable():
    floor = constant_predictor_floor(["a", "a", "b"], "a")
    assert floor["accuracy"] == pytest.approx(2 / 3)
    assert floor["per_class"]["a"]["precision"] == pytest.approx(2 / 3)
    assert floor["per_class"]["a"]["recall"] == pytest.approx(1.0)
    assert floor["per_class"]["a"]["f1"] == pytest.approx(0.8)
    assert floor["per_class"]["b"]["f1"] == pytest.approx(0.0)
    assert floor["macro_f1"] == pytest.approx(0.4)
    assert floor["ignores_the_input"] is True


def test_l1_4_fits_the_majority_class_on_train_only(report):
    floors = report["l1_4_floors"]
    assert floors["majority_class_fitted_on_train"] == {
        "error_type": "subtle_baseless_info",
        "mechanism": MECHANISM_BASELESS,
        "severity": SEVERITY_SUBTLE,
    }
    assert floors["base_rates"]["train"]["error_type"]["n"] == 5
    assert floors["base_rates"]["test"]["error_type"]["n"] == 2
    test_floor = floors["majority_class_floor"]["test"]["error_type"]
    assert test_floor["accuracy"] == pytest.approx(0.5)
    assert test_floor["macro_f1"] == pytest.approx(1 / 3)
    assert floors["majority_class_floor"]["train"]["error_type"]["accuracy"] == (
        pytest.approx(2 / 5)
    )


def test_every_span_lands_in_exactly_one_split(report):
    total = sum(
        block["error_type"]["n"] for block in report["l1_4_floors"]["base_rates"].values()
    )
    assert total == report["n_spans"]


# --------------------------------------------------------------------------
# Splits, tasks, plumbing
# --------------------------------------------------------------------------


def test_frozen_split_ids_override_the_corpus_field():
    mapping = split_by_response_id(RESPONSES, {"train": ["1"], "calib": ["2"]})
    assert mapping == {"1": "train", "2": "calib", "3": "test", "4": "test"}
    report = build_report(RESPONSES, split_ids={"train": ["1"], "calib": ["2"]})
    assert report["used_frozen_split_ids"] is True
    assert report["l1_4_floors"]["base_rates"]["calib"]["error_type"]["n"] == 2


def test_split_falls_back_to_the_corpus_field():
    mapping = split_by_response_id(RESPONSES, None)
    assert mapping["1"] == "train"
    assert mapping["3"] == "test"


def test_task_type_by_source_id_maps_the_raw_corpus_strings():
    rows = [
        {"source_id": "a", "task_type": "QA"},
        {"source_id": "b", "task_type": "Data2txt"},
        {"source_id": "c", "task_type": "Summary"},
    ]
    assert task_type_by_source_id(rows) == {
        "a": "qa",
        "b": "data2text",
        "c": "summarization",
    }


def test_task_type_by_source_id_rejects_an_unknown_task():
    with pytest.raises(KeyError):
        task_type_by_source_id([{"source_id": "a", "task_type": "translation"}])


def test_format_report_prints_the_floor_and_the_agreement(report):
    text = format_report(report)
    assert "assert_label_coverage: PASS" in text
    assert "mechanism  agreement" in text
    assert "independent-vocabulary agreement" in text
    assert "floor error_type" in text


def test_main_writes_the_artefact_and_returns_zero(tmp_path):
    responses = tmp_path / "response.jsonl"
    responses.write_text(
        "\n".join(json.dumps(r) for r in RESPONSES) + "\n", encoding="utf-8"
    )
    source_info = tmp_path / "source_info.jsonl"
    source_info.write_text(
        "\n".join(
            json.dumps({"source_id": sid, "task_type": raw})
            for sid, raw in (("s1", "QA"), ("s2", "Summary"))
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    code = main(
        [
            "--responses",
            str(responses),
            "--source-info",
            str(source_info),
            "--split-ids",
            "",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    written = json.loads((out_dir / "label_audit.json").read_text(encoding="utf-8"))
    assert written["n_spans"] == 7
    assert written["used_frozen_split_ids"] is False
