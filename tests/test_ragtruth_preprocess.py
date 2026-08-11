"""Tests for the RAGTruth preprocessing path.

The fixtures below are copied verbatim from the sample records published in the
RAGTruth README (https://github.com/ParticleMedia/RAGTruth), so these are checks
against the real published data shape, not against something invented here.
Response "1472" carries a label at [219:229] which the authors say is
"Gaza Strip"; test_published_sample_offsets_are_exact confirms that our reading
of the offset convention -- plain Python half-open character offsets -- agrees.

Nothing here needs a tokenizer or the network. The BIO tests supply an offset
mapping directly, which is also how the real tokenizer output is shaped.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.c1_detector.bio import (  # noqa: E402
    ANSWER_SEQUENCE_ID,
    B_HAL,
    I_HAL,
    IGNORE_INDEX,
    OUTSIDE,
    assert_round_trip,
    bio_to_binary,
    bio_to_char_spans,
    char_spans_to_bio,
)
from src.c1_detector.build_examples import (  # noqa: E402
    build_context_and_question,
    clean_spans,
    merge_overlapping,
)
from src.c1_detector.ragtruth_labels import (  # noqa: E402
    assert_label_coverage,
    error_type_of,
    task_type_of,
    unmapped_error_types,
)
from src.common.schema import ErrorType, TaskType  # noqa: E402

# Verbatim from the RAGTruth README's response.jsonl sample.
SAMPLE_RESPONSE = (
    "The Palestinian Authority has officially become the 123rd member of the "
    "International Criminal Court (ICC), giving the court jurisdiction over "
    "alleged crimes in Palestinian territories. This includes East Jerusalem "
    "and Gaza Strip, which are occupied by Israel. The signing of Rome Statute "
    "by Palestinians in January 2021 had already established ICC's jurisdiction "
    'over alleged crimes committed "since June 13, 2014" in these areas. Now, '
    "the court can open a preliminary investigation or formal investigation "
    "into the situation in Palestinian territories, potentially leading to war "
    "crimes probes against Israeli individuals. However, this could also lead "
    "to counter-charges against Palestinians. The ICC welcomed Palestine's "
    "accession, while Israel and the US, who are not ICC members, opposed the "
    "move."
)

SAMPLE_LABEL = {
    "start": 219,
    "end": 229,
    "text": "Gaza Strip",
    "meta": "HIGH INTRO OF NEW INFO\nIt is not mentioned in the original source...",
    "label_type": "Evident Baseless Info",
}


# --------------------------------------------------------------------------
# The published offset convention
# --------------------------------------------------------------------------


def test_published_sample_offsets_are_exact():
    """RAGTruth offsets are plain Python half-open character offsets.

    If this ever fails, every span in the corpus is being read wrong and no
    downstream number means anything.
    """
    assert SAMPLE_RESPONSE[219:229] == "Gaza Strip"
    assert len(SAMPLE_RESPONSE) == 803


def test_clean_spans_accepts_the_published_sample():
    spans, problems = clean_spans(SAMPLE_RESPONSE, [SAMPLE_LABEL], "1472")
    assert problems == []
    assert len(spans) == 1
    assert spans[0]["start"] == 219
    assert spans[0]["end"] == 229
    assert spans[0]["text"] == "Gaza Strip"
    assert spans[0]["error_type"] == ErrorType.EVIDENT_BASELESS.value
    assert spans[0]["implicit_true"] is False


# --------------------------------------------------------------------------
# Span validation
# --------------------------------------------------------------------------


def test_offset_text_mismatch_is_reported_and_dropped():
    bad = dict(SAMPLE_LABEL, start=218)  # shifted one character left
    spans, problems = clean_spans(SAMPLE_RESPONSE, [bad], "1472")
    assert spans == []
    assert len(problems) == 1
    assert "slices out" in problems[0]


def test_span_past_the_end_is_dropped():
    bad = dict(SAMPLE_LABEL, start=800, end=9999)
    spans, problems = clean_spans(SAMPLE_RESPONSE, [bad], "1472")
    assert spans == []
    assert "outside a 803-char response" in problems[0]


def test_spans_come_back_sorted():
    labels = [
        {"start": 219, "end": 229, "text": "Gaza Strip", "label_type": "Evident Conflict"},
        {"start": 4, "end": 15, "text": SAMPLE_RESPONSE[4:15], "label_type": "Subtle Conflict"},
    ]
    spans, problems = clean_spans(SAMPLE_RESPONSE, labels, "1472")
    assert problems == []
    assert [s["start"] for s in spans] == [4, 219]


# --------------------------------------------------------------------------
# Overlap merging
#
# 125 of RAGTruth's 14,289 spans overlap another span on the same response:
# 62 exact duplicates, 58 nested, 5 partial. Measured on the real corpus,
# 2026-08-11.
# --------------------------------------------------------------------------


def _span(start, end, text, error_type="evident_conflict", **kw):
    base = {
        "start": start,
        "end": end,
        "text": text,
        "error_type": error_type,
        "implicit_true": False,
        "due_to_null": False,
    }
    base.update(kw)
    return base


def test_non_overlapping_spans_pass_through_unchanged():
    spans = [_span(4, 15, SAMPLE_RESPONSE[4:15]), _span(219, 229, "Gaza Strip")]
    merged, collapsed = merge_overlapping(SAMPLE_RESPONSE, spans)
    assert collapsed == 0
    assert [(s["start"], s["end"]) for s in merged] == [(4, 15), (219, 229)]


def test_exact_duplicate_spans_collapse_to_one():
    """62 pairs in the corpus are byte-identical duplicates."""
    spans = [_span(219, 229, "Gaza Strip"), _span(219, 229, "Gaza Strip")]
    merged, collapsed = merge_overlapping(SAMPLE_RESPONSE, spans)
    assert collapsed == 1
    assert len(merged) == 1
    assert (merged[0]["start"], merged[0]["end"]) == (219, 229)


def test_nested_span_is_absorbed_by_the_outer_one():
    spans = [
        _span(219, 229, SAMPLE_RESPONSE[219:229]),
        _span(219, 250, SAMPLE_RESPONSE[219:250]),
    ]
    merged, collapsed = merge_overlapping(SAMPLE_RESPONSE, spans)
    assert collapsed == 1
    assert (merged[0]["start"], merged[0]["end"]) == (219, 250)


def test_partial_overlap_extends_to_cover_both():
    spans = [
        _span(219, 240, SAMPLE_RESPONSE[219:240]),
        _span(230, 260, SAMPLE_RESPONSE[230:260]),
    ]
    merged, collapsed = merge_overlapping(SAMPLE_RESPONSE, spans)
    assert collapsed == 1
    assert (merged[0]["start"], merged[0]["end"]) == (219, 260)


def test_merged_text_is_resliced_from_the_answer():
    """The merged span's text must come from the answer, never be concatenated."""
    spans = [
        _span(219, 240, SAMPLE_RESPONSE[219:240]),
        _span(230, 260, SAMPLE_RESPONSE[230:260]),
    ]
    merged, _ = merge_overlapping(SAMPLE_RESPONSE, spans)
    s = merged[0]
    assert s["text"] == SAMPLE_RESPONSE[s["start"] : s["end"]]


def test_longest_contributor_wins_the_category_and_all_are_kept():
    """31 overlapping pairs disagree on error_type."""
    spans = [
        _span(219, 229, SAMPLE_RESPONSE[219:229], error_type="evident_baseless_info"),
        _span(219, 260, SAMPLE_RESPONSE[219:260], error_type="evident_conflict"),
    ]
    merged, _ = merge_overlapping(SAMPLE_RESPONSE, spans)
    assert merged[0]["error_type"] == "evident_conflict"
    assert merged[0]["error_types"] == ["evident_baseless_info", "evident_conflict"]


def test_merged_flags_are_conservative():
    """implicit_true only survives if every contributor had it; due_to_null if any did."""
    spans = [
        _span(219, 229, SAMPLE_RESPONSE[219:229], implicit_true=True, due_to_null=True),
        _span(219, 260, SAMPLE_RESPONSE[219:260], implicit_true=False, due_to_null=False),
    ]
    merged, _ = merge_overlapping(SAMPLE_RESPONSE, spans)
    assert merged[0]["implicit_true"] is False
    assert merged[0]["due_to_null"] is True


def test_three_way_chain_merges_into_one():
    spans = [
        _span(219, 240, SAMPLE_RESPONSE[219:240]),
        _span(235, 260, SAMPLE_RESPONSE[235:260]),
        _span(255, 280, SAMPLE_RESPONSE[255:280]),
    ]
    merged, collapsed = merge_overlapping(SAMPLE_RESPONSE, spans)
    assert collapsed == 2
    assert (merged[0]["start"], merged[0]["end"]) == (219, 280)


def test_merging_empty_span_list():
    assert merge_overlapping(SAMPLE_RESPONSE, []) == ([], 0)


def test_merged_spans_never_overlap():
    """The postcondition the BIO round trip depends on."""
    spans = [
        _span(219, 229, SAMPLE_RESPONSE[219:229]),
        _span(219, 229, SAMPLE_RESPONSE[219:229]),
        _span(225, 260, SAMPLE_RESPONSE[225:260]),
        _span(300, 320, SAMPLE_RESPONSE[300:320]),
    ]
    merged, _ = merge_overlapping(SAMPLE_RESPONSE, spans)
    for a, b in zip(merged, merged[1:]):
        assert a["end"] <= b["start"]


def test_drop_implicit_true_flag():
    label = dict(SAMPLE_LABEL, implicit_true=True)
    kept, _ = clean_spans(SAMPLE_RESPONSE, [label], "1472", drop_implicit_true=False)
    dropped, _ = clean_spans(SAMPLE_RESPONSE, [label], "1472", drop_implicit_true=True)
    assert len(kept) == 1 and kept[0]["implicit_true"] is True
    assert dropped == []


def test_response_with_no_labels_yields_no_spans():
    spans, problems = clean_spans(SAMPLE_RESPONSE, [], "1472")
    assert spans == [] and problems == []


# --------------------------------------------------------------------------
# Label and task mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Evident Conflict", ErrorType.EVIDENT_CONFLICT),
        ("Subtle Conflict", ErrorType.SUBTLE_CONFLICT),
        ("Evident Baseless Info", ErrorType.EVIDENT_BASELESS),
        ("Subtle Baseless Info", ErrorType.SUBTLE_BASELESS),
        ("  evident   baseless   info  ", ErrorType.EVIDENT_BASELESS),
        ("something the annotators invented", ErrorType.UNKNOWN),
    ],
)
def test_error_type_mapping(raw, expected):
    assert error_type_of(raw) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("QA", TaskType.QA),
        ("Data2txt", TaskType.DATA2TEXT),
        ("Summary", TaskType.SUMMARIZATION),
        ("qa", TaskType.QA),
    ],
)
def test_task_type_mapping(raw, expected):
    assert task_type_of(raw) is expected


def test_unknown_task_type_is_a_hard_error():
    """Silently bucketing an unknown task would corrupt the per-task breakdown."""
    with pytest.raises(KeyError, match="unrecognised RAGTruth task_type"):
        task_type_of("Translation")


def test_label_coverage_check():
    assert unmapped_error_types(["Evident Conflict", "Subtle Conflict"]) == []
    assert unmapped_error_types(["Mystery Label"]) == ["mystery label"]
    assert_label_coverage(["Evident Conflict"])
    with pytest.raises(ValueError, match="no mapping"):
        assert_label_coverage(["Mystery Label"])


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------


def test_qa_context_uses_passages_and_question():
    context, question = build_context_and_question(
        TaskType.QA, {"question": "how to prepare beets", "passages": "passage 1: wash them"}
    )
    assert context == "passage 1: wash them"
    assert question == "how to prepare beets"


def test_data2text_context_is_deterministic_json():
    info = {"name": "Subway", "city": "Santa Barbara", "business_stars": 3.0}
    a, qa = build_context_and_question(TaskType.DATA2TEXT, info)
    b, _ = build_context_and_question(TaskType.DATA2TEXT, dict(reversed(list(info.items()))))
    assert a == b, "key order must not change the serialised context"
    assert "Subway" in a and qa


def test_summary_context_is_the_article_string():
    context, question = build_context_and_question(TaskType.SUMMARIZATION, "The article.")
    assert context == "The article."
    assert question


def test_qa_with_non_dict_source_info_raises():
    with pytest.raises(TypeError):
        build_context_and_question(TaskType.QA, "not a dict")


# --------------------------------------------------------------------------
# BIO tagging
# --------------------------------------------------------------------------


def word_offsets(text: str):
    """A stand-in tokenizer that splits on spaces.

    Produces the same (offsets, sequence_ids) shape a fast tokenizer returns for
    a sequence pair: a leading special, the answer tokens, a trailing special.
    """
    offsets = [(0, 0)]
    seq_ids = [None]
    cursor = 0
    for word in text.split(" "):
        offsets.append((cursor, cursor + len(word)))
        seq_ids.append(ANSWER_SEQUENCE_ID)
        cursor += len(word) + 1
    offsets.append((0, 0))
    seq_ids.append(None)
    return offsets, seq_ids


ANSWER = "Sigiriya was built in 477 CE by 25,000 workers"
#         0         1         2         3         4
#         0123456789012345678901234567890123456789012345


def test_non_answer_tokens_are_ignored():
    offsets, seq_ids = word_offsets(ANSWER)
    labels = char_spans_to_bio(offsets, seq_ids, [])
    assert labels[0] == IGNORE_INDEX
    assert labels[-1] == IGNORE_INDEX
    assert all(label == OUTSIDE for label in labels[1:-1])


def test_single_span_gets_b_then_i():
    offsets, seq_ids = word_offsets(ANSWER)
    span = (ANSWER.index("477"), ANSWER.index("477") + len("477 CE"))
    labels = char_spans_to_bio(offsets, seq_ids, [span])
    tagged = [label for label in labels if label in (B_HAL, I_HAL)]
    assert tagged == [B_HAL, I_HAL]


def test_two_adjacent_spans_each_start_with_b():
    """The whole reason for BIO over binary labels."""
    offsets, seq_ids = word_offsets(ANSWER)
    a = (ANSWER.index("477"), ANSWER.index("477") + 3)
    b = (ANSWER.index("CE"), ANSWER.index("CE") + 2)
    labels = char_spans_to_bio(offsets, seq_ids, [a, b])
    tagged = [label for label in labels if label in (B_HAL, I_HAL)]
    assert tagged == [B_HAL, B_HAL]
    assert len(bio_to_char_spans(labels, offsets, seq_ids)) == 2


def test_partial_token_overlap_counts_as_hallucinated():
    """A span covering part of a token must still tag that token.

    Otherwise the model learns to shave the edge off every span.
    """
    offsets, seq_ids = word_offsets(ANSWER)
    start = ANSWER.index("25,000")
    labels = char_spans_to_bio(offsets, seq_ids, [(start, start + 2)])  # just "25"
    assert sum(1 for label in labels if label in (B_HAL, I_HAL)) == 1


def test_round_trip_recovers_the_original_span():
    offsets, seq_ids = word_offsets(ANSWER)
    start = ANSWER.index("25,000 workers")
    span = (start, start + len("25,000 workers"))
    labels = char_spans_to_bio(offsets, seq_ids, [span])
    assert bio_to_char_spans(labels, offsets, seq_ids) == [span]
    assert_round_trip([span], labels, offsets, seq_ids, ANSWER, tolerance=0)


def test_round_trip_detects_a_dropped_span():
    offsets, seq_ids = word_offsets(ANSWER)
    span = (ANSWER.index("477"), ANSWER.index("477") + 3)
    labels = char_spans_to_bio(offsets, seq_ids, [span])
    broken = [OUTSIDE if label in (B_HAL, I_HAL) else label for label in labels]
    with pytest.raises(AssertionError, match="span count"):
        assert_round_trip([span], broken, offsets, seq_ids, ANSWER)


def test_trimming_removes_a_leading_space_from_a_decoded_span():
    """ModernBERT's byte-level BPE puts the preceding space inside the token.

    Measured on 400 real training records: 369 decoded one character wide
    without trimming, none wider.
    """
    text = "a bb ccc"
    offsets = [(0, 0), (0, 1), (1, 4), (4, 8), (0, 0)]  # tokens carry leading spaces
    seq_ids = [None, ANSWER_SEQUENCE_ID, ANSWER_SEQUENCE_ID, ANSWER_SEQUENCE_ID, None]
    labels = [IGNORE_INDEX, OUTSIDE, B_HAL, OUTSIDE, IGNORE_INDEX]

    assert bio_to_char_spans(labels, offsets, seq_ids) == [(1, 4)]
    trimmed = bio_to_char_spans(labels, offsets, seq_ids, answer=text)
    assert trimmed == [(2, 4)]
    assert text[2:4] == "bb"


def test_trimming_drops_a_span_that_is_only_whitespace():
    text = "a   b"
    offsets = [(0, 0), (1, 4), (0, 0)]
    seq_ids = [None, ANSWER_SEQUENCE_ID, None]
    labels = [IGNORE_INDEX, B_HAL, IGNORE_INDEX]
    assert bio_to_char_spans(labels, offsets, seq_ids, answer=text) == []


def test_round_trip_flags_widening_beyond_tolerance():
    """A span starting mid-token decodes wider; tolerance controls acceptance."""
    offsets, seq_ids = word_offsets(ANSWER)
    start = ANSWER.index("25,000")
    span = (start + 3, start + 6)  # "000", inside the token "25,000"
    labels = char_spans_to_bio(offsets, seq_ids, [span])
    with pytest.raises(AssertionError, match="wider"):
        assert_round_trip([span], labels, offsets, seq_ids, ANSWER, tolerance=0)
    assert_round_trip([span], labels, offsets, seq_ids, ANSWER, tolerance=3)


def test_stray_i_without_b_still_decodes():
    offsets, seq_ids = word_offsets(ANSWER)
    labels = [IGNORE_INDEX] + [OUTSIDE] * (len(offsets) - 2) + [IGNORE_INDEX]
    labels[3] = I_HAL
    assert len(bio_to_char_spans(labels, offsets, seq_ids)) == 1


def test_bio_to_binary_preserves_the_loss_mask():
    labels = [IGNORE_INDEX, OUTSIDE, B_HAL, I_HAL, OUTSIDE, IGNORE_INDEX]
    assert bio_to_binary(labels) == [IGNORE_INDEX, 0, 1, 1, 0, IGNORE_INDEX]


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        char_spans_to_bio([(0, 1), (1, 2)], [ANSWER_SEQUENCE_ID], [])
