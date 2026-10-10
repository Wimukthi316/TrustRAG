"""The evidence-label extractor: the marker vocabulary, and the noise it must ignore."""

from __future__ import annotations

import pytest

from src.c4_attribution import evidence_labels as el


def test_marker_list_is_closed_and_lowercase():
    assert el.MARKERS == ("original", "generative", "generated", "aigc", "aogc")
    assert all(m == m.lower() for m in el.MARKERS)


@pytest.mark.parametrize(
    "meta,expected,marker",
    [
        ("EVIDENT CONFLICT\nOriginal: February 7 (1945)\nGenerated: February 7, 2022",
         "February 7 (1945)", "generated"),
        ("Original: the source says X\nGenerative: the model said Y",
         "the source says X", "generative"),
        ("Original: quoted text\nAIGC: generated text", "quoted text", "aigc"),
        ("Original: the quote runs to the end of the field", 
         "the quote runs to the end of the field", None),
    ],
)
def test_extraction_uses_every_marker(meta, expected, marker):
    assert el.original_snippet(meta) == expected
    assert el.closing_marker(meta) == marker


def test_a_generic_marker_pattern_would_truncate_a_real_quote():
    # The observed noise: Review:, Sunday:, Friday:, Weekends: appear INSIDE
    # quoted source text. A generic \n\s*Word\s*: scan stops there and destroys
    # the quote; the closed list does not.
    meta = "Original: Opening hours\nWeekends: 9am to 5pm\nGenerated: open 24 hours"
    assert el.original_snippet(meta) == "Opening hours\nWeekends: 9am to 5pm"
    assert el.closing_marker(meta) == "generated"
    assert "Weekends" in el.false_marker_candidates(meta)


def test_no_original_marker_is_none_not_empty_string():
    assert el.original_snippet("EVIDENT CONFLICT\nGenerated: something") is None
    assert el.has_original_marker("EVIDENT CONFLICT") is False


def test_the_two_conventions_are_allowed_to_disagree():
    meta = "Original: line one\nline two\nGenerated: something"
    assert el.original_snippet(meta) == "line one\nline two"
    assert el.original_snippet_first_line(meta) == "line one"


@pytest.mark.parametrize(
    "snippet,expected",
    [
        ("The date is not mentioned in the passage.", True),
        ("There is no information about the price.", True),
        ("The passage does not state the author.", True),
        ("February 7 (1945)", False),
        ("The building was completed in 1998.", False),
    ],
)
def test_no_evidence_phrase_set(snippet, expected):
    assert el.is_no_evidence(snippet) is expected


def test_attributable_requires_a_quote_that_is_not_a_denial():
    quote = el.label_from_annotation(
        "1",
        {"start": 0, "end": 3, "text": "abc", "meta": "Original: real source text",
         "label_type": "Evident Conflict", "implicit_true": False, "due_to_null": False},
    )
    denial = el.label_from_annotation(
        "1",
        {"start": 0, "end": 3, "text": "abc", "meta": "Original: not mentioned anywhere",
         "label_type": "Evident Baseless Info", "implicit_true": False, "due_to_null": False},
    )
    assert quote.attributable is True
    assert denial.attributable is False
    assert denial.no_evidence is True


def test_marker_counts_covers_every_case_including_none():
    metas = [
        "Original: a\nGenerated: b",
        "Original: a\nGenerative: b",
        "Original: a\nAIGC: b",
        "Original: a",
        "no marker here",
    ]
    counts = el.marker_counts(metas)
    assert counts["generated"] == 1
    assert counts["generative"] == 1
    assert counts["aigc"] == 1
    assert counts["none"] == 1
    assert sum(counts.values()) == 4


def test_the_hand_check_starts_as_a_todo_and_becomes_a_rate(tmp_path):
    """The no-evidence label is a heuristic until a human has read fifty of them."""
    import json

    from src.c4_attribution import evidence_audit

    path = tmp_path / "sample.json"
    path.write_text(json.dumps([{"human_says_no_evidence": None} for _ in range(50)]))
    block = evidence_audit.hand_check_agreement(path)
    assert block["agreement"] is None
    assert block["status"].startswith("TODO")

    rows = [{"human_says_no_evidence": True} for _ in range(46)]
    rows += [{"human_says_no_evidence": False} for _ in range(4)]
    path.write_text(json.dumps(rows))
    block = evidence_audit.hand_check_agreement(path)
    assert block["agreement"] == 0.92
    assert "may be quoted as a label" in block["verdict"]

    rows = [{"human_says_no_evidence": True} for _ in range(30)]
    rows += [{"human_says_no_evidence": False} for _ in range(20)]
    path.write_text(json.dumps(rows))
    block = evidence_audit.hand_check_agreement(path)
    assert block["agreement"] == 0.60
    assert "Do not quietly keep the regex" in block["verdict"]


def test_a_rerun_never_wipes_labels_somebody_entered_by_hand(tmp_path, monkeypatch):
    import json

    from src.c4_attribution import evidence_audit

    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps([{"snippet": "x", "human_says_no_evidence": True}]))
    monkeypatch.setattr(evidence_audit, "SAMPLE", sample)
    monkeypatch.setattr(evidence_audit, "SAMPLE_SHEET", tmp_path / "sheet.md")
    monkeypatch.setattr(evidence_audit, "OUT_DIR", tmp_path)
    monkeypatch.setattr(evidence_audit, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(evidence_audit, "span_records", lambda *_: [])
    monkeypatch.setattr(evidence_audit, "load_responses", lambda *a, **k: [])
    evidence_audit.main([])
    assert json.loads(sample.read_text())[0]["human_says_no_evidence"] is True


def test_committed_hand_check_is_complete_and_meets_threshold():
    """Verify the 50-snippet human audit on disk is 100% labelled and >= 0.90 agreement."""
    from src.c4_attribution import evidence_audit

    block = evidence_audit.hand_check_agreement()
    assert block["n_sampled"] == 50
    assert block["labelled"] == 50
    assert block["agreement"] == 0.94
    assert "may be quoted as a label" in block["verdict"]

