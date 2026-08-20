"""Tests for the LLM-judge harness. No network: the API call is never made here.

What is worth pinning is everything around the call. The cache has to save quota
across an interruption, the sample has to be stratified rather than the head of a
task-grouped file, and a quote the judge cannot place has to be counted rather
than dropped, because that count is the span-level column of this baseline.
"""

from __future__ import annotations


import pytest

from src.c1_detector.llm_judge import (
    Cache,
    build_prompt,
    cache_key,
    choose_model,
    evaluate_judge,
    format_report,
    locate,
    parse_verdict,
    stratified_sample,
)

ANSWER = "Paris is the capital of France and it rains there every single day."


def record(rid, task="qa", answer=ANSWER, gold=None):
    return {
        "id": rid,
        "task_type": task,
        "context": "France is a country in Europe. Its capital is Paris.",
        "question": "Tell me about France.",
        "answer": answer,
        "spans": gold or [],
    }


def test_the_prompt_carries_context_question_and_answer():
    prompt = build_prompt(record("a"))
    assert "Its capital is Paris." in prompt
    assert "Tell me about France." in prompt
    assert ANSWER in prompt


def test_the_prompt_braces_survive_formatting():
    """The JSON example in the prompt must reach the model as literal braces."""
    prompt = build_prompt(record("a"))
    assert '{"hallucinated": true or false' in prompt
    assert "{{" not in prompt


def test_the_cache_key_changes_with_the_prompt_and_with_the_model():
    a = cache_key("m1", "prompt one")
    assert a == cache_key("m1", "prompt one")
    assert a != cache_key("m1", "prompt two")
    assert a != cache_key("m2", "prompt one")


def test_the_cache_survives_a_restart(tmp_path):
    Cache(tmp_path).put("abc", {"hallucinated": True, "spans": ["x"]})
    assert Cache(tmp_path).get("abc")["spans"] == ["x"]


def test_a_missing_entry_reads_as_none(tmp_path):
    assert Cache(tmp_path).get("never-written") is None


def test_a_torn_write_is_re_asked_rather_than_trusted(tmp_path):
    cache = Cache(tmp_path)
    cache.path("half").write_text('{"hallucinated": tru', encoding="utf-8")
    assert cache.get("half") is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"hallucinated": true, "spans": ["it rains there every single day"]}',
        '```json\n{"hallucinated": true, "spans": ["it rains there every single day"]}\n```',
    ],
)
def test_the_verdict_parses_with_or_without_a_code_fence(raw):
    verdict = parse_verdict(raw)
    assert verdict["hallucinated"] is True
    assert verdict["spans"] == ["it rains there every single day"]
    assert verdict["parse_failed"] is False


def test_an_unparseable_reply_is_flagged_not_guessed():
    verdict = parse_verdict("I think the second sentence is made up.")
    assert verdict["parse_failed"] is True
    assert verdict["spans"] == []


def test_empty_spans_parse_as_no_hallucination():
    verdict = parse_verdict('{"hallucinated": false, "spans": []}')
    assert verdict["hallucinated"] is False
    assert verdict["spans"] == []


def test_a_verbatim_quote_becomes_character_offsets():
    spans, missing = locate(ANSWER, ["it rains there every single day"])
    assert missing == 0
    start, end = spans[0]
    assert ANSWER[start:end] == "it rains there every single day"


def test_a_paraphrased_quote_is_counted_not_dropped():
    """The judge's failure to quote exactly IS the span-level finding."""
    spans, missing = locate(ANSWER, ["it rains every day there"])
    assert spans == []
    assert missing == 1


def test_repeated_quotes_do_not_collapse_onto_one_offset():
    answer = "the cat sat. the cat sat."
    spans, missing = locate(answer, ["the cat sat", "the cat sat"])
    assert missing == 0
    assert spans[0] != spans[1]


def test_the_sample_is_stratified_rather_than_the_head_of_the_file():
    records = [record(f"s{i}", "summarization") for i in range(60)]
    records += [record(f"q{i}", "qa") for i in range(30)]
    records += [record(f"d{i}", "data2text") for i in range(10)]
    sample = stratified_sample(records, 20)
    tasks = {r["task_type"] for r in sample}
    assert tasks == {"summarization", "qa", "data2text"}


def test_the_sample_is_reproducible():
    records = [record(str(i), "qa" if i % 2 else "summarization") for i in range(50)]
    first = [r["id"] for r in stratified_sample(records, 10, seed=7)]
    second = [r["id"] for r in stratified_sample(records, 10, seed=7)]
    assert first == second


def test_asking_for_more_than_exists_returns_everything():
    records = [record(str(i)) for i in range(5)]
    assert len(stratified_sample(records, 500)) == 5


def test_the_newest_general_flash_model_is_chosen():
    available = [
        "gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.7-flash",
        "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-image",
        "gemini-2.5-pro",
    ]
    assert choose_model(available) == "gemini-3.7-flash"


def test_image_tts_and_preview_variants_are_not_treated_as_judges():
    available = ["gemini-3.1-flash-image", "gemini-3-flash-preview", "gemini-flash-latest"]
    assert choose_model(available) == "gemini-flash-latest"


def test_an_unusable_model_list_fails_loudly_rather_than_guessing():
    with pytest.raises(SystemExit, match="available models"):
        choose_model(["embedding-001", "aqa"])


@pytest.fixture
def judged():
    """One correct call, one missed, one false alarm, one clean."""
    return [
        {"id": "1", "task_type": "qa", "gold_spans": [(0, 5)], "pred_spans": [(0, 5)],
         "judge_said_hallucinated": True, "n_quotes": 1, "n_quotes_not_found": 0,
         "parse_failed": False},
        {"id": "2", "task_type": "qa", "gold_spans": [(0, 5)], "pred_spans": [],
         "judge_said_hallucinated": False, "n_quotes": 0, "n_quotes_not_found": 0,
         "parse_failed": False},
        {"id": "3", "task_type": "qa", "gold_spans": [], "pred_spans": [(2, 9)],
         "judge_said_hallucinated": True, "n_quotes": 1, "n_quotes_not_found": 0,
         "parse_failed": False},
        {"id": "4", "task_type": "qa", "gold_spans": [], "pred_spans": [],
         "judge_said_hallucinated": False, "n_quotes": 2, "n_quotes_not_found": 2,
         "parse_failed": False},
    ]


def test_the_verdict_and_the_quotes_are_scored_separately(judged):
    report = evaluate_judge(judged)
    assert report["example"]["tp"] == 1
    assert report["span_exact"]["tp"] == 1
    assert report["n"] == 4


def test_the_trivial_baseline_travels_with_the_score(judged):
    report = evaluate_judge(judged)
    assert report["positive_rate"] == pytest.approx(0.5)
    assert report["trivial_f1"] == pytest.approx(2 * 0.5 / 1.5)
    assert isinstance(report["clears_trivial"], bool)


def test_unlocatable_quotes_are_reported_as_a_rate(judged):
    report = evaluate_judge(judged)
    assert report["quotes_requested"] == 4
    assert report["quotes_not_found_verbatim"] == 2
    assert report["quote_location_failure_rate"] == pytest.approx(0.5)


def test_the_report_says_a_quote_that_cannot_be_placed_cannot_be_highlighted(judged):
    text = format_report({"model": "m", **evaluate_judge(judged)})
    assert "do not appear verbatim" in text
    assert "cannot be highlighted" in text


def test_an_empty_placeholder_does_not_shadow_the_real_key():
    """A .env that documents GEMINI_API_KEY= above the real value must still work."""
    from src.c1_detector.llm_judge import parse_env

    values = parse_env(
        "# keys\nGEMINI_API_KEY=\nGROQ_API_KEY=\n\nGEMINI_API_KEY=the-real-one\n"
    )
    assert values["GEMINI_API_KEY"] == "the-real-one"
    assert "GROQ_API_KEY" not in values


def test_quoted_values_are_unwrapped():
    from src.c1_detector.llm_judge import parse_env

    assert parse_env('A="x"\nB=\'y\'\n') == {"A": "x", "B": "y"}


def test_comments_and_blank_lines_are_ignored():
    from src.c1_detector.llm_judge import parse_env

    assert parse_env("\n# a comment\n\nA=1\nnot a pair\n") == {"A": "1"}


# ---------------------------------------------------------------------------
# A retired model answers 404 while ListModels still advertises it. That is a
# naming error, so it must abort at once rather than spend the failure budget.
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import io  # noqa: E402
from collections import Counter  # noqa: E402
import json  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from src.c1_detector import llm_judge as judge_module  # noqa: E402
from src.c1_detector.llm_judge import (  # noqa: E402
    JudgeUnavailable,
    ModelUnusable,
    call_model,
)


def http_error(code, message):
    body = io.BytesIO(json.dumps({"error": {"message": message}}).encode("utf-8"))
    return urllib.error.HTTPError("http://x", code, "err", {}, body)


def test_404_aborts_on_the_first_attempt_with_the_api_message(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        raise http_error(404, "This model models/gemini-2.5-flash is no longer available")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ModelUnusable) as caught:
        call_model("key", "gemini-2.5-flash", "prompt", attempts=8)

    assert len(calls) == 1, "a 404 must not be retried; no sample size fixes it"
    assert "no longer available" in str(caught.value)
    assert "gemini-2.5-flash" in str(caught.value)


def test_503_still_retries_and_then_succeeds(monkeypatch):
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise http_error(503, "high demand")
        payload = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
        return contextlib.closing(io.BytesIO(json.dumps(payload).encode("utf-8")))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge_module.time, "sleep", lambda _seconds: None)

    assert call_model("key", "gemini-3.6-flash", "prompt", attempts=3) == "{}"
    assert len(attempts) == 2


def test_a_spent_quota_is_still_a_skippable_failure(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(http_error(429, "quota"))
    )
    monkeypatch.setattr(judge_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(JudgeUnavailable):
        call_model("key", "gemini-3.6-flash", "prompt", attempts=2)


# ---------------------------------------------------------------------------
# The free quota stops a run partway through, so any prefix of the sample has to
# be usable. Grouped by task it was not: the first twenty answers were all one
# task type.
# ---------------------------------------------------------------------------

from src.c1_detector.llm_judge import stratified_sample  # noqa: E402


def corpus():
    return [
        {"id": f"{task}-{i}", "task_type": task, "answer": "a"}
        for task in ("data2text", "qa", "summarization")
        for i in range(300)
    ]


def test_a_partial_run_still_covers_every_task():
    sample = stratified_sample(corpus(), 150, seed=42)

    assert len(sample) == 150
    for cut in (3, 9, 21, 60):
        tasks = {record["task_type"] for record in sample[:cut]}
        assert tasks == {"data2text", "qa", "summarization"}, (
            f"the first {cut} judgements cover only {tasks}; a run the quota "
            "cut short would be a sample of one task type"
        )


def test_the_whole_sample_is_still_proportional():
    sample = stratified_sample(corpus(), 150, seed=42)
    counts = Counter(record["task_type"] for record in sample)
    assert counts == {"data2text": 50, "qa": 50, "summarization": 50}


def test_the_sample_is_reproducible_from_the_seed():
    first = [r["id"] for r in stratified_sample(corpus(), 30, seed=42)]
    second = [r["id"] for r in stratified_sample(corpus(), 30, seed=42)]
    other = [r["id"] for r in stratified_sample(corpus(), 30, seed=7)]

    assert first == second
    assert first != other
