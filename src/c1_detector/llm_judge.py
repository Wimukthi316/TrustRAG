"""An LLM judge on RAGTruth, and whether it can point at the words.

"Why not just use an LLM?" is the first question a panel asks, and answering it
with an opinion loses. This runs one and reports its number.

There is a second, sharper question underneath. An LLM can obviously say whether
an answer contains a hallucination. Can it say WHICH WORDS? This harness asks for
both in the same call: a yes or no, and the exact quoted text of every
unsupported span. Then it tries to find each quote in the answer. Quotes that do
not appear verbatim are counted, not silently dropped, because a judge that
cannot produce text you can locate cannot highlight anything either, and that is
the same gap C1 exists to measure.

Design decisions that matter more than the model choice:

Every judgement is cached on disk under a hash of the exact prompt. Free quota is
the binding constraint, a run will be interrupted, and nothing is more wasteful
than paying twice for the same answer. Re-running after an interruption costs
nothing for the records already done.

The prompt is a module constant, committed, and the hash covers it. Change the
prompt and the cache misses, which is the correct behaviour: those are different
judgements and pretending otherwise would mix two experiments in one table.

The model is discovered from the API rather than hard-coded. Model identifiers
change, a guessed one fails an hour into a run, and a wrong guess written into a
report is a fabricated citation. list_models asks the account what it can
actually call and the chosen id is recorded in the output.

Sampling is stratified over the three task types and seeded. A few hundred
responses is enough for a baseline row and the quota will not stretch to 2,700.

No new dependency: this uses urllib from the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.c1_detector.evaluate_c1 import (
    example_metrics,
    span_char_metrics,
    span_overlap_metrics,
)
from src.c1_detector.ood_operating_point import trivial_f1

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# A plain "gemini-<version>-flash", newest version first.
#
# Deliberately not the cheapest available model. The claim this baseline supports
# is that a capable general model still cannot say WHICH WORDS are unsupported,
# and running a deliberately weak judge would make C1 look good by rigging the
# comparison. Flash rather than Pro because the free tier has to carry a few
# hundred calls and this is a classification task, not a reasoning one.
GENERAL_FLASH = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash$")

# The prompt, verbatim and committed. It goes into the cache key, so editing it
# invalidates every cached judgement rather than quietly mixing two experiments.
PROMPT = """You are checking whether an AI-generated answer is fully supported by the source text it was given.

SOURCE TEXT:
{context}

REQUEST GIVEN TO THE AI:
{question}

THE AI'S ANSWER:
{answer}

A span is hallucinated if it states something the SOURCE TEXT does not support, whether it contradicts the source or simply adds information the source never mentions. Correct-sounding general knowledge still counts as hallucinated if the source does not state it.

Reply with JSON only, in exactly this form:
{{"hallucinated": true or false, "spans": ["exact quoted text from the answer", ...]}}

Rules for spans:
- Quote the answer EXACTLY, character for character. Do not paraphrase, correct, reorder or add ellipses.
- Quote the shortest span that carries the unsupported claim, not the whole sentence around it.
- If nothing is unsupported, use {{"hallucinated": false, "spans": []}}.
"""


def parse_env(text: str) -> Dict[str, str]:
    """KEY=VALUE lines to a dict, ignoring blanks, comments and empty values.

    Empty values are skipped on purpose. A .env commonly carries commented
    placeholder lines like `GEMINI_API_KEY=` as documentation, and a placeholder
    must not shadow the real value further down the file. Later non-empty lines
    win over earlier ones, which is what someone appending a key expects.
    """
    values: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value:
            values[key] = value
    return values


def load_env(path: Path | str = ".env") -> None:
    """Load .env without adding a dependency. A real environment variable wins."""
    file = Path(path)
    if not file.exists():
        return
    for key, value in parse_env(file.read_text(encoding="utf-8")).items():
        if not os.environ.get(key):
            os.environ[key] = value


def api_key() -> str:
    load_env()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    raise SystemExit(
        "no GEMINI_API_KEY with a value. Put it in .env (which is gitignored) as "
        "GEMINI_API_KEY=..., not on the command line"
    )


def _get(url: str, timeout: int = 60) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_models(key: str) -> List[str]:
    """Model ids this account can actually call generateContent on."""
    payload = _get(f"{API_ROOT}/models?key={key}&pageSize=200")
    names: List[str] = []
    for model in payload.get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            names.append(model["name"].removeprefix("models/"))
    return sorted(names)


def choose_model(available: Sequence[str]) -> str:
    """Newest plain flash model the account can call, and say so if there is none.

    Guessing an identifier is how a run dies an hour in, and how a model name that
    never existed ends up cited in a report. Image, TTS, robotics and preview
    variants are excluded: they are not general text judges, and a preview id can
    disappear between running the experiment and defending it.
    """
    versioned = [
        (float(match.group(1)), name)
        for name, match in ((n, GENERAL_FLASH.match(n)) for n in available)
        if match
    ]
    if versioned:
        return max(versioned)[1]
    for fallback in ("gemini-flash-latest", "gemini-pro-latest"):
        if fallback in available:
            return fallback
    raise SystemExit(
        f"no general flash model among the {len(available)} available models: "
        f"{sorted(available)[:20]}"
    )


def build_prompt(record: Dict[str, Any]) -> str:
    return PROMPT.format(
        context=record["context"],
        question=record.get("question", ""),
        answer=record["answer"],
    )


def cache_key(model: str, prompt: str) -> str:
    """The model and the prompt together. Same question, same model, same answer."""
    return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()


class Cache:
    """One small JSON file per judgement, named by hash.

    A directory of files rather than one big file: a run that dies mid-write
    loses one judgement, not all of them, and the free quota makes that the
    failure worth protecting against.
    """

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None  # a torn write; ask again rather than trust it

    def put(self, key: str, value: Dict[str, Any]) -> None:
        self.path(key).write_text(json.dumps(value), encoding="utf-8")


class JudgeUnavailable(RuntimeError):
    """The API would not answer this prompt after every retry."""


class ModelUnusable(RuntimeError):
    """The chosen model cannot be called at all, so no sample size will help.

    ListModels advertises generateContent for models that have since been
    retired: gemini-2.5-flash is listed and answers 404 "no longer available to
    new users". That is a naming error, not a busy server, so it aborts on the
    first record instead of spending the failure budget discovering it sixteen
    times.
    """


def _error_message(error: urllib.error.HTTPError) -> str:
    """The API's own explanation, which says what is wrong with the model id."""
    try:
        return json.loads(error.read().decode("utf-8", "replace"))["error"]["message"]
    except Exception:  # noqa: BLE001 - the body is best effort
        return str(error)


def call_model(
    key: str, model: str, prompt: str, attempts: int = 8, timeout: int = 120
) -> str:
    """One generateContent call, retrying on rate limits and transient errors.

    503 means the model is busy rather than the quota being spent, and on a free
    tier the newest model is the busiest. Backing off further is usually enough;
    when it is not, the caller records the failure and moves on rather than
    losing the rest of a run to one unlucky record.
    """
    url = f"{API_ROOT}/models/{model}:generateContent?key={key}"
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
    ).encode("utf-8")

    delay = 5.0
    last = "unknown"
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            candidates = payload.get("candidates") or []
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts") or []
            return "".join(part.get("text", "") for part in parts)
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}"
            if error.code == 404:
                raise ModelUnusable(f"{model}: {_error_message(error)}") from error
            if error.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(min(delay, 120.0))
                delay *= 2
                continue
            raise JudgeUnavailable(f"{last} after {attempt + 1} attempts") from error
        except (urllib.error.URLError, TimeoutError) as error:
            last = str(error)
            if attempt < attempts - 1:
                time.sleep(min(delay, 120.0))
                delay *= 2
                continue
            raise JudgeUnavailable(f"{last} after {attempts} attempts") from error
    raise JudgeUnavailable(f"{last} after {attempts} attempts")


def parse_verdict(text: str) -> Dict[str, Any]:
    """Read the judge's JSON, tolerating the usual code-fence wrapping."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"hallucinated": None, "spans": [], "parse_failed": True}
    spans = payload.get("spans") or []
    return {
        "hallucinated": bool(payload.get("hallucinated")),
        "spans": [s for s in spans if isinstance(s, str) and s],
        "parse_failed": False,
    }


def locate(answer: str, quotes: Sequence[str]) -> Tuple[List[Tuple[int, int]], int]:
    """Turn quoted text into character offsets. Returns (spans, quotes not found).

    The count of quotes that do not appear verbatim is a result, not a nuisance.
    A judge whose quotes cannot be located cannot highlight anything, which is
    the whole distinction C1 is measuring.
    """
    spans: List[Tuple[int, int]] = []
    missing = 0
    cursor = 0
    for quote in quotes:
        index = answer.find(quote, cursor)
        if index < 0:
            index = answer.find(quote)
        if index < 0:
            missing += 1
            continue
        spans.append((index, index + len(quote)))
        cursor = index + len(quote)
    return spans, missing


def stratified_sample(
    records: Sequence[Dict[str, Any]], n: int, seed: int = 42
) -> List[Dict[str, Any]]:
    """Proportional over task types, seeded, so the row is reproducible.

    Taking the head of the file would be worse than useless: build_examples
    writes records grouped by task, so the first few hundred are all one task.
    That exact mistake has already been made once on this project.
    """
    if n >= len(records):
        return list(records)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("task_type", "all")), []).append(record)

    rng = random.Random(seed)
    per_task: List[List[Dict[str, Any]]] = []
    for task in sorted(grouped):
        members = sorted(grouped[task], key=lambda r: str(r["id"]))
        rng.shuffle(members)
        take = max(1, round(n * len(grouped[task]) / len(records)))
        per_task.append(members[:take])

    # Round-robin across tasks rather than one task after another. The free
    # quota stops a run partway through, and every partial run so far ended
    # inside the first task, which is a sample of one task type and not a
    # baseline. Interleaved, whatever the quota allows stays balanced.
    chosen: List[Dict[str, Any]] = []
    for index in range(max(len(members) for members in per_task)):
        for members in per_task:
            if index < len(members):
                chosen.append(members[index])
    return chosen[:n]


def judge_records(
    records: Sequence[Dict[str, Any]],
    key: str,
    model: str,
    cache: Cache,
    sleep_seconds: float = 4.0,
) -> List[Dict[str, Any]]:
    """One row per record: the verdict, the located spans, and the gold spans."""
    rows: List[Dict[str, Any]] = []
    asked = served = failed = 0

    for index, record in enumerate(records, 1):
        prompt = build_prompt(record)
        digest = cache_key(model, prompt)
        cached = cache.get(digest)
        if cached is None:
            try:
                raw = call_model(key, model, prompt)
            except JudgeUnavailable as error:
                # Not cached: a record the API refused today should be asked
                # again tomorrow, not remembered as an answer of "no".
                failed += 1
                print(f"  record {record['id']}: {error}", flush=True)
                if failed > max(5, len(records) // 10):
                    raise SystemExit(
                        f"{failed} records failed; stopping rather than reporting a "
                        "baseline built on a mostly-unanswered sample. Everything "
                        "already answered is cached, so re-running resumes."
                    ) from error
                time.sleep(sleep_seconds)
                continue
            cached = {"raw": raw, **parse_verdict(raw)}
            cache.put(digest, cached)
            asked += 1
            time.sleep(sleep_seconds)
        else:
            served += 1

        spans, missing = locate(record["answer"], cached["spans"])
        gold = [(s["start"], s["end"]) for s in (record.get("spans") or [])]
        rows.append(
            {
                "id": record["id"],
                "task_type": record.get("task_type", "all"),
                "answer": record["answer"],
                "gold_spans": gold,
                "pred_spans": spans,
                "judge_said_hallucinated": cached["hallucinated"],
                "n_quotes": len(cached["spans"]),
                "n_quotes_not_found": missing,
                "parse_failed": cached.get("parse_failed", False),
            }
        )
        if index % 25 == 0:
            print(f"  {index}/{len(records)}  asked {asked} cached {served}", flush=True)

    print(f"done: {asked} API calls, {served} from cache, {failed} unanswered")
    return rows


def evaluate_judge(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Example level from the judge's own yes/no; span level from its quotes.

    The judge's verdict and its quotes are scored separately on purpose. It can
    be right that an answer contains a hallucination and still be unable to say
    which words, and that difference is the point of running it.
    """
    verdict_rows = [
        {
            "gold_spans": row["gold_spans"],
            "pred_spans": [(0, 1)] if row["judge_said_hallucinated"] else [],
        }
        for row in rows
    ]
    example = example_metrics(verdict_rows)
    positives = sum(1 for row in rows if row["gold_spans"])
    rate = positives / len(rows) if rows else 0.0

    quoted = sum(row["n_quotes"] for row in rows)
    not_found = sum(row["n_quotes_not_found"] for row in rows)
    return {
        "n": len(rows),
        "positive_rate": rate,
        "trivial_f1": trivial_f1(rate),
        "example": example,
        "clears_trivial": example["f1"] > trivial_f1(rate),
        "span_exact": span_char_metrics(rows),
        "span_overlap": span_overlap_metrics(rows),
        "quotes_requested": quoted,
        "quotes_not_found_verbatim": not_found,
        "quote_location_failure_rate": (not_found / quoted) if quoted else 0.0,
        "parse_failures": sum(1 for row in rows if row["parse_failed"]),
    }


def format_report(report: Dict[str, Any]) -> str:
    example = report["example"]
    exact = report["span_exact"]
    overlap = report["span_overlap"]
    lines = [
        f"model {report['model']}   n {report['n']:,}   "
        f"positive rate {report['positive_rate']:.3f}",
        "",
        f"{'level':<16}{'P':>9}{'R':>9}{'F1':>9}",
        f"{'example':<16}{example['precision']:>9.4f}{example['recall']:>9.4f}"
        f"{example['f1']:>9.4f}",
        f"{'span exact':<16}{exact['precision']:>9.4f}{exact['recall']:>9.4f}"
        f"{exact['f1']:>9.4f}",
        f"{'span overlap':<16}{overlap['precision']:>9.4f}{overlap['recall']:>9.4f}"
        f"{overlap['f1']:>9.4f}",
        "",
        f"trivial baseline at this positive rate: {report['trivial_f1']:.4f}   "
        f"clears it: {'yes' if report['clears_trivial'] else 'NO'}",
        "",
        f"quotes requested {report['quotes_requested']:,}, of which "
        f"{report['quotes_not_found_verbatim']:,} do not appear verbatim in the "
        f"answer ({100 * report['quote_location_failure_rate']:.1f}%)",
        f"unparseable replies: {report['parse_failures']}",
    ]
    if report["quote_location_failure_rate"] > 0.05:
        lines.append(
            "A quote that cannot be located cannot be highlighted. That failure "
            "rate is the span-level column of this baseline, whatever its F1 says."
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an LLM judge over a sample of the RAGTruth test split."
    )
    parser.add_argument("--data", default="data/processed/ragtruth_test.jsonl")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="omit to discover from the API")
    parser.add_argument("--sleep", type=float, default=4.0, help="seconds between calls")
    parser.add_argument("--out-dir", default="results/llm_judge")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)

    key = api_key()
    available = list_models(key)
    if args.list_models:
        print("\n".join(available))
        return 0

    model = args.model or choose_model(available)
    print(f"{len(available)} models available, using {model}")

    records = [
        json.loads(line)
        for line in Path(args.data).open(encoding="utf-8")
        if line.strip()
    ]
    sample = stratified_sample(records, args.sample, args.seed)
    tasks: Dict[str, int] = {}
    for record in sample:
        tasks[record.get("task_type", "all")] = tasks.get(record.get("task_type", "all"), 0) + 1
    print(f"sampled {len(sample):,} of {len(records):,} responses: {tasks}")

    out_dir = Path(args.out_dir)
    try:
        rows = judge_records(sample, key, model, Cache(out_dir / "cache"), args.sleep)
    except ModelUnusable as error:
        raise SystemExit(
            f"{error}\n"
            "ListModels advertises generateContent for retired models, so pick "
            "another id from that list and re-run; the cache keeps whatever this "
            "model already answered."
        ) from error

    report = {"model": model, "prompt": PROMPT, "seed": args.seed, **evaluate_judge(rows)}
    print()
    print(format_report(report))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llm_judge_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (out_dir / "llm_judge_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: v for k, v in row.items() if k != "answer"}) + "\n")
    print(f"\nwritten: {out_dir / 'llm_judge_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
