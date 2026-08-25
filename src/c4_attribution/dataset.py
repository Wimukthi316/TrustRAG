"""The join: RAGTruth's raw files, C1's frozen splits, and C4's evidence labels.

One place decides what a C4 record is, so that every layer scores the same rows.
Three rules are enforced here rather than in each layer:

* **The context is built by C1's own rule.** `build_context_and_question` is
  imported, never re-implemented. Different formatting changes the sentence
  split and therefore every `evidence_index`, so a second copy of that logic
  would quietly make C4's indices incomparable with the serving path's.
* **The calibration split is C1's frozen 1,509 response ids**, the same ones C2
  and C3 calibrate on. It is the only slice legitimately held out from C1's
  training, and re-drawing it would leak.
* **The test split is the corpus test split of 2,700**, identified by
  `split == "test"` in `response.jsonl` and untouched by any component.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from src.c1_detector.build_examples import build_context_and_question
from src.c1_detector.ragtruth_labels import task_type_of
from src.c4_attribution.evidence_labels import EvidenceLabel, label_from_annotation
from src.c4_attribution.sentences import (
    sentence_offsets,
    split_sentences,
    split_sentences_variant,
)

RAW_DIR = Path("data/raw")
SPLIT_IDS = Path("results/c1/modernbert-base/split_ids.json")

# The splits C4 reports on. `train` and `val` exist and are deliberately unused:
# C1 trained on them, so any evidence number measured there is contaminated by
# the detector having seen the answers.
REPORTED_SPLITS: Tuple[str, ...] = ("calib", "test")


@dataclass
class ResponseRecord:
    """One RAGTruth response, with the context C1 would have shown the model."""

    response_id: str
    source_id: str
    split: str
    task: str
    model: str
    question: str
    context: str
    answer: str
    labels: List[EvidenceLabel] = field(default_factory=list)
    _sentences: Optional[List[str]] = field(default=None, repr=False)
    _variant: Optional[List[str]] = field(default=None, repr=False)

    @property
    def sentences(self) -> List[str]:
        """The candidate pool, split exactly as the serving path splits it."""
        if self._sentences is None:
            self._sentences = split_sentences(self.context)
        return self._sentences

    @property
    def variant_sentences(self) -> List[str]:
        """The Layer 1c ablation pool. Never used for a reported index."""
        if self._variant is None:
            self._variant = split_sentences_variant(self.context)
        return self._variant


@dataclass
class SpanRecord:
    """One annotated span, with the response it lives in.

    This is the unit every C4 layer scores. `response` is shared, so the
    response-level cluster bootstrap can group by `response.response_id` without
    a second lookup.
    """

    response: ResponseRecord
    label: EvidenceLabel

    @property
    def response_id(self) -> str:
        return self.response.response_id

    @property
    def task(self) -> str:
        return self.response.task

    @property
    def sentences(self) -> List[str]:
        return self.response.sentences

    @property
    def span_text(self) -> str:
        return self.label.span_text

    def _answer_sentence_bounds(self) -> Tuple[int, int]:
        """Character bounds of the answer sentence holding this span.

        Uses the same splitter as the context so a span and a context sentence
        are cut by one rule, not two. Falls back to the span's own bounds when
        the splitter cannot place it, which keeps a hypothesis non-empty rather
        than returning the whole answer.
        """
        answer = self.response.answer
        pieces = split_sentences(answer)
        for start, end in sentence_offsets(answer, pieces):
            if start <= self.label.start < end:
                return (start, max(end, self.label.end))
        return (self.label.start, self.label.end)

    @property
    def answer_sentence(self) -> str:
        """The answer sentence containing this span.

        AlignScore and MiniCheck both operate on sentences, and a flagged span is
        often a fragment ("in 2019") that is not a proposition on its own. Layer
        4b measures whether that matters instead of assuming it.
        """
        start, end = self._answer_sentence_bounds()
        return self.response.answer[start:end].strip() or self.label.span_text

    @property
    def marked_answer_sentence(self) -> str:
        """The parent sentence with the span wrapped in << >>.

        The same input format `src/c3_explanation/type_model.py` uses, so the two
        components present a span to a model the same way.
        """
        answer = self.response.answer
        start, end = self._answer_sentence_bounds()
        head = answer[start:self.label.start]
        body = answer[self.label.start:self.label.end]
        tail = answer[self.label.end:end]
        return f"{head}<< {body} >>{tail}".strip() or self.label.span_text


def read_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_split_ids(path: Path | str = SPLIT_IDS) -> Dict[str, set]:
    """C1's frozen split of the corpus train half. Raises if it is missing.

    There is no fallback on purpose. A re-drawn split would look like it worked
    and would quietly leak C1's training data into C4's calibration set.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: set(map(str, ids)) for name, ids in payload.items()}


def split_of(response: Dict[str, Any], split_ids: Dict[str, set]) -> str:
    """calib / train / val for corpus-train responses, test for corpus-test ones."""
    if str(response.get("split")) == "test":
        return "test"
    rid = str(response["id"])
    for name in ("calib", "val", "train"):
        if rid in split_ids.get(name, ()):
            return name
    return "unassigned"


def load_responses(
    raw_dir: Path | str = RAW_DIR,
    split_ids_path: Path | str = SPLIT_IDS,
    splits: Optional[Sequence[str]] = REPORTED_SPLITS,
    with_labels_only: bool = True,
) -> List[ResponseRecord]:
    """Every response in `splits`, with its context built by C1's rule.

    `with_labels_only` keeps responses carrying at least one annotation, which
    is every response C4 can be scored on. Pass False when a layer needs the
    hallucination-free responses too (Layer 11 does, for the fusion table).
    """
    raw_dir = Path(raw_dir)
    split_ids = load_split_ids(split_ids_path)
    sources = {str(r["source_id"]): r for r in read_jsonl(raw_dir / "source_info.jsonl")}

    wanted = set(splits) if splits is not None else None
    out: List[ResponseRecord] = []
    for row in read_jsonl(raw_dir / "response.jsonl"):
        name = split_of(row, split_ids)
        if wanted is not None and name not in wanted:
            continue
        annotations = row.get("labels") or []
        if with_labels_only and not annotations:
            continue
        source = sources[str(row["source_id"])]
        task = task_type_of(str(source["task_type"]))
        context, question = build_context_and_question(task, source["source_info"])
        record = ResponseRecord(
            response_id=str(row["id"]),
            source_id=str(row["source_id"]),
            split=name,
            task=task.value,
            model=str(row.get("model") or ""),
            question=question,
            context=context,
            answer=str(row.get("response") or ""),
        )
        record.labels = [
            label_from_annotation(record.response_id, a) for a in annotations
        ]
        out.append(record)
    return out


def span_records(responses: Iterable[ResponseRecord]) -> List[SpanRecord]:
    """Flatten to one row per annotated span."""
    return [SpanRecord(response=r, label=label) for r in responses for label in r.labels]


def attributable(rows: Iterable[SpanRecord]) -> List[SpanRecord]:
    """Spans whose annotator quoted real source text. Layers 2, 3, 4, 5, 6 scope."""
    return [r for r in rows if r.label.attributable]


def no_evidence(rows: Iterable[SpanRecord]) -> List[SpanRecord]:
    """Spans whose annotator said the information is not in the source. Layer 7."""
    return [r for r in rows if r.label.snippet and r.label.no_evidence]


# --------------------------------------------------------------------------
# C1's predicted spans, for the exclusion counts and the fusion table
# --------------------------------------------------------------------------


def load_predicted_spans(path: Path | str) -> List[Dict[str, Any]]:
    """C1's probability dump: one record per response, with pred_spans.

    `is_hallucinated` on a predicted span is the *gold* label -- whether that
    candidate touches an annotated span -- not a model output. A candidate with
    `is_hallucinated` false is a spurious span: it has no gold type and no gold
    evidence, so it is excluded from attribution scoring and counted separately.
    """
    return list(read_jsonl(path))


def spurious_rate(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """How much of C1's output C4 cannot be scored on, stated up front."""
    total = 0
    spurious = 0
    for record in records:
        for span in record.get("pred_spans") or []:
            total += 1
            if not span.get("is_hallucinated"):
                spurious += 1
    return {
        "n_predicted_spans": total,
        "n_spurious": spurious,
        "spurious_rate": spurious / total if total else 0.0,
    }


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def predicted_span_records(
    dump_path: Path | str,
    responses: Sequence[ResponseRecord],
) -> List[SpanRecord]:
    """C1's predicted spans, each carrying the evidence label it inherits.

    A predicted span is matched to the annotation it shares the most characters
    with, which is the same rule `src/c3_explanation/explanation_eval.lookup_meta`
    uses to attach an annotator note to a processed gold span. A predicted span
    touching no annotation is **spurious**: it inherits an empty EvidenceLabel,
    it has no gold evidence, and every layer excludes it from attribution
    scoring and counts it separately. 47.45% of C1's predicted test spans are
    spurious, and that exclusion is stated up front rather than in a footnote.

    The point of this loader is the ablation the report needs: gold spans tell
    you how well attribution works when the localisation is right, predicted
    spans tell you what it is worth in the system that actually ships.
    """
    by_id = {r.response_id: r for r in responses}
    out: List[SpanRecord] = []
    for record in read_jsonl(dump_path):
        response = by_id.get(str(record["id"]))
        if response is None:
            continue
        annotations = {(lab.start, lab.end): lab for lab in response.labels}
        for span in record.get("pred_spans") or []:
            start, end = int(span["start"]), int(span["end"])
            best = None
            best_shared = 0
            for key, label in annotations.items():
                shared = _overlap((start, end), key)
                if shared > best_shared:
                    best, best_shared = label, shared
            text = response.answer[start:end]
            if best is None:
                inherited = EvidenceLabel(
                    response_id=response.response_id,
                    start=start,
                    end=end,
                    span_text=text,
                    label_type="",
                    raw_meta="",
                    implicit_true=False,
                    due_to_null=False,
                    has_original=False,
                    snippet=None,
                    no_evidence=False,
                    closing_marker=None,
                )
            else:
                inherited = EvidenceLabel(
                    response_id=response.response_id,
                    start=start,
                    end=end,
                    span_text=text,
                    label_type=best.label_type,
                    raw_meta=best.raw_meta,
                    implicit_true=best.implicit_true,
                    due_to_null=best.due_to_null,
                    has_original=best.has_original,
                    snippet=best.snippet,
                    no_evidence=best.no_evidence,
                    closing_marker=best.closing_marker,
                )
            row = SpanRecord(response=response, label=inherited)
            out.append(row)
    return out


def is_spurious(row: SpanRecord) -> bool:
    """A predicted span touching no annotation. No gold type, no gold evidence."""
    return not row.label.label_type and not row.label.has_original
