"""Rebuild a served payload from artefacts already on disk, for the offline demo.

This is not a second serving path and it must never become one. Every number it
produces was produced by something else and written down:

    C1  the span offsets, the per-token probabilities and the raw span scores
        come from `results/c1/{split}/probabilities.jsonl`, which a training run
        on Kaggle wrote.
    C2  the calibrated score and the flag/abstain/pass decision are computed by
        C2's own `Calibrator` and `SplitConformal`, loaded from
        `c2_artifact.json` and unchanged.
    C2  the distribution check is C2's own `ExchangeabilityReference`, and the
        wording is the same `distribution_message` the live endpoint prints.
    C4  the evidence sentence and its confidence come from C4's own
        `C4Attributor`, which is the class the live endpoint calls.

So a payload built here is what the live endpoint would have returned for that
response, minus the forward pass -- and the forward pass is exactly the part
that was already recorded.

What it is for: a version of the interface that runs with no backend, no model
download and no GPU, so the work can be shown to somebody on a laptop or a phone
without setting anything up. The alternative -- hand-written example payloads --
would put invented numbers on a screen next to real ones, which is the one thing
this project does not do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.c1_detector.build_examples import build_context_and_question
from src.c1_detector.ragtruth_labels import task_type_of
from src.c2_calibration.calibration import calibrator_from_dict
from src.c2_calibration.conformal import SplitConformal
from src.c2_calibration.exchangeability import (
    ExchangeabilityReference,
    distribution_message,
    features_from_record,
)
from src.c2_calibration.run_c2 import span_units
from src.c4_attribution.serving import C4Attributor
from src.common.schema import (
    AnalysisResult,
    ConformalDecision,
    DistributionCheck,
    FeatureCheck,
    Span,
    TaskType,
)

RAW_DIR = Path("data/raw")
CALIB_DUMP = Path("results/c1/calib/probabilities.jsonl")
TEST_DUMP = Path("results/c1/test/probabilities.jsonl")
C2_ARTIFACT = Path("results/c2/c1/c2_artifact.json")
C2_REFERENCE = Path("results/c2/c1/c2_ood_reference.json")

DEFAULT_ALPHA = 0.1


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class RecordedRuns:
    """The pieces of a served response that were recorded rather than recomputed."""

    def __init__(
        self,
        raw_dir: Path | str = RAW_DIR,
        calib_dump: Path | str = CALIB_DUMP,
        test_dump: Path | str = TEST_DUMP,
        artifact_path: Path | str = C2_ARTIFACT,
        reference_path: Path | str = C2_REFERENCE,
        c4_artifact: Optional[Path | str] = None,
    ) -> None:
        self.records = _read_jsonl(test_dump)
        self.by_id = {str(r["id"]): r for r in self.records}

        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        self.calibrator = calibrator_from_dict(artifact["calibrator"])

        calib_scores, calib_labels, _ = span_units(_read_jsonl(calib_dump))
        self.calib_scores = self.calibrator.transform(calib_scores)
        self.calib_labels = list(calib_labels)

        self.reference = (
            ExchangeabilityReference.load(reference_path)
            if Path(reference_path).exists()
            else None
        )
        self.c4 = (
            C4Attributor.load(c4_artifact) if c4_artifact else C4Attributor.load()
        )
        # C3 is loaded the same way the serving path loads it, and it returns
        # None rather than raising when results/c3 has not been generated. The
        # offline copy then shows exactly what the live endpoint would show.
        from backend.app.services.c3_typer import C3Typer

        self.c3 = C3Typer.load()

        # Answers are stored in the dump; the context is not, so it is rebuilt
        # with C1's own rule. Re-implementing that rule would change the
        # sentence split and therefore every evidence index.
        raw_dir = Path(raw_dir)
        self.sources = {
            str(r["source_id"]): r for r in _read_jsonl(raw_dir / "source_info.jsonl")
        }
        self.source_of = {
            str(r["id"]): str(r["source_id"])
            for r in _read_jsonl(raw_dir / "response.jsonl")
        }

    # -- the rebuild ------------------------------------------------------

    def build(self, response_id: str, alpha: float = DEFAULT_ALPHA) -> AnalysisResult:
        """The payload the live endpoint would have returned for this response."""
        record = self.by_id[str(response_id)]
        source = self.sources[self.source_of[str(response_id)]]
        task = task_type_of(str(source["task_type"]))
        context, question = build_context_and_question(task, source["source_info"])
        answer = str(record["answer"])

        candidates = [
            (int(s["start"]), int(s["end"])) for s in record.get("pred_spans") or []
        ]
        raw_scores = [float(s["mean_prob"]) for s in record.get("pred_spans") or []]
        calibrated = self.calibrator.transform(raw_scores) if raw_scores else []

        conformal = SplitConformal(alpha=alpha).fit(self.calib_scores, self.calib_labels)
        decisions = conformal.decisions(calibrated) if raw_scores else []

        from backend.app.services.detector import split_sentences

        sentences = split_sentences(context)

        spans: List[Span] = []
        for (start, end), raw, cal, decision in zip(
            candidates, raw_scores, calibrated, decisions
        ):
            # The serving path drops PASS spans: the rest of the answer is not
            # worth drawing. Keeping them here would make the offline copy show
            # something the live one never shows.
            if decision is ConformalDecision.PASS:
                continue
            evidence = self._attribute(answer[start:end], sentences, alpha)
            typing = self._type(
                answer, start, end, sentences, task, alpha, decision
            )
            spans.append(
                Span(
                    start=start,
                    end=end,
                    text=answer[start:end],
                    span_score=round(float(raw), 4),
                    calibrated_score=round(float(cal), 4),
                    nonconformity=round(1.0 - float(cal), 4),
                    conformal_decision=decision,
                    alpha=alpha,
                    error_type=typing.get("error_type"),
                    explanation=typing.get("explanation"),
                    escalated=typing.get("escalated"),
                    evidence_sentence=evidence.get("evidence_sentence"),
                    evidence_index=evidence.get("evidence_index"),
                    entailment_score=evidence.get("entailment_score"),
                )
            )

        check, guarantee = self._check(record)

        return AnalysisResult(
            question=question,
            context=context,
            answer=answer,
            context_sentences=sentences,
            spans=spans,
            task_type=task if isinstance(task, TaskType) else TaskType.OTHER,
            model_version="modernbert-base-e6+c2",
            alpha=alpha,
            distribution_check=check,
            guarantee_applies=guarantee,
            attribution_checked=self.c4 is not None and self.c4.available,
            latency_ms=None,
        )

    def _type(
        self,
        answer: str,
        start: int,
        end: int,
        sentences: Sequence[str],
        task: Any,
        alpha: float,
        decision: Optional[ConformalDecision],
    ) -> Dict[str, Any]:
        """C3's three fields, rebuilt through C3's own classes.

        Mirrors `lettucedetect_detector._type_span` step for step, including the
        broad except: a typing failure must never take a recorded case down with
        it, because the span and its decision are the product and the type is an
        addition to it.
        """
        if self.c3 is None or not self.c3.available:
            return {}
        try:
            from src.c3_explanation.grounding import span_features

            task_value = task.value if isinstance(task, TaskType) else "other"
            features = span_features(
                answer[start:end], answer, start, list(sentences), task_value
            )
            return self.c3.annotate(
                features, answer[start:end], list(sentences), alpha, decision
            )
        except Exception:  # pragma: no cover - defensive, as in serving
            return {}

    def _attribute(
        self, span_text: str, sentences: Sequence[str], alpha: float
    ) -> Dict[str, Any]:
        if not self.c4.available:
            return {}
        try:
            return self.c4.annotate(span_text, list(sentences), alpha)
        except Exception:  # pragma: no cover - defensive, as in serving
            return {}

    def _check(self, record: Dict[str, Any]) -> tuple[DistributionCheck, bool]:
        if self.reference is None:
            return (
                DistributionCheck(
                    checked=False,
                    in_distribution=True,
                    message=(
                        "No exchangeability reference is loaded, so it is not "
                        "known whether this input resembles the data the "
                        "guarantee was calibrated on."
                    ),
                ),
                True,
            )
        raw = self.reference.check(features_from_record(record))
        in_distribution = bool(raw["in_distribution"])
        return (
            DistributionCheck(
                checked=True,
                in_distribution=in_distribution,
                p_value=raw["p_value"],
                threshold=raw["threshold"],
                n_reference=raw["n_reference"],
                most_unusual=raw["most_unusual"],
                features=[FeatureCheck(**row) for row in raw["features"]],
                message=distribution_message(in_distribution, raw["most_unusual"]),
            ),
            in_distribution,
        )

    # -- picking cases to show --------------------------------------------

    def summarise(self, response_id: str, alpha: float = DEFAULT_ALPHA) -> Dict[str, Any]:
        """Counts for one response, so a caller can choose what to show."""
        result = self.build(response_id, alpha)
        counts = {"flag": 0, "abstain": 0}
        for span in result.spans:
            if span.conformal_decision is ConformalDecision.FLAG:
                counts["flag"] += 1
            elif span.conformal_decision is ConformalDecision.ABSTAIN:
                counts["abstain"] += 1
        return {
            "id": response_id,
            "task": result.task_type.value,
            "n_spans": len(result.spans),
            "n_gold": len(self.by_id[str(response_id)].get("gold_spans") or []),
            "with_evidence": sum(1 for s in result.spans if s.evidence_sentence),
            "without_evidence": sum(1 for s in result.spans if not s.evidence_sentence),
            "answer_chars": len(result.answer),
            "context_sentences": len(result.context_sentences),
            "guarantee_applies": result.guarantee_applies,
            **counts,
        }


# --------------------------------------------------------------------------
# The three cases the interface ships with
# --------------------------------------------------------------------------
#
# Chosen for what they show, not for how well the system does on them. Each one
# is named here with the reason, so a reader can check that the selection was
# not "whichever looked best".

CASES: tuple[dict[str, str], ...] = (
    {
        "id": "14297",
        "slug": "carbon-footprint",
        "title": "A confident answer with two invented claims",
        "why": (
            "All three verdicts on one paragraph, and both evidence states. The "
            "two passages the system flags have no supporting sentence in the "
            "document; the two it will not decide do. That is the mechanism "
            "working, not a coincidence -- an invented claim has nothing to "
            "point at."
        ),
    },
    {
        "id": "1826",
        "slug": "news-summary",
        "title": "A news summary with a detail the article never gave",
        "why": (
            "Prose rather than a passage list, so the sentence the attributor "
            "points at is a sentence a person would actually read."
        ),
    },
    {
        "id": "16359",
        "slug": "promise-withdrawn",
        "title": "An input the promise does not cover",
        "why": (
            "The run-time check found this input unlike the data the guarantee "
            "was calibrated on, so the coverage claim is withdrawn while the "
            "marks stay. This is the case worth showing on purpose: what is "
            "taken away is the promise, not the detection."
        ),
    },
)

OUT = Path("frontend/public/recorded-runs.json")


def build_cases(
    runs: Optional["RecordedRuns"] = None, alpha: float = DEFAULT_ALPHA
) -> List[Dict[str, Any]]:
    runs = runs or RecordedRuns()
    out: List[Dict[str, Any]] = []
    for case in CASES:
        result = runs.build(case["id"], alpha)
        out.append(
            {
                "slug": case["slug"],
                "title": case["title"],
                "why": case["why"],
                "response_id": case["id"],
                "source": "RAGTruth test split, recorded run",
                "result": json.loads(result.model_dump_json()),
            }
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    runs = RecordedRuns()
    payload = {
        "note": (
            "Recorded runs, not live calls. Every field was produced by the "
            "components themselves: the spans and token probabilities by C1's "
            "training run, the calibrated score and the decision by C2's own "
            "calibrator and conformal class, the distribution check by C2's "
            "reference, the evidence by C4's attributor. Regenerate with "
            "python -m src.common.recorded_analysis"
        ),
        "alpha": DEFAULT_ALPHA,
        "cases": build_cases(runs),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for case in payload["cases"]:
        spans = case["result"]["spans"]
        flags = sum(1 for s in spans if s["conformal_decision"] == "flag")
        evid = sum(1 for s in spans if s["evidence_sentence"])
        print(
            f"{case['slug']:22s} id={case['response_id']:>6s} "
            f"spans={len(spans)} flag={flags} with-evidence={evid} "
            f"guarantee={case['result']['guarantee_applies']}"
        )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} kB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
