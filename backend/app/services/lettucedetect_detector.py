"""A real detector for the API: LettuceDetect scores, C2 calibration and conformal.

This is step 2 of the three-step swap described in detector.py. Every number it
returns comes from a model and from a calibration set, not from a heuristic. The
one thing it is not is *our* detector -- that arrives when C1 finishes training,
and the change will be the model id and the artifact path.

The pipeline for one request:

    1. Format (question, context) into the prompt LettuceDetect expects.
    2. Tokenize (prompt, answer) and take P(hallucinated) per answer token.
    3. Merge runs of tokens above `candidate_threshold` into candidate spans.
    4. Score each candidate by its mean token probability, calibrate it, and let
       split conformal decide FLAG / ABSTAIN / PASS at the requested alpha.

Two design points worth defending:

**The prompt template.** Offline, `lettucedetect_adapter.py` feeds RAGTruth's own
published `prompt` field, because that is exactly what the checkpoint was trained
on and it is what reproduces their reported score. A web request has no such
field, so this file rebuilds the prompt from the templates in the LettuceDetect
repository -- which is what their own inference code does. Note the consequence,
because it is not obvious: their training prompt for summarization was
"Summarize the following news within N words:" while their serving template is
"Summarize the following text:". That mismatch is theirs, not ours, and it means
API results are not strictly comparable with the offline evaluation numbers.

**Candidates, then decisions.** `candidate_threshold` controls which stretches of
the answer are put forward at all; the conformal layer then decides what to do
with each. The default is 0.5, which is argmax on a two-label head -- the exact
operating point LettuceDetect itself uses and the one our offline evaluation
measured at example F1 0.7918. It is not a free parameter to tune until the demo
looks good.

That distinction was learned by getting it wrong. The default was briefly 0.3,
chosen so that more spans would land in the ABSTAIN band and the interface would
look livelier. On the canned Sigiriya example the model scores every answer
token above 0.38, so a 0.3 cutoff merged the entire answer into one span --
including the supported opening clause -- and destroyed the span-level
localisation that is the whole point. The model was right; the threshold was
wrong. If the demo needs more abstention, the honest lever is the alpha slider,
which is a research parameter with a stated meaning, not a hidden cutoff.

**Known limitation, and it is visible in the numbers.** Candidates are runs of
consecutive above-threshold tokens, so a hallucinated clause sitting next to a
supported one merges with it. That is why span-exact F1 is 0.1937 against
span-overlap 0.5969 in the offline evaluation. RAGTruth's gold spans are often
whole sentences, so merging is frequently right, but it is not always right and
the report should say so rather than quoting only the overlap figure.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.app.services.detector import split_sentences
from src.common.schema import (
    AnalysisResult,
    AnalyzeRequest,
    ConformalDecision,
    Span,
    TaskType,
)

DEFAULT_MODEL_ID = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
HALLUCINATED_INDEX = 1

# Verbatim from lettucedetect/prompts/*.txt in the KRLabsOrg repository. Copied
# rather than imported so the backend does not take a dependency on their
# package; if they change the templates, ours will not silently follow.
QA_TEMPLATE = (
    "Briefly answer the following question:\n"
    "{question}\n"
    "Bear in mind that your response should be strictly based on the following "
    "{num_passages} passages:\n"
    "{context}\n"
    'In case the passages do not contain the necessary information to answer the '
    'question, please reply with: "Unable to answer based on given passages."\n'
    "output:"
)
SUMMARY_TEMPLATE = "Summarize the following text:\n{text}\noutput:"


def format_prompt(question: str, context: str, task_type: TaskType) -> str:
    """Rebuild the prompt LettuceDetect was trained to read.

    Summarization and data-to-text have no user question in RAGTruth, so they
    take the summary template. Anything with a real question takes the QA one.
    """
    if task_type in (TaskType.SUMMARIZATION, TaskType.DATA2TEXT) or not question.strip():
        return SUMMARY_TEMPLATE.format(text=context)
    # RAGTruth's QA contexts already arrive with "passage 1:" markers embedded,
    # so the passage count is read off the text rather than invented.
    num_passages = max(1, context.count("passage "))
    return QA_TEMPLATE.format(
        question=question, num_passages=num_passages, context=context
    )


class C2Layer:
    """The calibration and conformal artifact written by run_c2.py."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        from src.c2_calibration.calibration import calibrator_from_dict

        self.payload = payload
        self.calibrator = calibrator_from_dict(payload["calibrator"])
        self.nonconformity: List[float] = list(payload["nonconformity_sorted"])
        self.n_calibration = int(payload["n_calibration"])

    @classmethod
    def load(cls, path: Path | str) -> "C2Layer":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def calibrate(self, scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        return [float(v) for v in self.calibrator.transform(scores)]

    def threshold(self, alpha: float) -> float:
        """The conformal quantile for this alpha, computed fresh, never interpolated."""
        from src.c2_calibration.conformal import conformal_quantile

        return conformal_quantile(self.nonconformity, alpha)

    def decide(self, calibrated: float, alpha: float) -> Tuple[ConformalDecision, float]:
        """Return the decision and the span's non-conformity score.

        The non-conformity reported is the one for the *flagged* hypothesis,
        1 - p, which is what the UI tooltip should show: how unusual it would be
        for this span to be a genuine hallucination.
        """
        q = self.threshold(alpha)
        keep_one = (1.0 - calibrated) <= q
        keep_zero = calibrated <= q
        if keep_one and not keep_zero:
            decision = ConformalDecision.FLAG
        elif keep_zero and not keep_one:
            decision = ConformalDecision.PASS
        else:
            decision = ConformalDecision.ABSTAIN
        return decision, 1.0 - calibrated


class LettuceDetectDetector:
    """Public LettuceDetect checkpoint, wrapped in the C2 layer."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        artifact_path: Optional[Path | str] = None,
        device: Optional[str] = None,
        max_length: int = 4096,
        candidate_threshold: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self.candidate_threshold = candidate_threshold
        self.c2 = C2Layer.load(artifact_path) if artifact_path else None
        self._device_preference = device
        self._model = None
        self._tokenizer = None

    @property
    def model_version(self) -> str:
        suffix = "+c2" if self.c2 else "+uncalibrated"
        return f"{self.model_id.split('/')[-1]}{suffix}"

    def _ensure_loaded(self) -> None:
        """Load on first use, not at import.

        A 1.6GB download inside module import makes the test suite depend on the
        network and makes uvicorn --reload unusable.
        """
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        device = torch.device(
            self._device_preference
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(self.model_id)
        if model.config.num_labels != 2:
            raise RuntimeError(
                f"{self.model_id} has {model.config.num_labels} labels; this "
                "detector assumes the binary LettuceDetect head"
            )
        self._model = model.to(device).eval()
        self._device = device

    def token_probabilities(
        self, prompt: str, answer: str
    ) -> Tuple[List[float], List[Tuple[int, int]]]:
        """P(hallucinated) and character offsets for each answer token."""
        import torch

        from src.c1_detector.bio import ANSWER_SEQUENCE_ID

        self._ensure_loaded()
        encoding = self._tokenizer(
            prompt,
            answer,
            truncation="only_first",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        sequence_ids = encoding.sequence_ids(0)
        offsets = encoding["offset_mapping"]

        input_ids = torch.tensor([encoding["input_ids"]], device=self._device)
        attention_mask = torch.tensor([encoding["attention_mask"]], device=self._device)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu()

        positions = [
            i
            for i, sid in enumerate(sequence_ids)
            if sid == ANSWER_SEQUENCE_ID and offsets[i][1] > offsets[i][0]
        ]
        return (
            [float(probs[i, HALLUCINATED_INDEX]) for i in positions],
            [tuple(offsets[i]) for i in positions],
        )

    def warmup(self) -> float:
        """Load the model and run throwaway forward passes. Returns seconds taken.

        Worth doing at server startup rather than on the first user click.
        Measured on the RTX 3050: the first call costs about 7 seconds -- weight
        loading, CUDA context creation and kernel autotuning -- while every warm
        call after it costs about 44 ms. Without this the first person to press
        Analyse waits nine seconds and concludes the thing is slow, when it is
        actually roughly twenty times faster than that.

        Warms through `analyze` rather than through the forward pass alone, and
        at several input lengths. Two reasons, both found by measuring:

        * CUDA autotunes per input shape, so a ten-token warmup leaves a
          realistic request paying its own autotune pass.
        * `analyze` does lazy imports and builds a pydantic model on the way
          out. Warming only `token_probabilities` skips all of that, which left
          the first real request at roughly 390 ms against a warm 45 ms.

        Exercising the whole path is the only version of this that actually
        works, and it is no harder to write.
        """
        started = time.perf_counter()
        self._ensure_loaded()
        sentence = (
            "passage 1: The archive records that construction began in the "
            "spring and continued for several years under royal patronage. "
        )
        for repeats in (1, 4, 16):
            self.analyze(
                AnalyzeRequest(
                    question="What does the archive record?",
                    context=sentence * repeats,
                    answer=sentence * min(repeats, 3),
                    alpha=0.1,
                    task_type=TaskType.QA,
                )
            )
        return time.perf_counter() - started

    def analyze(self, req: AnalyzeRequest) -> AnalysisResult:
        from src.c1_detector.evaluate_c1 import spans_from_token_mask

        started = time.perf_counter()
        prompt = format_prompt(req.question, req.context, req.task_type)
        token_probs, offsets = self.token_probabilities(prompt, req.answer)

        mask = [p >= self.candidate_threshold for p in token_probs]
        candidates = spans_from_token_mask(mask, offsets, req.answer)

        raw_scores: List[float] = []
        covered_probs: List[List[float]] = []
        for start, end in candidates:
            covered = [
                p
                for p, (offset_start, offset_end) in zip(token_probs, offsets)
                if offset_start < end and start < offset_end
            ]
            covered_probs.append(covered)
            raw_scores.append(sum(covered) / len(covered) if covered else 0.0)

        calibrated = self.c2.calibrate(raw_scores) if self.c2 else list(raw_scores)

        spans: List[Span] = []
        for (start, end), raw, cal, covered in zip(
            candidates, raw_scores, calibrated, covered_probs
        ):
            if self.c2:
                decision, nonconformity = self.c2.decide(cal, req.alpha)
            else:
                decision, nonconformity = None, None
            # A PASS span is the rest of the answer and is not worth drawing.
            if decision is ConformalDecision.PASS:
                continue
            spans.append(
                Span(
                    start=start,
                    end=end,
                    text=req.answer[start:end],
                    token_probs=[round(p, 6) for p in covered],
                    span_score=round(raw, 4),
                    calibrated_score=round(cal, 4),
                    nonconformity=round(nonconformity, 4)
                    if nonconformity is not None
                    else None,
                    conformal_decision=decision,
                    alpha=req.alpha if self.c2 else None,
                )
            )

        return AnalysisResult(
            question=req.question,
            context=req.context,
            answer=req.answer,
            context_sentences=split_sentences(req.context),
            spans=spans,
            task_type=req.task_type,
            model_version=self.model_version,
            alpha=req.alpha,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


def build_from_env() -> Optional[LettuceDetectDetector]:
    """Construct from environment, or return None to leave the stub in place.

        TRUSTRAG_DETECTOR       "lettucedetect" to switch on; anything else keeps the stub
        TRUSTRAG_C2_ARTIFACT    path to c2_artifact.json
        TRUSTRAG_MODEL_ID       override the checkpoint
        TRUSTRAG_DEVICE         cuda / cpu
        TRUSTRAG_CANDIDATE_THRESHOLD

    Defaulting to off is deliberate: the test suite and a plain `uvicorn` start
    must not trigger a 1.6GB download.
    """
    if os.environ.get("TRUSTRAG_DETECTOR", "").lower() != "lettucedetect":
        return None

    artifact = os.environ.get("TRUSTRAG_C2_ARTIFACT")
    if artifact and not Path(artifact).exists():
        raise SystemExit(
            f"TRUSTRAG_C2_ARTIFACT points at {artifact}, which does not exist. "
            "Run scripts\\run_lettucedetect_baseline.ps1 to produce it."
        )

    return LettuceDetectDetector(
        model_id=os.environ.get("TRUSTRAG_MODEL_ID", DEFAULT_MODEL_ID),
        artifact_path=artifact,
        device=os.environ.get("TRUSTRAG_DEVICE") or None,
        candidate_threshold=float(
            os.environ.get("TRUSTRAG_CANDIDATE_THRESHOLD", "0.5")
        ),
    )
