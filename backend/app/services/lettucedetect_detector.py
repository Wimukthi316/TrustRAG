"""A real detector for the API: model scores, C2 calibration and conformal.

Serves either checkpoint. `TRUSTRAG_DETECTOR=c1` runs our own trained
ModernBERT-base from `results/c1`; `TRUSTRAG_DETECTOR=lettucedetect` runs the
public baseline. Every number returned comes from a model and a calibration set,
not from a heuristic.

The two are not interchangeable at the input, which is the one non-obvious thing
in this file. C1 was trained on a bare `question\\n\\ncontext` first sequence at
max_length 3,072; the public checkpoint was trained on its own instruction
templates at 4,096. Sending either the other's format costs accuracy silently,
with no error anywhere, so the format is chosen by `prompt_style` and defaulted
per detector in `build_from_env` rather than left to the caller.

The pipeline for one request:

    1. Format (question, context) into the first sequence this detector expects.
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

**Known limitation, and it is visible in the numbers.** Under `decode="threshold"`
candidates are runs of consecutive above-threshold tokens, so a hallucinated
clause sitting next to a supported one merges with it. That is why span-exact F1
is 0.1937 against span-overlap 0.5969 in the offline evaluation of the baseline.

Do not excuse this by claiming RAGTruth annotates whole sentences. Measured on
the processed test split, 2026-08-13: only 11.1% of the 1,517 gold spans are
exactly one sentence, the median span is 35 characters, and the median response
has 7% of its answer marked. The paper calls the task word-level detection and
the data agrees. Merging is a real cost, and both figures belong in the report.
`decode="bio_argmax"`, which C1 uses, avoids it: B-HAL restarts a span.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.app.services.detector import split_sentences
from src.c2_calibration.exchangeability import (
    FEATURE_LABELS,
    ExchangeabilityReference,
    response_features,
)
from src.common.schema import (
    AnalysisResult,
    AnalyzeRequest,
    ConformalDecision,
    DistributionCheck,
    FeatureCheck,
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


def format_prompt(
    question: str,
    context: str,
    task_type: TaskType,
    style: str = "lettucedetect",
) -> str:
    """Rebuild the first sequence the detector was trained to read.

    Two styles, because the two checkpoints were trained on different inputs and
    feeding one the other's format is a silent train/serve mismatch:

    * ``lettucedetect`` -- the public checkpoint's own prompt templates, copied
      from their repository. Summarization and data-to-text have no user
      question in RAGTruth, so they take the summary template; anything with a
      real question takes the QA one.
    * ``c1`` -- our own detector. `src/c1_detector/bio.py:encode_example` builds
      the first sequence as ``question + "\\n\\n" + context``, or the context
      alone when there is no question, with no instruction template at all. That
      is what every one of C1's 15,090 training records looked like, so it is
      what the server must send.
    """
    if style == "c1":
        from src.c1_detector.bio import build_first_sequence

        return build_first_sequence(question, context)
    if style != "lettucedetect":
        raise ValueError(f"unknown prompt style {style!r}; use 'lettucedetect' or 'c1'")
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
        prompt_style: str = "lettucedetect",
        label: Optional[str] = None,
        decode: str = "threshold",
        reference_path: Optional[Path | str] = None,
    ) -> None:
        if decode not in ("threshold", "bio_argmax"):
            raise ValueError(f"unknown decode {decode!r}")
        self.model_id = model_id
        self.max_length = max_length
        self.candidate_threshold = candidate_threshold
        self.prompt_style = prompt_style
        self.label = label
        self.decode = decode
        self.c2 = C2Layer.load(artifact_path) if artifact_path else None
        # The exchangeability reference is loaded separately from the C2
        # artifact on purpose. It answers a different question -- "may the
        # promise be shown for this input" rather than "what is the threshold" --
        # and keeping it out of c2_artifact.json means adding it changes nothing
        # about the file the coverage numbers were verified against.
        self.reference = (
            ExchangeabilityReference.load(reference_path) if reference_path else None
        )
        self._device_preference = device
        self._model = None
        self._tokenizer = None

    @property
    def model_version(self) -> str:
        """What /api/health reports and the UI badge shows.

        `label` exists because a local checkpoint path ends in a directory name
        like `best`, and "best+c2" tells a demo audience nothing about which
        model produced the scores on screen.
        """
        suffix = "+c2" if self.c2 else "+uncalibrated"
        stem = self.label or self.model_id.replace("\\", "/").rstrip("/").split("/")[-1]
        return f"{stem}{suffix}"

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
        # The two checkpoints have different heads and the decode has to match:
        # LettuceDetect is binary (not-hallucinated / hallucinated), C1 is
        # 3-label BIO (O / B-HAL / I-HAL). Checking here rather than trusting
        # configuration means a wrong pairing fails at startup with a readable
        # message instead of producing plausible nonsense.
        expected = 3 if self.decode == "bio_argmax" else 2
        if model.config.num_labels != expected:
            raise RuntimeError(
                f"{self.model_id} has {model.config.num_labels} labels but "
                f"decode={self.decode!r} expects {expected}. The binary "
                "LettuceDetect head needs decode='threshold'; C1's BIO head "
                "needs decode='bio_argmax'."
            )
        self._model = model.to(device).eval()
        self._device = device

    def token_probabilities(
        self, prompt: str, answer: str
    ) -> Tuple[List[float], List[Tuple[int, int]]]:
        """P(hallucinated) and character offsets for each answer token.

        Collapses whichever head this checkpoint has down to one number, so the
        rest of the pipeline does not care which model is loaded. Binary:
        P(label 1). BIO: P(B-HAL) + P(I-HAL), matching `evaluate_c1.predict`
        exactly -- that sum is the score C2 was calibrated on.
        """
        rows, offsets = self.label_probabilities(prompt, answer)
        if self.decode == "bio_argmax":
            from src.c1_detector.bio import B_HAL, I_HAL

            return [row[B_HAL] + row[I_HAL] for row in rows], offsets
        return [row[HALLUCINATED_INDEX] for row in rows], offsets

    def label_probabilities(
        self, prompt: str, answer: str
    ) -> Tuple[List[List[float]], List[Tuple[int, int]]]:
        """The full label distribution per answer token, plus character offsets.

        Kept separate from `token_probabilities` because BIO decoding needs the
        argmax over all three labels, which a collapsed score cannot give back.
        """
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
            [[float(v) for v in probs[i]] for i in positions],
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

    def candidate_spans(
        self, prompt: str, answer: str
    ) -> Tuple[List[Tuple[int, int]], List[float], List[Tuple[int, int]]]:
        """Propose spans, and return the token scores and offsets behind them.

        Two decoders, one per head, and the choice is not cosmetic: C2's
        calibrator and conformal quantile were fitted on the spans the offline
        evaluation produced, so serving has to produce the same population.

        bio_argmax  argmax over O / B-HAL / I-HAL, decoded by
                    `evaluate_c1.spans_from_bio_ids`. This is what every reported
                    C1 number used, and B-HAL restarts a span so two adjacent
                    hallucinations stay separate.
        threshold   runs of tokens with P(hallucinated) >= candidate_threshold.
                    The binary head has no B/I distinction to exploit, so
                    adjacent spans merge. That merging is why span-exact F1 sits
                    far below span-overlap F1 in the baseline numbers.
        """
        from src.c1_detector.evaluate_c1 import (
            spans_from_bio_ids,
            spans_from_token_mask,
        )

        if self.decode == "bio_argmax":
            from src.c1_detector.bio import B_HAL, I_HAL

            rows, offsets = self.label_probabilities(prompt, answer)
            token_probs = [row[B_HAL] + row[I_HAL] for row in rows]
            tag_ids = [max(range(len(row)), key=row.__getitem__) for row in rows]
            return spans_from_bio_ids(tag_ids, offsets, answer), token_probs, offsets

        token_probs, offsets = self.token_probabilities(prompt, answer)
        mask = [p >= self.candidate_threshold for p in token_probs]
        return spans_from_token_mask(mask, offsets, answer), token_probs, offsets

    def analyze(self, req: AnalyzeRequest) -> AnalysisResult:
        started = time.perf_counter()
        prompt = format_prompt(
            req.question, req.context, req.task_type, style=self.prompt_style
        )
        candidates, token_probs, offsets = self.candidate_spans(prompt, req.answer)

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

        # Asked before any span is dropped, because the reference was built from
        # unfiltered decoder output and a filtered count would not compare.
        check, guarantee_applies = self._distribution_check(
            token_probs, candidates, covered_probs, req.answer
        )

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
            distribution_check=check,
            guarantee_applies=guarantee_applies,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _distribution_check(
        self,
        token_probs: Sequence[float],
        candidates: Sequence[Tuple[int, int]],
        covered_probs: Sequence[Sequence[float]],
        answer: str,
    ) -> Tuple[DistributionCheck, bool]:
        """Decide whether the coverage guarantee may be shown for this input.

        Three outcomes, and they are three different sentences on screen.

        No conformal layer at all: nothing to promise, and `guarantee_applies`
        is false for that reason rather than because of anything about the input.

        A conformal layer but no reference: the promise is shown, and the check
        says it did not run. That is the behaviour this endpoint had before the
        check existed, so a missing reference file degrades to the old
        behaviour rather than to a broken page -- but it says so.

        Both present: the p-value decides. When it fails, the spans are still
        returned and only the promise is withdrawn. A detector that stops
        detecting because it is unsure would be a worse product than one that
        detects and admits the promise does not hold.
        """
        if self.c2 is None:
            return (
                DistributionCheck(
                    checked=False,
                    in_distribution=True,
                    message=(
                        "No calibration layer is attached, so there is no "
                        "coverage guarantee to check against."
                    ),
                ),
                False,
            )
        if self.reference is None:
            return (
                DistributionCheck(
                    checked=False,
                    in_distribution=True,
                    message=(
                        "No exchangeability reference is loaded, so it is not "
                        "known whether this input resembles the data the "
                        "guarantee was calibrated on. Build one with "
                        "src.c2_calibration.exchangeability."
                    ),
                ),
                True,
            )

        features = response_features(
            token_probs,
            candidates,
            [len(covered) for covered in covered_probs],
            answer,
        )
        raw = self.reference.check(features)
        in_distribution = bool(raw["in_distribution"])
        odd = FEATURE_LABELS.get(raw["most_unusual"], raw["most_unusual"])

        if in_distribution:
            message = (
                "Nothing about this input looks unlike the data the guarantee "
                "was calibrated on. That is not proof it is in distribution -- "
                "the check sees a handful of features, and a shift can be real "
                "and invisible to all of them."
            )
        else:
            message = (
                f"This input is unlike the data the guarantee was calibrated "
                f"on -- most obviously its {odd}. The highlighting below is "
                "still the detector's, but the coverage promise does not apply "
                "to it. Recalibrate on your own data before trusting the dial."
            )

        check = DistributionCheck(
            checked=True,
            in_distribution=in_distribution,
            p_value=raw["p_value"],
            threshold=raw["threshold"],
            n_reference=raw["n_reference"],
            most_unusual=raw["most_unusual"],
            features=[FeatureCheck(**row) for row in raw["features"]],
            message=message,
        )
        return check, in_distribution


def build_from_env() -> Optional[LettuceDetectDetector]:
    """Construct from environment, or return None to leave the stub in place.

        TRUSTRAG_DETECTOR       "c1" for our own trained detector, "lettucedetect"
                                for the public baseline checkpoint. Anything else,
                                including unset, keeps the stub.
        TRUSTRAG_C2_ARTIFACT    path to c2_artifact.json
        TRUSTRAG_MODEL_ID       override the checkpoint (required for "c1", which
                                lives on disk and has no hub id)
        TRUSTRAG_DEVICE         cuda / cpu
        TRUSTRAG_CANDIDATE_THRESHOLD
        TRUSTRAG_PROMPT_STYLE   override the input format; normally left to the
                                detector default, which is the one it trained on
        TRUSTRAG_MAX_LENGTH
        TRUSTRAG_MODEL_LABEL    what /api/health calls the model

    The two detectors carry different defaults because they were trained on
    different inputs and at different lengths, and getting either wrong is a
    train/serve mismatch that shows up as quietly worse scores rather than as an
    error. C1 trained at max_length 3,072 on a bare question/context pair; the
    public checkpoint trained at 4,096 on its own instruction templates.

    Defaulting to off is deliberate: the test suite and a plain `uvicorn` start
    must not trigger a 1.6GB download.
    """
    choice = os.environ.get("TRUSTRAG_DETECTOR", "").lower()
    if choice == "c1":
        defaults = {
            "prompt_style": "c1",
            "max_length": 3072,
            "label": "trustrag-c1",
            "decode": "bio_argmax",
        }
    elif choice == "lettucedetect":
        defaults = {
            "prompt_style": "lettucedetect",
            "max_length": 4096,
            "label": None,
            "decode": "threshold",
        }
    else:
        return None

    artifact = os.environ.get("TRUSTRAG_C2_ARTIFACT")
    if artifact and not Path(artifact).exists():
        raise SystemExit(
            f"TRUSTRAG_C2_ARTIFACT points at {artifact}, which does not exist. "
            "Run scripts\\run_lettucedetect_baseline.ps1 for the baseline, or "
            "src.c2_calibration.run_c2 against results/c1 for our own detector."
        )

    # Defaults to the reference sitting beside the C2 artifact, because that is
    # where run_c2 and exchangeability both write for a given detector. Setting
    # it to an empty string turns the check off explicitly, which is different
    # from it being missing by accident.
    reference = os.environ.get("TRUSTRAG_OOD_REFERENCE")
    if reference is None and artifact:
        beside = Path(artifact).with_name("c2_ood_reference.json")
        reference = str(beside) if beside.exists() else None
    if reference and not Path(reference).exists():
        raise SystemExit(
            f"TRUSTRAG_OOD_REFERENCE points at {reference}, which does not "
            "exist. Build it with src.c2_calibration.exchangeability, or set "
            "the variable to an empty string to serve without the check."
        )

    model_id = os.environ.get("TRUSTRAG_MODEL_ID", DEFAULT_MODEL_ID)
    if choice == "c1" and model_id == DEFAULT_MODEL_ID:
        raise SystemExit(
            "TRUSTRAG_DETECTOR=c1 needs TRUSTRAG_MODEL_ID pointing at the local "
            "checkpoint, e.g. results\\c1\\modernbert-base\\best"
        )

    return LettuceDetectDetector(
        model_id=model_id,
        artifact_path=artifact,
        device=os.environ.get("TRUSTRAG_DEVICE") or None,
        candidate_threshold=float(
            os.environ.get("TRUSTRAG_CANDIDATE_THRESHOLD", "0.5")
        ),
        prompt_style=os.environ.get("TRUSTRAG_PROMPT_STYLE") or defaults["prompt_style"],
        max_length=int(
            os.environ.get("TRUSTRAG_MAX_LENGTH") or defaults["max_length"]
        ),
        label=os.environ.get("TRUSTRAG_MODEL_LABEL") or defaults["label"],
        decode=defaults["decode"],
        reference_path=reference or None,
    )
