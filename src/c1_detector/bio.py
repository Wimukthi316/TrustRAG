"""Character spans to BIO token labels, and back again.

The core functions here take an offset mapping rather than a tokenizer, so the
labelling logic can be tested exhaustively without downloading a model or
touching the network. `encode_example` is the thin wrapper that supplies a real
tokenizer's output.

Label scheme
    0  O      token is supported by the context
    1  B-HAL  first token of a hallucinated span
    2  I-HAL  continuation of a hallucinated span
    -100      not part of the answer (context, question, padding, specials);
              excluded from the loss, which is how the detector is trained to
              judge only the generated answer

Why BIO and not plain binary: two hallucinated spans that sit next to each other
are indistinguishable under binary labels, so they merge into one span at decode
time and the span-boundary F1 is wrong. B- marks the restart. `bio_to_binary`
collapses to the binary scheme if a run needs it, so nothing is lost by tagging
BIO first.

The critical invariant, enforced by `assert_round_trip`: decoding the BIO labels
back into character spans must return the spans we started from, modulo the
tokenizer's own granularity. If that fails, every downstream number is wrong.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

IGNORE_INDEX = -100

LABEL_NAMES = ["O", "B-HAL", "I-HAL"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

# Spelled out rather than using "O" as an identifier, which is easy to misread
# as a zero at a glance.
OUTSIDE, B_HAL, I_HAL = 0, 1, 2

# The answer is the second sequence in the encoder input, so its tokens carry
# sequence id 1. Context and question are sequence 0; specials are None.
ANSWER_SEQUENCE_ID = 1

Span = Tuple[int, int]


def char_spans_to_bio(
    offsets: Sequence[Tuple[int, int]],
    sequence_ids: Sequence[Optional[int]],
    spans: Sequence[Span],
) -> List[int]:
    """Label each token, given its character offsets into the answer.

    `offsets[i]` is the (start, end) of token i within whichever sequence it
    belongs to. Only tokens with sequence id ANSWER_SEQUENCE_ID are labelled;
    everything else gets IGNORE_INDEX.

    A token is hallucinated if its character range overlaps a hallucinated span
    at all. Partial overlap counts: subword tokenizers routinely split across a
    span boundary, and treating a half-covered token as supported would let the
    model learn to shave the edges off every span.
    """
    if len(offsets) != len(sequence_ids):
        raise ValueError(
            f"offsets ({len(offsets)}) and sequence_ids ({len(sequence_ids)}) "
            "must be the same length"
        )

    ordered = sorted(spans)
    labels: List[int] = []
    # Tracks whether the previous answer token was inside the same span, which
    # is what distinguishes B-HAL from I-HAL.
    open_span: Optional[Span] = None

    for (start, end), seq_id in zip(offsets, sequence_ids):
        if seq_id != ANSWER_SEQUENCE_ID or end <= start:
            # Special tokens report (0, 0); they are not part of the answer.
            labels.append(IGNORE_INDEX)
            continue

        hit = _first_overlap(ordered, start, end)
        if hit is None:
            labels.append(OUTSIDE)
            open_span = None
        elif hit == open_span:
            labels.append(I_HAL)
        else:
            labels.append(B_HAL)
            open_span = hit

    return labels


def _first_overlap(ordered: Sequence[Span], start: int, end: int) -> Optional[Span]:
    """Return the first span overlapping [start, end), or None.

    Linear scan. Spans per response are in the single digits, so a bisect would
    add risk of an off-by-one for no measurable gain.
    """
    for span in ordered:
        if span[0] < end and start < span[1]:
            return span
    return None


def bio_to_char_spans(
    labels: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
    sequence_ids: Sequence[Optional[int]],
    answer: Optional[str] = None,
) -> List[Span]:
    """Decode BIO labels back into character spans over the answer.

    This is also the inference-time decoder: the model emits label ids, this
    turns them into the `start`/`end` that src/common/schema.py's Span expects.

    Pass `answer` to trim whitespace off each decoded span. ModernBERT's
    tokenizer is byte-level BPE, so a word-initial token carries its preceding
    space and its offset starts one character early. Without trimming almost
    every decoded span is one character too wide; every highlight in the UI
    would start on a space, and span-boundary F1 would be penalised for a
    tokenizer detail rather than a modelling error.

    With trimming, measured over all 17,790 records with
    answerdotai/ModernBERT-base at max_length 4096 on 2026-08-11:

        span count preserved   7,664 / 7,664 responses that carry spans
        decoded exactly        7,595  (99.1%)
        one character wide        68
        two characters wide        1
        answers truncated          0  (longest sequence was 2,628 tokens)

    The 69 that stay wide are gold spans whose end offset falls inside a token,
    almost always just before trailing punctuation -- the annotator marked
    "attac" where the tokenizer's token is "attack". That is a granularity limit
    of subword tokenisation, not a bug, and it caps span-exact-match at roughly
    99% no matter how good the model is. Worth stating in the paper's metrics
    section rather than discovering during the viva.
    """
    spans: List[Span] = []
    current: Optional[List[int]] = None

    for label, (start, end), seq_id in zip(labels, offsets, sequence_ids):
        if seq_id != ANSWER_SEQUENCE_ID or label == IGNORE_INDEX:
            continue

        if label == B_HAL:
            if current is not None:
                spans.append((current[0], current[1]))
            current = [start, end]
        elif label == I_HAL:
            if current is None:
                # I- without a preceding B-. Model output can look like this;
                # treat it as the start of a span rather than dropping it.
                current = [start, end]
            else:
                current[1] = end
        else:  # OUTSIDE
            if current is not None:
                spans.append((current[0], current[1]))
                current = None

    if current is not None:
        spans.append((current[0], current[1]))

    if answer is None:
        return spans
    return [t for t in (_trim(answer, s, e) for s, e in spans) if t is not None]


def _trim(answer: str, start: int, end: int) -> Optional[Span]:
    """Shrink a span past leading and trailing whitespace. None if nothing left."""
    while start < end and answer[start].isspace():
        start += 1
    while end > start and answer[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def bio_to_binary(labels: Sequence[int]) -> List[int]:
    """Collapse BIO to the binary supported/hallucinated scheme.

    IGNORE_INDEX is preserved so the loss mask survives the collapse.
    """
    return [
        label if label == IGNORE_INDEX else (1 if label in (B_HAL, I_HAL) else 0)
        for label in labels
    ]


def assert_round_trip(
    spans: Sequence[Span],
    labels: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
    sequence_ids: Sequence[Optional[int]],
    answer: str,
    tolerance: int = 0,
) -> None:
    """Check that decoding the labels recovers the spans we encoded.

    Decoded spans are snapped to token boundaries, so they can be slightly wider
    than the published character spans when a label starts mid-token. `tolerance`
    is the number of characters of widening allowed per edge. Zero is the right
    default for a hand check on a handful of examples; raise it only if you have
    confirmed the widening comes from tokenisation and not from a bug.

    Whitespace trimming is applied, which is what makes tolerance=0 achievable
    on real RAGTruth data with ModernBERT's tokenizer.
    """
    decoded = bio_to_char_spans(labels, offsets, sequence_ids, answer=answer)
    original = sorted(spans)

    if len(decoded) != len(original):
        raise AssertionError(
            f"round trip changed the span count: encoded {len(original)} "
            f"{original}, decoded {len(decoded)} {decoded}"
        )

    for (os_, oe), (ds, de) in zip(original, decoded):
        if ds > os_ or de < oe:
            raise AssertionError(
                f"decoded span [{ds}:{de}] ({answer[ds:de]!r}) does not cover the "
                f"original [{os_}:{oe}] ({answer[os_:oe]!r})"
            )
        if (os_ - ds) > tolerance or (de - oe) > tolerance:
            raise AssertionError(
                f"decoded span [{ds}:{de}] ({answer[ds:de]!r}) is wider than the "
                f"original [{os_}:{oe}] ({answer[os_:oe]!r}) by more than "
                f"{tolerance} characters"
            )


# --------------------------------------------------------------------------
# Tokenizer wrapper
# --------------------------------------------------------------------------


def build_first_sequence(question: Optional[str], context: str) -> str:
    """The first encoder sequence for C1: question, blank line, context.

    Lives here and is imported by the serving code rather than being written out
    twice. Training reads it through `encode_example`; the API reads it through
    `backend/app/services/lettucedetect_detector.format_prompt(style="c1")`. If
    the two ever disagree the model is served inputs it never saw in training,
    which costs accuracy and raises no error anywhere -- so there is exactly one
    implementation and a test asserts the server uses it.

    Deliberately has no instruction template. RAGTruth's own `prompt` field and
    LettuceDetect's templates both add one; ours does not, and that difference is
    recorded in the report's method section.
    """
    return f"{question}\n\n{context}" if question and question.strip() else context


def encode_example(
    tokenizer: Any,
    record: Dict[str, Any],
    max_length: int = 4096,
) -> Dict[str, Any]:
    """Tokenize one build_examples record and attach BIO labels.

    The context and question go in as the first sequence and the answer as the
    second, so `sequence_ids()` cleanly separates them and the answer's offset
    mapping is relative to the answer string.

    truncation="only_first" is deliberate and load-bearing: it truncates the
    context, never the answer. If the answer were truncated, its trailing
    hallucination labels would be silently dropped and both the training signal
    and the evaluation would be quietly wrong.

    Requires a fast tokenizer -- offset mapping and sequence_ids are not
    available on the slow Python ones.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "a fast tokenizer is required for offset mapping; load it with "
            "AutoTokenizer.from_pretrained(..., use_fast=True)"
        )

    answer = record["answer"]
    first = build_first_sequence(record.get("question"), record["context"])

    encoding = tokenizer(
        first,
        answer,
        truncation="only_first",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    sequence_ids = encoding.sequence_ids(0)
    offsets = encoding["offset_mapping"]
    spans = [(s["start"], s["end"]) for s in record.get("spans", [])]

    answer_offsets = [
        off for off, sid in zip(offsets, sequence_ids) if sid == ANSWER_SEQUENCE_ID
    ]
    answer_truncated = bool(answer_offsets) and answer_offsets[-1][1] < len(answer)

    labels = char_spans_to_bio(offsets, sequence_ids, spans)

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "offset_mapping": offsets,
        "sequence_ids": sequence_ids,
        "answer_truncated": answer_truncated,
        "n_answer_tokens": len(answer_offsets),
    }
