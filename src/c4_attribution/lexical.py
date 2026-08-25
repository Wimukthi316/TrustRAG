"""Layer 3 -- the lexical attribution baseline, and the four floors it is read against.

`src/c3_explanation/grounding.best_matching_sentence` is the baseline the neural
scorer has to beat. It is re-implemented here rather than imported, for one
reason: C3 is not always on the branch C4 runs from, and a component that cannot
be evaluated because a sibling has not been pushed is a component that fails at
the worst possible moment. `tests/test_c4_lexical.py` asserts the two agree
whenever C3 *is* importable, so the copy can never silently drift from the
original.

The four floors are not decoration. Attribution over a median of sixteen
sentences is a task where a policy that ignores its input entirely can look
respectable, and three of these four do exactly that:

    first sentence     the cheapest possible policy
    longest sentence   length is the classic lexical confound
    uniform random     roughly 1/16 on the median example
    always abstain     the majority-correct policy on QA

The last one is the honest floor. On the QA task most annotator notes say the
information is not in the source at all, so a system that says nothing beats a
system that guesses, and the table has to show it.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Mirrors src/c3_explanation/grounding.py. The parity test pins them together.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.,'][A-Za-z0-9]+)*")

STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with without as is are was were be been being it its he she they
    them his her their there here not no do does did doing have has had having
    will would can could should may might must about into over under again
    further once other such only own same so too very s t just
    """.split()
)


def tokenise(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN.finditer(text or "")]


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenise(text) if t not in STOPWORDS and len(t) > 1]


def idf_weights(sentences: Sequence[str]) -> Dict[str, float]:
    """IDF over the record's own context sentences, so nothing leaks across records."""
    n = len(sentences)
    if n == 0:
        return {}
    document_frequency: Dict[str, int] = {}
    for sentence in sentences:
        for token in set(content_tokens(sentence)):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return {
        token: math.log((n + 1) / (df + 1)) + 1.0
        for token, df in document_frequency.items()
    }


def _weighted(tokens: Sequence[str], weights: Dict[str, float]) -> float:
    default = max(weights.values(), default=1.0)
    return sum(weights.get(t, default) for t in tokens)


def idf_overlap_scores(span_text: str, sentences: Sequence[str]) -> List[float]:
    """IDF-weighted share of the span's content tokens each sentence carries."""
    span_set = set(content_tokens(span_text))
    weights = idf_weights(sentences)
    span_mass = _weighted(sorted(span_set), weights)
    out: List[float] = []
    for sentence in sentences:
        shared = span_set & set(content_tokens(sentence))
        out.append(
            _weighted(sorted(shared), weights) / span_mass if span_mass > 0 else 0.0
        )
    return out


def best_sentence(span_text: str, sentences: Sequence[str]) -> Tuple[Optional[int], float]:
    """Index of the most plausible evidence sentence, and its score.

    Ties break on the lowest index, matching `grounding.best_matching_sentence`,
    so the choice never depends on iteration order.
    """
    scores = idf_overlap_scores(span_text, sentences)
    if not scores:
        return (None, 0.0)
    best_index = 0
    best_value = scores[0]
    for index in range(1, len(scores)):
        if scores[index] > best_value:
            best_index, best_value = index, scores[index]
    return (best_index, float(best_value))


# --------------------------------------------------------------------------
# Containment: the metric the Layer 1 audit forced
# --------------------------------------------------------------------------


def containment(snippet: str, sentence: str) -> float:
    """Share of the annotator's quoted content tokens present in one sentence.

    Fewer than half the human evidence quotes appear verbatim in the source --
    annotators paraphrased, trimmed and re-punctuated -- so a string match
    scores a system against a reference that moves. Content-token containment is
    what the audit found to be measurable, and it is reported beside the
    verbatim rate rather than instead of it.
    """
    quoted = set(content_tokens(snippet))
    if not quoted:
        return 0.0
    return len(quoted & set(content_tokens(sentence))) / len(quoted)


def containment_scores(snippet: str, sentences: Sequence[str]) -> List[float]:
    return [containment(snippet, s) for s in sentences]


def best_containment(snippet: str, sentences: Sequence[str]) -> Tuple[Optional[int], float]:
    """The sentence holding most of the quote, ties to the lowest index."""
    scores = containment_scores(snippet, sentences)
    if not scores:
        return (None, 0.0)
    best_index = int(np.argmax(scores))
    return (best_index, float(scores[best_index]))


# --------------------------------------------------------------------------
# The floors
# --------------------------------------------------------------------------


def floor_first(sentences: Sequence[str]) -> Optional[int]:
    """Always the first context sentence."""
    return 0 if sentences else None


def floor_longest(sentences: Sequence[str]) -> Optional[int]:
    """Always the longest context sentence, ties to the lowest index."""
    if not sentences:
        return None
    lengths = [len(s) for s in sentences]
    return int(np.argmax(lengths))


def floor_random(sentences: Sequence[str], seed: int) -> Optional[int]:
    """A uniformly random sentence, seeded per row so the run reproduces."""
    if not sentences:
        return None
    return int(np.random.default_rng(seed).integers(0, len(sentences)))


FLOOR_NAMES: Tuple[str, ...] = ("first", "longest", "random", "abstain")


def floor_prediction(name: str, sentences: Sequence[str], seed: int) -> Optional[int]:
    """One floor policy's prediction. `abstain` returns None, which is the point."""
    if name == "first":
        return floor_first(sentences)
    if name == "longest":
        return floor_longest(sentences)
    if name == "random":
        return floor_random(sentences, seed)
    if name == "abstain":
        return None
    raise ValueError(f"unknown floor {name!r}; expected one of {FLOOR_NAMES}")
