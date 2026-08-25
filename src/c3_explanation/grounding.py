"""Grounding features: what a span shares with the context it was generated from.

Three parts of C3 need the same two questions answered -- which context sentence
is this span about, and how much of the span is in the context at all -- so they
are answered once, here:

    Layer 3.1  the CPU feature baseline for the type model. Axis A (conflict vs
               baseless) is exactly the question "is there a sentence this
               contradicts, or is there simply nothing about it", so a handful of
               overlap features should carry most of it. If they do, the
               transformer has to beat something honest.
    Layer 5.1  the template explanation quotes the best-matching sentence.
    Add-on 6   the self-faithfulness check asks whether the sentence C3 quoted
               really appears in the context.

Everything is deterministic, CPU-only, and adds no dependency. There is no
embedding model here on purpose: sentence-transformers would be a new dependency
and the project rule is to ask first, so the lexical version ships and the
embedding version stays an offer.

The scoring is IDF-weighted token overlap where the IDF comes from the record's
own context sentences. That keeps every number self-contained per record -- no
corpus-level statistic leaks from train into test -- and it costs nothing.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

# Mirrors backend/app/services/detector.py. tests/test_c3_grounding.py asserts the
# two stay identical; if that test fails, one of them was edited alone.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.,'][A-Za-z0-9]+)*")
_NUMBER = re.compile(r"^\d[\d.,]*$")

# A short, explicit stop list. Long enough to stop "the" dominating an overlap
# score, short enough that nothing meaning-bearing is thrown away.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with without as is are was were be been being it its he she they
    them his her their there here not no do does did doing have has had having
    will would can could should may might must about into over under again
    further once other such only own same so too very s t just
    """.split()
)


def split_sentences(text: str) -> List[str]:
    """Deliberately naive sentence splitter, identical to the serving path's."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def tokenise(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN.finditer(text or "")]


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenise(text) if t not in STOPWORDS and len(t) > 1]


def number_tokens(text: str) -> List[str]:
    return [t for t in tokenise(text) if _NUMBER.match(t)]


def capitalised_tokens(text: str) -> List[str]:
    """Rough proxy for named entities: mid-sentence capitalised words.

    A proxy, not a tagger. It is named that way in the feature list so nobody
    reads it as entity linking.
    """
    tokens = _TOKEN.findall(text or "")
    return [t.lower() for t in tokens[1:] if t[:1].isupper() and not t.isupper()]


def idf_weights(sentences: Sequence[str]) -> Dict[str, float]:
    """Inverse document frequency over the record's own context sentences."""
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
    # An unseen token gets the weight of a token in no sentence at all, which is
    # the largest value the formula can produce.
    default = max(weights.values(), default=1.0)
    return sum(weights.get(t, default) for t in tokens)


def sentence_scores(
    span_text: str, sentences: Sequence[str]
) -> List[Dict[str, float]]:
    """Per sentence: containment, Jaccard and IDF-weighted overlap with the span."""
    span_tokens = content_tokens(span_text)
    span_set = set(span_tokens)
    weights = idf_weights(sentences)
    span_mass = _weighted(sorted(span_set), weights)
    out: List[Dict[str, float]] = []
    for sentence in sentences:
        sentence_set = set(content_tokens(sentence))
        shared = span_set & sentence_set
        union = span_set | sentence_set
        out.append(
            {
                "containment": len(shared) / len(span_set) if span_set else 0.0,
                "jaccard": len(shared) / len(union) if union else 0.0,
                "idf_overlap": (
                    _weighted(sorted(shared), weights) / span_mass
                    if span_mass > 0
                    else 0.0
                ),
            }
        )
    return out


def best_matching_sentence(
    span_text: str, sentences: Sequence[str], key: str = "idf_overlap"
) -> Tuple[Optional[int], float]:
    """Index of the context sentence a span is most plausibly about, and its score.

    Ties break on the lowest index so the choice does not depend on iteration
    order. Returns (None, 0.0) when there are no sentences.
    """
    scores = sentence_scores(span_text, sentences)
    if not scores:
        return (None, 0.0)
    best_index = 0
    best_value = scores[0][key]
    for index in range(1, len(scores)):
        if scores[index][key] > best_value:
            best_index, best_value = index, scores[index][key]
    return (best_index, float(best_value))


FEATURE_NAMES: Tuple[str, ...] = (
    "max_containment",
    "max_jaccard",
    "max_idf_overlap",
    "unseen_token_rate",
    "number_present_rate",
    "has_number",
    "entity_present_rate",
    "has_entity",
    "span_content_tokens",
    "span_char_len",
    "span_relative_position",
    "best_sentence_content_tokens",
    "n_context_sentences",
    "task_qa",
    "task_data2text",
    "task_summarization",
)


def span_features(
    span_text: str,
    answer: str,
    span_start: int,
    sentences: Sequence[str],
    task_type: str,
) -> Dict[str, float]:
    """The feature vector for one span. Keys are exactly FEATURE_NAMES.

    Every feature is computable for a gold span and for a predicted span alike.
    The detector's own score is deliberately NOT here: the model trains on gold
    spans, which have no detector score, so including it would give the two
    settings different feature spaces and make the numbers incomparable.
    """
    scores = sentence_scores(span_text, sentences)
    span_tokens = content_tokens(span_text)
    span_set = set(span_tokens)

    context_tokens = set()
    for sentence in sentences:
        context_tokens |= set(content_tokens(sentence))
    unseen = span_set - context_tokens
    unseen_rate = len(unseen) / len(span_set) if span_set else 0.0

    span_numbers = set(number_tokens(span_text))
    context_numbers = set()
    span_entities = set(capitalised_tokens(span_text))
    context_entities = set()
    for sentence in sentences:
        context_numbers |= set(number_tokens(sentence))
        context_entities |= set(capitalised_tokens(sentence))

    best_index, _ = best_matching_sentence(span_text, sentences)
    best_len = (
        len(content_tokens(sentences[best_index])) if best_index is not None else 0
    )

    task = (task_type or "").lower()
    return {
        "max_containment": max((s["containment"] for s in scores), default=0.0),
        "max_jaccard": max((s["jaccard"] for s in scores), default=0.0),
        "max_idf_overlap": max((s["idf_overlap"] for s in scores), default=0.0),
        "unseen_token_rate": unseen_rate,
        "number_present_rate": (
            len(span_numbers & context_numbers) / len(span_numbers)
            if span_numbers
            else 1.0
        ),
        "has_number": 1.0 if span_numbers else 0.0,
        "entity_present_rate": (
            len(span_entities & context_entities) / len(span_entities)
            if span_entities
            else 1.0
        ),
        "has_entity": 1.0 if span_entities else 0.0,
        "span_content_tokens": float(len(span_tokens)),
        "span_char_len": float(len(span_text or "")),
        "span_relative_position": (
            span_start / len(answer) if answer else 0.0
        ),
        "best_sentence_content_tokens": float(best_len),
        "n_context_sentences": float(len(sentences)),
        "task_qa": 1.0 if task == "qa" else 0.0,
        "task_data2text": 1.0 if task == "data2text" else 0.0,
        "task_summarization": 1.0 if task == "summarization" else 0.0,
    }


def feature_vector(features: Dict[str, float]) -> List[float]:
    """FEATURE_NAMES order, so a matrix column always means the same thing."""
    return [float(features[name]) for name in FEATURE_NAMES]
