"""Mapping from RAGTruth's raw string labels onto the enums in src/common/schema.py.

schema.py points at this file for the ErrorType mapping, so it is the single place
where a raw dataset string is turned into one of our identifiers.

Everything here was read off the published dataset documentation at
https://github.com/ParticleMedia/RAGTruth (README, "Dataset" section) and the
sample records it publishes. The raw strings are matched case-insensitively and
whitespace-normalised because annotator-entered strings are not guaranteed to be
uniform across 14,289 spans.

IMPORTANT: `assert_label_coverage()` exists so that an unseen raw string is a loud
failure rather than a silent UNKNOWN. Run it once over the real file before
training. If it raises, add the string here -- do not widen the fallback.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from src.common.schema import ErrorType, TaskType

# RAGTruth's four annotated hallucination categories. Keys are normalised
# (lowercased, runs of whitespace collapsed) forms of the raw `label_type` value.
_ERROR_TYPE_BY_RAW: Dict[str, ErrorType] = {
    "evident conflict": ErrorType.EVIDENT_CONFLICT,
    "subtle conflict": ErrorType.SUBTLE_CONFLICT,
    "evident baseless info": ErrorType.EVIDENT_BASELESS,
    "subtle baseless info": ErrorType.SUBTLE_BASELESS,
}

# source_info.jsonl uses these three task_type values.
_TASK_TYPE_BY_RAW: Dict[str, TaskType] = {
    "qa": TaskType.QA,
    "data2txt": TaskType.DATA2TEXT,
    "summary": TaskType.SUMMARIZATION,
}

_WS = re.compile(r"\s+")


def normalise(raw: str) -> str:
    return _WS.sub(" ", raw.strip().lower())


def error_type_of(raw: str) -> ErrorType:
    """Map a raw `label_type` string to an ErrorType.

    Falls back to UNKNOWN so that one odd annotation cannot crash a whole
    preprocessing run. Use assert_label_coverage() to find those cases up front.
    """
    return _ERROR_TYPE_BY_RAW.get(normalise(raw), ErrorType.UNKNOWN)


def task_type_of(raw: str) -> TaskType:
    """Map a raw `task_type` string to a TaskType.

    Unlike error types, an unrecognised task type is a hard error: results are
    reported per task, so a record silently landing in OTHER would corrupt the
    per-task breakdown that PP2 requires.
    """
    key = normalise(raw)
    if key not in _TASK_TYPE_BY_RAW:
        raise KeyError(
            f"unrecognised RAGTruth task_type {raw!r}; expected one of "
            f"{sorted(_TASK_TYPE_BY_RAW)}"
        )
    return _TASK_TYPE_BY_RAW[key]


def unmapped_error_types(raws: Iterable[str]) -> List[str]:
    """Return the distinct raw label_type strings that do not map to a category."""
    seen = {normalise(r) for r in raws}
    return sorted(s for s in seen if s not in _ERROR_TYPE_BY_RAW)


def assert_label_coverage(raws: Iterable[str]) -> None:
    """Raise if any raw label_type in the corpus is unmapped."""
    missing = unmapped_error_types(raws)
    if missing:
        raise ValueError(
            "RAGTruth label_type values with no mapping in ragtruth_labels.py: "
            f"{missing}. Add them explicitly rather than letting them fall through "
            "to ErrorType.UNKNOWN."
        )
