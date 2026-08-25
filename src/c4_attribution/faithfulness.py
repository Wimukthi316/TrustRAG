"""Layer 9 -- is the sentence C4 quotes actually in the context, and how wrong is it when it is wrong?

Two checks, and the second is the interesting one.

**Self-faithfulness.** Every returned `evidence_sentence` must appear verbatim
in `context_sentences[evidence_index]`. C3 measured its own version of this and
got 1.0000, and measured it anyway -- because an off-by-one in the index, or a
template quoting text it invented, would show up here and nowhere else. An
off-by-one is the single most likely C4 bug and the one that would look worst in
a live demo, so it is asserted in a test as well as reported as a rate.

**Post-rationalisation.** When the system returns a sentence the annotator did
*not* quote, is that sentence at least in the right neighbourhood? A system
whose errors are all near-misses is a different product from one whose errors
are random, and the histogram of sentence-index distance says which one was
built. The design document raises this citing "Correctness is not Faithfulness";
this is the cheap, measurable version of that question.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from src.c4_attribution.attribution_eval import Prediction
from src.c4_attribution.dataset import SpanRecord
from src.c4_attribution.matching import Match


def check(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    predictions: Sequence[Prediction],
) -> Dict[str, Any]:
    """Self-faithfulness and the error-distance histogram for one system."""
    quoted_ok = 0
    quoted_total = 0
    index_out_of_range = 0
    distances: List[int] = []
    near_miss = 0
    far_miss = 0

    for row, gold, prediction in zip(rows, matches, predictions):
        top1 = prediction.top1
        if top1 is None:
            continue
        sentences = row.sentences
        quoted_total += 1
        if 0 <= top1 < len(sentences):
            quoted = sentences[top1]
            if quoted and quoted in row.response.context:
                quoted_ok += 1
        else:
            index_out_of_range += 1

        target = gold.convention_a
        if target is None or target == top1:
            continue
        distance = abs(top1 - target)
        distances.append(distance)
        if distance <= 1:
            near_miss += 1
        else:
            far_miss += 1

    histogram: Dict[str, int] = {}
    for distance in distances:
        bucket = str(distance) if distance <= 5 else "6+"
        histogram[bucket] = histogram.get(bucket, 0) + 1

    return {
        "n_returned": quoted_total,
        "self_faithfulness": quoted_ok / quoted_total if quoted_total else float("nan"),
        "index_out_of_range": index_out_of_range,
        "n_errors": len(distances),
        "near_miss": near_miss,
        "far_miss": far_miss,
        "share_near_miss": near_miss / len(distances) if distances else float("nan"),
        "mean_index_distance": float(np.mean(distances)) if distances else float("nan"),
        "median_index_distance": float(np.median(distances)) if distances else float("nan"),
        "index_distance_histogram": histogram,
        "note": (
            "self_faithfulness below 1.0 means the system quoted text that is "
            "not in the context it was given, which is an index bug rather than "
            "a modelling weakness. index_out_of_range counts the case that would "
            "crash the serving path, and it must be zero."
        ),
    }
