"""Does this input look like the data the guarantee was calibrated on?

Block C measured what happens when it does not: coverage falls from 0.8808 to
0.7410 against a 0.900 target, seven noise bands out, and neither reweighting
repairs it. That is a finding in a table. This module turns it into something
the running system can act on.

**The problem it solves.** The conformal guarantee is a promise about data drawn
exchangeably with the calibration split. Nothing in the serving path checks
that. A user pastes an answer from a different domain, written by a different
model, ten times longer than anything in RAGTruth -- and the interface prints a
calibrated score and a coverage badge with exactly the same confidence it shows
on a corpus record. That is not a small cosmetic problem. It is the demo
contradicting the result.

**What this does.** Before the badge is shown, compare the input against the
calibration split on five quantities the serving path already has, and return a
p-value for the null hypothesis "this response is exchangeable with the
calibration set". If it is small, the interface keeps the spans and drops the
promise.

**How the p-value is built, and why this shape.** It reuses the machinery the
rest of C2 already runs on rather than inventing a second, unrelated statistic.

1. For each feature j, the calibration split gives an empirical distribution.
   For a response x, take its two-sided tail depth

        d_j(x) = min( P(F_j <= f_j(x)) , P(F_j >= f_j(x)) )

   which is near 0.5 at the median of that feature and near 0 in either tail.
2. Reduce to one number by taking the worst feature, D(x) = min_j d_j(x). A
   response is unusual if it is unusual in *any* one of the five ways, which is
   the behaviour wanted here -- an answer of ordinary length with a wildly
   unusual score distribution should still trip.
3. Compute D for every calibration response, then read off the conformal
   p-value with the same finite-sample correction used everywhere else:

        p(x) = ( 1 + #{ i : D_i <= D(x) } ) / ( n + 1 )

   Warn when p < threshold. The threshold is therefore a **false-alarm rate**,
   not a magic number: at 0.01, about one in a hundred genuinely in-distribution
   responses is expected to trip, and that is a property that can be measured
   rather than hoped for.

**Two honesty caveats that belong on the screen, not only in this docstring.**

*It is a smoke alarm, not a proof.* The test is one-sided. A large p-value does
not establish that the input is in-distribution; it only says nothing obviously
unusual was found in these five features. Shift can be real and invisible here.

*The p-value is approximate, not exact.* The calibration depths D_i are computed
against an empirical distribution that includes the point itself, while a new
response is not in it. The asymmetry is of order 1/n and n is 1,509, so it is
small -- but "small" is not "zero", and the honest check is empirical: the
builder measures the actual trip rate on the held-out test split and records it
in the artifact. If that number is not close to the threshold, the threshold is
lying and the artifact says so.

**Feature parity is structural, not documented.** `features_from_record` reads
an offline probability dump and `response_features` reads what the serving path
holds in memory, and both call the same `_features` underneath. A test asserts
they agree on real records. Two implementations of "how long is this answer"
that drift apart would make the reference meaningless in exactly the silent way
this project keeps finding.

Build the reference with:

    python -m src.c2_calibration.exchangeability \\
        --calib results/c1/calib/probabilities.jsonl \\
        --test  results/c1/test/probabilities.jsonl \\
        --out   results/c2/c1/c2_ood_reference.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

DEFAULT_THRESHOLD = 0.01

# The eight quantities. Every one is available both in an offline probability
# dump and in the serving path's own memory, which is the whole constraint on
# this list. Context length is not here: the dump does not carry the context, so
# there would be nothing to compare a served context against.
#
# The last three are span geometry, and they are here because **Block C's domain
# classifier said so**. Fitted on the estimation half of RAGBench, its largest
# standardised coefficients were span token count and span character length --
# the shift between those two corpora lives more in the shape of the phrases the
# detector proposes than in how suspicious it finds them. They were taken from
# that diagnosis, not by sweeping feature sets until the trip rate looked good.
#
# Worth being clear about what adding features can and cannot do here. The
# p-value compares a response's worst-feature depth against the calibration
# distribution of the same statistic, so the false-alarm rate stays at the
# threshold whatever the feature list is. More features can only change the
# alarm's *power*; they cannot quietly break its *validity*. That is the
# property that makes extending this list a safe thing to do.
FEATURE_NAMES: Tuple[str, ...] = (
    "log_answer_tokens",
    "log_candidate_spans",
    "mean_token_prob",
    "max_token_prob",
    "p90_token_prob",
    "log_mean_span_chars",
    "log_mean_span_tokens",
    "answer_fraction_covered",
)

FEATURE_LABELS: Dict[str, str] = {
    "log_answer_tokens": "answer length",
    "log_candidate_spans": "number of highlighted phrases",
    "mean_token_prob": "average suspicion across the answer",
    "max_token_prob": "peak suspicion",
    "p90_token_prob": "suspicion in the top tenth of the answer",
    "log_mean_span_chars": "average length of a highlighted phrase",
    "log_mean_span_tokens": "average word count of a highlighted phrase",
    "answer_fraction_covered": "share of the answer that got highlighted",
}


# --------------------------------------------------------------------------
# Features. One implementation, two callers.
# --------------------------------------------------------------------------


def _features(
    token_probs: Sequence[float],
    span_char_lengths: Sequence[int],
    span_token_counts: Sequence[int],
    answer_chars: int,
) -> Dict[str, float]:
    """The eight numbers, from what both callers have.

    Degenerate inputs get zeros rather than exceptions: an answer that tokenises
    to nothing the model scores, or one where the detector proposed no spans at
    all. Neither is an error, and both are genuinely unusual -- the reference
    will say so on its own without this function needing an opinion about it.
    """
    probs = np.asarray(list(token_probs), dtype=np.float64)
    chars = np.asarray(list(span_char_lengths), dtype=np.float64)
    tokens = np.asarray(list(span_token_counts), dtype=np.float64)

    covered = float(chars.sum()) if chars.size else 0.0
    return {
        "log_answer_tokens": math.log1p(probs.size),
        "log_candidate_spans": math.log1p(chars.size),
        "mean_token_prob": float(probs.mean()) if probs.size else 0.0,
        "max_token_prob": float(probs.max()) if probs.size else 0.0,
        "p90_token_prob": float(np.percentile(probs, 90)) if probs.size else 0.0,
        "log_mean_span_chars": math.log1p(float(chars.mean())) if chars.size else 0.0,
        "log_mean_span_tokens": math.log1p(float(tokens.mean())) if tokens.size else 0.0,
        "answer_fraction_covered": (
            min(1.0, covered / answer_chars) if answer_chars > 0 else 0.0
        ),
    }


def response_features(
    token_probs: Sequence[float],
    candidate_spans: Sequence[Tuple[int, int]],
    covered_token_counts: Sequence[int],
    answer: str,
) -> Dict[str, float]:
    """Serving-side entry point.

    `candidate_spans` must be what the **decoder** produced, before any are
    dropped for coming back PASS. The offline dump's `pred_spans` is unfiltered,
    so filtering here would compare a filtered count against an unfiltered
    reference and shift three features at once, silently.
    """
    return _features(
        token_probs,
        [int(end) - int(start) for start, end in candidate_spans],
        covered_token_counts,
        len(answer),
    )


def features_from_record(record: Dict[str, Any]) -> Dict[str, float]:
    """Offline entry point, reading one line of a probabilities.jsonl dump."""
    spans = record.get("pred_spans", [])
    return _features(
        record.get("token_probs", []),
        [int(span["end"]) - int(span["start"]) for span in spans],
        [int(span.get("n_tokens", 0)) for span in spans],
        len(record.get("answer", "")),
    )


def feature_matrix(
    records: Sequence[Dict[str, Any]], features: Sequence[str] = FEATURE_NAMES
) -> Dict[str, np.ndarray]:
    """Per-feature sorted arrays over a whole dump, ready for lookups."""
    rows = [features_from_record(record) for record in records]
    return {
        name: np.sort(np.asarray([row[name] for row in rows], dtype=np.float64))
        for name in features
    }


# --------------------------------------------------------------------------
# The reference, and the p-value it answers with
# --------------------------------------------------------------------------


def _tail_depth(sorted_values: np.ndarray, value: float) -> Tuple[float, float]:
    """(depth, percentile) of one value against one sorted calibration array.

    Depth is how far into the nearer tail the value sits: 0.5 at the median,
    approaching 0 at either extreme. Percentile is the fraction of calibration
    values at or below it, kept separately because it is what a human reads --
    "longer than 99% of what we calibrated on" is a sentence; "depth 0.004" is
    not.
    """
    n = sorted_values.size
    if n == 0:
        return 0.5, 0.5
    at_or_below = float(np.searchsorted(sorted_values, value, side="right")) / n
    at_or_above = 1.0 - float(np.searchsorted(sorted_values, value, side="left")) / n
    return min(at_or_below, at_or_above), at_or_below


@dataclass
class ExchangeabilityReference:
    """What the calibration split looked like, and how odd a new response is."""

    sorted_values: Dict[str, np.ndarray]
    sorted_depths: np.ndarray
    threshold: float = DEFAULT_THRESHOLD
    source: str = ""
    measured_false_alarm_rate: Optional[float] = None
    measured_on: str = ""
    # Which features this particular reference was built on. Carried rather than
    # assumed, so an artifact written before the list changed is refused on load
    # instead of being read against a list it never saw.
    features: Tuple[str, ...] = FEATURE_NAMES

    @property
    def n(self) -> int:
        return int(self.sorted_depths.size)

    def depth(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Tuple[float, float]]]:
        """The worst-feature depth, and every feature's (depth, percentile)."""
        per_feature: Dict[str, Tuple[float, float]] = {}
        for name in self.features:
            per_feature[name] = _tail_depth(
                self.sorted_values[name], float(features[name])
            )
        worst = min(value[0] for value in per_feature.values())
        return worst, per_feature

    def p_value(self, features: Dict[str, float]) -> float:
        """P(a calibration response is at least this unusual), (n+1)-corrected."""
        worst, _ = self.depth(features)
        at_or_below = int(np.searchsorted(self.sorted_depths, worst, side="right"))
        return (1.0 + at_or_below) / (self.n + 1.0)

    def check(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Everything the API and the interface need, in one call."""
        worst, per_feature = self.depth(features)
        at_or_below = int(np.searchsorted(self.sorted_depths, worst, side="right"))
        p_value = (1.0 + at_or_below) / (self.n + 1.0)
        in_distribution = p_value >= self.threshold

        most_unusual = min(self.features, key=lambda name: per_feature[name][0])
        return {
            "checked": True,
            "in_distribution": bool(in_distribution),
            "p_value": float(p_value),
            "threshold": float(self.threshold),
            "n_reference": self.n,
            "most_unusual": most_unusual,
            "features": [
                {
                    "name": name,
                    "label": FEATURE_LABELS[name],
                    "value": float(features[name]),
                    "percentile": float(per_feature[name][1]),
                    "unusual": bool(per_feature[name][0] < self.threshold),
                }
                for name in self.features
            ],
        }


    # -- serialisation ----------------------------------------------------
    #
    # JSON, not pickle, for the same reason the calibrator is JSON: this file is
    # loaded by a web server at startup, and unpickling is arbitrary code
    # execution. It also stays readable, which matters when someone asks at a
    # viva what the deployed system is actually comparing against.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "exchangeability_reference",
            "features": list(self.features),
            "threshold": self.threshold,
            "n": self.n,
            "source": self.source,
            "measured_false_alarm_rate": self.measured_false_alarm_rate,
            "measured_on": self.measured_on,
            "sorted_values": {
                name: [float(v) for v in self.sorted_values[name]]
                for name in self.features
            },
            "sorted_depths": [float(v) for v in self.sorted_depths],
            "note": (
                "A one-sided smoke alarm. A small p-value is evidence the input "
                "is not exchangeable with the calibration split, so the coverage "
                "guarantee does not apply to it. A large p-value is NOT evidence "
                "that it is: the check sees a handful of features and shift can "
                "be real and invisible to all of them."
            ),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExchangeabilityReference":
        features = tuple(payload.get("features") or FEATURE_NAMES)
        unknown = [name for name in features if name not in FEATURE_NAMES]
        if unknown:
            raise ValueError(
                f"reference names features this module does not compute "
                f"{unknown}; it must be regenerated"
            )
        missing = [name for name in features if name not in payload["sorted_values"]]
        if missing:
            raise ValueError(
                f"reference is missing values for {missing}; it was written by "
                "an older version of this module and must be regenerated"
            )
        return cls(
            sorted_values={
                name: np.asarray(payload["sorted_values"][name], dtype=np.float64)
                for name in features
            },
            sorted_depths=np.asarray(payload["sorted_depths"], dtype=np.float64),
            threshold=float(payload.get("threshold", DEFAULT_THRESHOLD)),
            source=str(payload.get("source", "")),
            measured_false_alarm_rate=payload.get("measured_false_alarm_rate"),
            measured_on=str(payload.get("measured_on", "")),
            features=features,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ExchangeabilityReference":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def build(
        cls,
        records: Sequence[Dict[str, Any]],
        threshold: float = DEFAULT_THRESHOLD,
        source: str = "",
        feature_subset: Sequence[str] = FEATURE_NAMES,
    ) -> "ExchangeabilityReference":
        """Fit the reference on a calibration dump.

        The depths of the calibration responses themselves are computed against
        an empirical distribution that contains them, while a new response is
        not in it. That asymmetry is what makes the p-value approximate rather
        than exact, and it is why `validate` exists.

        `feature_subset` exists so the feature list itself can be investigated
        rather than assumed. It is a research handle, not a tuning dial: the
        served reference uses the full list, and any narrowing has to be
        justified by a measurement and written down.
        """
        if not records:
            raise ValueError("cannot build a reference from an empty dump")
        features = tuple(feature_subset)
        unknown = [name for name in features if name not in FEATURE_NAMES]
        if unknown:
            raise ValueError(f"unknown features {unknown}")
        sorted_values = feature_matrix(records, features)
        reference = cls(
            sorted_values=sorted_values,
            sorted_depths=np.zeros(0),
            threshold=threshold,
            source=source,
            features=features,
        )
        depths = np.asarray(
            [
                reference.depth(features_from_record(record))[0]
                for record in records
            ],
            dtype=np.float64,
        )
        reference.sorted_depths = np.sort(depths)
        return reference

    def validate(self, records: Sequence[Dict[str, Any]], label: str = "") -> Dict[str, Any]:
        """Measure the real false-alarm rate on data the reference never saw.

        The threshold claims to be a false-alarm rate. This is the only thing
        that makes that claim checkable, and a run whose measured rate is far
        from its threshold is a run whose threshold means nothing.
        """
        p_values = np.asarray(
            [self.p_value(features_from_record(record)) for record in records],
            dtype=np.float64,
        )
        rate = float((p_values < self.threshold).mean()) if p_values.size else float("nan")
        self.measured_false_alarm_rate = rate
        self.measured_on = label

        # Label-conditional rates, because of something this check turned out to
        # do that it was not designed to.
        #
        # A response that really does contain hallucinations has more spans,
        # longer spans and higher scores, so it sits further into the tail of
        # the calibration distribution than a clean one. The alarm therefore
        # fires on hallucinated responses roughly twice as often as on clean
        # ones. That is not a defect in the statistic; it is a fact about the
        # data, and it has a consequence worth stating plainly in the report:
        # **the coverage guarantee is least well-supported on exactly the
        # responses that matter most.**
        #
        # Dropping the score-derived features was tried as a fix and made it
        # worse -- span geometry is only defined when spans exist, so it is just
        # as label-correlated. Measured on the test split: 66.7% of alarms are
        # gold-positive with all eight features, 78.9% with span shape alone,
        # 85.0% with span shape plus count. The full list is both the least
        # biased and the most sensitive, so it is what ships.
        positive = np.asarray(
            [bool(record.get("gold_spans")) for record in records], dtype=bool
        )
        trips = p_values < self.threshold
        bias: Dict[str, Any] = {}
        if positive.any() and (~positive).any():
            bias = {
                "trip_rate_gold_positive": float(trips[positive].mean()),
                "trip_rate_gold_negative": float(trips[~positive].mean()),
                "share_of_alarms_gold_positive": (
                    float(positive[trips].mean()) if trips.any() else None
                ),
                "gold_positive_base_rate": float(positive.mean()),
            }

        return {
            "n": int(p_values.size),
            "threshold": self.threshold,
            "measured_false_alarm_rate": rate,
            "ratio_to_threshold": rate / self.threshold if self.threshold else float("nan"),
            "p_value_quantiles": {
                "p01": float(np.percentile(p_values, 1)) if p_values.size else None,
                "p05": float(np.percentile(p_values, 5)) if p_values.size else None,
                "p50": float(np.percentile(p_values, 50)) if p_values.size else None,
            },
            "label_conditional": bias,
        }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def format_report(
    reference: ExchangeabilityReference,
    validation: Dict[str, Any],
    shifted: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "EXCHANGEABILITY REFERENCE",
        f"  built on {reference.n:,} calibration responses from {reference.source}",
        f"  threshold {reference.threshold} -- the false-alarm rate it claims",
        "",
        "  feature                        calibration median   p01        p99",
        "  " + "-" * 68,
    ]
    for name in reference.features:
        values = reference.sorted_values[name]
        lines.append(
            f"  {name:<30} {np.median(values):<20.4f} "
            f"{np.percentile(values, 1):<10.4f} {np.percentile(values, 99):.4f}"
        )
    lines += [
        "",
        f"VALIDATION on {validation['n']:,} held-out responses ({reference.measured_on})",
        f"  claimed false-alarm rate  {validation['threshold']:.4f}",
        f"  measured                  {validation['measured_false_alarm_rate']:.4f}"
        f"   ({validation['ratio_to_threshold']:.2f}x the claim)",
        "  A measured rate far above the claim means the threshold is lying and",
        "  the alarm will cry wolf on real inputs. Far below means it is deaf.",
    ]
    bias = validation.get("label_conditional") or {}
    if bias:
        lines += [
            "",
            "  who the alarm fires on:",
            f"    responses that DO contain hallucinations   "
            f"{bias['trip_rate_gold_positive']:.4f}",
            f"    responses that do not                      "
            f"{bias['trip_rate_gold_negative']:.4f}",
            f"    share of alarms that are hallucinated      "
            f"{bias['share_of_alarms_gold_positive']:.3f}"
            f"   (base rate {bias['gold_positive_base_rate']:.3f})",
            "  A hallucinated response has more spans, longer spans and higher",
            "  scores, so it sits further into the tail. The guarantee is",
            "  therefore least well-supported on the responses that matter most.",
            "  Report this. Do not drop features to hide it -- that was tried and",
            "  it made the skew worse, not better.",
        ]
    if shifted:
        lines += [
            "",
            f"ON THE SHIFTED CORPUS ({shifted['label']}, {shifted['n']:,} responses)",
            f"  tripped on {shifted['trip_rate']:.4f} of them",
            "  This corpus is the one Block C measured the guarantee breaking on,",
            "  so a rate far above the false-alarm rate is the alarm working.",
        ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.c2_calibration.run_c2 import read_probability_file

    parser = argparse.ArgumentParser(
        description="Build the serving-time exchangeability reference."
    )
    parser.add_argument("--calib", default="results/c1/calib/probabilities.jsonl")
    parser.add_argument("--test", default="results/c1/test/probabilities.jsonl")
    parser.add_argument(
        "--shifted",
        default="results/ood/ragbench-probs/probabilities.jsonl",
        help=(
            "optional out-of-domain dump. Not used to build anything -- only to "
            "report how often the alarm trips on data the guarantee is already "
            "known to fail on"
        ),
    )
    parser.add_argument("--out", default="results/c2/c1/c2_ood_reference.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    calib = read_probability_file(args.calib)
    test = read_probability_file(args.test)
    print(f"calibration {len(calib):,} responses | held-out {len(test):,} responses")

    reference = ExchangeabilityReference.build(
        calib, threshold=args.threshold, source=args.calib
    )
    validation = reference.validate(test, label=args.test)

    shifted_report = None
    shifted_path = Path(args.shifted)
    if shifted_path.exists():
        shifted_records = read_probability_file(shifted_path)
        p_values = np.asarray(
            [reference.p_value(features_from_record(r)) for r in shifted_records]
        )
        shifted_report = {
            "label": args.shifted,
            "n": int(p_values.size),
            "trip_rate": float((p_values < reference.threshold).mean()),
        }
    else:
        print(f"note: {args.shifted} not found; skipping the shifted-corpus report")

    print()
    print(format_report(reference, validation, shifted_report))

    payload = reference.to_dict()
    payload["validation"] = validation
    if shifted_report:
        payload["shifted_corpus_trip_rate"] = shifted_report

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
