"""The C4 driver. One command regenerates every artefact, in dependency order.

    python -m src.c4_attribution.run_c4

Layers that need the pair-score dump are skipped with a stated reason when it is
absent, rather than being silently dropped or filled with a plausible number.
Build it first with `python -m src.c4_attribution.run_scorer calib test`.

The order below is the build order, and each step is independently reportable:
the audit, the protocol, the scored baseline and the abstention result exist
before any model runs, so a component exists even if everything after Layer 3
fails.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


from src.c4_attribution import evidence_audit, matching, run_lexical, scorer, serving
from src.c4_attribution.attribution_eval import (
    Prediction,
    aggregate,
    by_group,
    floor_rows,
    prediction_from_scores,
    score_rows,
)
from src.c4_attribution.cost import curve as cost_curve, headline as cost_headline
from src.c4_attribution.dataset import (
    SpanRecord,
    load_responses,
    predicted_span_records,
    read_jsonl,
    span_records,
    spurious_rate,
)
from src.c4_attribution.evidence_conformal import (
    ConformalRow,
    compare_formalisms,
    coverage_table,
    evaluate as conformal_evaluate,
    normalise_scores,
    per_task,
    risk_control_cross_check,
    set_shape,
)
from src.c4_attribution.faithfulness import check as faithfulness_check
from src.c4_attribution import fusion as fusion_mod
from src.c4_attribution.matching import Match, match_all
from src.c4_attribution.no_evidence import (
    choose_threshold as choose_abstention,
    evaluate as evaluate_abstention,
    floors as abstention_floors,
    gold_abstain,
    in_scope,
)
from src.c4_attribution.score_calibration import evaluate as calibration_evaluate
from src.c4_attribution.transfer import check_ragbench, fold

RESULTS = Path("results/c4")
SPLITS = ("calib", "test")
ALPHA = 0.10


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"  wrote {path}")
    return path


def _key(row: SpanRecord) -> str:
    return f"{row.response_id}:{row.label.start}:{row.label.end}"


def _predictions(
    rows: Sequence[SpanRecord],
    dump: Dict[str, Dict[str, Any]],
    backend: str,
    hypothesis: str = "sentence",
) -> List[Prediction]:
    """One backend's Prediction per row, zero-filled where the dump has no record."""
    out: List[Prediction] = []
    for row in rows:
        record = dump.get(_key(row))
        if record is None:
            out.append(Prediction((), None, 0.0))
            continue
        out.append(
            prediction_from_scores(scorer.backend_scores(record, backend, hypothesis))
        )
    return out


def _conformal_rows(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    dump: Dict[str, Dict[str, Any]],
    backend: str,
    hypothesis: str = "sentence",
) -> List[ConformalRow]:
    out: List[ConformalRow] = []
    for row, gold in zip(rows, matches):
        record = dump.get(_key(row))
        scores = (
            scorer.backend_scores(record, backend, hypothesis)
            if record is not None
            else [0.0] * len(row.sentences)
        )
        out.append(
            ConformalRow(
                response_id=row.response_id,
                task=row.task,
                probabilities=normalise_scores(scores),
                gold_index=gold.convention_a,
            )
        )
    return out


def _select_backend(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    dump: Dict[str, Dict[str, Any]],
) -> Tuple[str, str, Dict[str, Any]]:
    """Pick the operating backend and hypothesis ON CALIBRATION ONLY.

    The test split is scored with every backend afterwards so the ablation table
    is complete, but the choice is made here, on data the report is not scored
    on. A number chosen after seeing test is a knob, not a result.

    **The selection is made on Convention B, not Convention A.** Convention A
    resolves gold by taking the argmax of content-token containment, and the
    lexical backend also takes an argmax of token overlap, so selecting on A
    would be handing the lexical baseline a structural advantage in the very
    comparison meant to decide whether the neural model earns its keep.
    Convention B's gold is exact string containment of the annotator's quote and
    carries no such coupling. Both columns are recorded either way.
    """
    table: Dict[str, Any] = {}
    best = ("lexical", "sentence", -1.0)
    for backend in scorer.BACKENDS:
        for hypothesis in scorer.hypotheses_for(backend):
            predictions = _predictions(rows, dump, backend, hypothesis)
            block_a = aggregate(
                score_rows(rows, matches, predictions, "A"), with_intervals=False
            )
            block_b = aggregate(
                score_rows(rows, matches, predictions, "B"), with_intervals=False
            )
            table[f"{backend}|{hypothesis}"] = {
                "convention_a": {
                    key: block_a.get(key) for key in ("n", "top1_accuracy", "mrr", "recall_at_5")
                },
                "convention_b": {
                    key: block_b.get(key) for key in ("n", "top1_accuracy", "mrr", "recall_at_5")
                },
            }
            value = block_b.get("top1_accuracy") or 0.0
            if value > best[2]:
                best = (backend, hypothesis, value)
    return best[0], best[1], table


def main(argv: Optional[Sequence[str]] = None) -> int:
    started = time.perf_counter()
    print("C4 -- retrieval attribution\n")

    print("Layer 1  evidence-label archaeology")
    evidence_audit.main([])

    print("\nLayer 2  the attribution protocol")
    matching.main([])

    print("\nLayer 3  the lexical baseline and its four floors")
    run_lexical.main([])

    all_responses = span_records(load_responses())
    rows = {s: [r for r in all_responses if r.response.split == s] for s in SPLITS}
    matches = {s: match_all(rows[s]) for s in SPLITS}

    dumps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for split in SPLITS:
        records = scorer.load_pair_scores(split)
        dumps[split] = {r["key"]: r for r in records}
        print(f"\npair-score dump [{split}]: {len(records)} records")

    have_scores = all(dumps[s] for s in SPLITS)
    if not have_scores:
        print(
            "\nNo pair-score dump on disk, so Layers 4 to 11 are skipped. Build "
            "it with: python -m src.c4_attribution.run_scorer calib test"
        )
        _write(
            RESULTS / "run_status.json",
            {
                "layers_1_to_3": "complete",
                "layers_4_to_11": "skipped",
                "reason": "results/c4/scores/pair_scores_*.jsonl not found",
            },
        )
        return 0

    print("\nLayer 4  the neural scorers, the pre-filter ceiling and the ablations")
    backend, hypothesis, calib_table = _select_backend(
        rows["calib"], matches["calib"], dumps["calib"]
    )
    print(f"  chosen on calibration: backend={backend} hypothesis={hypothesis}")

    # The cross-encoders only ever see the pre-filter's top k, so their ceiling
    # is its recall. Comparing them with an unfiltered scorer without saying so
    # would be comparing a model against a pipeline. The conditional slice --
    # the spans whose gold sentence survived the pre-filter -- removes that
    # confound, and both columns are reported.
    survived = [
        gold.convention_a is not None
        and gold.convention_a in (dumps["test"].get(_key(row), {}).get("topk") or [])
        for row, gold in zip(rows["test"], matches["test"])
    ]
    survivor_rows = [r for r, keep in zip(rows["test"], survived) if keep]
    survivor_matches = [m for m, keep in zip(matches["test"], survived) if keep]

    test_table: Dict[str, Any] = {}
    for name in scorer.BACKENDS:
        for kind in scorer.hypotheses_for(name):
            predictions = _predictions(rows["test"], dumps["test"], name, kind)
            scored = score_rows(rows["test"], matches["test"], predictions)
            block = aggregate(scored, with_intervals=False)
            block["by_task"] = by_group(scored, "task")
            conditional = aggregate(
                score_rows(
                    survivor_rows,
                    survivor_matches,
                    [p for p, keep in zip(predictions, survived) if keep],
                ),
                with_intervals=False,
            )
            block["top1_given_gold_survived_prefilter"] = conditional.get("top1_accuracy")
            block["n_given_gold_survived_prefilter"] = conditional.get("n")
            block["sees_only_prefiltered_sentences"] = name not in ("lexical", "embed")
            test_table[f"{name}|{kind}"] = block

    chosen_predictions = {
        split: _predictions(rows[split], dumps[split], backend, hypothesis)
        for split in SPLITS
    }

    _write(
        RESULTS / "scores" / "scorer_metrics.json",
        {
            "models": {
                "prefilter": scorer.EMBED_MODEL,
                "nli": scorer.NLI_MODEL,
                "hhem": scorer.HHEM_MODEL,
            },
            "licences_checked_2026_08_24": {
                scorer.EMBED_MODEL: "Apache-2.0",
                scorer.NLI_MODEL: "MIT",
                scorer.HHEM_MODEL: "Apache-2.0",
                "bespokelabs/Bespoke-MiniCheck-7B": (
                    "CC BY-NC 4.0, non-commercial -- rejected on licence "
                    "grounds, which is the evidence the check was made"
                ),
            },
            "prefilter": scorer.prefilter_report(list(dumps["test"].values())),
            "selected_on_calibration": {
                "backend": backend,
                "hypothesis": hypothesis,
                "selected_by": (
                    "Convention B top-1 on the calibration split. B's gold is "
                    "exact string containment and is not coupled to a lexical "
                    "argmax the way A's is, so it does not flatter the lexical "
                    "baseline in the comparison meant to judge it."
                ),
                "table": calib_table,
            },
            "test_table": test_table,
            "floors_test": floor_rows(rows["test"], matches["test"], with_intervals=False),
            "direction_note": (
                "nli_entail is P(entailment); nli_signed is "
                "max(P_entail, P_contradict). For a hallucinated span the "
                "sentence a human would cite is often the one it contradicts, "
                "so entailment alone is the wrong score and both are reported."
            ),
        },
    )

    print("\nLayer 5  is the attribution score a probability?")
    calibration: Dict[str, Any] = {}
    for name in ("lexical", "embed", backend):
        calib_predictions = _predictions(rows["calib"], dumps["calib"], name, hypothesis)
        test_predictions = _predictions(rows["test"], dumps["test"], name, hypothesis)
        calib_scored = score_rows(rows["calib"], matches["calib"], calib_predictions)
        test_scored = score_rows(rows["test"], matches["test"], test_predictions)
        calibration[name] = calibration_evaluate(
            [r["score"] for r in calib_scored],
            [int(r["correct"]) for r in calib_scored],
            [r["score"] for r in test_scored],
            [int(r["correct"]) for r in test_scored],
        )
    _write(RESULTS / "calibration" / "score_calibration.json", calibration)

    print("\nLayer 6  conformal evidence sets")
    calib_conformal = _conformal_rows(
        rows["calib"], matches["calib"], dumps["calib"], backend, hypothesis
    )
    test_conformal = _conformal_rows(
        rows["test"], matches["test"], dumps["test"], backend, hypothesis
    )
    table = coverage_table(calib_conformal, test_conformal)
    at_alpha = next(r for r in table if abs(r["alpha"] - ALPHA) < 1e-9)
    crc = risk_control_cross_check(calib_conformal, test_conformal, ALPHA)
    conformal_payload = {
        "backend": backend,
        "hypothesis": hypothesis,
        "coverage_table": table,
        "per_task": per_task(calib_conformal, test_conformal, ALPHA),
        "risk_control": crc,
        "formalism_comparison": compare_formalisms(at_alpha, crc),
        "set_shape": set_shape(calib_conformal, test_conformal, ALPHA),
        "c3_comparison": {
            "c3_mean_set_size": 1.9538,
            "c3_singleton_rate": 0.2643,
            "c3_n_classes": 4,
            "c4_mean_pool_size": at_alpha.get("mean_pool_size"),
            "note": (
                "C3 hedged over four error types. C4 hedges over the sentences "
                "of one context, a median of sixteen on these splits. The "
                "comparison is the point: the same machinery, a much larger "
                "label space."
            ),
        },
    }
    _write(RESULTS / "conformal" / "c4_conformal.json", conformal_payload)

    print("\nLayer 7  'no evidence exists' as a measured claim")
    keep = in_scope(matches["calib"]).astype(bool)
    calib_confidence = [
        p.score for p, k in zip(chosen_predictions["calib"], keep) if k
    ]
    calib_gold = gold_abstain(matches["calib"])[keep]
    chosen = choose_abstention(calib_confidence, calib_gold)
    threshold = float(chosen["threshold"]) if chosen.get("threshold") is not None else 0.0
    abstention = evaluate_abstention(
        rows["test"], matches["test"], [p.score for p in chosen_predictions["test"]], threshold
    )
    empty_sets = conformal_evaluate(calib_conformal, test_conformal, ALPHA, False)
    _write(
        RESULTS / "analysis" / "no_evidence.json",
        {
            "backend": backend,
            "threshold_fitted_on": "calib",
            "threshold": threshold,
            "threshold_selection": {
                k: v for k, v in chosen.items() if k != "curve"
            },
            "test": abstention,
            "floors": abstention_floors(rows["test"], matches["test"]),
            "empty_conformal_set_rate": empty_sets.get("empty_set_rate"),
            "note": (
                "A thresholded abstention and an empty conformal set are two "
                "different routes to the same answer and they are different "
                "objects. Both rates are printed so a reader can see whether "
                "they agree."
            ),
        },
    )

    print("\nLayer 8  cross-task transfer")
    folds = []
    for held_out in ("qa", "data2text", "summarization"):
        out_index = [i for i, r in enumerate(rows["calib"]) if r.task != held_out]
        in_index = [i for i, r in enumerate(rows["calib"]) if r.task == held_out]
        test_index = [i for i, r in enumerate(rows["test"]) if r.task == held_out]
        if not out_index or not test_index:
            folds.append({"held_out_task": held_out, "n_test_spans": 0, "skipped": True})
            continue

        def _fold(calib_index: Sequence[int], name: str) -> Dict[str, Any]:
            return fold(
                [rows["calib"][i] for i in calib_index],
                [matches["calib"][i] for i in calib_index],
                [calib_conformal[i].probabilities for i in calib_index],
                [chosen_predictions["calib"][i] for i in calib_index],
                [rows["test"][i] for i in test_index],
                [matches["test"][i] for i in test_index],
                [test_conformal[i].probabilities for i in test_index],
                [chosen_predictions["test"][i] for i in test_index],
                name,
                ALPHA,
            )

        transferred = _fold(out_index, held_out)
        transferred["in_task"] = (
            {
                key: _fold(in_index, held_out)[key]
                for key in ("conformal", "abstention", "n_calibration_spans")
            }
            if in_index
            else {"note": "no calibration spans for this task"}
        )
        folds.append(transferred)
    _write(
        RESULTS / "transfer" / "transfer_metrics.json",
        {
            "ragbench": check_ragbench(),
            "folds": folds,
            "in_task_reference": by_group(
                score_rows(rows["test"], matches["test"], chosen_predictions["test"]),
                "task",
            ),
            "reading_note": (
                f"The selected attribution backend is `{backend}`, which has no "
                "parameter fitted on a calibration split -- it scores a span "
                "against the sentences of its own context and nothing else. Its "
                "top-1 accuracy therefore CANNOT degrade under a task shift, and "
                "a zero drop in that column is a property of the scorer rather "
                "than evidence of robustness. What genuinely transfers, or fails "
                "to, is the conformal quantile and the abstention threshold, and "
                "both are reported against their in-task counterparts."
            ),
        },
    )

    print("\nLayer 9  attribution faithfulness")
    _write(
        RESULTS / "analysis" / "faithfulness.json",
        {
            "selected": {
                "backend": backend,
                "hypothesis": hypothesis,
                **faithfulness_check(
                    rows["test"], matches["test"], chosen_predictions["test"]
                ),
            },
            "lexical": faithfulness_check(
                rows["test"],
                matches["test"],
                _predictions(rows["test"], dumps["test"], "lexical"),
            ),
        },
    )

    print("\nLayer 10  the review-cost curve")
    curve = cost_curve(calib_conformal, test_conformal)
    _write(
        RESULTS / "analysis" / "review_cost.json",
        {"backend": backend, "curve": curve, "headline": cost_headline(curve, ALPHA)},
    )

    print("\nLayer 11  the fusion table")
    _write(RESULTS / "fusion" / "fusion_metrics.json", _fusion(threshold, backend))

    print("\nLayer 12  the serving artefact")
    lac = {
        str(row["alpha"]): row["threshold"]
        for row in coverage_table(
            _conformal_rows(rows["calib"], matches["calib"], dumps["calib"], "lexical"),
            _conformal_rows(rows["test"], matches["test"], dumps["test"], "lexical"),
        )
    }
    lexical_calib = _predictions(rows["calib"], dumps["calib"], "lexical")
    lexical_keep = in_scope(matches["calib"]).astype(bool)
    lexical_choice = choose_abstention(
        [p.score for p, k in zip(lexical_calib, lexical_keep) if k],
        gold_abstain(matches["calib"])[lexical_keep],
    )
    serving.write_artifact(
        serving.build_artifact(
            backend="lexical",
            abstention_threshold=lexical_choice.get("threshold"),
            lac_thresholds=lac,
            offline_backend=f"{backend}|{hypothesis}",
            notes=[
                "The shipped attributor is lexical: the serving process already "
                "holds ModernBERT for C1, and a second cross-encoder per request "
                "would cost hundreds of milliseconds and hundreds of megabytes "
                "on a 6 GB laptop for an addition to the product rather than the "
                "product itself.",
                "The abstention threshold and the LAC thresholds here are the "
                "LEXICAL ones, so the served path is internally consistent. The "
                "neural backend's numbers live in results/c4/scores and "
                "results/c4/conformal and are not mixed with these.",
            ],
        )
    )

    print(f"\ndone in {time.perf_counter() - started:.1f}s")
    return 0


def _fusion(gold_span_threshold: float, backend: str) -> Dict[str, Any]:
    """Layer 11, in its own function because it loads a different set of inputs.

    One discipline point that is easy to get wrong: the abstention threshold used
    here is fitted on the **calibration split's predicted spans**, not on the
    gold-span threshold Layer 7 chose. A predicted span is a boundary-shifted
    fragment of a gold span roughly half the time and is spurious the other half,
    so its score distribution is not the gold one, and carrying the gold
    threshold across would be applying a number fitted on a different
    population. The gold-span threshold is still recorded for comparison.
    """
    artifact = json.loads(fusion_mod.ARTIFACT.read_text(encoding="utf-8"))
    calib_records = list(read_jsonl(fusion_mod.CALIB_DUMP))
    test_records = list(read_jsonl(fusion_mod.TEST_DUMP))
    rebuild = fusion_mod.rebuild_c2_decisions(artifact, calib_records, test_records)
    if not rebuild["matches_recorded"]:
        raise SystemExit(
            "the rebuilt C2 coverage does not match the artefact "
            f"({rebuild['rebuilt_coverage']} against {rebuild['recorded_coverage']}). "
            "One of the frozen inputs has moved. Refusing to report a fusion number."
        )

    from src.c4_attribution.lexical import idf_overlap_scores

    responses = load_responses(splits=None, with_labels_only=False)
    predicted = {
        split: predicted_span_records(path, responses)
        for split, path in (("calib", fusion_mod.CALIB_DUMP), ("test", fusion_mod.TEST_DUMP))
    }

    def confidence(rows: Sequence[SpanRecord]) -> List[float]:
        out = []
        for row in rows:
            scores = idf_overlap_scores(row.span_text, row.sentences)
            out.append(max(scores) if scores else 0.0)
        return out

    calib_predicted_matches = match_all(predicted["calib"])
    calib_keep = in_scope(calib_predicted_matches).astype(bool)
    calib_confidence = [
        c for c, k in zip(confidence(predicted["calib"]), calib_keep) if k
    ]
    calib_gold = gold_abstain(calib_predicted_matches)[calib_keep]
    chosen = (
        choose_abstention(calib_confidence, calib_gold)
        if len(calib_confidence)
        else {"threshold": None}
    )
    threshold = (
        float(chosen["threshold"]) if chosen.get("threshold") is not None else 0.0
    )

    test_confidence = confidence(predicted["test"])
    test_flag = {
        (row.response_id, row.label.start, row.label.end): value < threshold
        for row, value in zip(predicted["test"], test_confidence)
    }

    gold = fusion_mod.response_labels(test_records)
    ids = list(gold)

    rules = {
        name: fusion_mod.example_f1(
            [gold[i] for i in ids],
            [fusion_mod.fuse(test_records, rebuild["rows"], test_flag, name)[i] for i in ids],
        )
        for name in fusion_mod.RULES
    }
    c1_scores = rules["c1_alone"]
    fused_scores = rules["c4_tie_break"]

    cells = fusion_mod.disagreement_cells(
        [1 if row["decision"] == "flag" else 0 for row in rebuild["rows"]],
        [
            0 if test_flag.get((row["response_id"], row["start"], row["end"]), False) else 1
            for row in rebuild["rows"]
        ],
    )

    spurious = spurious_rate(test_records)
    predicted_matches = match_all(predicted["test"])
    gold_vs_predicted = {
        "n_predicted_spans": spurious["n_predicted_spans"],
        "n_spurious": spurious["n_spurious"],
        "spurious_rate": spurious["spurious_rate"],
        "n_predicted_with_evidence_label": sum(
            1 for r in predicted["test"] if r.label.has_original
        ),
        "n_predicted_scorable": sum(1 for m in predicted_matches if m.scorable),
        "top1_on_predicted_spans": aggregate(
            score_rows(
                predicted["test"],
                predicted_matches,
                [
                    prediction_from_scores(idf_overlap_scores(r.span_text, r.sentences))
                    for r in predicted["test"]
                ],
            ),
            with_intervals=False,
        ).get("top1_accuracy"),
        "note": (
            "A spurious predicted span touches no annotation, so it has no gold "
            "type and no gold evidence. It is excluded from attribution scoring "
            "and counted here rather than in a footnote. The top-1 row is the "
            "gold-spans-versus-predicted-spans ablation: what attribution is "
            "worth in the system that actually ships, rather than when the "
            "localisation is already right."
        ),
    }

    return {
        "c2_rebuild": {
            "rebuilt_coverage": rebuild["rebuilt_coverage"],
            "recorded_coverage": rebuild["recorded_coverage"],
            "matches_recorded": rebuild["matches_recorded"],
        },
        "attributor": "lexical",
        "abstention_threshold": threshold,
        "threshold_fitted_on": "calibration split, predicted spans",
        "gold_span_threshold_for_comparison": gold_span_threshold,
        "rules": rules,
        "c1_alone": c1_scores,
        "fused": fused_scores,
        "headline_rule": "c4_tie_break",
        "c4_contribution": {
            "against": "c1_c2_flag_only",
            "delta_f1_points": (
                rules["c4_tie_break"]["f1"] - rules["c1_c2_flag_only"]["f1"]
            ) * 100.0,
            "note": (
                "This is the isolated contribution: the two rules differ only in "
                "whether an ABSTAIN span is promoted when C4 finds no supporting "
                "evidence for it. C2 abstains rather than decides on a large "
                "share of spans, and an abstention is a span nobody has decided."
            ),
        },
        "comparison": fusion_mod.compare_to_floors(fused_scores["f1"]),
        "c1_alone_reproduces_reported": abs(c1_scores["f1"] - 0.7623) < 5e-4,
        "disagreement_cells": cells,
        "gold_vs_predicted_spans": gold_vs_predicted,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
