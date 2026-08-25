"""Tests for the C3 table and figure generators.

The generators exist so that no number is ever typed into the report by hand, so
the property worth testing is the failure mode: a missing artefact must leave a
visible TODO, never a silently shorter table.
"""

from __future__ import annotations

import json

import pytest

from src.c3_explanation import figures as F
from src.c3_explanation import tables as T

CLASSES = [
    "evident_conflict",
    "subtle_conflict",
    "evident_baseless_info",
    "subtle_baseless_info",
]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def artefacts(tmp_path):
    root = tmp_path / "c3"
    write(
        root / "analysis" / "label_audit.json",
        {
            "n_spans": 100,
            "l1_1_label_coverage": {
                "assert_label_coverage_passed": True,
                "n_distinct": 2,
                "distinct_raw_label_types": {"Evident Conflict": 60, "Subtle Conflict": 40},
            },
            "l1_2_meta_agreement": {
                "prefix_mapped_rate": 0.9,
                "mechanism": {"agreement": 0.99, "n_agree": 89, "n_comparable": 90},
                "severity_by_annotator_word": {
                    "table": {"high": {"evident": 40, "subtle": 1}, "evident": {"evident": 20}},
                    "independent_words": ["high", "low"],
                    "independent": {"agreement": 0.98, "n_agree": 40, "n_comparable": 41},
                },
                "structure": {"counts": {"original": 63}, "rates": {"original": 0.63}},
            },
            "l1_3_census": {
                "implicit_true": {"n": 13, "rate": 0.13},
                "due_to_null": {"n": 11, "rate": 0.11},
            },
            "l1_4_floors": {
                "base_rates": {
                    "train": {"error_type": {"n": 80}},
                    "test": {"error_type": {"n": 20}},
                },
                "majority_class_floor": {
                    "train": {
                        "error_type": {"accuracy": 0.4, "macro_f1": 0.15},
                        "severity": {"accuracy": 0.8, "macro_f1": 0.44},
                    },
                    "test": {
                        "error_type": {"accuracy": 0.5, "macro_f1": 0.16},
                        "severity": {"accuracy": 0.88, "macro_f1": 0.47},
                    },
                },
            },
        },
    )
    write(
        root / "analysis" / "typed_matching_test.json",
        {
            "n_gold_spans": 1517,
            "n_predicted_spans": 2390,
            "predicted_by_bucket": {"exact": 290, "spurious": 1134},
            "typeable": {
                "n": 1256,
                "rate_of_predicted": 0.5255,
                "n_excluded_spurious": 1134,
                "excluded_rate": 0.4745,
            },
            "footnote_6": {
                "n_multi_gold_components": 19,
                "n_multi_gold_crossing": 6,
                "crossing_rate_of_multi_gold": 0.3158,
                "n_typeable_predictions_affected": 6,
                "affected_rate_of_typeable": 0.0048,
            },
            "gold_multi_type": {"n": 4, "rate_of_gold": 0.0026},
        },
    )
    write(
        root / "type" / "baseline_metrics.json",
        {
            "evaluations": {
                "predicted_test": {
                    "n": 1256,
                    "n_responses": 658,
                    "mechanism": {
                        "labels": ["baseless", "conflict"],
                        "model": {"accuracy": 0.75, "macro_f1": 0.73, "confusion": {}},
                        "floor": {"accuracy": 0.61, "macro_f1": 0.38, "predicted_class": "baseless"},
                        "intervals": {"macro_f1": {"ci_low": 0.70, "ci_high": 0.75}},
                    },
                    "severity": {
                        "labels": ["evident", "subtle"],
                        "model": {
                            "accuracy": 0.88,
                            "macro_f1": 0.52,
                            "confusion": {
                                "evident": {"evident": 1100, "subtle": 33},
                                "subtle": {"evident": 114, "subtle": 9},
                            },
                        },
                        "floor": {"accuracy": 0.90, "macro_f1": 0.47, "predicted_class": "evident"},
                        "intervals": {"macro_f1": {"ci_low": 0.49, "ci_high": 0.56}},
                    },
                }
            }
        },
    )
    write(
        root / "type" / "c3_conformal.json",
        {
            "classes": CLASSES,
            "headline_alpha": 0.1,
            "calibration": [
                {"method": "identity", "ece": 0.09, "mce": 0.67, "brier": 0.5,
                 "nll": 0.84, "accuracy": 0.64},
                {"method": "constant base rate", "ece": 0.05, "mce": 0.05, "brier": 0.58,
                 "nll": 0.99, "accuracy": 0.52, "is_floor": True},
            ],
            "coverage": [
                {
                    "alpha": 0.1,
                    "target_coverage": 0.9,
                    "empirical_coverage": 0.9029,
                    "band": 0.0388,
                    "inside_band": True,
                    "average_set_size": 1.95,
                    "singleton_rate": 0.26,
                    "empty_set_rate": 0.0,
                    "threshold": 0.72,
                    "set_shapes": {"singleton": 332, "pair_severity_undecided": 200},
                    "set_shape_rates": {"singleton": 0.62, "pair_severity_undecided": 0.38},
                    "coverage_interval": {"ci_low": 0.885, "ci_high": 0.922},
                }
            ],
            "per_class_coverage": {
                name: {
                    "n_test": 10,
                    "n_calibration": 0 if name == "subtle_conflict" else 10,
                    "empirical_coverage": 0.9,
                    "average_set_size": 2.0,
                    "measurable": name != "subtle_conflict",
                }
                for name in CLASSES
            },
            "ood_note": "RAGBench carries no error-type annotation.",
        },
    )
    write(
        root / "explanation" / "explanation_metrics.json",
        {
            "systems": {
                "constant_floor": {
                    "n_spans": 1517,
                    "snippet": {"mean_containment": 0.0, "rate_at_threshold": {"1.0": 0.0}},
                    "self_faithfulness": {"verbatim_rate": float("nan")},
                    "rouge_l_vs_meta": {"mean_f1": 0.085, "n": 1512, "caption": "single reference"},
                    "by_task": {},
                },
                "template_gold_type": {
                    "n_spans": 1517,
                    "snippet": {"mean_containment": 0.59, "rate_at_threshold": {"1.0": 0.49}},
                    "self_faithfulness": {"verbatim_rate": 1.0},
                    "rouge_l_vs_meta": {"mean_f1": 0.257, "n": 1512, "caption": "single reference"},
                    "by_task": {"qa": {"n": 231, "n_with_reference": 129, "mean_containment": 0.18}},
                },
            },
            "llm_comparison": {"n": 0, "status": "not run"},
            "snippet_metric_note": "chosen by grounding, not by type",
        },
    )
    write(
        root / "analysis" / "addons.json",
        {
            "layer_6_escalation_budget": {
                "rule": "abstain and set size >= 2",
                "n_spans": 2390,
                "n_abstained": 1148,
                "abstention_rate": 0.4803,
                "per_alpha": {"0.1": {"n_escalated": 894, "share_of_all_spans": 0.374,
                                      "share_of_abstained": 0.779}},
                "loop_status": "not run",
            },
            "addon_1_type_conditional_calibration": {
                "overall": {"n": 1256, "ece": 0.0649, "mean_confidence": 0.62, "accuracy": 0.64},
                "by_error_type": {
                    name: {"n": 100, "ece": 0.2, "mean_confidence": 0.6, "accuracy": 0.5}
                    for name in CLASSES
                },
            },
            "addon_2_severity_weighted_risk_control": {
                "loss": "weighted share missed",
                "weights": {"evident": 1.0, "subtle": 0.5, "implicit_true": 0.0},
                "weights_note": "a stated value judgement",
                "rows": [
                    {"alpha": 0.1, "feasible": False, "reason": "no threshold"},
                    {"alpha": 0.4, "feasible": True, "threshold": 0.44, "test_risk": 0.39,
                     "bound_held": True, "span_flag_rate": 0.65, "n_spans_flagged": 1555},
                ],
            },
            "addon_3_generator_by_error_type": {
                "by_model": {
                    "gpt-4-0613": {"n_spans": 482, "spans_per_response": 0.163,
                                   "rates": {name: 0.25 for name in CLASSES}}
                }
            },
            "addon_4_type_conditional_detectability": {
                "by_error_type": {
                    name: {"n_gold": 100, "overlap_recall": 0.6,
                           "mean_conformal_set_size": 2.0}
                    for name in CLASSES
                }
            },
            "addon_5_implicit_true_ablation": {
                "n_spans_flagged": 96,
                "n_responses_that_change_label": 49,
                "example_before": {"f1": 0.7623},
                "example_after": {"f1": 0.7488},
                "delta_f1": -0.0135,
            },
        },
    )
    return root


def paths_for(root):
    return {
        "label_audit": root / "analysis" / "label_audit.json",
        "typed_matching": root / "analysis" / "typed_matching_test.json",
        "addons": root / "analysis" / "addons.json",
        "baseline_metrics": root / "type" / "baseline_metrics.json",
        "conformal": root / "type" / "c3_conformal.json",
        "explanation": root / "explanation" / "explanation_metrics.json",
    }


def test_every_table_is_present_when_every_artefact_is(artefacts):
    text = T.build(paths_for(artefacts))
    for heading in (
        "C3-1 Label audit",
        "C3-2 Typed span matching",
        "C3-3 Type prediction",
        "C3-4 Type calibration",
        "C3-5 Conformal type sets",
        "C3-6 Severity-weighted conformal risk control",
        "C3-7 Explanations",
        "C3-8 Generator model against error type",
        "C3-9 Escalation budget",
    ):
        assert heading in text


def test_a_missing_artefact_becomes_a_visible_todo(artefacts, tmp_path):
    paths = paths_for(artefacts)
    paths["conformal"] = tmp_path / "absent.json"
    text = T.build(paths)
    assert "TODO" in text
    assert "absent.json` not on disk" in text
    # The tables that do not depend on it must still be there.
    assert "C3-2 Typed span matching" in text


def test_the_floor_row_is_always_labelled(artefacts):
    text = T.build(paths_for(artefacts))
    assert "ignores the input" in text
    assert "floor, ignores the input" in text


def test_an_unreachable_risk_bound_says_so_instead_of_printing_a_number(artefacts):
    text = T.build(paths_for(artefacts))
    assert "not reachable at this alpha" in text


def test_a_class_absent_from_calibration_is_marked_not_scored(artefacts):
    text = T.build(paths_for(artefacts))
    assert "n/a - absent from calibration" in text


def test_number_formatting_handles_nan_and_none():
    assert T.number(None) == "TODO"
    assert T.number(float("nan")) == "n/a"
    assert T.number(float("inf")) == "inf"
    assert T.number(0.5) == "0.5000"
    assert T.count(None) == "TODO"
    assert T.count(1234) == "1,234"


def test_main_writes_the_markdown(artefacts, tmp_path):
    out = tmp_path / "tables.md"
    assert T.main(["--results-dir", str(artefacts), "--out", str(out)]) == 0
    assert "C3-1 Label audit" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def test_every_figure_is_written_as_png_and_pdf(artefacts, tmp_path):
    out_dir = tmp_path / "figures"
    assert F.main(["--results-dir", str(artefacts), "--out-dir", str(out_dir)]) == 0
    for name in (
        "c3_severity_confusion",
        "c3_set_size_vs_alpha",
        "c3_coverage_vs_alpha",
        "c3_type_conditional_ece",
        "c3_escalation_budget",
    ):
        assert (out_dir / f"{name}.png").exists()
        assert (out_dir / f"{name}.pdf").exists()


def test_a_missing_artefact_skips_its_figure_without_failing(tmp_path, capsys):
    out_dir = tmp_path / "figures"
    assert F.main(["--results-dir", str(tmp_path / "nothing"), "--out-dir", str(out_dir)]) == 0
    printed = capsys.readouterr().out
    assert "skipped" in printed
    assert not list(out_dir.glob("*.png")) if out_dir.exists() else True
