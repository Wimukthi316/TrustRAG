"""The table generator must print TODO where a number is missing, never a guess."""

import json

import pytest

from src.c1_detector.tables import TODO, build, table_1, table_3, trivial_f1


def write(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def metrics(f1_value):
    return {
        "overall": {
            "token": {"f1": f1_value},
            "span_exact": {"f1": f1_value / 3},
            "span_overlap": {"f1": f1_value},
            "example": {
                "f1": f1_value + 0.2,
                "tp": 715,
                "fp": 218,
                "fn": 228,
                "tn": 1539,
            },
        }
    }


def test_an_empty_results_tree_produces_todo_and_not_a_crash(tmp_path):
    text = build(tmp_path)
    assert TODO in text
    for title in ("Table 1", "Table 2", "Table 3", "Table 4", "Table 5"):
        assert title in text


def test_no_number_is_invented_for_a_missing_baseline(tmp_path):
    write(tmp_path, "c1/test/metrics.json", metrics(0.5304))
    text = "\n".join(table_1(tmp_path))

    assert "0.5304" in text
    # HHEM and the judge were never run in this tree.
    assert text.count(TODO) >= 4


def test_the_trivial_floor_is_computed_from_the_confusion_counts(tmp_path):
    write(tmp_path, "c1/test/metrics.json", metrics(0.5304))
    text = "\n".join(table_1(tmp_path))

    # 943 positives of 2,700 responses -> p = 0.3493, floor 2p/(1+p) = 0.5177.
    assert "p=0.3493" in text
    assert "0.5177" in text


def test_trivial_f1_is_the_flag_everything_score():
    assert trivial_f1(0.0) == 0.0
    assert trivial_f1(1.0) == pytest.approx(1.0)
    assert trivial_f1(0.3493) == pytest.approx(0.5177, abs=5e-5)


def test_a_response_level_model_reads_n_a_in_the_span_columns(tmp_path):
    write(tmp_path, "c1/test/metrics.json", metrics(0.5304))
    write(
        tmp_path,
        "hhem/hhem_metrics.json",
        {
            "chosen_threshold": 0.18,
            "test": {"adapted": {"f1": 0.7012}, "at_half": {"f1": 0.6256}},
        },
    )
    row = [line for line in table_1(tmp_path) if "HHEM" in line and "0.7012" in line]
    assert row, "the HHEM row should be present"
    assert row[0].count("n/a") == 3, "token and both span columns are not applicable"


def test_impossible_cells_read_as_a_dash_rather_than_zero(tmp_path):
    write(
        tmp_path,
        "c1/analysis/localisation_report.json",
        {
            "n_gold_spans": 1517,
            "n_pred_spans": 2390,
            "overall": {
                "buckets": {
                    "exact": {"gold": 290, "pred": 290},
                    "missed": {"gold": 503, "pred": 0},
                    "spurious": {"gold": 0, "pred": 1134},
                }
            },
            "tokenisation_ceiling": {"span_exact": {"f1": 0.9967}},
            "span_lengths": {"ALL": {"gold_chars": {"p50": 35}, "pred_chars": {"p50": 17}}},
        },
    )
    lines = table_3(tmp_path)
    missed = next(line for line in lines if line.startswith("| missed"))
    spurious = next(line for line in lines if line.startswith("| spurious"))

    # A predicted-span count of 0 for "missed" is a definition, not a measurement.
    assert missed.endswith("| - | - |")
    assert "| - | - |" in spurious
    assert "0.9967" in "\n".join(lines)
