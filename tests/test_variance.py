"""Repeat-run variance: the reader must refuse to invent a sigma from two runs."""

import json

import pytest

from src.c1_detector.variance import (
    MIN_RUNS_FOR_SD,
    format_report,
    read_run,
    sample_sd,
    summarise,
)


def write_run(root, name, f1, seed=42):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "overall": {
                    "token": {"f1": f1},
                    "span_exact": {"f1": f1 / 3},
                    "span_overlap": {"f1": f1},
                    "example": {"f1": f1 + 0.2},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"config": {"seed": seed}}), encoding="utf-8"
    )
    return run_dir


def test_sd_is_none_below_three_runs():
    assert sample_sd([0.50]) is None
    assert sample_sd([0.50, 0.52]) is None
    assert sample_sd([0.50, 0.52, 0.54]) is not None


def test_sd_matches_the_hand_computed_value():
    # mean 2, deviations -1/0/+1, sample variance 2/2 = 1
    assert sample_sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_two_runs_report_a_range_but_no_sigma(tmp_path):
    runs = [
        read_run(write_run(tmp_path, "run-1", 0.5304)),
        read_run(write_run(tmp_path, "run-2", 0.5435)),
    ]
    report = summarise(runs)

    assert report["n_runs"] == 2
    assert report["sd_reported"] is False
    token = report["levels"]["token"]
    assert token["sd"] is None
    assert token["range_points"] == pytest.approx(1.31, abs=1e-6)

    text = format_report(report)
    assert "not enough for a standard deviation" in text
    assert "n/a" in text


def test_five_runs_report_a_sigma(tmp_path):
    runs = [
        read_run(write_run(tmp_path, f"run-{i}", 0.53 + 0.001 * i, seed=40 + i))
        for i in range(MIN_RUNS_FOR_SD + 2)
    ]
    report = summarise(runs)

    assert report["sd_reported"] is True
    assert report["levels"]["token"]["sd"] is not None
    assert report["levels"]["token"]["sd_points"] == pytest.approx(
        100 * report["levels"]["token"]["sd"]
    )


def test_mixed_seeds_are_flagged_because_they_confound_two_sources(tmp_path):
    same = [
        read_run(write_run(tmp_path, "a", 0.51, seed=42)),
        read_run(write_run(tmp_path, "b", 0.52, seed=42)),
    ]
    assert summarise(same)["mixes_seeds"] is False
    assert "GPU non-determinism alone" in format_report(summarise(same))

    mixed = same + [read_run(write_run(tmp_path, "c", 0.53, seed=7))]
    report = summarise(mixed)
    assert report["mixes_seeds"] is True
    assert report["distinct_seeds"] == [7, 42]
    assert "separates neither" in format_report(report)


def test_seed_is_read_from_the_run_not_from_the_config_file(tmp_path):
    # --seed overrides the YAML, and summary.json records what the run really used.
    run_dir = write_run(tmp_path, "override", 0.53, seed=2024)
    assert read_run(run_dir)["seed"] == 2024


def test_missing_metrics_is_an_error_not_a_silent_skip(tmp_path):
    empty = tmp_path / "no-metrics"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_run(empty)


def test_summarise_refuses_an_empty_run_list():
    with pytest.raises(ValueError):
        summarise([])


def test_the_seed_can_be_stated_when_the_metrics_dir_has_no_summary(tmp_path):
    # Evaluation output lands in results/c1/test-seed7, while summary.json stays
    # with the training run, so DIR=SEED has to be accepted.
    from src.c1_detector.variance import main

    run_dir = write_run(tmp_path, "test-seed7", 0.53, seed=42)
    (run_dir / "summary.json").unlink()
    assert read_run(run_dir)["seed"] is None

    out = tmp_path / "variance.json"
    main([f"{run_dir}=7", "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["distinct_seeds"] == [7]
