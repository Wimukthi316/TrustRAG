"""Network-free tests for the medical conformal selective-prediction driver.

They exercise the file reader, the stratified split, the selective operating
points and the full analyse() path on a small synthetic scores set. They need
numpy (via conformal.py) but no checkpoint and no download.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.c2_calibration.medical_conformal import (
    analyse,
    difficulty_shift,
    read_answer_scores,
    selective_operating_points,
    stratified_split,
)


def _write_scores(path: Path, n: int = 800, seed: int = 0) -> None:
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n):
            diff = rng.choice(["easy", "medium", "hard"])
            gold = i % 2
            sep = {"easy": 0.25, "medium": 0.15, "hard": 0.05}[diff]
            base = 0.5 + (sep if gold else -sep)
            s = min(0.999, max(0.001, rng.gauss(base, 0.25)))
            handle.write(json.dumps({
                "id": f"medhallu:pqa_labeled:{i}:{'hall' if gold else 'true'}",
                "config": "pqa_labeled", "difficulty": diff, "category": "",
                "gold_positive": bool(gold), "pred_positive": s > 0.5, "score": s,
            }) + "\n")


def test_read_answer_scores_parses_score_and_label(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_scores(p, n=100)
    scores, labels, difficulty = read_answer_scores(p)
    assert len(scores) == len(labels) == len(difficulty) == 100
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert set(labels) <= {0, 1}
    assert set(difficulty) <= {"easy", "medium", "hard"}


def test_read_answer_scores_config_filter(tmp_path):
    p = tmp_path / "s.jsonl"
    with p.open("w") as h:
        h.write(json.dumps({"score": 0.9, "gold_positive": True, "config": "pqa_labeled"}) + "\n")
        h.write(json.dumps({"score": 0.1, "gold_positive": False, "config": "pqa_artificial"}) + "\n")
    scores, labels, _ = read_answer_scores(p, config="pqa_labeled")
    assert scores == [0.9] and labels == [1]


def test_stratified_split_preserves_class_rate_and_is_disjoint():
    labels = [0, 1] * 100
    calib, test = stratified_split(labels, calib_fraction=0.3, seed=1)
    assert set(calib).isdisjoint(test)
    assert sorted(calib + test) == list(range(200))
    # ~30% of each class in calibration
    calib_pos = sum(labels[i] for i in calib)
    assert 25 <= calib_pos <= 35


def test_selective_operating_points_are_monotone_in_budget():
    rng = random.Random(0)
    scores, labels = [], []
    for i in range(600):
        gold = i % 2
        scores.append(min(0.999, max(0.001, rng.gauss(0.5 + (0.2 if gold else -0.2), 0.25))))
        labels.append(gold)
    ops = selective_operating_points(scores, labels, target_risks=(0.1, 0.2, 0.3))
    met = [o for o in ops if o["met"]]
    # a looser error budget never buys less coverage
    covs = [o["coverage"] for o in met]
    assert covs == sorted(covs)
    for o in met:
        assert o["risk"] <= o["target_risk"] + 1e-9


def test_difficulty_shift_reports_both_directions():
    rng = random.Random(0)
    scores, labels, diff = [], [], []
    for i in range(900):
        d = rng.choice(["easy", "medium", "hard"])
        gold = i % 2
        sep = {"easy": 0.3, "medium": 0.2, "hard": 0.05}[d]
        scores.append(min(0.999, max(0.001, rng.gauss(0.5 + (sep if gold else -sep), 0.2))))
        labels.append(gold)
        diff.append(d)
    out = difficulty_shift(scores, labels, diff, alpha=0.1, seed=0)
    assert out is not None
    assert "calibrated_on_easy_tested_on_hard" in out
    assert "calibrated_on_hard_tested_on_hard" in out
    assert 0.0 <= out["calibrated_on_easy_tested_on_hard"]["empirical_coverage"] <= 1.0


def test_analyse_runs_end_to_end_and_coverage_tracks_target(tmp_path):
    p = tmp_path / "s.jsonl"
    _write_scores(p, n=1500, seed=2)
    scores, labels, difficulty = read_answer_scores(p)
    result = analyse(scores, labels, difficulty, alphas=(0.1, 0.2), calib_fraction=0.3, seed=7)
    assert result["n_total"] == 1500
    assert "coverage_table" in result and len(result["coverage_table"]) == 2
    # a correct split-conformal run tracks target within a few points on 1,500
    for row in result["coverage_table"]:
        assert abs(row["empirical_coverage"] - row["target_coverage"]) < 0.08
    assert isinstance(result["guarantee_holds"], bool)
    assert result["risk_coverage"]["aurc"] >= 0.0
