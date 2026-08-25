"""The three proof-page figures.

These are the only numbers a non-technical reader of the interface ever sees
argued, so the properties worth pinning are the ones that would let the page
mislead someone: a band that is not the band C2 computes, a "held" flag that
does not follow from its own inputs, and a saving that is not the saving the
underlying sets actually give.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common import proof_charts


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_coverage_band_comes_from_c2_not_from_here(tmp_path):
    """The tolerance must be C2's own, for the split sizes of that row."""
    from src.c2_calibration.conformal import coverage_tolerance

    results = _write(
        tmp_path / "c2.json",
        {
            "span": {
                "conformal": {
                    "coverage": [
                        {
                            "alpha": 0.1,
                            "target_coverage": 0.9,
                            "empirical_coverage": 0.8808,
                            "n_calibration": 1509,
                            "n_test": 2390,
                            "abstention_rate": 0.4803,
                        }
                    ]
                }
            }
        },
    )

    rows = proof_charts.coverage(results)
    assert len(rows) == 1
    row = rows[0]
    assert row["band"] == pytest.approx(
        round(coverage_tolerance(0.1, 1509, 2390), 4)
    )
    assert row["promised"] == 0.9
    assert row["delivered"] == 0.8808


@pytest.mark.parametrize(
    "delivered, held",
    [
        (0.9, True),  # exactly on target
        (0.8808, True),  # under target but inside the band
        (0.5, False),  # under target and well outside it
        (0.95, True),  # over target
    ],
)
def test_held_follows_from_the_band(tmp_path, delivered, held):
    """`held` is a claim about the band, not a bare comparison to the target."""
    results = _write(
        tmp_path / "c2.json",
        {
            "span": {
                "conformal": {
                    "coverage": [
                        {
                            "alpha": 0.1,
                            "target_coverage": 0.9,
                            "empirical_coverage": delivered,
                            "n_calibration": 1509,
                            "n_test": 2390,
                            "abstention_rate": 0.4,
                        }
                    ]
                }
            }
        },
    )
    assert proof_charts.coverage(results)[0]["held"] is held


def test_evidence_share_is_the_ratio_of_the_means(tmp_path):
    """Not the mean of the ratios -- the cost artefact reports that one.

    Two different numbers both called "reduction" on one page is how a reader
    stops trusting either, so this one is named `share_returned` and has to be
    exactly what that name says.
    """
    conformal = _write(
        tmp_path / "c4_conformal.json",
        {
            "coverage_table": [
                {
                    "alpha": 0.2,
                    "empirical_coverage": 0.80,
                    "mean_pool_size": 10.0,
                    "mean_set_size": 3.0,
                    "singleton_rate": 0.4,
                    "n_test": 100,
                },
                {
                    "alpha": 0.1,
                    "empirical_coverage": 0.9003,
                    "mean_pool_size": 20.0,
                    "mean_set_size": 5.0,
                    "singleton_rate": 0.5,
                    "n_test": 582,
                },
            ]
        },
    )
    cost = _write(tmp_path / "cost.json", {"headline": {"sentence": "a sentence"}})

    out = proof_charts.evidence(conformal, cost)
    assert out is not None
    # The alpha=0.1 row, not the first row in the file.
    assert out["n"] == 582
    assert out["pool"] == 20.0
    assert out["returned"] == 5.0
    assert out["share_returned"] == 0.25
    assert out["sentence"] == "a sentence"


def test_evidence_is_absent_rather_than_invented(tmp_path):
    """No C4 run means no figure. It must not fall back to a plausible number."""
    assert proof_charts.evidence(tmp_path / "missing.json") is None

    empty = _write(tmp_path / "c4_conformal.json", {"coverage_table": []})
    assert proof_charts.evidence(empty) is None

    wrong_alpha = _write(
        tmp_path / "other.json",
        {
            "coverage_table": [
                {
                    "alpha": 0.25,
                    "empirical_coverage": 0.75,
                    "mean_pool_size": 9.0,
                    "mean_set_size": 4.0,
                    "singleton_rate": 0.3,
                    "n_test": 10,
                }
            ]
        },
    )
    assert proof_charts.evidence(wrong_alpha) is None


def test_the_shipped_file_matches_what_the_page_expects():
    """The interface reads these keys by name; a rename would blank the page."""
    path = Path("frontend/public/proof-charts.json")
    if not path.exists():
        pytest.skip("proof-charts.json not generated in this checkout")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert {"generated", "source", "reliability", "coverage", "evidence"} <= set(payload)

    reliability = payload["reliability"]
    assert reliability["before"] and reliability["after"]
    for series in ("before", "after"):
        for point in reliability[series]:
            assert {"confidence", "actual", "n"} == set(point)
            assert 0.0 <= point["confidence"] <= 1.0
            assert 0.0 <= point["actual"] <= 1.0
            assert point["n"] > 0
    # The whole first claim on the page is that correction moved it closer.
    assert reliability["ece_after"] < reliability["ece_before"]

    assert payload["coverage"]
    for row in payload["coverage"]:
        assert {"promised", "delivered", "band", "held", "sent_to_person"} == set(row)
        assert row["held"] == (row["delivered"] >= row["promised"] - row["band"])

    if payload["evidence"] is not None:
        ev = payload["evidence"]
        assert ev["returned"] < ev["pool"]
        assert 0.0 < ev["share_returned"] < 1.0
