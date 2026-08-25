"""Layer 3's driver: score the lexical baseline and its four floors, and write one artefact.

Kept separate from `attribution_eval` so the harness stays importable by the
later layers without running anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from src.c4_attribution.attribution_eval import build_report, prediction_from_scores
from src.c4_attribution.dataset import load_responses, span_records
from src.c4_attribution.lexical import idf_overlap_scores
from src.c4_attribution.matching import match_all

OUT = Path("results/c4/lexical/lexical_metrics.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    rows_all = span_records(load_responses())
    payload = {}
    for split in ("calib", "test"):
        rows = [r for r in rows_all if r.response.split == split]
        matches = match_all(rows)
        predictions = [
            prediction_from_scores(idf_overlap_scores(r.span_text, r.sentences))
            for r in rows
        ]
        payload[split] = build_report(rows, matches, predictions, "lexical_idf")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    test = payload["test"]
    overall = test["convention_a"]["overall"]
    print(f"lexical top-1 (A) {overall['top1_accuracy']:.4f} on n={overall['n']}")
    print("per task, lexical against its floors, Convention A:")
    header = f"  {'task':<14}{'n':>6}{'lexical':>10}{'first':>10}{'longest':>10}{'random':>10}"
    print(header)
    floors = test["floors"]["by_task_convention_a"]
    for task, block in test["convention_a"]["by_task"].items():
        row = f"  {task:<14}{block['n']:>6}{block['top1_accuracy']:>10.4f}"
        for name in ("first", "longest", "random"):
            value = floors[name].get(task, {}).get("top1_accuracy", float("nan"))
            row += f"{value:>10.4f}"
        print(row)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
