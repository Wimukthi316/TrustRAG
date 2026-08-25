"""Layer 4's driver: build the pair-score dump for one split.

Run as a module so an interrupted run can be restarted and will resume from the
records already on disk:

    python -m src.c4_attribution.run_scorer calib
    python -m src.c4_attribution.run_scorer test

The dump is `*.jsonl` and therefore gitignored. It is regenerated, not
committed, exactly like C1's probability dumps.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from src.c4_attribution.dataset import load_responses, span_records
from src.c4_attribution.matching import match_all
from src.c4_attribution.scorer import build_pair_scores


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    splits = argv or ["calib", "test"]
    rows_all = span_records(load_responses())
    for split in splits:
        rows = [r for r in rows_all if r.response.split == split]
        matches = match_all(rows)
        build_pair_scores(rows, matches, split)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
