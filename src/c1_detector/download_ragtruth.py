"""Download the RAGTruth corpus into data/raw/.

RAGTruth is MIT licensed and published as two JSONL files in the authors'
repository. They are small enough (about 37 MB combined) to fetch directly, so
there is no need to clone the whole repo or go through the Hugging Face Hub.

Usage:
    python -m src.c1_detector.download_ragtruth
    python -m src.c1_detector.download_ragtruth --dest data/raw --force

On Kaggle, run the same command after cloning this repo into the notebook.
Kaggle notebooks need the internet toggle enabled for this to work.

Nothing here is cached across Kaggle sessions, so budget about a minute for the
download each time a fresh session starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"

# Expected byte sizes as reported by the GitHub contents API on 2026-08-11.
# These are a sanity check on a truncated download, not a security control --
# the upstream repo can legitimately update the files, in which case the sizes
# change and this script warns rather than fails.
FILES: Dict[str, int] = {
    "response.jsonl": 21_458_735,
    "source_info.jsonl": 15_117_971,
}

# Line counts published in the RAGTruth README's statistics table.
EXPECTED_LINES = {
    "response.jsonl": 17_790,
    "source_info.jsonl": 2_965,
}


def download(name: str, dest: Path, force: bool = False) -> Path:
    out = dest / name
    if out.exists() and not force:
        print(f"{name}: already present at {out} ({out.stat().st_size:,} bytes), skipping")
        return out

    url = f"{BASE_URL}/{name}"
    print(f"{name}: downloading from {url}")
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as fh:
            downloaded = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                print(f"\r  {downloaded / 1e6:6.1f} MB", end="", flush=True)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"download of {name} failed: {exc}\n"
            "If this is a Kaggle notebook, check that the internet toggle is on."
        ) from exc
    print()

    tmp.replace(out)
    return out


def verify(path: Path) -> bool:
    """Check size and line count. Returns True if everything looks right."""
    name = path.name
    size = path.stat().st_size
    expected_size = FILES[name]
    ok = True

    if size != expected_size:
        print(
            f"  size {size:,} != expected {expected_size:,}. "
            "Upstream may have updated the file; check the line count below "
            "before trusting it."
        )
        ok = False

    lines = 0
    bad_json = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            lines += 1
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1

    expected_lines = EXPECTED_LINES[name]
    print(f"  {lines:,} JSON lines (expected {expected_lines:,})")
    if bad_json:
        print(f"  {bad_json} lines failed to parse as JSON")
        ok = False
    if lines != expected_lines:
        print("  line count does not match the published statistics table")
        ok = False

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"  sha256 {digest}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="directory to download into (default: data/raw)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file exists"
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for name in FILES:
        path = download(name, args.dest, force=args.force)
        all_ok &= verify(path)

    if all_ok:
        print("\nRAGTruth downloaded and verified.")
        print("Next: python -m src.c1_detector.build_examples")
        return 0

    print(
        "\nOne or more checks failed. Do not train on this data until you know why.\n"
        "Re-run with --force to redownload."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
