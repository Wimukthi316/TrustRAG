# Regenerate every C4 artefact from the frozen inputs, in dependency order.
#
# Reads:  data/raw, results/c1, results/c2            (all frozen)
# Writes: results/c4, paper/figures/c4_*
#
# It never writes to results/c1, results/c2, results/c3 or data. If it did,
# every number those components reported would need re-verifying.
#
# Two phases, because they cost very different amounts of time:
#
#   1. the pair-score dump. A MiniLM pre-filter, a DeBERTa NLI cross-encoder and
#      HHEM over the surviving pairs. CPU-only and slow -- hours on a laptop --
#      so it is resumable: rerun it and it picks up from the records already on
#      disk. Pass -SkipScores to leave the dump alone.
#   2. everything else. Minutes.
#
# Layers 1, 2 and 3 need no model at all and run even when phase 1 has never
# been run: the audit, the protocol and the scored lexical baseline are already
# a component.

param(
    [switch]$SkipScores
)

$ErrorActionPreference = "Stop"
$repo = "D:\SLIIT\Research\trustrag"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location $repo

function Step($name, $arguments) {
    Write-Host ""
    Write-Host "=== $name ===" -ForegroundColor Cyan
    & $py @arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$name FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if (-not $SkipScores) {
    Step "Layer 4 - pair-score dump (slow, resumable)" @(
        "-m", "src.c4_attribution.run_scorer", "calib", "test"
    )
}

Step "Layers 1 to 12 - the driver" @("-m", "src.c4_attribution.run_c4")

Step "tables" @("-m", "src.c4_attribution.tables", "--write")

Step "figures" @("-m", "src.c4_attribution.figures")

Write-Host ""
Write-Host "=== verification ===" -ForegroundColor Cyan
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "TESTS FAILED" -ForegroundColor Red; exit $LASTEXITCODE }

& $py -m ruff check --isolated --select E4,E7,E9,F src tests backend
if ($LASTEXITCODE -ne 0) { Write-Host "RUFF FAILED" -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host ""
Write-Host "C4 regenerated. Remaining TODO markers in results/c4/tables.md:" -ForegroundColor Green
Select-String -Path "results\c4\tables.md" -Pattern "TODO" | Measure-Object | Select-Object -ExpandProperty Count
