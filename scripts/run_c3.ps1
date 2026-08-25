# Regenerate every C3 artefact from the frozen inputs, in dependency order.
#
# Reads:  data/raw, data/processed, results/c1, results/c2   (all frozen)
# Writes: results/c3, paper/figures/c3_*
#
# It never writes to results/c1, results/c2 or data. If it did, every C2 number
# in the handoff would need re-verifying.
#
# Roughly ten minutes end to end on the laptop. The slow steps are the feature
# extraction in Layer 3 and the cluster bootstraps in Layers 3 and 4.

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

Step "Layer 1 - label audit" @("-m", "src.c3_explanation.labels_audit")

Step "Layer 2 - typed span matching, test" @(
    "-m", "src.c3_explanation.typed_matching",
    "--probs", "results/c1/test/probabilities.jsonl",
    "--processed", "data/processed/ragtruth_test.jsonl",
    "--split", "test"
)

Step "Layer 2 - typed span matching, calib" @(
    "-m", "src.c3_explanation.typed_matching",
    "--probs", "results/c1/calib/probabilities.jsonl",
    "--processed", "data/processed/ragtruth_train.jsonl",
    "--split", "calib"
)

Step "Layer 3.1 - CPU type baseline" @("-m", "src.c3_explanation.type_baseline")

Step "Layer 4 - conformal type sets" @("-m", "src.c3_explanation.type_conformal")

Step "Layer 4 - conformal, axis-product distribution" @(
    "-m", "src.c3_explanation.type_conformal",
    "--probs-key", "probs_axis_product",
    "--out-name", "c3_conformal_axis_product.json"
)

Step "Layer 4 - conformal on gold spans" @(
    "-m", "src.c3_explanation.type_conformal",
    "--calib", "results/c3/type/baseline_probs_gold_calib.jsonl",
    "--test", "results/c3/type/baseline_probs_gold_test.jsonl",
    "--out-name", "c3_conformal_gold.json"
)

Step "Layer 5.2 - explanation scoring" @("-m", "src.c3_explanation.explanation_eval")

Step "Layer 6 and the add-ons" @("-m", "src.c3_explanation.run_c3")

Step "tables" @("-m", "src.c3_explanation.tables")

Step "figures" @("-m", "src.c3_explanation.figures")

Write-Host ""
Write-Host "=== verification ===" -ForegroundColor Cyan
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "TESTS FAILED" -ForegroundColor Red; exit $LASTEXITCODE }

& $py -m ruff check --isolated --select E4,E7,E9,F src tests backend
if ($LASTEXITCODE -ne 0) { Write-Host "RUFF FAILED" -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host ""
Write-Host "C3 regenerated. Remaining TODO markers in results/c3/tables.md:" -ForegroundColor Green
Select-String -Path "results\c3\tables.md" -Pattern "TODO" | Measure-Object | Select-Object -ExpandProperty Count
