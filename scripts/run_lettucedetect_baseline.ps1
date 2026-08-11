# Score RAGTruth with the public LettuceDetect checkpoint, then run C2 on the result.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_lettucedetect_baseline.ps1
#
# Produces two things the project does not otherwise have until C1 finishes
# training:
#
#   1. A baseline measured under OUR evaluation code, so the comparison against
#      the published 79.22% is like-for-like rather than our metric definition
#      against theirs.
#   2. Real probabilities on a held-out calibration split, which is everything
#      C2 needs. This is the parallelism the plan depends on -- C2 does not have
#      to wait for C1.
#
# The calibration split is re-derived with the same seed and fractions
# configs/c1_base.yaml uses, so it is exactly the set C1 will hold out.
#
# Inference only, no training. Runs on the 6GB RTX 3050 at batch size 1.

param(
    [switch]$SkipTest,
    [switch]$SkipCalib,
    [switch]$SkipC2
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "no virtualenv at $python" }
if (-not (Test-Path (Join-Path $repo "data\raw\source_info.jsonl"))) {
    throw "data\raw\source_info.jsonl missing -- run scripts\prepare_ragtruth.ps1 first"
}

if (-not $SkipTest) {
    Write-Host "`n== LettuceDetect on the RAGTruth test split ==" -ForegroundColor Cyan
    & $python -u -m src.c1_detector.lettucedetect_adapter --split test --dump-probs
    if ($LASTEXITCODE -ne 0) { throw "test split failed" }
}

if (-not $SkipCalib) {
    # Scored for reference only. This split comes out of RAGTruth TRAIN, which
    # the public checkpoint was trained on, so it is in-sample: example F1 0.9267
    # here against 0.7918 on the real test split. It must NOT be used to
    # calibrate -- doing so drove empirical coverage to 0.769 against a 0.900
    # target. It is kept because that gap is itself a result worth reporting.
    Write-Host "`n== LettuceDetect on the C1 calibration split (in-sample, reference only) ==" -ForegroundColor Cyan
    & $python -u -m src.c1_detector.lettucedetect_adapter --split calib --dump-probs
    if ($LASTEXITCODE -ne 0) { throw "calibration split failed" }
}

if (-not $SkipC2) {
    # --self-split halves the held-out test file into calibration and evaluation.
    # Both halves are out-of-sample for this checkpoint, which is what
    # exchangeability requires. Our own C1 will not need this: train_c1.py holds
    # its calibration split out of training already.
    Write-Host "`n== C2: calibration and split conformal ==" -ForegroundColor Cyan
    & $python -u -m src.c2_calibration.run_c2 `
        --test results/lettucedetect/test/probabilities.jsonl `
        --self-split `
        --out-dir results/c2/lettucedetect
    $c2Exit = $LASTEXITCODE
    if ($c2Exit -ne 0) {
        Write-Host "`nC2 reported a COVERAGE VIOLATION." -ForegroundColor Red
        Write-Host "Empirical coverage fell below the target. That is not a weak" -ForegroundColor Red
        Write-Host "result, it is a wrong one -- debug the maths before anything else." -ForegroundColor Red
        exit $c2Exit
    }
}

Write-Host "`n== done ==" -ForegroundColor Green
Write-Host "results\lettucedetect\test\metrics.json    baseline vs the published 79.22%"
Write-Host "results\c2\lettucedetect\c2_results.json   ECE, coverage, risk-coverage"
