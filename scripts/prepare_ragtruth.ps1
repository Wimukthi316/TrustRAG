# Download RAGTruth and build the training examples, then stop for the hand check.
#
#   powershell -ExecutionPolicy Bypass -File scripts\prepare_ragtruth.ps1
#
# Skips the download if data\raw already holds both files. Pass -Force to
# redownload.

param(
    [switch]$Force,
    [switch]$StatsOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "no virtualenv at $python -- run scripts\setup_env.ps1 first"
}

Write-Host "`n== downloading RAGTruth ==" -ForegroundColor Cyan
$dlArgs = @("-m", "src.c1_detector.download_ragtruth")
if ($Force) { $dlArgs += "--force" }
& $python @dlArgs
if ($LASTEXITCODE -ne 0) { throw "download failed" }

Write-Host "`n== building examples ==" -ForegroundColor Cyan
$buildArgs = @("-m", "src.c1_detector.build_examples")
if ($StatsOnly) { $buildArgs += "--stats-only" }
& $python @buildArgs
$buildExit = $LASTEXITCODE

if ($buildExit -ne 0) {
    Write-Host "`nbuild_examples exited $buildExit -- the reproduction check against" -ForegroundColor Red
    Write-Host "the published statistics table did not pass. Read the table above" -ForegroundColor Red
    Write-Host "before going any further. Do not start training." -ForegroundColor Red
    exit $buildExit
}

if ($StatsOnly) { exit 0 }

Write-Host "`n== next: check ten examples by hand ==" -ForegroundColor Cyan
Write-Host "  $python -m src.c1_detector.inspect_examples --n 10 --with-spans"
Write-Host "  $python -m src.c1_detector.inspect_examples --task data2text --n 5 --with-spans"
Write-Host ""
Write-Host "Read them. Confirm the >>> <<< brackets land on word boundaries and the" -ForegroundColor Yellow
Write-Host "bracketed text really is unsupported by the context printed above it." -ForegroundColor Yellow
