# Boot the API with the REAL detector and hit it over HTTP.
#
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_live_real.ps1
#
# Proves the whole serving path end to end: LettuceDetect probabilities, the C2
# calibration artifact, the conformal decision, and a schema-valid response the
# React frontend can render. smoke_live.ps1 covers the same endpoints with the
# stub; this one covers them with a model.
#
# First run downloads ~1.6GB of checkpoint. Later runs use the HF cache.

param(
    [int]$Port = 8011,
    [double]$Alpha = 0.1
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "no virtualenv at $python" }

$artifact = Join-Path $repo "results\c2\lettucedetect\c2_artifact.json"
if (-not (Test-Path $artifact)) {
    throw "no C2 artifact at $artifact -- run scripts\run_lettucedetect_baseline.ps1 first"
}

# Environment variables must be set here, in a script file. Passing them inside
# a powershell -Command string gets them expanded by the outer shell first.
$env:TRUSTRAG_DETECTOR = "lettucedetect"
$env:TRUSTRAG_C2_ARTIFACT = $artifact
$env:PYTHONPATH = $repo

$log = Join-Path $repo "results\smoke_live_real.log"
Write-Host "starting uvicorn on port $Port (first run downloads the checkpoint)" -ForegroundColor Cyan

$server = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--port", "$Port", "--host", "127.0.0.1" `
    -WorkingDirectory $repo -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
    -WindowStyle Hidden -PassThru

try {
    $healthy = $false
    foreach ($attempt in 1..90) {
        Start-Sleep -Seconds 2
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
            $healthy = $true
            break
        } catch { }
    }
    if (-not $healthy) { throw "server never became healthy; see $log" }

    Write-Host "`n== /api/health ==" -ForegroundColor Cyan
    $health | Format-List | Out-String | Write-Host

    if (-not $health.detector_loaded) {
        throw "detector_loaded is false -- the stub is serving, not the model. See $log"
    }

    Write-Host "== /api/analyze on the canned example ==" -ForegroundColor Cyan
    $example = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/example"
    $body = @{
        question  = $example.question
        context   = $example.context
        answer    = $example.answer
        alpha     = $Alpha
        task_type = "qa"
    } | ConvertTo-Json

    $result = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/analyze" `
        -Method Post -Body $body -ContentType "application/json"

    Write-Host "model_version : $($result.model_version)"
    Write-Host "latency_ms    : $($result.latency_ms)"
    Write-Host "spans         : $($result.spans.Count)"
    Write-Host ""
    foreach ($span in $result.spans) {
        Write-Host ("  [{0,3}:{1,3}] {2,-9} raw {3:F4}  calibrated {4:F4}  nonconf {5:F4}  {6}" -f `
            $span.start, $span.end, $span.conformal_decision, $span.span_score, `
            $span.calibrated_score, $span.nonconformity, ("'" + $span.text + "'"))
    }

    if ($result.spans.Count -eq 0) {
        Write-Host "`nNo spans returned. Not necessarily wrong -- but check the log." -ForegroundColor Yellow
    }

    Write-Host "`n== alpha sweep on the same answer ==" -ForegroundColor Cyan
    foreach ($a in 0.05, 0.1, 0.2, 0.4) {
        $body = @{
            question = $example.question; context = $example.context
            answer = $example.answer; alpha = $a; task_type = "qa"
        } | ConvertTo-Json
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/analyze" `
            -Method Post -Body $body -ContentType "application/json"
        # @() forces an array: a single-element filter result has no .Count in
        # Windows PowerShell and prints as blank.
        $flag = @($r.spans | Where-Object { $_.conformal_decision -eq "flag" }).Count
        $abstain = @($r.spans | Where-Object { $_.conformal_decision -eq "abstain" }).Count
        Write-Host ("  alpha {0,-5} spans {1,-3} flag {2,-3} abstain {3}" -f $a, @($r.spans).Count, $flag, $abstain)
    }

    Write-Host "`n== live smoke passed ==" -ForegroundColor Green
}
finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
        Write-Host "server stopped"
    }
}
