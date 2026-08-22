# Boot the API with a REAL detector and hit it over HTTP.
#
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_live_real.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_live_real.ps1 -Detector lettucedetect
#
# Proves the whole serving path end to end: model probabilities, the C2
# calibration artifact, the conformal decision, and a schema-valid response the
# React frontend can render. smoke_live.ps1 covers the same endpoints with the
# stub; this one covers them with a model.
#
# -Detector c1            our own trained ModernBERT-base from results\c1.
#                         Nothing to download; the checkpoint is on disk.
# -Detector lettucedetect the public baseline. First run downloads ~1.6GB.

param(
    [ValidateSet("c1", "lettucedetect")]
    [string]$Detector = "c1",
    [int]$Port = 8011,
    [double]$Alpha = 0.1
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "no virtualenv at $python" }

# Environment variables must be set here, in a script file. Passing them inside
# a powershell -Command string gets them expanded by the outer shell first.
#
# TRUSTRAG_PROMPT_STYLE and TRUSTRAG_MAX_LENGTH are deliberately NOT set: each
# detector's own training-time defaults live in build_from_env, and overriding
# them from a smoke script is how a train/serve mismatch gets normalised.
if ($Detector -eq "c1") {
    $artifact = Join-Path $repo "results\c2\c1\c2_artifact.json"
    $checkpoint = Join-Path $repo "results\c1\modernbert-base\best"
    if (-not (Test-Path $artifact)) {
        throw "no C2 artifact at $artifact -- run src.c2_calibration.run_c2 against results\c1 first"
    }
    if (-not (Test-Path $checkpoint)) {
        throw "no C1 checkpoint at $checkpoint -- unpack the Kaggle output first"
    }
    $env:TRUSTRAG_MODEL_ID = $checkpoint
} else {
    $artifact = Join-Path $repo "results\c2\lettucedetect\c2_artifact.json"
    if (-not (Test-Path $artifact)) {
        throw "no C2 artifact at $artifact -- run scripts\run_lettucedetect_baseline.ps1 first"
    }
    Remove-Item Env:\TRUSTRAG_MODEL_ID -ErrorAction SilentlyContinue
}

$env:TRUSTRAG_DETECTOR = $Detector
$env:TRUSTRAG_C2_ARTIFACT = $artifact
$env:PYTHONPATH = $repo
Write-Host "detector $Detector  artifact $artifact" -ForegroundColor Cyan

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

    Write-Host "`n== does the guarantee apply to this input? ==" -ForegroundColor Cyan
    $check = $result.distribution_check
    if ($null -eq $check) {
        throw "the response carries no distribution_check"
    }
    Write-Host ("  checked {0}  in_distribution {1}  p {2:F4}  reference n {3}" -f `
        $check.checked, $check.in_distribution, $check.p_value, $check.n_reference)
    if (-not $check.checked) {
        Write-Host "  no reference loaded: $($check.message)" -ForegroundColor Yellow
    } elseif (-not $result.guarantee_applies) {
        # The record the demo opens on is a real RAGTruth response and must be
        # unremarkable. A warning here would appear in front of a panel.
        throw "the demo record tripped the out-of-distribution warning (p $($check.p_value))"
    } else {
        Write-Host "  demo record keeps its guarantee, as it must" -ForegroundColor Green

        Write-Host "`n== and the hand-written example, which should NOT ==" -ForegroundColor Cyan
        $hand = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/example?name=handwritten"
        $handBody = @{
            question = $hand.question; context = $hand.context
            answer = $hand.answer; alpha = $Alpha; task_type = "qa"
        } | ConvertTo-Json
        $handResult = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/analyze" `
            -Method Post -Body $handBody -ContentType "application/json"
        $handCheck = $handResult.distribution_check
        Write-Host ("  p {0:F4}  most unusual: {1}" -f $handCheck.p_value, $handCheck.most_unusual)
        if ($handResult.guarantee_applies) {
            throw "the hand-written example kept its guarantee; the alarm is deaf"
        }
        Write-Host "  guarantee correctly withdrawn, spans still returned ($($handResult.spans.Count))" -ForegroundColor Green
        foreach ($f in $handCheck.features | Where-Object { $_.unusual }) {
            Write-Host ("    unusual: {0,-38} {1:P1}" -f $f.label, $f.percentile)
        }
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

    Write-Host "`n== /api/metrics ==" -ForegroundColor Cyan
    $metrics = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/metrics" -TimeoutSec 20
    if (-not $metrics.available) {
        throw "metrics tab has nothing to show -- run src.c2_calibration.run_c2 first"
    }
    Write-Host ("  calibrator {0}   ECE {1:F4} -> {2:F4}   AUROC {3:F4}" -f `
        $metrics.selected_calibrator, $metrics.ece_before, $metrics.ece_after, $metrics.auroc)
    $floor = @($metrics.calibration | Where-Object { $_.is_floor })
    if ($floor.Count -ne 1) {
        throw "the constant-base-rate floor row is missing; an ECE must never be shown without it"
    }
    Write-Host ("  uninformative floor ECE {0:F4} -- read the column against this, not zero" -f $floor[0].ece)
    foreach ($row in $metrics.coverage) {
        if ($row.band -le 0) { throw "coverage row at alpha $($row.alpha) carries no band" }
        Write-Host ("  alpha {0,-5} coverage {1:F4}  band +/-{2:F4}  {3}" -f `
            $row.alpha, $row.empirical_coverage, $row.band,
            $(if ($row.inside_band) { "in band" } else { "OUTSIDE BAND" }))
    }
    if ($metrics.shift_available) {
        $shift = @($metrics.shift | Where-Object { $_.alpha -eq 0.1 })[0]
        Write-Host ("  under shift at alpha 0.10: in-domain {0:F4}  shifted(VOID) {1:F4}  best repair {2:F4}" -f `
            $shift.in_domain, $shift.shifted, $shift.repaired)
    }

    Write-Host "`n== /api/figures ==" -ForegroundColor Cyan
    if ($metrics.figures.Count -eq 0) {
        Write-Host "  none generated; run python -m src.c2_calibration.figures" -ForegroundColor Yellow
    }
    # -UseBasicParsing is not optional here. Without it Windows PowerShell 5.1
    # hands the response to the Internet Explorer engine to parse, and on a
    # machine where IE has never been configured that call hangs forever with no
    # error and no timeout. Cost an evening once; it is not going to cost
    # another one.
    foreach ($name in $metrics.figures) {
        $figure = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/figures/$name.png" `
            -TimeoutSec 15 -UseBasicParsing
        if ($figure.Headers["Content-Type"] -notlike "image/png*") {
            throw "$name did not come back as a PNG"
        }
        Write-Host ("  {0,-28} {1:N0} bytes" -f $name, $figure.RawContentLength)
    }
    $refused = $false
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/figures/secrets.png" `
            -TimeoutSec 10 -UseBasicParsing | Out-Null
    } catch {
        $refused = $true
    }
    if (-not $refused) {
        throw "an unknown figure name was served; the route must reject anything off the list"
    }
    Write-Host "  unknown figure name correctly refused"

    Write-Host "`n== live smoke passed ==" -ForegroundColor Green
}
finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
        Write-Host "server stopped"
    }
}
