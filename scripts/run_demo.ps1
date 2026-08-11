# Start the demo: FastAPI with the real detector, plus the Vite dev server.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
#
# Then open http://localhost:5173
#
# Leaves both processes running in the background and writes their PIDs to
# results\demo_pids.txt. Stop them with:
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Stop
#
# The backend takes ~30s on first start because it loads a 400M-parameter model.
# Pass -Stub to serve the placeholder detector instead, which starts instantly
# and is useful for frontend work.

param(
    [switch]$Stop,
    [switch]$Stub,
    [int]$ApiPort = 8000,
    [int]$UiPort = 5173
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$pidFile = Join-Path $repo "results\demo_pids.txt"

if ($Stop) {
    if (Test-Path $pidFile) {
        foreach ($line in Get-Content $pidFile) {
            if ($line -match '^\d+$') {
                Stop-Process -Id ([int]$line) -Force -ErrorAction SilentlyContinue
                Write-Host "stopped $line"
            }
        }
        Remove-Item $pidFile -Force
    } else {
        Write-Host "no $pidFile -- nothing recorded as running"
    }
    exit 0
}

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "no virtualenv at $python" }

# nvm keeps Node 18 first on PATH and Vite 6 needs Node 20+.
$env:Path = "C:\Program Files\nodejs;" + $env:Path
$env:NODE_ENV = "development"

if ($Stub) {
    Remove-Item Env:\TRUSTRAG_DETECTOR -ErrorAction SilentlyContinue
    Write-Host "serving the PLACEHOLDER detector -- scores are not model output" -ForegroundColor Yellow
} else {
    $artifact = Join-Path $repo "results\c2\lettucedetect\c2_artifact.json"
    if (-not (Test-Path $artifact)) {
        throw "no C2 artifact at $artifact -- run scripts\run_lettucedetect_baseline.ps1 first"
    }
    $env:TRUSTRAG_DETECTOR = "lettucedetect"
    $env:TRUSTRAG_C2_ARTIFACT = $artifact
}
$env:PYTHONPATH = $repo

New-Item -ItemType Directory -Force -Path (Join-Path $repo "results") | Out-Null

Write-Host "starting API on http://127.0.0.1:$ApiPort" -ForegroundColor Cyan
$api = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
    -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $repo "results\demo_api.log") `
    -RedirectStandardError (Join-Path $repo "results\demo_api.err.log") `
    -WindowStyle Hidden -PassThru

Write-Host "starting Vite on http://localhost:$UiPort" -ForegroundColor Cyan
$ui = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm.cmd run dev -- --port $UiPort" `
    -WorkingDirectory (Join-Path $repo "frontend") `
    -RedirectStandardOutput (Join-Path $repo "results\demo_ui.log") `
    -RedirectStandardError (Join-Path $repo "results\demo_ui.err.log") `
    -WindowStyle Hidden -PassThru

Set-Content -Path $pidFile -Value @($api.Id, $ui.Id)

Write-Host "`nwaiting for the API (first start loads the model, ~30s)..." -ForegroundColor Cyan
$health = $null
foreach ($attempt in 1..60) {
    Start-Sleep -Seconds 2
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 5
        break
    } catch { }
}
if (-not $health) { throw "API never came up; see results\demo_api.err.log" }

Write-Host "  model_version   : $($health.model_version)"
Write-Host "  detector_loaded : $($health.detector_loaded)"
if (-not $Stub -and -not $health.detector_loaded) {
    Write-Host "  WARNING: the stub is serving, not the model. Check the log." -ForegroundColor Red
}

$uiUp = $false
foreach ($attempt in 1..30) {
    Start-Sleep -Seconds 2
    try {
        Invoke-WebRequest -Uri "http://localhost:$UiPort" -TimeoutSec 5 -UseBasicParsing | Out-Null
        $uiUp = $true
        break
    } catch { }
}
if (-not $uiUp) { throw "Vite never came up; see results\demo_ui.err.log" }

Write-Host "`n== demo is up ==" -ForegroundColor Green
Write-Host "  UI   http://localhost:$UiPort"
Write-Host "  API  http://127.0.0.1:$ApiPort/docs"
Write-Host "`nstop it with:  powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Stop"
