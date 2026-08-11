# Live end-to-end check: boots the real uvicorn server, hits every endpoint over
# HTTP, then boots the Vite dev server and confirms it serves. Catches things the
# in-process TestClient cannot (import paths, CORS config, port binding, proxying).
#
# Run:  powershell -ExecutionPolicy Bypass -File scripts\smoke_live.ps1

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:Path = "C:\Program Files\nodejs;" + $env:Path
$env:NODE_ENV = "development"

$fail = 0

Write-Host "=============== BACKEND ==============="
$api = Start-Process -FilePath "$repo\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--port", "8000" `
    -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$repo\_api.log" -RedirectStandardError "$repo\_api_err.log"

Start-Sleep -Seconds 8

try {
    $health = Invoke-RestMethod "http://127.0.0.1:8000/api/health" -TimeoutSec 10
    Write-Host "health        : $($health.status) | schema $($health.schema_version) | model $($health.model_version)"

    $example = Invoke-RestMethod "http://127.0.0.1:8000/api/example" -TimeoutSec 10
    Write-Host "example       : loaded, answer is $($example.answer.Length) chars"

    $body = $example | ConvertTo-Json -Depth 5
    $result = Invoke-RestMethod "http://127.0.0.1:8000/api/analyze" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec 20
    Write-Host "analyze       : $($result.spans.Count) spans in $($result.latency_ms) ms"

    # Every span must slice back out of the answer exactly.
    $bad = 0
    foreach ($s in $result.spans) {
        if ($result.answer.Substring($s.start, $s.end - $s.start) -ne $s.text) { $bad++ }
    }
    if ($bad -gt 0) { Write-Host "OFFSET MISMATCH on $bad spans"; $fail++ }
    else { Write-Host "span offsets  : all $($result.spans.Count) verified against the answer text" }

    $decisions = ($result.spans | Group-Object conformal_decision | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join " "
    Write-Host "decisions     : $decisions"

    # A bad alpha must be rejected, not silently accepted.
    try {
        Invoke-RestMethod "http://127.0.0.1:8000/api/analyze" -Method Post `
            -ContentType "application/json" `
            -Body '{"question":"q","context":"c","answer":"a","alpha":1.5}' -TimeoutSec 10 | Out-Null
        Write-Host "validation    : FAILED - alpha=1.5 was accepted"; $fail++
    } catch {
        Write-Host "validation    : alpha=1.5 correctly rejected"
    }
} catch {
    Write-Host "BACKEND ERROR : $_"
    Get-Content "$repo\_api_err.log" -ErrorAction SilentlyContinue | Select-Object -Last 15
    $fail++
} finally {
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=============== FRONTEND DEV SERVER ==============="
$vite = Start-Process -FilePath "C:\Program Files\nodejs\npm.cmd" `
    -ArgumentList "run", "dev" -WorkingDirectory "$repo\frontend" `
    -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$repo\_vite.log" -RedirectStandardError "$repo\_vite_err.log"

# Note: Vite binds to "localhost", which on Windows resolves to ::1 first. Probing
# 127.0.0.1 fails even though the server is up -- always use the localhost name.
$page = $null
for ($i = 1; $i -le 6; $i++) {
    Start-Sleep -Seconds 4
    try {
        $page = Invoke-WebRequest "http://localhost:5173/" -TimeoutSec 5 -UseBasicParsing
        break
    } catch {
        Write-Host "  waiting for vite... ($($i * 4)s)"
    }
}

try {
    if (-not $page) { throw "vite did not answer on :5173 within 60s" }
    if ($page.StatusCode -eq 200 -and $page.Content -match "root") {
        Write-Host "vite dev      : serving on :5173 (HTTP $($page.StatusCode))"
    } else {
        Write-Host "vite dev      : unexpected response"; $fail++
    }
} catch {
    Write-Host "VITE ERROR    : $_"
    Write-Host "--- vite stdout ---"
    Get-Content "$repo\_vite.log" -ErrorAction SilentlyContinue | Select-Object -Last 20
    Write-Host "--- vite stderr ---"
    Get-Content "$repo\_vite_err.log" -ErrorAction SilentlyContinue | Select-Object -Last 20
    $fail++
} finally {
    Stop-Process -Id $vite.Id -Force -ErrorAction SilentlyContinue
    Get-Process node -ErrorAction SilentlyContinue |
        Where-Object { $_.StartTime -gt (Get-Date).AddMinutes(-2) } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

Remove-Item "$repo\_api*.log", "$repo\_vite*.log" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=============== RESULT ==============="
if ($fail -eq 0) { Write-Host "ALL LIVE CHECKS PASSED" } else { Write-Host "$fail CHECK(S) FAILED" }
