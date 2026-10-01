# Ask a running demo whether all four components are actually answering.
#
#   powershell -ExecutionPolicy Bypass -File scripts\check_live_components.ps1
#
# Start the server first with scripts\run_demo.ps1; this script does not boot
# anything, it interrogates what is already up.
#
# Why it exists. run_demo.ps1 prints model_version and detector_loaded, which
# say that C1 and C2 are attached and nothing at all about C3 or C4. Those two
# load their own artefacts from disk, and when a path is wrong they do not
# fail -- they return None on every span, exactly as they would if the layers
# had never been written. That is how C3 stayed invisible in the demo for six
# days. A field nothing prints is a field nobody notices is missing.
#
# So this walks one real request and reports, per component, whether the
# fields that component owns came back populated.
#
# -Alpha  the setting to check at. 0.4 is worth running too: it is the only
#         place the empty prediction set shows up.

param(
    [int]$Port = 8000,
    [double]$Alpha = 0.1,
    [string]$Example = "ragtruth_qa"
)

$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:$Port/api"
$failed = @()

function Show-Line($name, $ok, $detail) {
    $mark = if ($ok) { "ok  " } else { "MISSING" }
    $colour = if ($ok) { "Green" } else { "Red" }
    Write-Host ("  {0,-8} {1,-26} {2}" -f $mark, $name, $detail) -ForegroundColor $colour
}

Write-Host "== /api/health ==" -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 10
} catch {
    throw "no API on port $Port. Start it with scripts\run_demo.ps1 -Detector c1"
}
Write-Host "  model_version   : $($health.model_version)"
Write-Host "  detector_loaded : $($health.detector_loaded)"
if (-not $health.detector_loaded) {
    Write-Host "  the placeholder is serving, not the model -- every number below is fake" -ForegroundColor Red
    exit 1
}

Write-Host "`n== one real request at alpha $Alpha ==" -ForegroundColor Cyan
$request = Invoke-RestMethod -Uri "$base/example?name=$Example" -TimeoutSec 10
$request.alpha = $Alpha
$result = Invoke-RestMethod -Uri "$base/analyze" -Method Post `
    -ContentType "application/json" `
    -Body ($request | ConvertTo-Json -Depth 8) -TimeoutSec 120

$spans = @($result.spans)
$n = $spans.Count
Write-Host ("  {0} passages marked, {1:F0} ms, guarantee applies: {2}" -f `
    $n, $result.latency_ms, $result.guarantee_applies)

if ($n -eq 0) {
    Write-Host "  nothing was marked, so there is nothing to check the layers against." -ForegroundColor Yellow
    Write-Host "  Not a failure by itself -- try another example." -ForegroundColor Yellow
    exit 0
}

Write-Host "`n== C1, the detector ==" -ForegroundColor Cyan
$withRaw = @($spans | Where-Object { $null -ne $_.span_score }).Count
$withTokens = @($spans | Where-Object { $null -ne $_.token_probs }).Count
Show-Line "span score" ($withRaw -eq $n) "$withRaw of $n"
Show-Line "per-token probabilities" ($withTokens -gt 0) "$withTokens of $n"
if ($withRaw -ne $n) { $failed += "C1" }

Write-Host "`n== C2, calibration and the guarantee ==" -ForegroundColor Cyan
$withCal = @($spans | Where-Object { $null -ne $_.calibrated_score }).Count
$withDecision = @($spans | Where-Object { $_.conformal_decision }).Count
$withSet = @($spans | Where-Object { $null -ne $_.prediction_set }).Count
$empty = @($spans | Where-Object { $null -ne $_.prediction_set -and @($_.prediction_set).Count -eq 0 }).Count
$check = $result.distribution_check
Show-Line "corrected score" ($withCal -eq $n) "$withCal of $n"
Show-Line "flag / not sure / in" ($withDecision -eq $n) "$withDecision of $n"
Show-Line "prediction set" ($withSet -eq $n) "$withSet of $n, $empty of them empty"
Show-Line "distribution check" ($check -and $check.checked) $(
    if ($check -and $check.checked) {
        "p {0:F4}, against {1:N0} reference texts" -f $check.p_value, $check.n_reference
    } else { "no reference loaded" }
)
if ($withCal -ne $n -or $withDecision -ne $n -or $withSet -ne $n) { $failed += "C2" }
if (-not ($check -and $check.checked)) { $failed += "C2 reference" }

# The set is the reason this field exists: conformal_decision folds the empty
# set and the two-label set both onto "abstain", and they are opposite states.
if ($empty -gt 0) {
    Write-Host "  $empty passage(s) came back with an EMPTY set: at this setting" -ForegroundColor Yellow
    Write-Host "  the system will not stand behind either answer for them." -ForegroundColor Yellow
}

Write-Host "`n== C3, what kind of mistake ==" -ForegroundColor Cyan
$withType = @($spans | Where-Object { $_.error_type -and $_.error_type -ne "unknown" }).Count
$withWhy = @($spans | Where-Object { $_.explanation }).Count
Show-Line "error type" ($withType -gt 0) "$withType of $n"
Show-Line "written reason" ($withWhy -gt 0) "$withWhy of $n"
if ($withType -eq 0 -and $withWhy -eq 0) {
    Write-Host "  C3 answered nothing on every span. Either its artefact is not on" -ForegroundColor Red
    Write-Host "  disk or the path resolved somewhere else. Check results\c3\." -ForegroundColor Red
    $failed += "C3"
}

Write-Host "`n== C4, where to check in the document ==" -ForegroundColor Cyan
$withEvidence = @($spans | Where-Object { $_.evidence_sentence }).Count
Show-Line "the check ran" ([bool]$result.attribution_checked) "attribution_checked = $($result.attribution_checked)"
Show-Line "cited a sentence" ($withEvidence -ge 0) "$withEvidence of $n"
if (-not $result.attribution_checked) {
    Write-Host "  C4 is not attached. Its artefact is missing or its path is wrong." -ForegroundColor Red
    $failed += "C4"
} elseif ($withEvidence -eq 0) {
    Write-Host "  The check ran and cited nothing, which is a result rather than a" -ForegroundColor Yellow
    Write-Host "  fault: it abstains on about half of all spans at its calibrated" -ForegroundColor Yellow
    Write-Host "  threshold, because most flagged passages are things the document" -ForegroundColor Yellow
    Write-Host "  never mentions and there is no sentence to point at. To see it" -ForegroundColor Yellow
    Write-Host "  cite one, use a conflict-style input rather than a made-up one." -ForegroundColor Yellow
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "== all four components answered ==" -ForegroundColor Green
    exit 0
}
Write-Host ("== silent: {0} ==" -f ($failed -join ", ")) -ForegroundColor Red
Write-Host "Nothing on the screen would look wrong. That is the point of this check."
exit 1
