# Rebuilds git history from scratch so no removed file survives in an old commit,
# then force-pushes. Only safe on a repo nobody else has cloned.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\git_reset_clean.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Remove-Item "$repo\_*.log" -ErrorAction SilentlyContinue
Remove-Item "$repo\.git" -Recurse -Force -ErrorAction SilentlyContinue

git init --quiet
git branch -M main
git config user.name "Wimukthi316"
git config user.email "wimukthi316@gmail.com"

git add -A

# Abort rather than publish a secret or a private note.
foreach ($bad in @(".env", "notes/STATUS.md", "notes/ACCOUNTS.md", "CLAUDE.md")) {
    if (git ls-files $bad) {
        Write-Error "Refusing to commit: $bad is staged."
        exit 1
    }
}

git commit --quiet -m "Initial scaffold: data contract, FastAPI backend, React frontend

- src/common/schema.py defines the Span and AnalysisResult contract shared by the
  detector, the calibration layer, the API and the UI.
- backend exposes /api/health, /api/analyze and /api/example. The detector is a
  placeholder until C1 is trained.
- frontend renders span highlighting with flag/abstain/pass states, calibrated
  confidence on hover, and an alpha slider.
- 10 tests covering contract validation and API round-trip."

git remote add origin "https://github.com/Wimukthi316/TrustrRAG.git"
git push --force -u origin main

Write-Host ""
Write-Host "push exit: $LASTEXITCODE"
git log --oneline
Write-Host ""
Write-Host "Tracked files: $((git ls-files).Count)"
Write-Host "--- checking nothing private survived ---"
git ls-files | Select-String -Pattern "notes/|CLAUDE|\.env$" | ForEach-Object { Write-Host "LEAK: $_" }
Write-Host "(no LEAK lines above means clean)"
