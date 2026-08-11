# First commit. Safe to re-run: skips init if .git already exists.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\git_init.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Remove-Item "$repo\_*.log" -ErrorAction SilentlyContinue

if (-not (Test-Path "$repo\.git")) {
    git init
    git branch -M main
}

# Set identity locally only if it is not already configured globally.
if (-not (git config user.name)) { git config user.name "Wimukthi" }
if (-not (git config user.email)) { git config user.email "wimukthi316@gmail.com" }

git add -A
git commit -m "Scaffold TrustRAG: JSON contract, FastAPI backend, React+Vite+Tailwind frontend

- src/common/schema.py: frozen Span/AnalysisResult contract shared by C1, C2,
  the API and the UI. frontend/src/types.ts mirrors it.
- backend: FastAPI with /api/health, /api/analyze, /api/example. Detector is a
  STUB - keyword matching with placeholder scores, no model, no real numbers.
- frontend: span-highlight UI with flag/abstain/pass colouring, calibrated
  confidence on hover, alpha slider.
- tests: 10 passing, covering contract validation and API round-trip.
- CLAUDE.md, notes/STATUS.md, notes/ACCOUNTS.md.

Stack decision: React + Vite + Tailwind + FastAPI (replaces the earlier
Streamlit plan). Python 3.11, Node 24 LTS."

Write-Host ""
git log --oneline
Write-Host ""
git status --short
Write-Host ""
Write-Host "Files tracked: $((git ls-files).Count)"
