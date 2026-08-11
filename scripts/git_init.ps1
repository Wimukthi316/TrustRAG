# Initialises the repository and makes the first commit.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\git_init.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Remove-Item "$repo\_*.log" -ErrorAction SilentlyContinue

if (-not (Test-Path "$repo\.git")) {
    git init
    git branch -M main
}

if (-not (git config user.name)) { git config user.name "Wimukthi316" }
if (-not (git config user.email)) { git config user.email "wimukthi316@gmail.com" }

git add -A
git commit -m "Initial scaffold: data contract, FastAPI backend, React frontend"

Write-Host ""
git log --oneline
Write-Host ""
Write-Host "Files tracked: $((git ls-files).Count)"
