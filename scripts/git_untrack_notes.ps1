# Stops tracking the personal notes and the assistant context file, then pushes.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\git_untrack_notes.ps1

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Remove-Item "$repo\_*.log" -ErrorAction SilentlyContinue

git rm -r --cached notes --quiet --ignore-unmatch
git rm --cached CLAUDE.md --quiet --ignore-unmatch

git add -A
git commit -m "Remove personal working notes from version control"
git push origin main

Write-Host ""
Write-Host "push exit: $LASTEXITCODE"
Write-Host "Tracked files: $((git ls-files).Count)"
Write-Host "--- anything private still tracked? ---"
$leaks = git ls-files | Select-String -Pattern "notes/|CLAUDE|\.env$"
if ($leaks) { $leaks | ForEach-Object { Write-Host "STILL TRACKED: $_" } }
else { Write-Host "none - clean" }
