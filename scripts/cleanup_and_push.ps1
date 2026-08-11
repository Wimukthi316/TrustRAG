# Removes the one-time git helper scripts that have already served their purpose,
# then commits and pushes.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\cleanup_and_push.ps1

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# These were single-use fixes: adding the remote, untracking the notes folder,
# and repointing at the renamed repository. All done, all clutter now.
foreach ($f in @("git_push.ps1", "git_untrack_notes.ps1", "git_fix_remote.ps1")) {
    Remove-Item "$repo\scripts\$f" -Force -ErrorAction SilentlyContinue
}
Remove-Item "$repo\_*.log" -ErrorAction SilentlyContinue

if (git ls-files ".env") {
    Write-Error "SECRET LEAK: .env is tracked. Run 'git rm --cached .env' first."
    exit 1
}

git add -A
git commit -m "Correct repository name in README, drop one-time setup scripts"
git push origin main

Write-Host "push exit: $LASTEXITCODE"
Write-Host ""
Write-Host "--- scripts remaining ---"
Get-ChildItem "$repo\scripts" | Select-Object -Expand Name
