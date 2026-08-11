# Non-interactive remote check. Fails fast instead of hanging on a credential prompt.
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:GIT_TERMINAL_PROMPT = "0"
$env:GCM_INTERACTIVE = "never"

Write-Host "remote: $(git remote get-url origin 2>&1)"
Write-Host "branch: $(git rev-parse --abbrev-ref HEAD)"
Write-Host "--- ls-remote ---"
git ls-remote --heads origin 2>&1 | Select-Object -First 5
Write-Host "exit: $LASTEXITCODE"
