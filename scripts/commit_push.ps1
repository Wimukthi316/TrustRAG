# Commit everything and push. Pass a message:
#   powershell -ExecutionPolicy Bypass -File scripts\commit_push.ps1 -m "your message"
param([Parameter(Mandatory = $true)][Alias("m")][string]$Message)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (git ls-files ".env") {
    Write-Error "SECRET LEAK: .env is tracked. Run 'git rm --cached .env' first."
    exit 1
}

git add -A
git commit -m $Message
git push origin main
Write-Host "push exit: $LASTEXITCODE"
git log --oneline -3
