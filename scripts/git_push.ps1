# Adds the GitHub remote (if missing) and pushes main.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\git_push.ps1

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$url = "https://github.com/Wimukthi316/TrustrRAG.git"

# Safety: refuse to push if .env somehow got staged.
$tracked = git ls-files ".env"
if ($tracked) {
    Write-Error "SECRET LEAK: .env is tracked by git. Run 'git rm --cached .env' before pushing."
    exit 1
}

if (git remote | Select-String -Quiet "^origin$") {
    git remote set-url origin $url
} else {
    git remote add origin $url
}

Write-Host "remote: $(git remote get-url origin)"
Write-Host ""
git push -u origin main
Write-Host ""
Write-Host "exit code: $LASTEXITCODE"
