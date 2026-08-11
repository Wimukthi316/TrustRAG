# The GitHub repo was renamed TrustrRAG -> TrustRAG. Point the remote at the new URL.
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Remove-Item "$repo\scripts\git_reset_clean.ps1" -Force -ErrorAction SilentlyContinue
Remove-Item "$repo\_*.log" -ErrorAction SilentlyContinue

git remote set-url origin "https://github.com/Wimukthi316/TrustRAG.git"
Write-Host "remote: $(git remote get-url origin)"

git add -A
git commit -m "Point remote at renamed repository"
git push origin main
Write-Host "push exit: $LASTEXITCODE"
