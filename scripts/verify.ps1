# Full verification: backend tests + frontend production build.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\verify.ps1

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# nvm keeps an old Node (v18) on PATH ahead of the winget install; Vite needs 20+.
$env:Path = "C:\Program Files\nodejs;" + $env:Path
$env:NODE_ENV = "development"

Write-Host "=============== VERSIONS ==============="
& "$repo\.venv\Scripts\python.exe" --version
node --version
Write-Host ""

Write-Host "=============== TORCH / CUDA ==============="
& "$repo\.venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
Write-Host ""

Write-Host "=============== PYTEST ==============="
& "$repo\.venv\Scripts\python.exe" -m pytest -q
$pytestExit = $LASTEXITCODE
Write-Host ""

Write-Host "=============== FRONTEND BUILD ==============="
Set-Location "$repo\frontend"
& "C:\Program Files\nodejs\npm.cmd" run build
$buildExit = $LASTEXITCODE
Set-Location $repo
Write-Host ""

Write-Host "=============== SUMMARY ==============="
Write-Host "pytest exit code : $pytestExit  (0 = pass)"
Write-Host "vite build exit  : $buildExit  (0 = pass)"
