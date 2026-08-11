# Creates the Python 3.11 virtual environment and installs requirements.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py311 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
if (-not (Test-Path $py311)) {
    Write-Error "Python 3.11 not found at $py311. Install it with: winget install Python.Python.3.11"
}

Write-Host "Creating .venv with $py311"
& $py311 -m venv .venv

$venvPy = Join-Path $repo ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip setuptools wheel
& $venvPy -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Verifying:"
& $venvPy -c "import torch, transformers, fastapi, netcal; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('transformers', transformers.__version__); print('fastapi', fastapi.__version__)"
