# Replaces the CPU-only torch build with the CUDA 12.6 build.
# Roughly a 2.5 GB download. Run:
#   powershell -ExecutionPolicy Bypass -File scripts\install_torch_cuda.ps1
#
# CUDA 12.6 chosen deliberately: the installed driver (610.88, CUDA 13.3 capable)
# is newer than 12.6 and CUDA is backward compatible, so 12.6 wheels run fine and
# have the widest ecosystem support.

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = "$repo\.venv\Scripts\python.exe"

Write-Host "Removing the CPU build first so pip does not keep it."
& $py -m pip uninstall -y torch

Write-Host ""
Write-Host "Installing torch (CUDA 12.6)."
# torchvision is not installed: this project is text-only and it would add
# several hundred MB for nothing.
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu126

Write-Host ""
Write-Host "=============== VERIFY ==============="
& $py -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('vram GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1) if torch.cuda.is_available() else 0)"
