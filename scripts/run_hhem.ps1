# Run HHEM-2.1-Open over C1's calibration split and the RAGTruth test split.
#
# Downloads roughly 1.5 GB from the Hub on first use and runs the model's own
# code via trust_remote_code, which is what its published interface requires.
#
# The tool call that launches this returns immediately; poll results/_hhem.log.

$ErrorActionPreference = "Stop"
$repo = "D:\SLIIT\Research\trustrag"
$python = Join-Path $repo ".venv\Scripts\python.exe"

$argList = @(
    "-m", "src.c1_detector.hhem_baseline",
    "--batch-size", "8",
    "--out-dir", "results/hhem"
)

$proc = Start-Process -FilePath $python -ArgumentList $argList `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $repo "results\_hhem.log") `
    -RedirectStandardError (Join-Path $repo "results\_hhem.err.log")

$proc.Id | Out-File -FilePath (Join-Path $repo "results\_hhem.pid") -Encoding ascii
Write-Output "started pid $($proc.Id) at $(Get-Date -Format 'HH:mm:ss')"
