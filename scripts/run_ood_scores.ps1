# Re-run the RAGBench evaluation with --dump-scores, in the background.
#
# Output goes to results/ood/ragbench-scores, NOT results/ood/ragbench. The
# original directory holds the numbers already reported, and this run differs
# from it only by the new flag, so writing over it would risk the record for no
# gain. Comparing the two afterwards is free and doubles as a reproducibility
# check on the whole inference path.
#
# ~1h20m on the RTX 3050 at batch 4, peak 1.6 GB of 6 GB. The tool call that
# launches this returns immediately; poll results/_ood_rerun.log.

$ErrorActionPreference = "Stop"
$repo = "D:\SLIIT\Research\trustrag"
$python = Join-Path $repo ".venv\Scripts\python.exe"

$argList = @(
    "-m", "src.c1_detector.evaluate_ood",
    "--checkpoint", "results/c1/modernbert-base/best",
    "--batch-size", "4",
    "--out-dir", "results/ood/ragbench-scores",
    "--dump-scores"
)

$proc = Start-Process -FilePath $python -ArgumentList $argList `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $repo "results\_ood_rerun.log") `
    -RedirectStandardError (Join-Path $repo "results\_ood_rerun.err.log")

$proc.Id | Out-File -FilePath (Join-Path $repo "results\_ood_rerun.pid") -Encoding ascii
Write-Output "started pid $($proc.Id) at $(Get-Date -Format 'HH:mm:ss')"
Write-Output "poll: results\_ood_rerun.log"
