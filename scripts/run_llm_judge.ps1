# Run the LLM judge over a stratified sample of the RAGTruth test split.
#
# Network-bound, not GPU-bound, so it can run alongside the HHEM job.
#
# 6 seconds between calls is 10 requests per minute, chosen to sit under the free
# tier rather than to finish fast. Every judgement is cached under a hash of its
# prompt, so if the quota cuts the run short, re-running it costs nothing for the
# records already done and picks up where it stopped.

$ErrorActionPreference = "Stop"
$repo = "D:\SLIIT\Research\trustrag"
$python = Join-Path $repo ".venv\Scripts\python.exe"

$argList = @(
    "-m", "src.c1_detector.llm_judge",
    "--sample", "300",
    "--sleep", "6",
    "--out-dir", "results/llm_judge"
)

$proc = Start-Process -FilePath $python -ArgumentList $argList `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $repo "results\_llm_judge.log") `
    -RedirectStandardError (Join-Path $repo "results\_llm_judge.err.log")

$proc.Id | Out-File -FilePath (Join-Path $repo "results\_llm_judge.pid") -Encoding ascii
Write-Output "started pid $($proc.Id) at $(Get-Date -Format 'HH:mm:ss')"
