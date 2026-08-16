# Run the LLM judge over a stratified sample of the RAGTruth test split.
#
# Network-bound, not GPU-bound, so it can run alongside a GPU job.
#
# Model and pace are parameters because the free tier decides them, not us. The
# newest model is also the most contended: gemini-3.7-flash returned 429 on four
# of five records and produced one judgement in forty-five minutes. An older
# flash model is a perfectly capable judge for a yes/no plus a quote, and its
# quota is not being fought over.
#
# Every judgement is cached under a hash of its prompt, so an interrupted run
# resumes for free and a model switch simply misses the cache, which is correct:
# a different judge is a different experiment.

param(
    [string]$Model = "gemini-2.5-flash",
    [int]$Sample = 150,
    [int]$Sleep = 8
)

$ErrorActionPreference = "Stop"
$repo = "D:\SLIIT\Research\trustrag"
$python = Join-Path $repo ".venv\Scripts\python.exe"

$argList = @(
    "-m", "src.c1_detector.llm_judge",
    "--model", $Model,
    "--sample", "$Sample",
    "--sleep", "$Sleep",
    "--out-dir", "results/llm_judge"
)

$proc = Start-Process -FilePath $python -ArgumentList $argList `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $repo "results\_llm_judge.log") `
    -RedirectStandardError (Join-Path $repo "results\_llm_judge.err.log")

$proc.Id | Out-File -FilePath (Join-Path $repo "results\_llm_judge.pid") -Encoding ascii
Write-Output "started pid $($proc.Id) at $(Get-Date -Format 'HH:mm:ss') model $Model sample $Sample sleep $Sleep"
