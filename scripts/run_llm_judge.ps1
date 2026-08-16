# Run the LLM judge over a stratified sample of the RAGTruth test split.
#
# Network-bound, not GPU-bound, so it can run alongside a GPU job.
#
# Model and pace are parameters because the free tier decides them, not us, and
# picking the model is not a one-line choice:
#   gemini-3.7-flash   429 on four of five records, one judgement in 45 minutes
#   gemini-2.5-flash   404 "no longer available to new users" - ListModels still
#                      advertises generateContent for it
#   gemini-flash-latest 429 immediately
#   gemini-3.6-flash   answered 21 of 150, then 429: the daily free tier is
#                      per model, and 21 judgements spent it
#   gemini-3.5-flash   503 while busy, answers later
# Quota is per model and it resets, so the model that works is a function of the
# hour. Switching model misses the cache by design, because a different judge is
# a different experiment: partial runs are not pooled across models.
#
# Every judgement is cached under a hash of its prompt, so an interrupted run
# resumes for free and a model switch simply misses the cache, which is correct:
# a different judge is a different experiment.

param(
    [string]$Model = "gemini-3.5-flash",
    [int]$Sample = 150,
    [int]$Sleep = 6
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
