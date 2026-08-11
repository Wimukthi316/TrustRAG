# Installs frontend dependencies.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\setup_frontend.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

# winget installs Node here; a shell opened before the install won't have it on PATH.
$env:Path = "C:\Program Files\nodejs;" + $env:Path

Set-Location (Join-Path $repo "frontend")
Write-Host "node $(node --version) / npm $(npm --version)"

# NODE_ENV=production in the user environment makes npm skip devDependencies,
# which silently breaks the build (no vite, no tailwind). Force them in.
$env:NODE_ENV = "development"
npm install --include=dev

Write-Host "Frontend dependencies installed."
