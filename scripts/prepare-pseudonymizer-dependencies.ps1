param(
    [ValidateSet('Prepare', 'Update')]
    [string]$Mode = 'Prepare'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$depsDir = Join-Path $repoRoot '.deps'
$cacheDir = Join-Path $depsDir 'pip-cache'
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker CLI was not found.'
}
Write-Host "Mode: $Mode. Persistent pip cache: $cacheDir"
docker run --rm `
    --mount "type=bind,source=$repoRoot,target=/repo,readonly" `
    --mount "type=bind,source=$depsDir,target=/deps" `
    --env PIP_CACHE_DIR=/deps/pip-cache `
    python:3.11.12-slim python /repo/scripts/pip_cache.py $Mode
if ($LASTEXITCODE -ne 0) { throw "Dependency preparation failed: $LASTEXITCODE" }
