param(
    [ValidateSet('Prepare', 'Update', 'Promote')]
    [string]$Mode = 'Prepare'
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$depsDir = Join-Path $repoRoot '.deps'
New-Item -ItemType Directory -Force -Path $depsDir | Out-Null

$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    throw 'Docker CLI was not found. Install Docker Desktop, then rerun this command.'
}

Write-Host "Running wheelhouse mode '$Mode' in Python 3.11.12 Linux."
Write-Host "Persistent wheels and pip cache: $depsDir"

docker run --rm `
    --mount "type=bind,source=$repoRoot,target=/repo" `
    --mount "type=bind,source=$depsDir,target=/deps" `
    --env PIP_CACHE_DIR=/deps/pip-cache `
    --workdir /repo `
    python:3.11.12-slim `
    python /repo/scripts/wheelhouse.py $Mode

if ($LASTEXITCODE -ne 0) {
    throw "Wheelhouse preparation failed with exit code $LASTEXITCODE."
}
