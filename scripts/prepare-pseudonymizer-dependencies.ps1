$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$depsDir = Join-Path $repoRoot '.deps'
$archive = Join-Path $depsDir 'pseudonymizer-deps-py311-cu128-v1.tar'
$image = 'persdatagate-pseudonymizer-deps:py311-cu128-v1'

New-Item -ItemType Directory -Force -Path $depsDir | Out-Null
Push-Location $repoRoot
try {

$imagePresent = $false
docker image inspect $image *> $null
if ($LASTEXITCODE -eq 0) {
    $imagePresent = $true
}

if (-not $imagePresent -and (Test-Path -LiteralPath $archive)) {
    Write-Host "Loading saved dependency image from $archive"
    docker load --input $archive
    if ($LASTEXITCODE -ne 0) { throw 'Could not load the saved dependency image.' }
    $imagePresent = $true
}

if (-not $imagePresent) {
    Write-Host 'Building the pinned pseudonymizer dependency image. This downloads CUDA wheels once.'
    docker build --file pseudonymizer/Dockerfile.dependencies --tag $image .
    if ($LASTEXITCODE -ne 0) { throw 'Could not build the dependency image.' }
}

if (-not (Test-Path -LiteralPath $archive)) {
    Write-Host "Saving dependency image to $archive"
    docker save --output $archive $image
    if ($LASTEXITCODE -ne 0) { throw 'Could not save the dependency image archive.' }
}

Write-Host "Ready: $image"
Write-Host 'Now run: docker compose up -d --build'
}
finally {
    Pop-Location
}
