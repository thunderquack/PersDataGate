$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$depsDir = Join-Path $repoRoot '.deps'
$archive = Join-Path $depsDir 'pseudonymizer-deps-py311-cu128-v1.tar'
$manifest = Join-Path $depsDir 'pseudonymizer-deps-py311-cu128-v1.json'
$partialArchive = "$archive.partial"
$image = 'persdatagate-pseudonymizer-deps:py311-cu128-v1'

$requirementsHash = (Get-FileHash (Join-Path $repoRoot 'pseudonymizer/requirements.txt') -Algorithm SHA256).Hash.ToLowerInvariant()
$dockerfileHash = (Get-FileHash (Join-Path $repoRoot 'pseudonymizer/Dockerfile.dependencies') -Algorithm SHA256).Hash.ToLowerInvariant()
$fingerprintInput = "requirements=$requirementsHash`ndockerfile=$dockerfileHash`n"
$fingerprintBytes = [System.Text.Encoding]::UTF8.GetBytes($fingerprintInput)
$fingerprint = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($fingerprintBytes)).ToLowerInvariant()

New-Item -ItemType Directory -Force -Path $depsDir | Out-Null
Push-Location $repoRoot
try {

$getImageFingerprint = {
    $label = docker image inspect --format '{{ index .Config.Labels "org.persdatagate.dependencies-fingerprint" }}' $image 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    return [string]$label
}

$imageFingerprint = & $getImageFingerprint
$imagePresent = $null -ne $imageFingerprint

if (-not $imagePresent -and (Test-Path -LiteralPath $archive)) {
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw "Dependency archive has no manifest. Preserve it and restore the tagged image manually, or remove the archive before preparing dependencies."
    }
    $savedManifest = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
    if ($savedManifest.fingerprint -ne $fingerprint) {
        throw 'Saved dependency archive is stale for the current pinned requirements or dependency Dockerfile. Bump the dependency image version.'
    }
    $actualArchiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($savedManifest.archive_sha256 -ne $actualArchiveHash) {
        throw 'Saved dependency archive checksum does not match its manifest.'
    }
    Write-Host "Loading saved dependency image from $archive"
    docker load --input $archive
    if ($LASTEXITCODE -ne 0) { throw 'Could not load the saved dependency image.' }
    $imageFingerprint = & $getImageFingerprint
    $imagePresent = $null -ne $imageFingerprint
}

if ($imagePresent -and $imageFingerprint -ne $fingerprint) {
    throw 'The tagged dependency image is stale for the current pinned requirements or dependency Dockerfile. Bump the dependency image version.'
}

if (-not $imagePresent) {
    Write-Host 'Building the pinned pseudonymizer dependency image. This downloads CUDA wheels once.'
    docker build --file pseudonymizer/Dockerfile.dependencies --build-arg "DEPS_FINGERPRINT=$fingerprint" --tag $image .
    if ($LASTEXITCODE -ne 0) { throw 'Could not build the dependency image.' }
    $imageFingerprint = & $getImageFingerprint
    if ($imageFingerprint -ne $fingerprint) { throw 'Built dependency image is missing the expected fingerprint label.' }
}

if (-not (Test-Path -LiteralPath $archive) -or -not (Test-Path -LiteralPath $manifest)) {
    Write-Host "Saving dependency image to $archive"
    Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath $partialArchive
    docker save --output $partialArchive $image
    if ($LASTEXITCODE -ne 0) { throw 'Could not save the dependency image archive.' }
    $partialHash = (Get-FileHash -LiteralPath $partialArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    Move-Item -Force -LiteralPath $partialArchive -Destination $archive
    $manifestData = [ordered]@{
        image = $image
        fingerprint = $fingerprint
        requirements_sha256 = $requirementsHash
        dockerfile_sha256 = $dockerfileHash
        archive_sha256 = $partialHash
    } | ConvertTo-Json
    $partialManifest = "$manifest.partial"
    Set-Content -LiteralPath $partialManifest -Value $manifestData -Encoding utf8
    Move-Item -Force -LiteralPath $partialManifest -Destination $manifest
}

Write-Host "Ready: $image"
Write-Host 'Now run: docker compose up -d --build'
}
finally {
    Pop-Location
}
