<#
.SYNOPSIS
    Install the pinned llama.cpp CUDA build used to host the supervisor.

.DESCRIPTION
    The build number is pinned rather than tracking "latest", because the
    verifier's behaviour is part of the experimental record. Re-running this
    script on another machine reproduces the same runtime.

    CUDA variant selection matters: 12.4 matches the CUDA 12.4 PyTorch wheels
    this project already uses and works on driver 560.x. The 13.x builds need a
    newer driver (580+) and will fail to load on the reference machine.

.EXAMPLE
    .\scripts\install_llamacpp.ps1
    .\scripts\install_llamacpp.ps1 -Build b10500 -Cuda 13.3
#>
[CmdletBinding()]
param(
    [string]$Build = 'b10453',
    [ValidateSet('12.4', '13.3')]
    [string]$Cuda = '12.4',
    [string]$InstallDir = 'C:\thesis\tools\llama.cpp',
    [switch]$SkipPath
)

$ErrorActionPreference = 'Stop'
$bin = Join-Path $InstallDir 'bin'
New-Item -ItemType Directory -Force -Path $bin | Out-Null

$base = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"
$files = @(
    "llama-$Build-bin-win-cuda-$Cuda-x64.zip",   # the binaries
    "cudart-llama-bin-win-cuda-$Cuda-x64.zip"    # the CUDA runtime DLLs they need
)

foreach ($file in $files) {
    $zip = Join-Path $InstallDir $file
    if (Test-Path $zip) {
        Write-Host "Already downloaded: $file"
    } else {
        Write-Host "Downloading $file ..." -ForegroundColor Cyan
        # curl.exe, not Invoke-WebRequest: -C - resumes a partial file, which
        # matters on a flaky connection and for these sizes (250-400 MB).
        & curl.exe -L -C - --retry 5 --retry-delay 5 -o $zip "$base/$file"
        if ($LASTEXITCODE -ne 0) { throw "download failed: $file" }
    }
    Write-Host "Extracting $file ..."
    Expand-Archive -Path $zip -DestinationPath $bin -Force
}

$exe = Join-Path $bin 'llama-server.exe'
if (-not (Test-Path $exe)) { throw "llama-server.exe not found under $bin after extraction" }

Write-Host ""
& $exe --version
Write-Host ""
& $exe --list-devices

if (-not $SkipPath) {
    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    if ($userPath -notlike "*$bin*") {
        [Environment]::SetEnvironmentVariable('PATH', "$userPath;$bin", 'User')
        Write-Host "`nAdded $bin to your user PATH (open a new terminal to pick it up)." -ForegroundColor Green
    } else {
        Write-Host "`n$bin is already on your user PATH." -ForegroundColor Green
    }
}

Write-Host "`nNext: .\scripts\serve_verifier.ps1 -Model glm -Download" -ForegroundColor Green
