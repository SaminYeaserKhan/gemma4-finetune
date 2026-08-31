<#
.SYNOPSIS
    Bring the cascade run back up after a power cut, crash, or reboot.

.DESCRIPTION
    Run this once when the machine is back. It is idempotent: it starts only
    what is not already running, so it is safe if you are unsure whether
    anything survived.

      1. starts the verifier (llama-server) detached, if it is not up
      2. waits for it to finish loading the 17.5 GB model
      3. reports how many rows each arm has completed
      4. restarts the arms detached, with --resume

    Nothing is lost in an interruption. Rows are appended one per question as
    they finish, so only the question actually in flight is lost, and --resume
    skips every id already written. Attempt-1 verdicts replay free from
    outputs/verdict_cache.jsonl, so re-running costs no supervisor calls for
    work already judged.

    A power cut can leave a half-written final row. supervise.py repairs that
    automatically before reading (io_utils.repair_jsonl) and prints what it
    dropped -- that question is simply answered again.

.EXAMPLE
    .\scripts\resume_after_interruption.ps1
#>
[CmdletBinding()]
param(
    [string]$ServerUrl = 'http://127.0.0.1:8080',
    [int]$NCpuMoe = 38,
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

function Test-Verifier {
    try {
        Invoke-RestMethod -Uri "$ServerUrl/v1/models" -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

Write-Host '=== Resuming FYDP 3 cascade ===' -ForegroundColor Cyan

if (Test-Verifier) {
    Write-Host 'Verifier already running.' -ForegroundColor Green
} else {
    Write-Host 'Starting verifier...' -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot 'serve_verifier.ps1') -Model glm -NCpuMoe $NCpuMoe -Detached

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while (-not (Test-Verifier)) {
        if ((Get-Date) -gt $deadline) {
            throw "Verifier did not come up within $TimeoutSeconds s. Check outputs\llama-server.log."
        }
        Start-Sleep -Seconds 5
    }
    Write-Host 'Verifier ready.' -ForegroundColor Green
}

Write-Host ''
Write-Host 'Completed rows (of 1319 each):' -ForegroundColor Cyan
$arms = @('10_pipeline_glm30b_hint_none.jsonl',
          '11_pipeline_glm30b_hint_short.jsonl',
          '12_pipeline_glm30b_hint_full.jsonl')
for ($level = 0; $level -lt $arms.Count; $level++) {
    $path = Join-Path $repo "outputs\predictions\$($arms[$level])"
    if (Test-Path $path) {
        $n = (Get-Content $path -ReadCount 0).Count
        Write-Host ("  L{0}: {1}/1319" -f $level, $n)
    } else {
        Write-Host ("  L{0}: not started" -f $level)
    }
}

# The arm runner skips levels that are already complete only in the sense that
# --resume finds every id present and writes nothing new, which takes seconds.
# So restarting all three is correct regardless of where it stopped.
Write-Host ''
Write-Host 'Restarting arms...' -ForegroundColor Yellow
& (Join-Path $PSScriptRoot 'run_cascade_arms.ps1') -Detached -ServerUrl $ServerUrl

Write-Host ''
Write-Host 'Done. Watch progress with:' -ForegroundColor Green
Write-Host '  Get-Content outputs\cascade_arms.log.err -Wait'
