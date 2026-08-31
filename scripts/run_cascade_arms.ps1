<#
.SYNOPSIS
    Run the L0 / L1 / L2 cascade arms back to back - the RQ4 result.

.DESCRIPTION
    Three arms differing in one variable only: how much of the supervisor's
    reply the model is allowed to see when it retries.

      L0  verdict only ("this is wrong, try again")
      L1  verdict + pointer  (names the first bad step)
      L2  verdict + pointer + correction

    All three reject the identical set of attempt-1 answers, because those
    verdicts replay from `outputs/verdict_cache.jsonl` rather than being
    re-asked. That is what makes the comparison valid: llama.cpp is not
    bit-reproducible, so without the cache the arms would differ by *which*
    questions were retried as well as by hint content, and the two effects
    could not be separated.

    Each arm takes ~5 hours (measured: ~48 s per escalated question, 396 of
    1,319 escalated at a 30% rate). Run detached - see -Detached below - since
    any runner with a timeout shorter than the arm will kill it partway.

    Interruptible and safe to re-run: --resume skips ids already written, and
    re-judging a cached verdict costs nothing.

.EXAMPLE
    # start the verifier first, then:
    .\scripts\run_cascade_arms.ps1 -Detached
    Get-Content outputs\cascade_arms.log -Wait
#>
[CmdletBinding()]
param(
    # Comma-separated rather than [int[]]: relaunching this script detached
    # goes through `powershell -File`, which binds every argument as a string.
    # An array parameter receiving "0,1,2" there collapses to the integer 12,
    # and argparse rejects it -- which is how this was caught, but only because
    # the loop below refuses to continue past a failed arm.
    [string]$Levels = '0,1,2',
    [double]$EscalationRate = 0.3,
    [string]$Gate = 'disagreement',
    [string]$Provider = 'llamacpp',
    [string]$ServerUrl = 'http://127.0.0.1:8080',
    [switch]$Detached,
    [string]$LogFile = 'outputs/cascade_arms.log'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'

if ($Detached) {
    # Re-invoke this same script without -Detached, in its own process.
    New-Item -ItemType Directory -Force -Path (Join-Path $repo (Split-Path -Parent $LogFile)) | Out-Null
    $inner = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
        '-Levels', $Levels, '-EscalationRate', $EscalationRate,
        '-Gate', $Gate, '-Provider', $Provider, '-ServerUrl', $ServerUrl
    )
    $proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $inner `
        -RedirectStandardOutput (Join-Path $repo $LogFile) `
        -RedirectStandardError (Join-Path $repo "$LogFile.err") `
        -WindowStyle Hidden -PassThru
    Write-Host "Detached: pid $($proc.Id), logging to $LogFile" -ForegroundColor Green
    Write-Host "Stop it with: Stop-Process -Id $($proc.Id)"
    exit 0
}

# Fail before spending hours if the verifier is not actually answering.
try {
    Invoke-RestMethod -Uri "$ServerUrl/v1/models" -TimeoutSec 10 | Out-Null
} catch {
    throw "No verifier at $ServerUrl. Start it with scripts\serve_verifier.ps1 -Detached."
}

$bank = @(
    'outputs/predictions/03_answers_finetuned_try2.jsonl',
    'outputs/predictions/04_answers_finetuned_try3.jsonl'
)

$levelValues = $Levels -split ',' | ForEach-Object { [int]$_.Trim() }
foreach ($level in $levelValues) {
    Write-Host "=== L$level  ($(Get-Date -Format o)) ===" -ForegroundColor Cyan
    & $python (Join-Path $repo 'supervise.py') `
        --provider $Provider `
        --supervisor-url $ServerUrl `
        --gate $Gate `
        --escalation-rate $EscalationRate `
        --feedback-level $level `
        --resume `
        --sample-bank $bank[0] `
        --sample-bank $bank[1]
    if ($LASTEXITCODE -ne 0) {
        # Stop rather than continue: a dead verifier would otherwise turn the
        # remaining arms into silent accept-everything runs.
        throw "L$level exited $LASTEXITCODE - stopping before the remaining arms."
    }
    Write-Host "=== L$level done ($(Get-Date -Format o)) ===" -ForegroundColor Green
}

Write-Host "All arms complete." -ForegroundColor Green
