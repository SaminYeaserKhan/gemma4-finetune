<#
.SYNOPSIS
    Score the sample-bank answers for the model's own confidence.

.DESCRIPTION
    Confidence has so far been measured only for greedy attempt 1, which is
    enough to use it as a gate but not enough to compare the three sampled
    answers against each other. Scoring the other two lets the pipeline pick
    which answer to send to the supervisor, instead of always sending the
    first one -- on the 492 questions where all three samples differ, the
    first answer is right 99 times but a correct answer is present 239 times.

    One forward pass per stored answer. Nothing is regenerated.
    Interruptible: re-run to resume.
#>
[CmdletBinding()]
param([string]$AdapterDir = 'gemma4-gsm8k-final')

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'

# try2/try3 are the two extra sampled answers; 06/07 are their confidence files.
$pairs = @(
    @{ Src = '03_answers_finetuned_try2.jsonl'; Out = '06_confidence_for_try2.jsonl'; Tag = 'try 2' },
    @{ Src = '04_answers_finetuned_try3.jsonl'; Out = '07_confidence_for_try3.jsonl'; Tag = 'try 3' }
)
foreach ($pair in $pairs) {
    $tag = $pair.Tag
    $src = Join-Path $repo "outputs\predictions\$($pair.Src)"
    $out = Join-Path $repo "outputs\predictions\$($pair.Out)"
    Write-Host ""
    Write-Host "=== scoring $tag ===" -ForegroundColor Cyan
    Write-Host "  $src"
    Write-Host "  -> $out"
    & $python (Join-Path $repo 'experiments\score_confidence.py') `
        --predictions $src --output $out --adapter-dir $AdapterDir --resume
    if ($LASTEXITCODE -ne 0) { throw "scoring $tag exited with code $LASTEXITCODE" }
}

Write-Host ""
Write-Host "Both sample sets scored." -ForegroundColor Green
