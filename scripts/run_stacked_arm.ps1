<#
.SYNOPSIS
    L2 cascade at the disagreement gate's natural threshold (37.3%).

.DESCRIPTION
    The 3-sample disagreement gate takes only three values: 0 (all agree),
    1/3 (two agree), 2/3 (all differ). 492 of the 1,319 test questions sit at
    2/3, which is 37.3% -- so a 30% escalation rate cuts *inside* a tie group
    and picks 396 of those 492 arbitrarily. This arm escalates the whole group,
    which is the only threshold the gate can actually express.

    It also supplies the missing 96 rows needed to evaluate the stacked
    configuration (vote first, escalate only the split votes) at that
    threshold, and re-runs 396 sampled retries as an independent check on
    run-to-run variance.

    Attempt-1 verdicts for the 396 already-judged questions replay free from
    outputs/verdict_cache.jsonl, so this costs ~96 new judgments, not 492.

    Interruptible: re-run with the same arguments to resume.
#>
[CmdletBinding()]
param(
    [string]$ServerUrl = 'http://127.0.0.1:8080',
    [double]$Rate = 0.373,
    [int]$Level = 2
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'
# Named for what the run is, not for its parameters -- see
# outputs/predictions/README.md. Level 2 is the configuration that was reported.
$file = if ($Level -eq 2) { '13_pipeline_glm30b_hint_full_more_escalation.jsonl' }
        else { "13_pipeline_glm30b_hint_level${Level}_more_escalation.jsonl" }
$out = Join-Path $repo "outputs\predictions\$file"

# A dead verifier turns the whole run into a silent accept-everything pass,
# which looks like a completed experiment and is worthless. Check first.
try {
    Invoke-RestMethod -Uri "$ServerUrl/v1/models" -TimeoutSec 10 | Out-Null
} catch {
    throw "Verifier not reachable at $ServerUrl. Start it with: .\scripts\serve_verifier.ps1 -Model glm -NCpuMoe 38 -Detached"
}

Write-Host "=== L$Level arm @ $($Rate.ToString('P1')) escalation ===" -ForegroundColor Cyan
Write-Host "Output: $out"
Write-Host ""

& $python (Join-Path $repo 'supervise.py') `
    --provider llamacpp --gate disagreement --escalation-rate $Rate `
    --feedback-level $Level --resume --output $out `
    --sample-bank (Join-Path $repo 'outputs\predictions\03_answers_finetuned_try2.jsonl') `
    --sample-bank (Join-Path $repo 'outputs\predictions\04_answers_finetuned_try3.jsonl')

if ($LASTEXITCODE -ne 0) { throw "Arm exited with code $LASTEXITCODE" }
Write-Host ""
Write-Host "Done. Analyse with scripts/analyze_all.ps1" -ForegroundColor Green
