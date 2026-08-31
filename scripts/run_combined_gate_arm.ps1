<#
.SYNOPSIS
    L2 cascade using the combined gate (agreement, ties broken by confidence).

.DESCRIPTION
    Same feedback level, same escalation budget and same supervisor as the L2
    arm in dossier 5.7 -- the only thing that changes is which 396 questions
    get sent up. That makes it a clean paired test of the gate itself.

    The combined gate catches 325 of the 582 errors at a 30% budget against
    disagreement's 315, and is far better at small budgets (125 vs 105 at 10%).
    Whether that converts into accuracy depends on the supervisor actually
    repairing the extra errors it now sees, which is what this run measures.

    Attempt-1 verdicts replay free from outputs/verdict_cache.jsonl for every
    question, since the verdict-only pass covered all 1,319. Only retry
    judgments cost new calls.

    Interruptible: re-run with the same arguments to resume.
#>
[CmdletBinding()]
param(
    [ValidateSet('glm', 'glm-reap', 'qwen')]
    [string]$Model = 'glm',
    [string]$ServerUrl = 'http://127.0.0.1:8080',
    [double]$Rate = 0.30,
    [int]$Level = 2,
    # Qwen averages 97 output tokens against GLM's 57 and truncated 5 of 30
    # replies at the 200-token default. A truncated reply becomes an error and
    # an error becomes an accept, so a tight budget silently costs recall.
    [int]$MaxOutputTokens = 512
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'
$tag = [int]($Rate * 100)
$names = @{
    'glm'      = 'GLM-4.7-Flash-UD-Q4_K_XL'
    'glm-reap' = 'GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS'
    'qwen'     = 'Qwen3.5-9B-UD-Q4_K_XL'
}
$name = $names[$Model]
# Named for what the run is, not for its parameters -- see
# outputs/predictions/README.md. The two reported configurations get their
# reported names so --resume finds the existing files.
$known = @{
    'glm-30-2'  = '14_pipeline_glm30b_hint_full_smart_gate.jsonl'
    'qwen-30-2' = '15_pipeline_qwen9b_BEST_RESULT.jsonl'
}
$key = "$Model-$tag-$Level"
$file = if ($known.ContainsKey($key)) { $known[$key] }
        else { "pipeline_${Model}_hint_level${Level}_smart_gate_${tag}pct.jsonl" }
$out = Join-Path $repo "outputs\predictions\$file"

# The model id goes into the verdict-cache key, so a mismatch between what is
# requested and what the server actually holds writes one judge's verdicts
# under another's and silently corrupts the cache every arm replays from.
try {
    $served = (Invoke-RestMethod -Uri "$ServerUrl/v1/models" -TimeoutSec 10).models[0].name
} catch {
    throw "Verifier not reachable at $ServerUrl. Start it with: .\scripts\serve_verifier.ps1 -Model $Model"
}
if ($served -ne $name) {
    throw "Server is serving '$served' but -Model $Model expects '$name'. Restart the server, or pass the matching -Model."
}
$env:SUPERVISOR_MAX_OUTPUT_TOKENS = "$MaxOutputTokens"

$conf = Join-Path $repo 'outputs\predictions\05_confidence_for_try1.jsonl'
if (-not (Test-Path $conf)) {
    throw "Confidence scores missing: $conf. Run experiments/score_confidence.py first."
}

Write-Host "=== L$Level arm, COMBINED gate @ $($Rate.ToString('P0')) ===" -ForegroundColor Cyan
Write-Host "  judge:  $served"
Write-Host "  budget: $MaxOutputTokens output tokens"
Write-Host "  output: $out"
Write-Host ""

& $python (Join-Path $repo 'supervise.py') `
    --provider llamacpp --gate combined --escalation-rate $Rate `
    --supervisor-model $name `
    --feedback-level $Level --resume --output $out --confidence $conf `
    --sample-bank (Join-Path $repo 'outputs\predictions\03_answers_finetuned_try2.jsonl') `
    --sample-bank (Join-Path $repo 'outputs\predictions\04_answers_finetuned_try3.jsonl')

if ($LASTEXITCODE -ne 0) { throw "Arm exited with code $LASTEXITCODE" }
Write-Host ""
Write-Host "Done." -ForegroundColor Green
