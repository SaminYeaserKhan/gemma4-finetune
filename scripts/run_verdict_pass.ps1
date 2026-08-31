<#
.SYNOPSIS
    Full 1,319-question verdict pass for one rung of the verifier ladder.

.DESCRIPTION
    Judges every cached attempt-1 answer and stops -- no retries, so the solver
    is never loaded. That is all RQ5 needs: precision, recall and false-reject
    rate against GSM8K ground truth.

    This script exists because the model id MUST be passed explicitly and is
    easy to forget. verdict_key hashes provider|model|system_prompt|question|
    candidate_answer, so judging with Qwen while the config still names GLM
    writes Qwen's verdicts under GLM's key and silently corrupts
    outputs/verdict_cache.jsonl -- which every cascade arm replays from, and
    which is the control that makes L0/L1/L2 comparable. The preflight was
    observed printing GLM's name while Qwen was answering, because the client
    reports the configured name rather than the served one.

    Interruptible: re-run with the same arguments to resume.

.EXAMPLE
    .\scripts\serve_verifier.ps1 -Model qwen
    .\scripts\run_verdict_pass.ps1 -Model qwen
#>
[CmdletBinding()]
param(
    [ValidateSet('glm', 'glm-reap', 'qwen')]
    [string]$Model = 'qwen',
    [string]$ServerUrl = 'http://127.0.0.1:8080',
    # Qwen truncated 5 of 30 replies at the 200-token default and 1 of 30 at
    # 512. A truncated reply becomes an error, and an error becomes an accept,
    # so a tight budget silently costs recall.
    [int]$MaxOutputTokens = 512
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'

$names = @{
    'glm'      = 'GLM-4.7-Flash-UD-Q4_K_XL'
    'glm-reap' = 'GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS'
    'qwen'     = 'Qwen3.5-9B-UD-Q4_K_XL'
}
$name = $names[$Model]

# Refuse to run against a server holding a different model: the id below goes
# into the cache key, so a mismatch here is exactly the corruption above.
try {
    $served = (Invoke-RestMethod -Uri "$ServerUrl/v1/models" -TimeoutSec 10).models[0].name
} catch {
    throw "Verifier not reachable at $ServerUrl. Start it with: .\scripts\serve_verifier.ps1 -Model $Model"
}
if ($served -ne $name) {
    throw "Server is serving '$served' but -Model $Model expects '$name'. Restart the server, or pass the matching -Model."
}

$out = Join-Path $repo "outputs\predictions\verdict_${Model}_test.jsonl"
$env:SUPERVISOR_MAX_OUTPUT_TOKENS = "$MaxOutputTokens"

Write-Host "=== Verdict pass: $name ===" -ForegroundColor Cyan
Write-Host "  server verified serving: $served"
Write-Host "  max output tokens:       $MaxOutputTokens"
Write-Host "  output:                  $out"
Write-Host ""

& $python (Join-Path $repo 'supervise.py') `
    --provider llamacpp --supervisor-model $name `
    --gate none --verdict-only --resume --output $out

if ($LASTEXITCODE -ne 0) { throw "Verdict pass exited with code $LASTEXITCODE" }
Write-Host ""
Write-Host "Done. Analyse with: analyze_supervision.py --cascade $Model=$out" -ForegroundColor Green
