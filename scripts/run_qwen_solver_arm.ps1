<#
.SYNOPSIS
    Re-run the cascade with Qwen2.5-1.5B-Instruct as the SOLVER, not the checker.

.DESCRIPTION
    Answers the one question the thesis could not otherwise answer: is the gate a
    property of our pipeline, or a property of the one model we happened to
    fine-tune? Every number in the dossier comes from a single solver, so a
    reviewer can fairly say the gate might be exploiting a quirk of Gemma.

    This runs the identical pipeline on a different base model, from a different
    family, that is already reported at 73.2% on GSM8K -- above our full
    cascade. Two things come out of it:

      1. What that model actually scores on OUR harness (4-bit, strict `#### N`
         extraction, full 1,319 test set). The published figure is 4-shot
         bfloat16 with lenient extraction and is not comparable.
      2. Whether the disagreement gate, the confidence tie-break, and the
         stacking result reproduce on a solver we did not tune anything against.

    The solver is used STOCK -- no fine-tuning, no adapter. That is deliberate:
    the claim under test is about the routing mechanism, not about training, and
    holding the solver untouched removes any suspicion that the gate was tuned
    alongside it.

    Prompting is ZERO-SHOT, unlike our Gemma baseline which is 8-shot. Measured,
    not assumed: on a 8-question smoke test the 8-shot completion-style prompt
    scored 3/8 and produced one degenerate 512-token repetition, while zero-shot
    scored 3/5 with coherent, extractable reasoning. Qwen2.5-Instruct is a chat
    model and few-shot completion prompting takes it out of distribution. This
    is consistent with the prompting sensitivity reported for GSM8K in
    arXiv:2604.07035.

    Token budget is raised to 768 (default 512). Zero-shot Qwen is verbose and
    one smoke-test answer reached 502 tokens; truncating its reasoning would
    understate it and make the comparison unfair.

.NOTES
    Phases 1-3 only (generation and confidence). The cascade itself needs
    llama-server up and is run separately -- see the end of this script.

    Resumable. Each phase counts the rows already written and restarts from
    there, so a power cut costs minutes, not hours. Never delete a partial file
    to "start clean"; that is what the offset is for.

    Runtime is dominated by phase 2: roughly 11 s per generation on an RTX 4080
    SUPER, three generations per question, 1,319 questions.
#>

[CmdletBinding()]
param(
    [string]$ModelName = 'Qwen/Qwen2.5-1.5B-Instruct',
    [int]$MaxNewTokens = 768,
    [int]$Limit = 1319,
    [switch]$SkipConfidence
)

# Deliberately NOT 'Stop'. tqdm draws its progress bar on stderr, and
# PowerShell 5.1 wraps every stderr line from a native exe in an ErrorRecord;
# under 'Stop' the first progress tick kills the run. Failures are caught by
# the explicit $LASTEXITCODE checks after each call instead, which is the
# reliable signal for a native process anyway.
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo 'venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "No venv python at $python" }

$pred = Join-Path $repo 'outputs\predictions'
New-Item -ItemType Directory -Force -Path $pred | Out-Null

$env:MAX_NEW_TOKENS = "$MaxNewTokens"

function Get-DoneCount([string]$Path) {
    # Rows already written. generate.py truncates when --offset is 0 and appends
    # otherwise, so this doubles as the resume point.
    if (-not (Test-Path $Path)) { return 0 }
    $n = (Get-Content $Path | Where-Object { $_.Trim().Length -gt 0 } | Measure-Object).Count
    return $n
}

function Invoke-Sample {
    param(
        [string]$OutFile,
        [double]$Temperature,
        [string]$RunName
    )
    $done = Get-DoneCount $OutFile
    if ($done -ge $Limit) {
        Write-Host "  [skip] $RunName already has $done rows" -ForegroundColor DarkGray
        return
    }
    if ($done -gt 0) {
        Write-Host "  [resume] $RunName from row $done" -ForegroundColor Yellow
    }
    $remaining = $Limit - $done
    & $python generate.py `
        --model-name $ModelName `
        --limit $remaining `
        --offset $done `
        --few-shot 0 `
        --temperature $Temperature `
        --max-new-tokens $MaxNewTokens `
        --run-name $RunName `
        --output $OutFile
    if ($LASTEXITCODE -ne 0) { throw "$RunName failed with exit code $LASTEXITCODE" }
}

function Invoke-Confidence {
    param([string]$Source, [string]$OutFile)
    & $python experiments/score_confidence.py `
        --model-name $ModelName `
        --adapter-dir '' `
        --predictions $Source `
        --output $OutFile `
        --resume
    if ($LASTEXITCODE -ne 0) { throw "confidence scoring failed for $Source" }
}

$try1 = Join-Path $pred '17_qwen15b_answers_try1.jsonl'
$try2 = Join-Path $pred '18_qwen15b_answers_try2.jsonl'
$try3 = Join-Path $pred '19_qwen15b_answers_try3.jsonl'
$conf1 = Join-Path $pred '20_qwen15b_confidence_try1.jsonl'
$conf2 = Join-Path $pred '21_qwen15b_confidence_try2.jsonl'
$conf3 = Join-Path $pred '22_qwen15b_confidence_try3.jsonl'

Write-Host ''
Write-Host '=== Qwen2.5-1.5B-Instruct as SOLVER: does the gate transfer? ===' -ForegroundColor Cyan
Write-Host "model      $ModelName  (stock, no adapter)"
Write-Host "prompting  zero-shot, max_new_tokens=$MaxNewTokens"
Write-Host "questions  $Limit"
Write-Host ''

$started = Get-Date

Write-Host '[1/3] Greedy attempt 1 -- this alone gives the headline number' -ForegroundColor Green
Invoke-Sample -OutFile $try1 -Temperature 0.0 -RunName 'qwen15b_try1'

Write-Host '[2/3] Two sampled attempts at temperature 0.7 -- feeds the vote and the gate' -ForegroundColor Green
Invoke-Sample -OutFile $try2 -Temperature 0.7 -RunName 'qwen15b_try2'
Invoke-Sample -OutFile $try3 -Temperature 0.7 -RunName 'qwen15b_try3'

if ($SkipConfidence) {
    Write-Host '[3/3] Confidence scoring skipped by request' -ForegroundColor DarkGray
}
else {
    Write-Host '[3/3] Confidence scoring -- one forward pass each, no regeneration' -ForegroundColor Green
    Invoke-Confidence -Source $try1 -OutFile $conf1
    Invoke-Confidence -Source $try2 -OutFile $conf2
    Invoke-Confidence -Source $try3 -OutFile $conf3
}

$elapsed = (Get-Date) - $started
Write-Host ''
Write-Host ("Done in {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Cyan
Write-Host ''
Write-Host 'Next -- the cascade itself needs the checker up:' -ForegroundColor Yellow
Write-Host '  .\scripts\serve_verifier.ps1 -Model qwen'
Write-Host '  python supervise.py --provider llamacpp --no-adapter \'
Write-Host '      --model-name Qwen/Qwen2.5-1.5B-Instruct \'
Write-Host '      --gate combined --escalation-rate 0.3 --feedback-level 2 \'
Write-Host "      --attempt1-cache $try1 \"
Write-Host "      --confidence $conf1 \"
Write-Host "      --sample-bank $try2 --sample-bank $try3 \"
Write-Host '      --output outputs/predictions/23_qwen15b_pipeline_cascade.jsonl --resume'
