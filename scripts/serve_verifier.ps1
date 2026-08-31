<#
.SYNOPSIS
    Serve a GGUF supervisor model through llama-server for `supervise.py --provider llamacpp`.

.DESCRIPTION
    The supervisor runs in its own process rather than inside the thesis venv.
    That keeps the working transformers/torch/peft stack for Gemma 4 untouched,
    lets a 30B MoE be swapped for a 9B by changing one argument instead of any
    code, and gives access to GGUF quantisation and MoE-aware CPU offload that
    bitsandbytes cannot provide.

    Fitting GLM-4.7-Flash beside Gemma on a 16 GB card relies on -NCpuMoe:
    it parks whole expert tensors in system RAM while attention stays on the
    GPU. Only ~3B of the model's 30B parameters are active per token, so the
    experts that get read are few and the transfer cost stays small. Raise the
    value if you see an OOM, lower it for more speed.

.EXAMPLE
    .\scripts\serve_verifier.ps1 -Model glm -Download
    .\scripts\serve_verifier.ps1 -Model qwen
    .\scripts\serve_verifier.ps1 -Model glm -NCpuMoe 28   # more offload, less VRAM
#>
[CmdletBinding()]
param(
    [ValidateSet('glm', 'glm-reap', 'qwen')]
    [string]$Model = 'glm',

    # MoE expert layers kept in system RAM. Higher = less VRAM, slower.
    [int]$NCpuMoe = -1,

    # Transformer layers on the GPU. 999 = all of them; 0 = pure CPU. Lower this
    # when something else is using the card -- a sample-bank generation run
    # holds ~8.7 GB, which does not leave room for the default placement.
    [int]$NGpuLayers = 999,

    [int]$Port = 8080,

    # 4096 is ample: a judging prompt is ~350 tokens and the reply is capped at
    # 200. A larger context would reserve KV cache VRAM for nothing.
    [int]$Ctx = 4096,

    [string]$ModelDir = "$env:USERPROFILE\.cache\llama-gguf",

    [switch]$Download,

    # Run the server in its own process and return immediately, logging to
    # -LogFile. Without this the server lives and dies with the shell that
    # started it, which is fine for an interactive window and fatal under any
    # runner that imposes a timeout -- a cascade arm takes ~4 hours and an
    # interactive-length timeout would kill the verifier mid-run.
    [switch]$Detached,

    [string]$LogFile = "outputs/llama-server.log"
)

$ErrorActionPreference = 'Stop'

# Repo, exact file, served name, default expert offload. The file name is
# pinned rather than globbed: the quantisation is part of the experimental
# record, and "whatever matched the pattern today" is not reproducible.
$presets = @{
    'glm' = @{
        Repo    = 'unsloth/GLM-4.7-Flash-GGUF'
        File    = 'GLM-4.7-Flash-UD-Q4_K_XL.gguf'
        Bytes   = 17520169312
        Name    = 'GLM-4.7-Flash-UD-Q4_K_XL'
        CpuMoe  = 22
        Note    = '30B total / 3B active, ~17.5 GB - primary supervisor'
    }
    'glm-reap' = @{
        Repo    = 'unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF'
        File    = 'GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS.gguf'
        Bytes   = 0
        Name    = 'GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS'
        CpuMoe  = 0
        Note    = 'expert-pruned to 23B, ~12.6 GB - fits fully in VRAM, use if glm is too slow'
    }
    'qwen' = @{
        Repo    = 'unsloth/Qwen3.5-9B-GGUF'
        File    = 'Qwen3.5-9B-UD-Q4_K_XL.gguf'
        Bytes   = 5966095584
        Name    = 'Qwen3.5-9B-UD-Q4_K_XL'
        CpuMoe  = 0
        Note    = '5.6 GB - middle rung of the verifier ladder'
    }
}

$preset = $presets[$Model]
if ($NCpuMoe -lt 0) { $NCpuMoe = $preset.CpuMoe }

Write-Host "Model:  $($preset.Name)" -ForegroundColor Cyan
Write-Host "        $($preset.Note)"

$target = Join-Path $ModelDir $Model

if ($Download) {
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $dest = Join-Path $target $preset.File
    # curl against the resolve URL, not `hf download`: the hf CLI stalled at
    # zero bytes with no error in this environment, and curl's -C - gives a
    # resumable transfer, which matters for a 17.5 GB file.
    $url = "https://huggingface.co/$($preset.Repo)/resolve/main/$($preset.File)"
    Write-Host "Downloading $($preset.File) -> $target" -ForegroundColor Cyan
    Write-Host "(17.5 GB for glm; resumable, safe to re-run if interrupted)"
    & curl.exe -L -C - --retry 5 --retry-delay 5 -o $dest $url
    if ($LASTEXITCODE -ne 0) { throw "download failed" }

    if ($preset.Bytes -gt 0) {
        $actual = (Get-Item $dest).Length
        if ($actual -ne $preset.Bytes) {
            throw "size mismatch for $($preset.File): got $actual, expected $($preset.Bytes)"
        }
        Write-Host "Verified: $actual bytes" -ForegroundColor Green
    }
}

if (-not (Test-Path $target)) {
    throw "No weights at $target. Re-run with -Download."
}

# Large GGUFs ship split; llama-server takes the first shard and finds the rest.
$gguf = Get-ChildItem -Path $target -Recurse -Filter '*.gguf' | Sort-Object Name | Select-Object -First 1
if (-not $gguf) { throw "No .gguf under $target. Re-run with -Download." }

$server = Get-Command llama-server -ErrorAction SilentlyContinue
if (-not $server) {
    # Where this repo's install puts it, for shells that have not picked up the
    # PATH change yet.
    $fallback = 'C:\thesis\tools\llama.cpp\bin\llama-server.exe'
    if (Test-Path $fallback) {
        $server = Get-Item $fallback
    } else {
        Write-Host ""
        Write-Host "llama-server not found." -ForegroundColor Yellow
        Write-Host "Download the CUDA build matching your driver from:"
        Write-Host "  https://github.com/ggml-org/llama.cpp/releases"
        Write-Host "  (llama-*-bin-win-cuda-12.4-x64.zip plus cudart-llama-bin-win-cuda-12.4-x64.zip)"
        Write-Host "Unzip both into C:\thesis\tools\llama.cpp\bin, then re-run this script."
        exit 1
    }
}
$serverPath = if ($server.Source) { $server.Source } else { $server.FullName }

Write-Host "Weights: $($gguf.FullName)"
Write-Host ("Serving on http://127.0.0.1:{0}  (--n-gpu-layers {1}, --n-cpu-moe {2}, --ctx-size {3})" -f $Port, $NGpuLayers, $NCpuMoe, $Ctx) -ForegroundColor Green

# --jinja is required for the model's own chat template, which is what makes
# chat_template_kwargs.enable_thinking=false take effect.
$serverArgs = @(
    '--model', $gguf.FullName,
    '--alias', $preset.Name,
    '--host', '127.0.0.1',
    '--port', $Port,
    '--ctx-size', $Ctx,
    '--n-gpu-layers', $NGpuLayers,
    '--n-cpu-moe', $NCpuMoe,
    '--jinja',
    '--temp', '0',
    '--parallel', '1'
)

if ($Detached) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null
    $proc = Start-Process -FilePath $serverPath -ArgumentList $serverArgs `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" `
        -WindowStyle Hidden -PassThru
    Write-Host "Detached: pid $($proc.Id), logging to $LogFile" -ForegroundColor Green
    Write-Host "Stop it with: Stop-Process -Id $($proc.Id)"
    exit 0
}

Write-Host "Leave this window open; run supervise.py in another one." -ForegroundColor Green
Write-Host ""
& $serverPath @serverArgs
