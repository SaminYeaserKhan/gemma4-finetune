<#
.SYNOPSIS
    Redraw the thesis diagrams, in both wordings.

.DESCRIPTION
    Two Markdown files describe the same six diagrams:

      docs/DIAGRAMS.md                               plain language, for teammates
      docs/diagrams-technical/DIAGRAMS_TECHNICAL.md  standard terms, for the paper

    Same system, same measured numbers, same output file names -- only the wording
    inside the boxes differs. Pictures go to reports/figures/ and
    reports/figures/technical/ respectively.

    The Markdown is the master copy of every diagram. These images are generated
    from it and should never be edited by hand.

.EXAMPLE
    .\scripts\render_diagrams.ps1              # both wordings
    .\scripts\render_diagrams.ps1 -Plain       # plain language only
    .\scripts\render_diagrams.ps1 -Technical   # technical only
    .\scripts\render_diagrams.ps1 -Only fig2   # one figure, both wordings
#>

[CmdletBinding()]
param(
    [string]$Only,
    [switch]$Plain,
    [switch]$Technical
)

# Not 'Stop': mermaid-cli writes progress and npm notices to stderr, and
# PowerShell 5.1 turns a native process's stderr into fatal ErrorRecords under
# 'Stop'. The exit code is checked explicitly instead.
$ErrorActionPreference = 'Continue'

if ($Plain -and $Technical) {
    throw "Pass -Plain or -Technical, not both. Omit both to render each of them."
}

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "No venv python at $python" }

$arguments = @((Join-Path $repo 'scripts\render_diagrams.py'))
if ($Only) { $arguments += @('--only', $Only) }
if ($Plain) { $arguments += @('--variant', 'plain') }
if ($Technical) { $arguments += @('--variant', 'technical') }

& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "render_diagrams.py failed with exit code $LASTEXITCODE" }
