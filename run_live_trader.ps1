<#
.SYNOPSIS
    Launches the Gotti Live News Trader Service.
.DESCRIPTION
    Runs News_LiquiditySweep and MarketReactor (Fade Mode) continuously with
    Lumibot and Alpaca, ingesting live feedback from MySQL signals.
.EXAMPLE
    .\run_live_trader.ps1 -Mode paper
    .\run_live_trader.ps1 -DryRun
    .\run_live_trader.ps1 -Once
#>
param (
    [ValidateSet("paper", "live")]
    [string]$Mode = "paper",

    [switch]$DryRun,

    [int]$PollInterval = 20,

    [switch]$Once,

    [string]$Strategies = "News_LiquiditySweep,MarketReactor_Fade"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$Arguments = @("services\live_news_trader_service.py", "--mode", $Mode, "--poll-interval", $PollInterval, "--strategies", $Strategies)

if ($DryRun) {
    $Arguments += "--dry-run"
}

if ($Once) {
    $Arguments += "--once"
}

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host " Starting Gotti Live News Trader Service" -ForegroundColor Green
Write-Host " Mode: $Mode | Poll Interval: ${PollInterval}s | Strategies: $Strategies" -ForegroundColor Yellow
Write-Host "=========================================================================" -ForegroundColor Cyan

& $PythonExe $Arguments
