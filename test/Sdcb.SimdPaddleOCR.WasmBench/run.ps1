<#
.SYNOPSIS
构建并启动浏览器 WebAssembly 性能与方向验证工具。
.PARAMETER Orientation
改为运行四方向模型及完整 OCR 的数值、文本、坐标验证，并保存方向报告。
.PARAMETER BrowserNoSandbox
仅在受限环境中的本机可信测试启用，向测试浏览器传入 --no-sandbox；默认不启用。
#>
[CmdletBinding()]
param(
    [int]$Workers = 4,
    [string]$Model = "tiny",
    [int]$Count = 0,
    [string]$Out = "",
    [switch]$Open,
    [switch]$NoBuild,
    [switch]$NoAot,
    [switch]$Orientation,
    [switch]$BrowserNoSandbox,
    # Runtime environment variables, e.g. -Env PPOCR_DET_THREADS=4,PPOCR_NHWC=0
    [string[]]$Env = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = $ScriptDir

if (-not $NoBuild) {
    Write-Host "[WasmBench] Building and publishing WebAssembly app (AOT: $(-not $NoAot))..." -ForegroundColor Cyan
    $PublishArgs = @($ProjectDir, "-c", "Release", "--nologo")
    if ($NoAot) {
        $PublishArgs += @("-p:RunAOTCompilation=false")
    }
    dotnet publish @PublishArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to publish WasmBench project."
    }
}

$ServerScript = Join-Path $ProjectDir "server.mjs"
$NodeArgs = @($ServerScript, "--workers", $Workers, "--model", $Model)
if ($Orientation) { $NodeArgs += "--orientation" }
if ($BrowserNoSandbox) { $NodeArgs += "--browser-no-sandbox" }
if ($Count -gt 0) {
    $NodeArgs += @("--count", $Count)
}
if ($Out -ne "") {
    $NodeArgs += @("--out", $Out)
}
if ($Open) {
    $NodeArgs += @("--open")
}
foreach ($pair in $Env) {
    $NodeArgs += @("--env", $pair)
}

Write-Host "[WasmBench] Launching benchmark harness..." -ForegroundColor Cyan
& node @NodeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark execution failed with exit code $LASTEXITCODE"
}
