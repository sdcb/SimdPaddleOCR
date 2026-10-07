# Alternating A/B of one harness binary across engines (default: vulkan vs RapidOcrNet CUDA).
# Needs a build with -p:EnableRapidOcr=true and the CUDA runtime from rapid-cuda-runtime.ps1.
# Writes bench-out/<tag>/<prefix>-<engine>-<model>-r<round>.json and summarizes them.
param(
    [string[]]$models = @("tiny", "small", "medium"),
    [int]$rounds = 3,
    [string]$tag = "rapid",
    [string]$label = "",
    [int]$workers = 4,
    [int]$count = 0,
    [int]$warmup = 1,
    [string[]]$engines = @("vulkan", "rapid"),
    [string]$extra = "",
    [string]$input = "",
    [switch]$skipSummary
)
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$exe = "$root\test\Sdcb.SimdPaddleOCR.Tests\bin\Release\net10.0\Sdcb.SimdPaddleOCR.Tests.exe"
# ONNX Runtime >= 1.23 resolves CUDA/cuDNN at load time, so they have to be on PATH.
$runtime = "$root\bench-out\rapid\cuda-runtime"
if (Test-Path $runtime) { $env:PATH = "$runtime;$env:PATH" }
$dir = "$root\bench-out\$tag"
if (!$input) { $input = "$root\dataset" }
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$prefix = if ($label) { "$tag-$label" } else { $tag }
$shared = @("--workers", "$workers", "--benchmark-kind", "simd", "--warmup", "$warmup", "--input", $input)
if ($count -gt 0) { $shared += @("--count", "$count") }
if ($extra) { $shared += $extra.Split(" ") }

$files = @()
for ($r = 1; $r -le $rounds; $r++) {
    foreach ($m in $models) {
        foreach ($e in $engines) {
            $out = "$dir\$prefix-$e-$m-r$r.json"
            $a = $shared + @("--model", $m, "--engine", $e, "--case-id", "$m-$e", "--replica", "$r", "--out", $out)
            Write-Host ":: [$prefix r$r] $e $m"
            & $exe @a 2>&1 | Select-String "median|accuracy|loaded" | Select-Object -Last 3 |
                ForEach-Object { "   $($_.Line.Trim())" }
            $files += $out
        }
    }
}
if ($skipSummary) { return }
& $exe --summarize @files --input $input --out-md "$dir\$prefix-summarize.md"
