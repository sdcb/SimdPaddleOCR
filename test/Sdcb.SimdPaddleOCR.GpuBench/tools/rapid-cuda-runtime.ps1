# Assembles the CUDA 12 / cuDNN 9 runtime DLLs that ONNX Runtime's CUDA execution
# provider loads by name (ORT >= 1.23 no longer ships them in the NuGet package).
# Copies from any -Local directory that already has them, otherwise downloads the
# NVIDIA wheels from PyPI. Files land in -Out, which the runner prepends to PATH.
param(
    [string]$Out,
    [string[]]$Local = @(
        "C:\Users\$env:USERNAME\source\repos\PaddleSharpOCRDemo\PaddleSharpOCRDemo\bin\Debug\net48",
        "C:\Users\$env:USERNAME\source\repos\TensorSharp\ExternalProjects\cudnn\bin\x64"
    ),
    [switch]$Force
)
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
if (!$Out) { $Out = "$root\bench-out\rapid\cuda-runtime" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$runtime = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
function Missing() { @($runtime | Where-Object { !(Test-Path "$Out\$_") }) }

if ($Force) { Get-ChildItem "$Out\*.dll" -ErrorAction SilentlyContinue | Remove-Item -Force }
$todo = Missing
foreach ($dir in $Local) {
    if ($todo.Count -eq 0) { break }
    if (!(Test-Path $dir)) { continue }
    foreach ($name in $todo) {
        Get-ChildItem $dir -Recurse -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1 |
            ForEach-Object { Copy-Item $_.FullName "$Out\$name" -Force; Write-Host "copied $name from $dir" }
    }
    $todo = Missing
}
# cuDNN ships as a loader plus per-library components; take the whole family.
foreach ($dir in $Local) {
    if (Test-Path $dir) {
        Get-ChildItem $dir -Recurse -Filter "cudnn_*.dll" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName "$Out\$($_.Name)" -Force }
    }
}

if ((Missing).Count -gt 0) {
    $wheels = @{
        "cudart64_12.dll"   = "nvidia-cuda-runtime-cu12"
        "cublas64_12.dll"   = "nvidia-cublas-cu12"
        "cublasLt64_12.dll" = "nvidia-cublas-cu12"
        "cudnn64_9.dll"     = "nvidia-cudnn-cu12"
    }
    $cache = "$root\bench-out\rapid\wheels"
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $packages = @($wheels.Keys | Where-Object { !(Test-Path "$Out\$_") } | ForEach-Object { $wheels[$_] } | Select-Object -Unique)
    foreach ($package in $packages) {
        $json = Invoke-RestMethod "https://pypi.org/pypi/$package/json"
        $file = $json.releases.($json.info.version) | Where-Object { $_.filename -match 'win_amd64' } | Select-Object -First 1
        if (!$file) { throw "$package has no win_amd64 wheel" }
        $path = "$cache\$($file.filename)"
        if (!(Test-Path $path)) {
            Write-Host "downloading $($file.filename) ($([math]::Round($file.size / 1MB)) MB)"
            Invoke-WebRequest $file.url -OutFile $path
        }
        $zip = [System.IO.Compression.ZipFile]::OpenRead($path)
        foreach ($entry in $zip.Entries) {
            if ($entry.Name -notlike "*.dll") { continue }
            $wanted = $runtime -contains $entry.Name -or $entry.Name -like "cudnn_*.dll"
            if ($wanted -and !(Test-Path "$Out\$($entry.Name)")) {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, "$Out\$($entry.Name)")
                Write-Host "extracted $($entry.Name)"
            }
        }
        $zip.Dispose()
    }
}

$todo = Missing
if ($todo.Count -gt 0) { throw "CUDA runtime is still missing: $($todo -join ', ')" }
Get-ChildItem "$Out\*.dll" | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize
Write-Host "cuda runtime ready: $Out"
