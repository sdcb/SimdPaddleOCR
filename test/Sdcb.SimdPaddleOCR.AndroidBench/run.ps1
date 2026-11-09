# One command for the phone: [build] -> [adb install] -> [push models/dataset] -> launch ->
# wait for <run>.done -> pull files/out/* into bench-out/android/. On a crash or a
# process that dies without writing .done, logcat is saved next to the log.
#
#   .\run.ps1 -Build -Install -Push -Bench "--e2e --model tiny --backend cpu"
#   .\run.ps1 -Bench "--e2e --model tiny --backend vulkan --out e2e-tiny-vk.json" -Env "SIMD_OCR_GPU_TIME=1"
param(
    [string]$Bench = "--e2e",
    [string]$Env = "",
    [string]$Run = "",
    [switch]$Build,
    [switch]$Install,
    [switch]$Push,
    [int]$TimeoutSec = 3600,
    [string]$Adb = $(if ($env:ADB) { $env:ADB } else { "D:\_\3rd\android-sdk\platform-tools\adb.exe" }),
    [string]$AndroidSdk = $(if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { "D:\_\3rd\android-sdk" }),
    [string]$Jdk = $(if ($env:JAVA_HOME -and (Test-Path "$env:JAVA_HOME\bin\javac.exe") -and ($env:JAVA_HOME -notmatch "jdk-2[2-9]")) { $env:JAVA_HOME } else { "C:\Program Files\Android\openjdk\jdk-21.0.8" })
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$root = (Resolve-Path "$here\..\..").Path
$pkg = "com.sdcb.simdocr.bench"
$remote = "/sdcard/Android/data/$pkg/files"
$local = "$root\bench-out\android"
New-Item -ItemType Directory -Force $local | Out-Null
if (-not $Run) { $Run = "r" + (Get-Date -Format "MMdd-HHmmss") }

if ($Build) {
    Push-Location $here
    try {
        dotnet build -c Release "-p:AndroidSdkDirectory=$AndroidSdk" "-p:JavaSdkDirectory=$Jdk" -v:q -nologo -clp:ErrorsOnly | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "build failed" }
    } finally { Pop-Location }
}
if ($Install) {
    $apk = "$here\bin\Release\net10.0-android\android-arm64\$pkg-Signed.apk"
    & $Adb install -r -g $apk | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "adb install failed" }
}

function Start-Bench([string]$id, [string]$argv, [string]$envs) {
    # a previous run may still be exiting: an intent delivered to it would be
    # swallowed by the dying activity instead of starting a fresh process
    & $Adb shell "am force-stop $pkg"
    for ($i = 0; $i -lt 20 -and ((& $Adb shell "pidof $pkg") -join ""); $i++) { Start-Sleep -Milliseconds 500 }
    & $Adb shell "rm -f $remote/out/$id.done $remote/out/$id.log"
    $cmd = "am start -S -n $pkg/$pkg.MainActivity --es run '$id' --es args '$argv'"
    if ($envs) { $cmd += " --es env '$envs'" }
    & $Adb shell $cmd | Out-Null
}

if ($Push) {
    # the app must create models/ dataset/ out/ itself: directories made by the
    # shell user are not readable by the app
    & $Adb shell "rm -rf $remote/models $remote/dataset $remote/out"
    Start-Bench "init" "--init" ""
    for ($i = 0; $i -lt 30 -and -not ((& $Adb shell "cat $remote/out/init.done 2>/dev/null") -join ""); $i++) { Start-Sleep 1 }
    & $Adb push "$root\bench-out\models\." "$remote/models/" | Out-Host
    & $Adb push "$root\dataset\." "$remote/dataset/" | Out-Host
}

& $Adb logcat -c
Start-Bench $Run $Bench $Env
Write-Output "[$Run] $Bench $(if ($Env) { "(env $Env)" })"

$t0 = Get-Date
$seen = 0
$started = $false
$result = $null
while (((Get-Date) - $t0).TotalSeconds -lt $TimeoutSec) {
    Start-Sleep -Seconds 2
    $done = (& $Adb shell "cat $remote/out/$Run.done 2>/dev/null") -join ""
    $log = & $Adb shell "cat $remote/out/$Run.log 2>/dev/null"
    if ($log) {
        $lines = @($log)
        for ($i = $seen; $i -lt $lines.Count; $i++) { Write-Output $lines[$i] }
        $seen = $lines.Count
    }
    if ($done -match "^\d+$") { $result = [int]$done; break }
    $procId = (& $Adb shell "pidof $pkg") -join ""
    if ($procId) { $started = $true }
    elseif ($started -or ((Get-Date) - $t0).TotalSeconds -gt 30) {
        # the app writes .done and then kills itself: look once more
        Start-Sleep -Seconds 1
        $done = (& $Adb shell "cat $remote/out/$Run.done 2>/dev/null") -join ""
        $lines = @(& $Adb shell "cat $remote/out/$Run.log 2>/dev/null")
        for ($i = $seen; $i -lt $lines.Count; $i++) { Write-Output $lines[$i] }
        if ($done -match "^\d+$") { $result = [int]$done }
        break
    }
}

& $Adb pull "$remote/out/." "$local" 2>&1 | Out-Null
if ($null -eq $result) {
    $lc = "$local\$Run-logcat.txt"
    & $Adb logcat -d -v time "SimdOcrBench:*" "DOTNET:*" "AndroidRuntime:*" "DEBUG:*" "libc:*" "ActivityManager:*" "lowmemorykiller:*" "*:F" "*:S" | Out-File -Encoding utf8 $lc
    Write-Output "[$Run] process ended without .done — logcat saved to $lc"
    exit 98
}
if ($result -ne 0) {
    & $Adb logcat -d -v time "SimdOcrBench:*" "DOTNET:*" "AndroidRuntime:*" "*:F" "*:S" | Out-File -Encoding utf8 "$local\$Run-logcat.txt"
}
Write-Output "[$Run] exit $result -> $local\$Run.log"
exit $result
