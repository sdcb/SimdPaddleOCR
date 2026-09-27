@echo off
rem Dev-time only: compile .comp GLSL -> .spv next to each source (committed).
rem Not part of `dotnet build` — SPIR-V is embedded as a resource.
setlocal
set "GLSLC=C:\VulkanSDK\1.4.357.0\Bin\glslc.exe"
if not exist "%GLSLC%" for %%i in (glslc.exe) do set "GLSLC=%%~$PATH:i"
if not exist "%GLSLC%" (
    where glslc >nul 2>nul && set "GLSLC=glslc"
)
if not exist "%GLSLC%" (
    echo glslc not found — install Vulkan SDK or put glslc.exe on PATH
    exit /b 1
)
set "DIR=%~dp0..\Shaders"
for %%f in ("%DIR%\*.comp") do (
    echo glslc %%~nxf
    "%GLSLC%" -O --target-env=vulkan1.1 "%%f" -o "%%~dpnf.spv" || exit /b 1
)
rem sg32 coopmat tile variants (128x64 / 128x32), SE-prescaled GEMM and the
rem implicit-GEMM kxk conv
set "SG32=%DIR%\conv1x1_cm_sg32.comp"
set "N64=-DSGX=2u -DSGY=2u -DTSM=4u -DTSN=2u -DNTHR=128"
set "N32=-DSGX=1u -DSGY=4u -DTSM=2u -DTSN=2u -DNTHR=128"
echo glslc sg32 variants
"%GLSLC%" -O --target-env=vulkan1.1 %N64% "%SG32%" -o "%DIR%\conv1x1_cm_sg32_n64.spv" || exit /b 1
"%GLSLC%" -O --target-env=vulkan1.1 %N32% "%SG32%" -o "%DIR%\conv1x1_cm_sg32_n32.spv" || exit /b 1
"%GLSLC%" -O --target-env=vulkan1.1 -DPRESCALE "%SG32%" -o "%DIR%\conv1x1_cm_sg32_ps.spv" || exit /b 1
"%GLSLC%" -O --target-env=vulkan1.1 -DPRESCALE %N64% "%SG32%" -o "%DIR%\conv1x1_cm_sg32_ps_n64.spv" || exit /b 1
"%GLSLC%" -O --target-env=vulkan1.1 -DPRESCALE %N32% "%SG32%" -o "%DIR%\conv1x1_cm_sg32_ps_n32.spv" || exit /b 1
"%GLSLC%" -O --target-env=vulkan1.1 -DCONVK "%SG32%" -o "%DIR%\convk_cm_sg32.spv" || exit /b 1
"%GLSLC%" -O --target-env=vulkan1.1 -DCONVK %N64% "%SG32%" -o "%DIR%\convk_cm_sg32_n64.spv" || exit /b 1
"%GLSLC%" -O --target-env=vulkan1.1 -DCONVK %N32% "%SG32%" -o "%DIR%\convk_cm_sg32_n32.spv" || exit /b 1
