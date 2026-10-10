# Sdcb.SimdPaddleOCR [![test](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml/badge.svg)](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml) [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE) [![QQ](https://img.shields.io/badge/QQ_Group-579060605-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)](https://qm.qq.com/q/bPw5jAK4qk)

[中文](README.md) | **English**

Pure C# PP-OCRv6 inference library: cross-platform hand-written kernels and GEMM, very high performance, low memory requirements, and very high accuracy.
It ships a managed ONNX interpreter and does not depend on Paddle Inference, ONNX Runtime, or OpenCV native libraries.
GPU support: Vulkan and Metal (macOS), .NET 10 only — still pure C#, with no bundled native binaries.

The core API accepts interleaved pixels (BGR24 by default; RGB24 / BGRA32 / RGBA32 are also first-class). It does not decode images, so ImageSharp, SkiaSharp, or OpenCvSharp are not required.

## Quick start

Install the core package and the tiny models (tiny transitively references CLS and `ModelProvider`):

```powershell
dotnet add package Sdcb.SimdPaddleOCR
dotnet add package Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny
dotnet add package SixLabors.ImageSharp --version 3.1.11
```

Models load from embedded assembly resources. They are not extracted or written to temp files. `stride = 0` means tightly packed (`width *` bytes-per-pixel). The default is `ImagePixelFormat.Bgr24`; other layouts are swizzled in-place during resize / crop and are not converted into an intermediate BGR image.

### ImageSharp 3 (recommended)

```csharp
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
using Image<Rgba32> image = await Image.LoadAsync<Rgba32>(CreateDecoderOptions(), "sample.jpg");
Rgba32[]? packedCopy = null;
if (!image.DangerousTryGetSinglePixelMemory(out Memory<Rgba32> memory))
{
    packedCopy = new Rgba32[checked(image.Width * image.Height)];
    image.CopyPixelDataTo(packedCopy);
    memory = packedCopy;
}

PaddleOcrResult result = ocr.Run(MemoryMarshal.AsBytes(memory.Span), image.Width, image.Height,
    format: ImagePixelFormat.Rgba32);
Console.WriteLine(result.Text);

// Clone Default so PNG/JPEG decoders stay registered. Do not set this on Configuration.Default.
// Default allocator splits pixels into 4MB chunks; large images then fail DangerousTryGetSinglePixelMemory.
static DecoderOptions CreateDecoderOptions()
{
    Configuration configuration = Configuration.Default.Clone();
    configuration.PreferContiguousImageBuffers = true;
    return new DecoderOptions { Configuration = configuration };
}
```

ImageSharp's default allocator splits pixels into 4MB chunks, so `DangerousTryGetSinglePixelMemory` fails on large images. Clone a `Configuration`, set `PreferContiguousImageBuffers` for a zero-copy buffer when possible, and fall back to `CopyPixelDataTo`. Do not mutate `Configuration.Default`.

The next three samples wrap a lock / native pointer as `ReadOnlySpan<byte>` before `Run`. Do not Unlock / Dispose the source until `Run` returns.

### SkiaSharp

```csharp
using SkiaSharp;

SKBitmap bitmap = SKBitmap.Decode("sample.jpg")
    ?? throw new InvalidDataException("Failed to read image");
if (bitmap.ColorType != SKColorType.Bgra8888)
    bitmap = bitmap.Copy(SKColorType.Bgra8888)
        ?? throw new InvalidDataException("Failed to convert to BGRA");
int stride = bitmap.RowBytes;
unsafe
{
    PaddleOcrResult result = ocr.Run(new ReadOnlySpan<byte>((byte*)bitmap.GetPixels(), stride * bitmap.Height),
        bitmap.Width, bitmap.Height, stride, ImagePixelFormat.Bgra32);
}
```

### OpenCvSharp5

```csharp
using OpenCvSharp;

using Mat image = Cv2.ImRead("sample.jpg", ImreadModes.Color);
if (image.Empty()) throw new InvalidDataException("Failed to read image");
int stride = (int)image.Step();
unsafe
{
    PaddleOcrResult result = ocr.Run(new ReadOnlySpan<byte>((byte*)image.Data, stride * image.Height),
        image.Width, image.Height, stride, ImagePixelFormat.Bgr24);
}
```

### Bitmap

```csharp
using System.Drawing;
using System.Drawing.Imaging;

using Bitmap bitmap = new("sample.jpg");
Rectangle rectangle = new(0, 0, bitmap.Width, bitmap.Height);
BitmapData data = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
try
{
    unsafe
    {
        PaddleOcrResult result = ocr.Run(new ReadOnlySpan<byte>((byte*)data.Scan0, data.Stride * bitmap.Height),
            bitmap.Width, bitmap.Height, data.Stride, ImagePixelFormat.Bgra32);
    }
}
finally
{
    bitmap.UnlockBits(data);
}
```

## NuGet packages

| NuGet package                                   | Version                                                                                                                                                                    | Description                                                                                           |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `Sdcb.SimdPaddleOCR`                            | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR)                                                       | Pure-managed inference core (`net10.0;netstandard2.0`)                                                |
| `Sdcb.SimdPaddleOCR.ModelProvider`              | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.ModelProvider.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.ModelProvider)                           | Model contracts (`IPaddleOcrModelProvider` / `PaddleOcrModelBundle`), usually referenced transitively |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny`       | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny)             | PP-OCRv6 tiny DET+REC+dictionary; `ChineseV6TinyModels.Default` includes CLS                          |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Small`      | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Small.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Small)           | PP-OCRv6 small; `ChineseV6SmallModels.Default`                                                        |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Medium`     | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Medium)         | PP-OCRv6 medium; `ChineseV6MediumModels.Default`                                                      |
| `Sdcb.SimdPaddleOCR.Models.TextLineOrientation` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.TextLineOrientation.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.TextLineOrientation) | PP-LCNet text-line orientation CLS, transitively referenced by the three Chinese model packages       |

Each `IPaddleOcrModelProvider` exposes `Name`, `Kind`, `Format`, language and version metadata, plus `OpenRead()` / `OpenReadAsync()`. A full OCR set is a `PaddleOcrModelBundle` (DET, REC, dictionary, and optional CLS). The current language code is `zh`. Individual models can also be consumed by other inference implementations, for example `ChineseV6TinyModel.Detection.OpenReadAsync()`. `Model`, `PaddleOcrDetector`, `PaddleOcrClassifier`, `PaddleOcrRecognizer`, and `PaddleOcrAll` all accept Stream load entry points; after parsing they do not keep the full raw ONNX bytes.

## Local models

The core does not download models. To use local DET, CLS, REC, and dictionary files:

```csharp
using Sdcb.SimdPaddleOCR;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(
    detectionPath: "models/det.onnx",
    classificationPath: "models/cls.onnx",
    recognitionPath: "models/rec.onnx",
    dictionaryPath: "models/ppocr_keys.txt");
```

Do not mix the two parallelism knobs in `PaddleOcrOptions`: `DetIntraOpThreads` is in-graph convolution threads for detection
(one session, default cap 8); `LineWorkerCount` is the number of CLS/REC worker lanes
(one session per lane, an upper bound, actually `min(requested, ProcessorCount)`; `0` means
`min(ProcessorCount, 4)`). Detection thresholds, min box side, orientation classification,
dynamic recognition width, and session cache limits live in the same options object.

## GPU backends

The default is `OcrBackend.Auto`: use a capable GPU when one is present, stay on CPU otherwise — no code changes needed. To pin a backend, the detection / orientation / recognition models can each be set independently:

```csharp
using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default, new PaddleOcrOptions
{
    Detector   = new PaddleOcrDetectorOptions   { Backend = OcrBackend.Vulkan },
    Recognizer = new PaddleOcrRecognizerOptions { Backend = OcrBackend.Vulkan },
    Classifier = new PaddleOcrClassifierOptions { Backend = OcrBackend.Vulkan },
});
```

Vulkan loads the system loader directly; Metal covers macOS arm64. Pass `OcrBackend.Cpu` to stay off the GPU entirely. GPU backends are only compiled in the `net10.0` target.

### CPU thread budgets in GPU mode

GPU OCR still runs the perspective crop, resize/normalize, and the vocabulary projection and ArgMax of the recognition results on the CPU.
`PreprocessWorkerCount` and `GpuCtcIntraOpThreads` control the thread counts of these stages respectively:

```csharp
var options = new PaddleOcrOptions
{
    Detector = new() { Backend = OcrBackend.Vulkan },
    Classifier = new() { Backend = OcrBackend.Vulkan },
    Recognizer = new() { Backend = OcrBackend.Vulkan },
    GpuCtcIntraOpThreads = 4,
    PreprocessWorkerCount = 6
};
using var ocr = PaddleOcrAll.Load(ChineseV6TinyModels.Default, options);
```

Both default to `0`, preserving the original automatic budgets; explicit values allow `1..16` and are capped by the logical processor count.
`GpuCtcIntraOpThreads` does not limit pure-CPU inference or the CPU graph operator budgets after a GPU failure.
`PreprocessWorkerCount` also limits the perspective crop in CPU mode, but does not change detector preprocessing or graph operator budgets.

Multiple images can share one instance, with image concurrency limited on the caller's side, e.g. `ParallelOptions.MaxDegreeOfParallelism = 2`.
Image concurrency and the per-line `LineWorkerCount` are separate dimensions. Lowering the per-call thread count may reduce process CPU usage, but may also reduce throughput or increase per-image latency;
compare after warmup with the same models, image sizes, and orientation-classification options, measuring batch throughput and per-image latency separately. The `4/6` above is a testable starting point, not an optimal parameter set for every device.

## Per-character boxes

Pass `returnCtcAlignment: true` to `Run` and each line carries `CtcSpans`; `EstimateCharacterBoxes()` then returns a per-character quad:

```csharp
PaddleOcrResult result = ocr.Run(bgr, width, height, returnCtcAlignment: true);
foreach (PaddleOcrLine line in result.Lines)
    foreach (PaddleOcrCharacterBox ch in line.EstimateCharacterBoxes())
        Console.WriteLine($"{ch.Text} @ ({ch.X1:F0},{ch.Y1:F0})");
```

Character boxes are estimated from CTC alignment (gaps are split at the midpoint), good enough for per-character overlays and redaction. The default is `false`, so you pay nothing unless you ask for it.

## Examples

All four samples share `examples/sample.jpg`. Each sample decodes the image into interleaved pixels (BGR / RGB / BGRA / RGBA are all accepted):

- `examples/ImageSharp.AspNetCore`: ASP.NET Core + ImageSharp 3, upload UI and `POST /api/ocr` JSON API.
- `examples/SkiaSharp.Avalonia`: Avalonia desktop sample, SkiaSharp decode.
- `examples/OpenCvSharp5.Wpf`: WPF sample, OpenCvSharp5 decode.
- `examples/SystemDrawing.WinForms`: dual-target WinForms sample for .NET 10 Windows / .NET Framework 4.8, using `Bitmap`/`LockBits`; install the .NET Framework 4.8 Developer Pack before running `net48`, and set the project platform to x64.

```powershell
dotnet run --project examples/ImageSharp.AspNetCore
dotnet run --project examples/OpenCvSharp5.Wpf -- path/to/image.jpg
dotnet run --project examples/SkiaSharp.Avalonia -- path/to/image.jpg
dotnet run --project examples/SystemDrawing.WinForms --framework net10.0-windows
```

Open the web sample in a browser to upload; the API is `POST /api/ocr` (`multipart/form-data` fields `file`, `model`), docs at `/scalar`.

`Sdcb.SimdPaddleOCR` also ships 4 LINQPad scripts (`imagesharp` / `skiasharp` / `opencvsharp5` / `bitmap`): download a sample image, run the tiny model, print text. `bitmap` runs on both LINQPad 5 / .NET Framework 4.8 and .NET 10. The package is tagged `linqpad-samples`, so the free edition can use them.

## FAQ

### Why is OCR recognition much slower while debugging?

The debugger can disable JIT optimization when modules load. This prevents the runtime from optimizing OCR's compute-intensive code, making recognition significantly slower.

Clear this option, then restart the debugging session:

- Visual Studio: `Tools > Options > Debugging > General` > clear `Suppress JIT optimization on module load (Managed only)`.
- Rider: `Build, Execution, Deployment > Debugger > JIT` > clear `Disable JIT optimization on module load`.
- VS Code: open `Settings (JSON)` and add `"csharp.debug.suppressJITOptimizations": false`.

### Why is Native AOT much slower than JIT?

When publishing Native AOT on x64, the executable project **must** set:

```xml
<IlcInstructionSet>avx2</IlcInstructionSet>
```

Without it, ILC targets the SSE2 / 128-bit `Vector<T>` baseline, `Avx2.IsSupported` is folded to `false`, the AVX2 kernels are stripped, and inference is much slower. Do not set this on CPUs without AVX2. ARM64 AOT already includes NEON / `AdvSimd` in the baseline, so you usually do not need `IlcInstructionSet`.

### ImageSharp throws "pixels are not a single contiguous buffer" or `DangerousTryGetSinglePixelMemory` fails?

ImageSharp's default allocator splits pixels into 4MB chunks, so large images do not get one packed buffer. Follow the ImageSharp 3 sample above: clone `Configuration`, set `PreferContiguousImageBuffers`, and fall back to `CopyPixelDataTo`. Do not mutate `Configuration.Default` or PNG/JPEG decoders disappear. The full pattern is in `examples/ImageSharp.AspNetCore/Program.cs`.

### Why isn't the GPU being used?

`Auto` picks by device capability: devices without cooperative matrix / subgroup support, or device types that measure slower than CPU, stay on CPU. To force GPU, pass `OcrBackend.Vulkan` / `OcrBackend.Metal` explicitly. GPU backends are only compiled for `net10.0`; `netstandard2.0` is CPU only.

### GPU (Vulkan/Metal) falls back to CPU with `BuildPlan failed at node 256 op=Transpose` on the medium model?

Check if you are loading an unoptimized third-party ONNX model (such as a raw paddle2onnx export without graph cleanup).

- **Cause**: SimdPaddleOCR's GPU scheduler uses a fused compute kernel (`attn`) for the SVTR recognition model's multi-head self-attention. This fusion relies on a normalized graph topology. If you load an unoptimized ONNX model containing redundant nodes (e.g. uncollapsed `Identity` nodes or dynamic `Shape`/`Reshape` chains), the graph planner may fail to match the attention fusion pattern. The un-fused 5-D Transpose is then emitted as a standalone operator, which throws `NotSupportedException` because standalone GPU transposition only supports channel-aligned layouts, causing the engine to fall back to CPU.
- **Solution**: Use the official model NuGet packages (such as `Sdcb.SimdPaddleOCR.Models.ChineseV6Medium`) or the optimized ONNX models distributed with this repository.

## Support

|                     | Notes                                                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Target frameworks   | Core `net10.0;netstandard2.0`; `ModelProvider` and all model packages are `netstandard2.0`                                                                               |
| Recommended runtime | .NET 10: full x86 SIMD and NativeAOT (`IsAotCompatible`)                                                                                                                 |
| Compatible runtime  | `netstandard2.0` can run on .NET Framework 4.8 and similar; AVX / AVX-512 / VNNI sources are excluded at compile time, falling back to `System.Numerics.Vector` / scalar |
| CI architectures    | Windows x64 / x86 / ARM64, Linux x64 / ARM64, macOS x64 / ARM64                                                                                                          |
| SIMD                | .NET 10 probes AVX → AVX2 → AVX-512 / VNNI at runtime; Vector/scalar when those ISAs are missing or on ARM                                                               |
| Input               | Interleaved pixels (BGR24 by default; RGB24 / BGRA32 / RGBA32 also accepted); no image path, file, or image-library API                                                  |
| Device              | CPU by default; `net10.0` also has optional GPU backends (`OcrBackend`): Vulkan loading the system loader directly (`vulkan-1.dll` on Windows, `libvulkan.so.1` on Linux, `libvulkan.so` on Android), Metal via the macOS system framework; `netstandard2.0` is CPU only |
| Android             | Currently run through the dev host `test/Sdcb.SimdPaddleOCR.AndroidBench` (`net10.0-android`, references the `net10.0` library, driven over adb) on a Snapdragon 8 Gen 3, CPU and Vulkan; see [`docs/vulkan-8gen3.md`](docs/vulkan-8gen3.md). Desktop Vulkan routing and shaders are unchanged |
| WebAssembly         | Benchmarked via dev host `test/Sdcb.SimdPaddleOCR.WasmBench` (`net10.0` `browser-wasm` multi-threaded Web Workers + SharedArrayBuffer) running pure CPU inference across tiny / small / medium models; see [`docs/wasm.md`](docs/wasm.md) |
| NativeAOT           | Keep the core assembly and the model assemblies you use when publishing trimmed                                                                                          |

## License and third-party components

Source code and documentation written in this repository are released under [Apache License 2.0](LICENSE).
Apache-2.0 includes an explicit patent grant, which is a better fit for a public library and NuGet packages.

Model assets and third-party code are not relicensed by this project:

- PP-OCRv6 DET/REC, TextLineOrientation CLS, and dictionaries come from the PaddleOCR ecosystem and are marked
  Apache-2.0 in their source materials; keep origin and license notices when publishing model packages.
- Sample dependencies follow their upstream licenses; in particular ImageSharp 3.x uses the Six Labors Split License,
  not a plain MIT license.

Full third-party attribution is in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). PaddleOCR, PP-OCR, and related names belong to their
respective owners. This project is not official and does not imply endorsement.

## Performance

**2.0** adds pure C# GPU backends (Vulkan / Metal). Medium model end-to-end versus the same machine's CPU:

| Device                            | Backend                     | vs same-machine CPU (medium)          |
| --------------------------------- | --------------------------- | ------------------------------------: |
| RTX 3080 Ti                       | Vulkan (cooperative matrix) |                             **14.4×** |
| Intel Arc B580                    | Vulkan (cooperative matrix) |                              **9.6×** |
| Apple M4 (VM)                     | Metal                       |                             **6.47×** |
| AMD Radeon 880M iGPU              | Vulkan (cooperative matrix) |                              **4.3×** |
| Snapdragon 8 Gen 3 / Adreno 750   | Vulkan (no coop matrix)     |                             **~2.2×** |
| Intel UHD 770 iGPU                | Vulkan (no coop matrix)     | Slower than CPU; `Auto` stays on CPU  |

The CPU path is another **4–15%** faster across the board in 2.0 (parallel CTC ArgMax, dynamic intra-op thread budget, rewritten DET postprocess; plus a hand-written AdvSIMD MatMul on ARM64).
Under WebAssembly (`browser-wasm`, multi-threaded + LLVM AOT), tiny runs at a **103 ms/image** median — about 2.3× the same machine's desktop native.

Full benchmarks, accuracy tables, and reproduction commands per device: [`docs/perf.md`](docs/perf.md) and the per-machine reports under `docs/` (`vulkan-*.md`, `metal-m4.md`, `wasm.md`).

## Reproducing performance

The [GitHub Actions `test` workflow](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml)
runs unit tests and benches tiny / small / medium on Windows / Linux / macOS across architectures
(including disabling AVX-512 / AVX2 / AVX / all hardware acceleration, and the `netstandard2.0` build). Summary output goes to the job summary and the `perf-report` artifact.

## WeChat group

![](https://io.starworks.cc:88/cv-public/2026/ocr-wxg-qr.png?1008)

If the WeChat QR code has expired, join the QQ group [C#/.NET Computer Vision 579060605](https://qm.qq.com/q/bPw5jAK4qk).
