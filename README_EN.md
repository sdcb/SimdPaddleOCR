# Sdcb.SimdPaddleOCR [![test](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml/badge.svg)](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml) [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE) [![QQ](https://img.shields.io/badge/QQ_Group-579060605-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)](https://qm.qq.com/q/bPw5jAK4qk)

[中文](README.md) | **English**

Pure C# PP-OCRv6 inference library: multi-platform SIMD, relatively low memory use, and high accuracy.
It ships a managed ONNX interpreter and does not depend on Paddle Inference, ONNX Runtime, or OpenCV native libraries.

The core API accepts packed BGR8 memory only. It does not decode images, so ImageSharp, SkiaSharp, or OpenCvSharp are not required.

## Quick start

Install the core package and the tiny models (tiny transitively references CLS and `ModelProvider`):

```powershell
dotnet add package Sdcb.SimdPaddleOCR
dotnet add package Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny
dotnet add package SixLabors.ImageSharp --version 3.1.11
```

Models load from embedded assembly resources. They are not extracted or written to temp files. Input is 8-bit BGR; `stride = 0` means tightly packed (`width * 3`).

### ImageSharp 3 (recommended)

```csharp
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
using Image<Bgr24> image = await Image.LoadAsync<Bgr24>("sample.jpg");
byte[] bgr = new byte[image.Width * image.Height * 3];
image.CopyPixelDataTo(bgr);
PaddleOcrResult result = ocr.Run(bgr, image.Width, image.Height);
Console.WriteLine(result.Text);
```

The next three samples only show how to decode to BGR. Loading and `Run` are the same as above.

### SkiaSharp

```csharp
using SkiaSharp;

SKBitmap bitmap = SKBitmap.Decode("sample.jpg")
    ?? throw new InvalidDataException("Failed to read image");
int stride = bitmap.Width * 3;
byte[] bgr = new byte[stride * bitmap.Height];
for (int y = 0; y < bitmap.Height; y++)
{
    for (int x = 0; x < bitmap.Width; x++)
    {
        SKColor color = bitmap.GetPixel(x, y);
        int offset = y * stride + x * 3;
        bgr[offset] = color.Blue;
        bgr[offset + 1] = color.Green;
        bgr[offset + 2] = color.Red;
    }
}
```

### OpenCvSharp5

```csharp
using System.Runtime.InteropServices;
using OpenCvSharp;

using Mat image = Cv2.ImRead("sample.jpg", ImreadModes.Color);
if (image.Empty()) throw new InvalidDataException("Failed to read image");
int rowBytes = image.Width * image.Channels();
byte[] bgr = new byte[rowBytes * image.Height];
for (int y = 0; y < image.Height; y++)
    Marshal.Copy(IntPtr.Add(image.Data, (int)(y * image.Step())), bgr, y * rowBytes, rowBytes);
```

### Bitmap

```csharp
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

using Bitmap bitmap = new("sample.jpg");
int stride = bitmap.Width * 3;
byte[] bgr = new byte[stride * bitmap.Height];
Rectangle rectangle = new(0, 0, bitmap.Width, bitmap.Height);
BitmapData data = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
try
{
    for (int y = 0; y < bitmap.Height; y++)
        Marshal.Copy(new IntPtr(data.Scan0.ToInt64() + y * (long)data.Stride), bgr, y * stride, stride);
}
finally
{
    bitmap.UnlockBits(data);
}
```

## NuGet packages

| NuGet package | Version | Description |
| --- | --- | --- |
| `Sdcb.SimdPaddleOCR` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR) | Pure-managed inference core (`net10.0;netstandard2.0`) |
| `Sdcb.SimdPaddleOCR.ModelProvider` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.ModelProvider.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.ModelProvider) | Model contracts (`IPaddleOcrModelProvider` / `PaddleOcrModelBundle`), usually referenced transitively |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny) | PP-OCRv6 tiny DET+REC+dictionary; `ChineseV6TinyModels.Default` includes CLS |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Small` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Small.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Small) | PP-OCRv6 small; `ChineseV6SmallModels.Default` |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Medium` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Medium) | PP-OCRv6 medium; `ChineseV6MediumModels.Default` |
| `Sdcb.SimdPaddleOCR.Models.TextLineOrientation` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.TextLineOrientation.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.TextLineOrientation) | PP-LCNet text-line orientation CLS, transitively referenced by the three Chinese model packages |

Each `IPaddleOcrModelProvider` exposes `Name`, `Kind`, `Format`, language and version metadata, plus `OpenRead()` / `OpenReadAsync()`. A full OCR set is a `PaddleOcrModelBundle` (DET, REC, dictionary, and optional CLS). The current language code is `zh`. Individual models can also be consumed by other inference implementations, for example `ChineseV6TinyModel.Detection.OpenReadAsync()`. `Model`, `Detector`, `Classifier`, `Recognizer`, and `PaddleOcrAll` all accept Stream load entry points; after parsing they do not keep the full raw ONNX bytes.

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

## Examples

All four samples share `examples/sample.jpg`. Each sample decodes the image and converts it to BGR:

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

## Support

| | Notes |
| --- | --- |
| Target frameworks | Core `net10.0;netstandard2.0`; `ModelProvider` and all model packages are `netstandard2.0` |
| Recommended runtime | .NET 10: full x86 SIMD and NativeAOT (`IsAotCompatible`) |
| Compatible runtime | `netstandard2.0` can run on .NET Framework 4.8 and similar; AVX / AVX-512 / VNNI sources are excluded at compile time, falling back to `System.Numerics.Vector` / scalar |
| CI architectures | Windows x64 / x86 / ARM64, Linux x64 / ARM64, macOS x64 / ARM64 |
| SIMD | .NET 10 probes AVX → AVX2 → AVX-512 / VNNI at runtime; Vector/scalar when those ISAs are missing or on ARM |
| Input | 8-bit BGR memory; no image path, file, or image-library API |
| Device | CPU only, no GPU |
| NativeAOT | Keep the core assembly and the model assemblies you use when publishing trimmed |

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

## Reproducing performance

The [GitHub Actions `test` workflow](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml)
runs unit tests and benches tiny / small / medium on Windows / Linux / macOS across architectures
(including disabling AVX-512 / AVX2 / AVX / all hardware acceleration, and the `netstandard2.0` build). Summary output goes to the job summary and the `report.md` artifact.

## WeChat group

![](https://io.starworks.cc:88/cv-public/2026/ocr-wxg-qr.png)

If the WeChat QR code has expired, join the QQ group [C#/.NET Computer Vision 579060605](https://qm.qq.com/q/bPw5jAK4qk).
