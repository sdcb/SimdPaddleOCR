# Sdcb.PaddleOCR

[![test](https://github.com/sdcb/Sdcb.PaddleOCR/actions/workflows/test.yml/badge.svg)](https://github.com/sdcb/Sdcb.PaddleOCR/actions/workflows/test.yml)
[![NuGet](https://img.shields.io/nuget/v/Sdcb.PaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.PaddleOCR)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

纯 C# PP-OCRv6 推理库：多平台 SIMD 优化、较低内存占用、高正确率。
自带托管 ONNX 解释器，不依赖 Paddle Inference、ONNX Runtime 或 OpenCV 原生库。

核心 API 只接收 BGR8 内存，不负责图片解码，因此不会强制引入 ImageSharp、SkiaSharp 或 OpenCvSharp。

## 快速开始

安装核心包和 tiny 模型（tiny 会传递引用 CLS 与 `ModelProvider`）：

```powershell
dotnet add package Sdcb.PaddleOCR
dotnet add package Sdcb.PaddleOCR.Models.ChineseV6Tiny
dotnet add package SixLabors.ImageSharp --version 3.1.11
```

模型从程序集嵌入资源直接加载，不会解压或写入临时文件。输入为 8-bit BGR，`stride = 0` 表示紧密排列（`width * 3`）。

### ImageSharp 3（推荐）

```csharp
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models.ChineseV6Tiny;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
using Image<Bgr24> image = await Image.LoadAsync<Bgr24>("sample.jpg");
byte[] bgr = new byte[image.Width * image.Height * 3];
image.CopyPixelDataTo(bgr);
PaddleOcrResult result = ocr.Run(bgr, image.Width, image.Height);
Console.WriteLine(result.Text);
```

后续三个示例只演示如何解码到 BGR，加载与 `Run` 与上面相同。

### SkiaSharp

```csharp
using SkiaSharp;

SKBitmap bitmap = SKBitmap.Decode("sample.jpg")
    ?? throw new InvalidDataException("无法读取图片");
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
if (image.Empty()) throw new InvalidDataException("无法读取图片");
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

## NuGet 包

| NuGet 包 | 版本 | 说明 |
| --- | --- | --- |
| `Sdcb.PaddleOCR` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.PaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.PaddleOCR) | 纯托管推理核心（`net10.0;netstandard2.0`） |
| `Sdcb.PaddleOCR.ModelProvider` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.PaddleOCR.ModelProvider.svg)](https://www.nuget.org/packages/Sdcb.PaddleOCR.ModelProvider) | 模型契约（`IPaddleOcrModelProvider` / `PaddleOcrModelBundle`），通常被传递引用 |
| `Sdcb.PaddleOCR.Models.ChineseV6Tiny` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.PaddleOCR.Models.ChineseV6Tiny.svg)](https://www.nuget.org/packages/Sdcb.PaddleOCR.Models.ChineseV6Tiny) | PP-OCRv6 tiny DET+REC+字典；`ChineseV6TinyModels.Default` 含 CLS |
| `Sdcb.PaddleOCR.Models.ChineseV6Small` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.PaddleOCR.Models.ChineseV6Small.svg)](https://www.nuget.org/packages/Sdcb.PaddleOCR.Models.ChineseV6Small) | PP-OCRv6 small；`ChineseV6SmallModels.Default` |
| `Sdcb.PaddleOCR.Models.ChineseV6Medium` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.PaddleOCR.Models.ChineseV6Medium.svg)](https://www.nuget.org/packages/Sdcb.PaddleOCR.Models.ChineseV6Medium) | PP-OCRv6 medium；`ChineseV6MediumModels.Default` |
| `Sdcb.PaddleOCR.Models.TextLineOrientation` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.PaddleOCR.Models.TextLineOrientation.svg)](https://www.nuget.org/packages/Sdcb.PaddleOCR.Models.TextLineOrientation) | PP-LCNet 文本行方向 CLS，被三个中文模型包传递引用 |

每个 `IPaddleOcrModelProvider` 提供 `Name`、`Kind`、`Format`、语言和版本元数据以及 `OpenRead()` / `OpenReadAsync()`。完整 OCR 组合由 `PaddleOcrModelBundle` 表达（DET、REC、字典和可选 CLS）。当前语言代码为 `zh`。单个模型也可被其他推理实现消费，例如 `ChineseV6TinyModel.Detection.OpenReadAsync()`。`Model`、`Detector`、`Classifier`、`Recognizer` 和 `PaddleOcrAll` 均提供 Stream 加载入口；解析完成后不会继续保留完整的 ONNX 原始字节。

## 使用本地模型

核心不下载模型。使用本地 DET、CLS、REC 和字典文件时：

```csharp
using Sdcb.PaddleOCR;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(
    detectionPath: "models/det.onnx",
    classificationPath: "models/cls.onnx",
    recognitionPath: "models/rec.onnx",
    dictionaryPath: "models/ppocr_keys.txt");
```

`PaddleOcrOptions` 里两套并行不要混用：`DetIntraOpThreads` 是检测图内的卷积线程
（一份 session，默认最多 8）；`LineWorkerCount` 是一行一组的 CLS/REC worker 路数
（每路一个 session，上限，实际 `min(请求, ProcessorCount)`；`0` 为
`min(ProcessorCount, 4)`）。检测阈值、边界长度、方向分类、
动态识别宽度和 Session 缓存上限等也在同一组 options 里。

## 示例

四个示例共用 `examples/sample.jpg`，图片由示例负责解码并转换为 BGR：

- `examples/ImageSharp.AspNetCore`：ASP.NET Core + ImageSharp 3，可上传体验与 `POST /api/ocr` JSON API。
- `examples/SkiaSharp.Avalonia`：Avalonia 桌面示例，SkiaSharp 解码。
- `examples/OpenCvSharp5.Wpf`：WPF 示例，OpenCvSharp5 解码。
- `examples/SystemDrawing.WinForms`：.NET 10 Windows / .NET Framework 4.8 双目标 WinForms 示例，使用 `Bitmap`/`LockBits`；运行 `net48` 前请安装 .NET Framework 4.8 Developer Pack，项目平台选择 x64。

```powershell
dotnet run --project examples/ImageSharp.AspNetCore
dotnet run --project examples/OpenCvSharp5.Wpf -- path/to/image.jpg
dotnet run --project examples/SkiaSharp.Avalonia -- path/to/image.jpg
dotnet run --project examples/SystemDrawing.WinForms --framework net10.0-windows
```

Web 示例打开站点即可上传；API 为 `POST /api/ocr`（`multipart/form-data` 字段 `file`、`model`），文档在 `/scalar`。

## 支持范围

| | 说明 |
| --- | --- |
| 目标框架 | 核心 `net10.0;netstandard2.0`；`ModelProvider` 与全部模型包为 `netstandard2.0` |
| 推荐运行时 | .NET 10：完整 x86 SIMD 与 NativeAOT（`IsAotCompatible`） |
| 兼容运行时 | `netstandard2.0` 可在 .NET Framework 4.8 等环境使用；编译时去掉 AVX / AVX-512 / VNNI 源，走 `System.Numerics.Vector` / 标量 |
| CI 架构 | Windows x64 / x86 / ARM64，Linux x64 / ARM64，macOS x64 / ARM64 |
| SIMD | .NET 10 运行时探测 AVX → AVX2 → AVX-512 / VNNI；无对应指令集或 ARM 时用 Vector/标量 |
| 输入 | 8-bit BGR 内存；无图片路径、文件或图片库 API |
| 设备 | CPU only，无 GPU |
| NativeAOT | 裁剪发布时请保留核心程序集和所用模型程序集 |

## 许可证与第三方组件

本仓库中由本项目编写的源代码和文档采用 [Apache License 2.0](LICENSE) 发布。
Apache-2.0 提供明确的专利授权条款，更适合公开发布的库和 NuGet 包。

模型资源和第三方代码不因本项目许可证而被重新授权：

- PP-OCRv6 DET/REC、TextLineOrientation CLS 及字典来自 PaddleOCR 生态，来源资料标记为
  Apache-2.0；发布模型包时请保留来源和许可证说明。
- 示例依赖遵循各自上游许可证；特别是 ImageSharp 3.x 使用 Six Labors Split License，
  不是普通 MIT 许可证。

完整的第三方归属和分发说明见
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)。PaddleOCR、PP-OCR 及相关名称归其
各自权利人所有，本项目不代表官方，也不构成官方背书。

## 性能复现

[GitHub Actions `test` 工作流](https://github.com/sdcb/Sdcb.PaddleOCR/actions/workflows/test.yml)
会跑单元测试，并在 Windows / Linux / macOS 多架构上对 tiny / small / medium 做 bench
（含关闭 AVX-512 / AVX2 / AVX / 全部硬件加速，以及 `netstandard2.0` 库）。汇总报告写入 job summary 与 `report.md` artifact。
