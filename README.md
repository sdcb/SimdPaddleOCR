# Sdcb.PaddleOCR

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

纯托管 PP-OCRv6 推理库。核心 API 只接收 BGR8 内存，不负责图片解码，
因此不会强制引入 ImageSharp、SkiaSharp 或 OpenCvSharp。

## 快速开始

核心包为 `Sdcb.PaddleOCR`。安装 `Sdcb.PaddleOCR.Models.ChineseV6Tiny` 后，可以通过
Bundle 使用嵌入资源中的 tiny DET、REC、CLS 和字典模型：

```csharp
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models.ChineseV6Tiny;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
PaddleOcrResult result = ocr.Run(bgr, width, height, stride);

Console.WriteLine(result.Text);
Console.WriteLine($"Lines: {result.Lines.Length}");
```

模型从程序集内存直接加载，不会解压或写入临时文件。

## 内置模型包

| NuGet 包 | 模型 | 工厂 |
| --- | --- | --- |
| `Sdcb.PaddleOCR.Models.ChineseV6Tiny` | PP-OCRv6 tiny | `ChineseV6TinyModels.Default` |
| `Sdcb.PaddleOCR.Models.ChineseV6Small` | PP-OCRv6 small | `ChineseV6SmallModels.Default` |
| `Sdcb.PaddleOCR.Models.ChineseV6Medium` | PP-OCRv6 medium | `ChineseV6MediumModels.Default` |
| `Sdcb.PaddleOCR.Models.TextLineOrientation` | 文本行方向分类 CLS | `TextLineOrientationModel.Provider` |

每个 `IPaddleOcrModelProvider` 表达一个模型，并提供 `Name`、`Kind`、`Format`、语言和版本
元数据以及 `OpenRead()` / `OpenReadAsync()`。完整 OCR 组合由 `PaddleOcrModelBundle` 表达，
包含 DET、REC、字典和可选 CLS。每个模型包都提供 `Default` 便捷入口和 `All` 变体集合；
当前内置组合语言代码为 `zh`，后续可扩展其他语言。单个模型也可以直接被其他推理实现消费，
例如 `ChineseV6TinyModel.Detection.OpenReadAsync()`。核心的 `Model`、`Detector`、`Classifier`、
`Recognizer` 和 `PaddleOcrAll` 也都提供 Stream 加载入口；模型解析完成后不会继续保留完整的
ONNX 原始字节，适合直接消费嵌入资源流。

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
动态识别宽度和 Session 缓存上限等也在同一组 options 里。输入约定为 8-bit BGR，
`stride = 0` 表示紧密排列。

## 示例

四个示例共用 `examples/sample.jpg`，图片由示例负责解码并转换为 BGR：

- `examples/ImageSharp.AspNetCore`：ASP.NET Core 示例，使用 ImageSharp 解码；提供可上传体验的前端，以及 `POST /api/ocr` JSON API，支持 tiny/small/medium 模型、检测框/红色识别文本叠加和耗时显示。
- `examples/SkiaSharp.Avalonia`：Avalonia 12.1.1 桌面示例，使用 SkiaSharp 4.151.1
  解码图片，支持 tiny/small/medium 模型单选、图片选择、重复运行、检测框/红色识别文本叠加和底部总耗时显示。
- `examples/OpenCvSharp5.Wpf`：.NET 10 WPF 示例，使用 OpenCvSharp5
  5.0.0.20260806 读取图片，支持 tiny/small/medium 模型单选；界面提供图片选择、重复运行、
  检测框/红色识别文本叠加和 OCR 总耗时显示。
- `examples/SystemDrawing.WinForms`：面向 .NET 10 Windows / .NET Framework 4.8 双目标的 WinForms 示例（默认优先 net10.0-windows），使用
  `System.Drawing.Bitmap`/`LockBits` 转换 BGR，支持图片及 DET/CLS/REC/字典路径选择，默认加载 tiny 模型并显示检测框和识别文本。
  运行前请安装 .NET Framework 4.8 Developer Pack，项目平台选择 x64。

Web 示例和三个桌面示例都会显示 OCR 结果；ImageSharp/OpenCvSharp/SkiaSharp 示例还会标明当前是 Debug 还是 Release 构建，方便比较性能。

运行 Web 示例：

```powershell
dotnet run --project examples/ImageSharp.AspNetCore
```

浏览器打开站点即可上传体验；API 为 `POST /api/ocr`（`multipart/form-data` 字段 `file`、`model`），文档在 `/scalar`。

运行桌面示例时可以传入图片路径：

```powershell
dotnet run --project examples/OpenCvSharp5.Wpf -- path/to/image.jpg
dotnet run --project examples/SkiaSharp.Avalonia -- path/to/image.jpg
dotnet run --project examples/SystemDrawing.WinForms --framework net10.0-windows
```

## 支持范围

核心包提供 `net10.0` 资产，使用 `System.Runtime.Intrinsics` SIMD 内核并支持
NativeAOT；模型提供包提供 `netstandard2.0` 资产，可被其他 .NET Framework/.NET Core
应用或推理引擎引用。核心推理最低运行时为 .NET 10。NativeAOT 裁剪发布时，请确保核心程序集和所使用的模型提供程序集
被保留。当前核心不提供图片路径、文件或图片库 API。

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

性能数据和 100 张图片 benchmark 位于 `benchmark/Sdcb.PaddleOCR.Benchmarks`、`perf-input/` 和
[`docs/perf.md`](docs/perf.md)。
