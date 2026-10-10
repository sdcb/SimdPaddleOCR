# Sdcb.SimdPaddleOCR [![test](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml/badge.svg)](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml) [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE) [![QQ](https://img.shields.io/badge/QQ_Group-579060605-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)](https://qm.qq.com/q/bPw5jAK4qk)

**中文** | [English](README_EN.md)

纯 C# PP-OCRv6 推理库：多平台手写 Kernel / GEMM 等算子、超高性能、低内存需求、超高准确率。
自带托管 ONNX 解释器，不依赖 Paddle Inference、ONNX Runtime 或 OpenCV 原生库。
GPU 支持 Vulkan 与 Metal（macOS），仅 .NET 10；同样纯 C#、不内嵌任何原生二进制。

核心 API 接收交错像素内存（默认 BGR24，也可直接传 RGB24 / BGRA32 / RGBA32），不负责图片解码，因此不会强制引入 ImageSharp、SkiaSharp 或 OpenCvSharp。

## 快速开始

安装核心包和 tiny 模型（tiny 会传递引用 CLS 与 `ModelProvider`）：

```powershell
dotnet add package Sdcb.SimdPaddleOCR
dotnet add package Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny
dotnet add package SixLabors.ImageSharp --version 3.1.11
```

模型从程序集嵌入资源直接加载，不会解压或写入临时文件。`stride = 0` 表示紧密排列（`width *` 每像素字节数）。默认 `ImagePixelFormat.Bgr24`；其它布局在 resize / crop 里就地 swizzle，不会先转成一张中间 BGR 图。

### ImageSharp 3（推荐）

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

// 必须 Clone Default：直接改 Configuration.Default 会丢掉 PNG/JPEG 解码器。
// 默认分配器按 4MB 分块，大图会让 DangerousTryGetSinglePixelMemory 失败。
static DecoderOptions CreateDecoderOptions()
{
    Configuration configuration = Configuration.Default.Clone();
    configuration.PreferContiguousImageBuffers = true;
    return new DecoderOptions { Configuration = configuration };
}
```

ImageSharp 默认分配器会把像素拆成 4MB 块，大图上 `DangerousTryGetSinglePixelMemory` 会失败。用独立 `Configuration` 打开 `PreferContiguousImageBuffers` 尽量零拷贝；仍不连续再 `CopyPixelDataTo`。不要改 `Configuration.Default`。

后续三个示例由调用方把 lock / 原生指针包成 `ReadOnlySpan<byte>` 再交给 `Run`，加载方式与上面相同。整段 `Run` 期间不要 Unlock / Dispose 源图。

### SkiaSharp

```csharp
using SkiaSharp;

SKBitmap bitmap = SKBitmap.Decode("sample.jpg")
    ?? throw new InvalidDataException("无法读取图片");
if (bitmap.ColorType != SKColorType.Bgra8888)
    bitmap = bitmap.Copy(SKColorType.Bgra8888)
        ?? throw new InvalidDataException("无法转换到 BGRA");
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
if (image.Empty()) throw new InvalidDataException("无法读取图片");
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

## NuGet 包

| NuGet 包                                        | 版本                                                                                                                                                                       | 说明                                                                           |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `Sdcb.SimdPaddleOCR`                            | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR)                                                       | 纯托管推理核心（`net10.0;netstandard2.0`）                                     |
| `Sdcb.SimdPaddleOCR.ModelProvider`              | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.ModelProvider.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.ModelProvider)                           | 模型契约（`IPaddleOcrModelProvider` / `PaddleOcrModelBundle`），通常被传递引用 |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny`       | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny)             | PP-OCRv6 tiny DET+REC+字典；`ChineseV6TinyModels.Default` 含 CLS               |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Small`      | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Small.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Small)           | PP-OCRv6 small；`ChineseV6SmallModels.Default`                                 |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Medium`     | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Medium)         | PP-OCRv6 medium；`ChineseV6MediumModels.Default`                               |
| `Sdcb.SimdPaddleOCR.Models.TextLineOrientation` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.TextLineOrientation.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.TextLineOrientation) | PP-LCNet 文本行方向 CLS，被三个中文模型包传递引用                              |

每个 `IPaddleOcrModelProvider` 提供 `Name`、`Kind`、`Format`、语言和版本元数据以及 `OpenRead()` / `OpenReadAsync()`。完整 OCR 组合由 `PaddleOcrModelBundle` 表达（DET、REC、字典和可选 CLS）。当前语言代码为 `zh`。单个模型也可被其他推理实现消费，例如 `ChineseV6TinyModel.Detection.OpenReadAsync()`。`Model`、`PaddleOcrDetector`、`PaddleOcrClassifier`、`PaddleOcrRecognizer` 和 `PaddleOcrAll` 均提供 Stream 加载入口；解析完成后不会继续保留完整的 ONNX 原始字节。

## 使用本地模型

核心不下载模型。使用本地 DET、CLS、REC 和字典文件时：

```csharp
using Sdcb.SimdPaddleOCR;

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

## GPU 后端

默认 `OcrBackend.Auto`：有满足条件的 GPU 就用，没有就留在 CPU，不用改代码。想固定后端，检测 / 方向 / 识别三个模型可以分别指定：

```csharp
using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default, new PaddleOcrOptions
{
    Detector   = new PaddleOcrDetectorOptions   { Backend = OcrBackend.Vulkan },
    Recognizer = new PaddleOcrRecognizerOptions { Backend = OcrBackend.Vulkan },
    Classifier = new PaddleOcrClassifierOptions { Backend = OcrBackend.Vulkan },
});
```

Vulkan 直连系统加载器，macOS arm64 走 Metal。完全不用 GPU 就显式 `OcrBackend.Cpu`。GPU 后端只在 `net10.0` 下可用。

### GPU 模式的 CPU 线程预算

GPU OCR 仍在 CPU 上执行裁剪、缩放归一化，以及识别结果的词表投影和 ArgMax。
可分别通过 `PreprocessWorkerCount` 与 `GpuCtcIntraOpThreads` 控制这些阶段的线程数：

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

两项默认 `0` 保留原自动预算；显式值允许 `1..16`，并限制到逻辑处理器数。
`GpuCtcIntraOpThreads` 不限制纯 CPU 推理或 GPU 失败后的 CPU 图算子预算。
`PreprocessWorkerCount` 在 CPU 模式也限制透视裁剪，但不改变检测预处理或图算子预算。

多张图片可共享同一实例，并由调用端限制图片并发，例如 `ParallelOptions.MaxDegreeOfParallelism = 2`。
图片并发与文本行 `LineWorkerCount` 是不同维度。降低每次调用的线程数可能降低进程 CPU，也可能降低吞吐或增加单张延迟；
应在相同模型、图片尺寸和方向分类选项下预热后比较，分别观察批量吞吐与单张延迟。上例的 `4/6` 是可测试的起点，不能视为所有设备的最优参数。

## 逐字坐标框

`Run` 传 `returnCtcAlignment: true` 后，每行会带上 `CtcSpans`，调 `EstimateCharacterBoxes()` 得到逐字符四边形坐标：

```csharp
PaddleOcrResult result = ocr.Run(bgr, width, height, returnCtcAlignment: true);
foreach (PaddleOcrLine line in result.Lines)
    foreach (PaddleOcrCharacterBox ch in line.EstimateCharacterBoxes())
        Console.WriteLine($"{ch.Text} @ ({ch.X1:F0},{ch.Y1:F0})");
```

字符框由 CTC 对齐估算（字符间空白按中点切），适合做按字覆盖、打码。默认 `false`，不需要就不付这部分开销。

## 示例

四个示例共用 `examples/sample.jpg`，图片由示例负责解码为交错像素（BGR / RGB / BGRA / RGBA 均可）：

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

`Sdcb.SimdPaddleOCR` 还带 4 个 LINQPad 脚本（`imagesharp` / `skiasharp` / `opencvsharp5` / `bitmap`）：下载示例图，tiny 模型识别，控制台输出文字。`bitmap` 同时能在 LINQPad 5 / .NET Framework 4.8 和 .NET 10 上跑。包打了 `linqpad-samples` 标签，免费版也能用。

## 常见问题

### 为什么调试时 OCR 识别特别慢？

调试器可能会在模块加载时取消 JIT 优化，使 OCR 的计算密集型代码无法获得应有的运行时优化，从而导致识别明显变慢。

请关闭该选项，然后重新启动调试会话：

- Visual Studio：`工具 > 选项 > 调试 > 常规`，取消勾选`在模块加载时取消 JIT 优化`。
- Rider：`构建、执行、部署 > 调试器 > JIT`，取消勾选`在加载模块时禁用 JIT 优化`。
- VS Code：打开`设置 (JSON)`，添加 `"csharp.debug.suppressJITOptimizations": false`。

### Native AOT 发布后为什么比 JIT 慢很多？

x64 发布 Native AOT 时，可执行项目里**必须**设置：

```xml
<IlcInstructionSet>avx2</IlcInstructionSet>
```

不设的话，ILC 按 SSE2 / 128-bit `Vector<T>` 基线编译，`Avx2.IsSupported` 会被折成 `false`，AVX2 内核整段裁掉，推理会慢一截。没有 AVX2 的 CPU 不要设这项。ARM64 的 AOT 基线已带 NEON / `AdvSimd`，一般不用写 `IlcInstructionSet`。

### ImageSharp 报「图片像素不是连续内存」或 `DangerousTryGetSinglePixelMemory` 失败？

ImageSharp 默认分配器会把像素拆成 4MB 块，大图上拿不到一整块连续缓冲。按上面「ImageSharp 3」示例：`Clone` 一份 `Configuration` 后打开 `PreferContiguousImageBuffers`，仍不连续再 `CopyPixelDataTo`。不要改 `Configuration.Default`，否则 PNG/JPEG 解码器会丢。完整写法见 `examples/ImageSharp.AspNetCore/Program.cs`。

### 为什么 GPU 没被用上？

`Auto` 按设备能力选：没有协作矩阵 / subgroup 不满足要求、或实测跑不过 CPU 的设备类型会留在 CPU。要强制走 GPU，显式指定 `OcrBackend.Vulkan` / `OcrBackend.Metal`。注意 GPU 后端只在 `net10.0` 下编译，`netstandard2.0` 只有 CPU。

### GPU（Vulkan/Metal）下 medium 模型报错 `BuildPlan failed at node 256 op=Transpose` 并回退 CPU？

请检查是否使用了第三方未优化的 ONNX 模型（例如直接通过 paddle2onnx 导出但未经过图优化清理的版本）。

- **原因**：SimdPaddleOCR 的 GPU 调度器为 SVTR 识别模型的注意力机制编写了专属的高性能融合算子（`attn` 着色器），该算子需要识别模型具备规整的计算图拓扑。如果使用含有大量未消除节点（如未折叠的 `Identity`、动态 `Shape/Reshape` 链）的外部直出 ONNX 模型，图构建器可能无法匹配该注意力融合模式，从而将内部的 5 维 Transpose 降级为独立算子调度；而 GPU 独立转置仅支持常规通道对齐维度，进而触发 `NotSupportedException` 并迫使引擎回退到 CPU 推理。
- **解决方式**：推荐直接使用 SimdPaddleOCR 官方配套的模型 NuGet 包（例如 `Sdcb.SimdPaddleOCR.Models.ChineseV6Medium`），或使用官方发布包中已做好图优化的 ONNX 模型。

## 支持范围

|            | 说明                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| 目标框架   | 核心 `net10.0;netstandard2.0`；`ModelProvider` 与全部模型包为 `netstandard2.0`                                              |
| 推荐运行时 | .NET 10：完整 x86 SIMD 与 NativeAOT（`IsAotCompatible`）                                                                    |
| 兼容运行时 | `netstandard2.0` 可在 .NET Framework 4.8 等环境使用；编译时去掉 AVX / AVX-512 / VNNI 源，走 `System.Numerics.Vector` / 标量 |
| CI 架构    | Windows x64 / x86 / ARM64，Linux x64 / ARM64，macOS x64 / ARM64                                                             |
| SIMD       | .NET 10 运行时探测 AVX → AVX2 → AVX-512 / VNNI；无对应指令集或 ARM 时用 Vector/标量                                         |
| 输入       | 交错像素内存（默认 BGR24，也可 RGB24 / BGRA32 / RGBA32）；无图片路径、文件或图片库 API                                      |
| 设备       | 默认 CPU；`net10.0` 另有可选 GPU 后端（`OcrBackend`）：Vulkan 直连系统加载器（Windows `vulkan-1.dll`、Linux `libvulkan.so.1`、Android `libvulkan.so`），Metal 走 macOS 系统框架；`netstandard2.0` 只有 CPU |
| 安卓       | 目前通过开发用宿主 `test/Sdcb.SimdPaddleOCR.AndroidBench`（`net10.0-android`，引用 `net10.0` 库，adb 驱动）在骁龙 8 Gen 3 上跑 CPU 与 Vulkan，见 [`docs/vulkan-8gen3.md`](docs/vulkan-8gen3.md)；桌面的 Vulkan 路由和 shader 没有变 |
| WebAssembly | 通过开发用宿主 `test/Sdcb.SimdPaddleOCR.WasmBench`（`net10.0` `browser-wasm` 多线程 Web Workers + SharedArrayBuffer）跑 CPU 推理，支持 tiny / small / medium，详见 [`docs/wasm.md`](docs/wasm.md) |
| NativeAOT  | 裁剪发布时请保留核心程序集和所用模型程序集                                                                                  |

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

## 性能

**2.0** 新增纯 C# GPU 后端（Vulkan / Metal），medium 端到端相对同机 CPU：

| 设备                      | 后端                 | 相对同机 CPU（medium）    |
| ------------------------- | -------------------- | ------------------------: |
| RTX 3080 Ti               | Vulkan（协作矩阵）   |                  **14.4×** |
| Intel Arc B580            | Vulkan（协作矩阵）   |                   **9.6×** |
| Apple M4（虚拟机）        | Metal                |                   **6.47×** |
| AMD Radeon 880M 核显      | Vulkan（协作矩阵）   |                   **4.3×** |
| 骁龙 8 Gen 3 / Adreno 750 | Vulkan（无协作矩阵） |                 **约 2.2×** |
| Intel UHD 770 核显        | Vulkan（无协作矩阵） | 慢于 CPU，`Auto` 默认走 CPU |

CPU 路径在 2.0 也普遍再快 **4–15%**（CTC ArgMax 并行化、动态 intra-op 线程预算、DET 后处理重写；ARM64 另有手写 AdvSIMD MatMul）。
WebAssembly（`browser-wasm` 多线程 + LLVM AOT）下 tiny 中位 **103 ms/张**，约为同机桌面原生的 2.3 倍。

每台设备的完整跑分、精度表和复现命令见 [`docs/perf.md`](docs/perf.md) 与 `docs/` 下各机型报告（`vulkan-*.md`、`metal-m4.md`、`wasm.md`）。

## 性能复现

[GitHub Actions `test` 工作流](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml)
会跑单元测试，并在 Windows / Linux / macOS 多架构上对 tiny / small / medium 做 bench
（含关闭 AVX-512 / AVX2 / AVX / 全部硬件加速，以及 `netstandard2.0` 库）。汇总报告写入 job summary 与 `perf-report` artifact。

## 微信群

![](https://io.starworks.cc:88/cv-public/2026/ocr-wxg-qr.png?1008)

如果微信群二维码过期了，请加入 QQ 群 [C#/.NET计算机视觉技术交流 579060605](https://qm.qq.com/q/bPw5jAK4qk)。
