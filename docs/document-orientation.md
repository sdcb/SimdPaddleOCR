# 四方向纸张识别

本机 CPU / GPU 持续负载的耗时、CPU/GPU 使用率及内存/显存测试见 [资源性能报告](resource-performance.md)。

`PaddleOcrAll` 支持可选的第一个阶段：纸张方向识别 → 整图旋转校正 → DET → 文本行 CLS → REC。

纸张模型为 [PaddlePaddle/PP-LCNet_x1_0_doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori)，标签按顺序为 0、90、180、270。用户指定的 [PP-LCNet_x1_0_textline_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori) 只有 0、180 两类，不能替代四方向纸张模型；它也已下载并独立转换为 `models/textline_ori_x1.onnx`，没有替换原有文本行模型。

## 独立 NuGet 与调用

新包 `Sdcb.SimdPaddleOCR.Models.DocumentOrientation` 只包含纸张模型和转换记录，并只依赖 `ModelProvider`。核心库与现有 ChineseV6Tiny/Small/Medium 模型包没有引用它，现有调用不会隐式启用该阶段。

```csharp
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(
    ChineseV6TinyModels.Default,
    documentOrientation: DocumentOrientationModel.Provider,
    options: new PaddleOcrOptions
    {
        DocumentOrientation = new()
        {
            Backend = OcrBackend.Auto, // 可指定 Cpu 或 Vulkan
            IntraOpThreads = 0        // 默认最多 4；允许 1..16
        }
    });

PaddleOcrResult result = ocr.Run(pixels, width, height, returnCtcAlignment: true);
Console.WriteLine(result.DocumentOrientation?.OrientationDegrees);
Console.WriteLine(result.AppliedDocumentRotationDegrees);
Console.WriteLine(result.Text);
```

传入 `options: null` 使用默认配置。`UseDocumentOrientationClassification = false` 关闭阶段；`DocumentOrientationThreshold` 默认 0，遵循模型 top-1，也可设置 0..1 的置信度门槛。`DocumentOrientation` 为 null 表示未运行纸张阶段。已有 `UseDirectionClassification` 仍控制裁剪后的文本行 CLS。

`OrientationDegrees` 是输入页面的顺时针方向；`AppliedDocumentRotationDegrees` 是实际执行的逆时针校正角度。旋转只使用整数像素拷贝，保留原像素与四种像素格式，支持行尾 padding；输入内存不会被修改。非零校正使用每次请求独立的池化缓冲。

返回的 `PaddleOcrLine.Box` 和 `EstimateCharacterBoxes()` 都使用原始输入图坐标，顶点保留校正后阅读方向的顺序，不能假设第一个点永远是原图中的左上角。行排序遵循校正后页面的阅读顺序。`DetectorResizedWidth/Height` 是校正后 DET 实际输入尺寸。行级 `AppliedRotationDegrees` 仍只描述文本行的 180° 校正。

单独使用或者从本机模型加载：

```csharp
using Stream stream = File.OpenRead("models/doc_ori.onnx");
using var direction = new PaddleOcrDocumentOrientationClassifier(stream,
    new() { Backend = OcrBackend.Cpu, IntraOpThreads = 4 });
var predicted = direction.Classify(pixels, width, height);
```

完整流程也可在 `PaddleOcrAll` 的 `Model` 构造函数最后传入 `documentOrientationModel`。调用方拥有传入的 `Model`；通过独立包的 Load/LoadAsync 加载时由 OCR 引擎统一释放。阶段共享不可变编译模型，池化每次调用的推理请求，支持并发 Run。

## 转换与数值验证

Paddle 模型由 `inference.json` 图结构和 `inference.pdiparams` 参数共同构成，不能只转换 pdiparams。原始文件在 `.tmp-artifacts/orientation-source/`，下载固定的仓库 revision，文件 SHA-256 记录于 `models/orientation-manifest.json`。

`build/convert-orientation.py` 使用隔离 Python 环境：

```powershell
python -m venv --system-site-packages .tmp-artifacts/orientation-venv
.tmp-artifacts/orientation-venv/Scripts/python -m pip install paddlepaddle==3.1.1 paddle2onnx==2.1.0 onnx==1.17.0 onnxsim==0.8.1 opencv-python-headless==4.13.0.92 opt-einsum==3.3.0 polygraphy==0.49.26 onnxruntime==1.24.3 requests
.tmp-artifacts/orientation-venv/Scripts/python build/convert-orientation.py
# 如本机不能直连 Hugging Face，可传 --endpoint https://hf-mirror.com
```

Windows 上这套 Paddle2ONNX 的预编译 DLL 与 Paddle 3.2.2/3.3.1 不兼容；已实测 3.1.1 可以导出。固定版本避免依赖更新改变转换结果。

流程为：Paddle2ONNX 不启用内置优化 → ONNX checker → 独立 onnxsim 简化并固定 `[1,3,224,224]` → 严格形状推断/checker → Paddle 原始推理、原始 ONNX、简化 ONNX 三方比较。ONNX Runtime 的图优化关闭，避免把修复交给运行时。全部数值检查通过后才写入交付模型。

每个模型测试 8 组随机输入、3 组常数输入和样例的 4 个方向。两模型均由 310 节点简化为 113 节点；纸张模型对 Paddle 的最大绝对误差为 `3.5763e-7`，指定文本行模型为 `4.1724e-7`，全部 top-1 标签一致。完整逐例分数、误差、转换器版本、revision 和哈希在 manifest 中。

## 算子与本机调优

简化后的 Gemm 导入时降低为已优化的 MatMul/Add。支持 transA/transB、alpha/beta、可选 bias 及广播；常量 B 转置与缩放在加载时完成，从而继续使用 CPU/WASM 打包权重和 GPU 算子。同步修复 Constant 内部 TensorProto 名称与输出边名称不同时丢失常量的问题。

预处理遵循 PaddleX 的 RGB、短边缩放至 256、中心裁剪 224、ImageNet 归一化。缩放/裁剪/归一化合并为一次写入，直接产生后端所需 NCHW/NHWC；仅计算中心裁剪区域，宽图不会分配一张完整的缩放图。OpenCV 生成的黄金值覆盖 RGB/BGR/RGBA/BGRA、非紧密 stride 和两种张量布局。

Vulkan 在 sg32 协作矩阵设备上，对不超过 256 行、256/512 输入通道的小 1×1 特征图选择 Split-K 点积，提高 LCNet 小特征图和分类颈部的占用率。`SIMD_OCR_NARROW_DOT=0` 可做旧路径对比。既有 Conv-HardSwish、5×5 depthwise、SE、池化融合由 CPU/WASM/Vulkan 的对应路径执行。默认 CPU 阶段选最多 4 线程，允许独立覆盖。

本机为 Intel i7-12700、RTX 3060，测试日期 2026-10-13。使用样例图的四方向，预热 3 次，20 次计时取中位数；以下是方向图推理耗时，预处理单独测量，模型加载、整页旋转和其余 OCR 阶段不包含在表内。

| 后端 | 设置 | 图推理约耗时 |
| --- | --- | --- |
| CPU | 1 线程 | 5.6 ms |
| CPU | 4 线程（默认） | 3.2 ms |
| Vulkan | 调优前 | 5.3 ms |
| Vulkan | 小特征图点积调优后 | 4.1 ms |
| WASM 解释执行 | 1 线程 | 54.6 ms |
| WASM 解释执行 | 4 线程 | 22.1 ms |
| WASM AOT | 1 线程 | 8.4 ms |
| WASM AOT | 4 线程（默认） | 4.0 ms |
| WASM AOT | 8 线程 | 3.7 ms |

CPU 预处理约 0.53 ms，WASM 解释执行预处理约 2.94 ms。GPU 的四方向最大分数误差约 `2.8e-4`，与 fp16 中间计算有关，top-1 全部一致。CPU/Vulkan/WASM 都验证了整条 OCR 的四方向文本、行框和字符框；每个方向均为 16 行、文本 hash `46d99468540b5eb7`。这组结果是本机样例的性能与正确性检查，不代表数据集方向识别准确率。

WASM AOT 的预处理约 0.75 ms。Windows GPU 时钟与调度会影响短测，调优路径在不同轮次的墙钟中位数约 2.1～4.1 ms；同一轮短测旧/新路径也曾为 4.06/4.01 ms。GPU profiler 的逐 dispatch 最小值之和由 4.52 ms 降至 2.40 ms，14×14 的 256→256 点积 dispatch 从约 0.22 ms 降至 0.07 ms。以上是算子路径选择的证据，不能将其等同于任意负载下整页 OCR 的固定加速倍数。

验证包括 .NET 10 的 131 项单元测试及 netstandard2.0 的 128 项单元测试，全部通过。完整浏览器 AOT 的最后一轮还逐点验证了四方向还原后的字符框坐标。数据快照见 [document-orientation-validation.json](document-orientation-validation.json)。

重现与打包：

```powershell
dotnet test test/Sdcb.SimdPaddleOCR.UnitTests -c Release
dotnet test test/Sdcb.SimdPaddleOCR.UnitTests -c Release -p:UseNs20Library=true
dotnet run --project test/Sdcb.SimdPaddleOCR.GpuBench -c Release -- --orientation --cpu
dotnet run --project test/Sdcb.SimdPaddleOCR.GpuBench -c Release -- --orientation
test/Sdcb.SimdPaddleOCR.WasmBench/run.ps1 -Orientation -NoAot -Out bench-out/orientation-wasm-interpreter.json
test/Sdcb.SimdPaddleOCR.WasmBench/run.ps1 -Orientation -Out bench-out/orientation-wasm-aot.json
dotnet pack src/models/Sdcb.SimdPaddleOCR.Models.DocumentOrientation -c Release -o build/nupkgs
```

在 Codex 的受限执行环境中，Edge 自身的沙箱可能阻止启动 GPU 子进程；本地受信任的 WASM 测试可给 run.ps1 加 `-BrowserNoSandbox`，普通环境保持默认。浏览器测试使用工作区中的独立临时 profile，不复用桌面浏览器 profile。输出 JSON 在 `bench-out/`，该目录不进入源码版本管理。
