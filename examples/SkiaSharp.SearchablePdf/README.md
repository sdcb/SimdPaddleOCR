# SkiaSharp 双层 PDF

使用本仓库的 PP-OCRv6 识别图片，将原图和可搜索、可选择、可复制的不可见文字写入多页 PDF。SkiaSharp 负责图片解码和透明图片的白底合成；轻量 PDF 写入器负责图片对象、逐字坐标和 Unicode 映射，不需要安装中文字体或商业 PDF 组件。

在仓库根目录运行（建议 Release）：

```powershell
dotnet run --project examples/SkiaSharp.SearchablePdf -c Release -- "E:\temp\22\33\17122502" "output\pdf\17122502-searchable.pdf"
```

输出 PDF，以及同名 `.txt` 识别文本和 `.ocr.json` 字符坐标报告。输入目录只读。图片按数字自然顺序排列，例如 `1.jpg, 2.jpg, ..., 10.jpg`；不递归子目录。也可以把第一个参数换成单张图片。

### 每张图片单独一个 PDF 和性能报告

`--separate` 把第二个参数作为输出目录，模型在整个进程中只加载一次，图片依次转换为 `1.pdf`、`2.pdf` 等独立文件：

```powershell
dotnet run --project examples/SkiaSharp.SearchablePdf -c Release -- "E:\temp\22\33\17122502" "output\pdf\single" --separate --repeat 3 --backend cpu
```

`--repeat` 默认 1，允许 1..10。第一轮记录冷启动后的完整批次；后续轮次记录预热性能，每一轮均重新运行 OCR 并生成 PDF，覆盖同名输出文件。性能测试使用 `--repeat 3`，正常逐张导出时只转换一次。

输出目录包含最终 PDF、同名 `.txt` 文本，以及：

- `benchmark.json`：环境、模型加载、首张耗时、每轮速度、预热统计、CPU 和内存峰值。
- `images.csv`：逐张、逐轮耗时，以及读取、解码、OCR、文字对齐和 PDF 写入阶段的耗时。
- `resources.csv`：每 50 ms 采样的 CPU 百分比、累计 CPU 时间、工作集、私有提交内存和托管堆。

CPU 百分比按逻辑处理器数量归一化，100% 表示整机全部逻辑处理器繁忙；平均核心数为进程 CPU 秒数除以实际经过秒数。内存使用 MiB（1 MiB = 1,048,576 字节），工作集是进程驻留内存，私有提交内存表示其私有内存承诺量，不包含独立显存。阶段计时包括正常缓冲写盘，不强制刷入物理磁盘；CSV、文本和报告的序列化不计入逐张转换耗时。

`GpuPolicySelected` 记录识别引擎是否选择 GPU，`ProbedGpu` 记录探测到的 Vulkan 设备；Auto 仍可能选择 CPU，也可能在个别 GPU 运算失败时回退到 CPU。

分别完成 Auto 和 CPU 两组、每组 3 轮测试后，可用下列命令汇总本次 53 张图片的中文性能报告：

```powershell
python examples/SkiaSharp.SearchablePdf/summarize_benchmark.py output/pdf/single-benchmark
```

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--model` | `medium` | `medium` / `small` / `tiny`，选择本地内嵌模型 |
| `--backend` | `auto` | `auto` / `cpu` / `vulkan` |
| `--dpi` | `300` | PDF 页面物理大小；不缩小原图、不改变比例 |
| `--det-size` | `1600` | 检测器长边上限，允许 `32..4096` |
| `--max-pages` | 全部 | 测试时限制页数 |
| `--separate` | 关闭 | 每张图片生成独立 PDF，第二个参数为输出目录 |
| `--repeat` | `1` | 逐张导出模式的完整批次次数；`3` 可统计预热性能 |

## 集成到自己的代码

引用 `src/Sdcb.SimdPaddleOCR.Pdf`，复用 OCR 引擎。必须在**同一张解码图片**上使用 `returnCtcAlignment: true`，这样文本层才有逐字位置：

```csharp
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Pdf;
using SkiaSharp;

byte[] encoded = File.ReadAllBytes(imagePath);
using SKBitmap bitmap = PdfImage.Decode(encoded);
PaddleOcrResult result = ocr.Run(bitmap.GetPixelSpan(), bitmap.Width, bitmap.Height,
    returnCtcAlignment: true, sourceStride: bitmap.RowBytes,
    format: ImagePixelFormat.Bgra32);

using FileStream output = File.Create(pdfPath);
using PaddleOcrPdfWriter pdf = new(output, dpi: 300);
pdf.AddPage(bitmap, result, encoded);
// 多页时重复 AddPage；按调用顺序形成页码。
pdf.Complete();
```

兼容的 RGB / 灰度 JPEG 原样嵌入，其他图片使用无损 RGB 压缩。文字使用 PDF `3 Tr` 不可见模式，Type 3 字体的空字形包含完整 `ToUnicode` 映射，支持中文、标点和补充 Unicode 字符。每个字符按 CTC 框的方向和尺寸放置，包含纸张方向校正后的原图坐标和文本行 180 度旋转。透视四边形使用逐字仿射近似。

CTC 字符框是估算值；复制得到的是 OCR 的识别结果，淡字、手写内容和复杂表格仍可能存在识别错误。图片层保持原始方向，文字层使用相同的原图坐标。多页 TIFF 只读取首帧。

识别模型有时会在右侧补白区域输出额外字符。该区域不属于原图，因此 PDF 层会排除这些字符；`.ocr.json` 的 `PaddingTokensRemoved` 记录数量，发生过滤时 `OriginalOcrText` 保留未过滤的原始识别结果。可调用 `PdfTextAlignment.RemovePaddingEmissions(result)` 取得与 PDF 一致的文本和字符坐标。

`PdfTextAlignment.EstimateCharacterBoxes(line)` 返回 PDF 实际使用的选择框：中间字符仍按 CTC 空白间隔的中点分界，首尾字符延伸到检测行框的外边界，避免大标题首尾字符只能选中一半。

## 验证

```powershell
dotnet test test/Sdcb.SimdPaddleOCR.Pdf.Tests -c Release
python -m pip install pymupdf pillow
python examples/SkiaSharp.SearchablePdf/verify_pdf.py output/pdf/17122502-searchable.pdf "E:\temp\22\33\17122502" --preview-dir tmp/pdfs
```

测试使用独立 PDF 解析器检查 Unicode、逐字基线、DPI、旋转、字体分组及原始 JPEG 保留。Python 脚本另行核对每页的默认文字提取顺序、逐字选择框与 OCR 框的偏差，并抽取首三页、中间页和末页渲染成 PNG；与仅含图片的页面逐像素比较，验证不可见文字不影响显示。生成的红框预览用于观察文字与原图的对应关系。
