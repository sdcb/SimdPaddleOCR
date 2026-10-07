# SimdPaddleOCR（Vulkan） vs RapidOcrNet（ONNX Runtime + CUDA）— 5800X / RTX 3080 Ti

问题：RapidOcr 系（ONNX Runtime，GPU）在本机同一套权重、同一份数据上，和本仓库的 Vulkan GPU 后端差多少。

实测机：HOME-MAIN，Windows 10.0.26200，电源方案“高性能”，Ryzen 7 5800X（8C/16T，AVX2，无 AVX-512），64 GB，RTX 3080 Ti（驱动 581.80）。运行时 .NET 10.0.11。本仓库 `2ee12a1`。`test/Sdcb.SimdPaddleOCR.Tests`，`--workers 4 --benchmark-kind simd --warmup 1`，n=99，同一个 `dataset/` 100 张变尺寸图（100 个不同尺寸，512–1536 px），张张新 shape。三方（Vulkan / RapidOcrNet CUDA / 本机 CPU）按 tiny → small → medium 每轮交替 3 轮。

**结论：RapidOcrNet 4.2.0（ONNX Runtime 1.29，CUDA EP）端到端中位是本仓库 Vulkan 的 13–18 倍慢（tiny 18.4× / small 16.4× / medium 13.3×）。把 RapidOcrNet 的检测输入几何对齐到本库的 960 max-side、并把 REC 并行度提到 16 之后仍慢 9–15 倍。瓶颈不在 GPU 算力：RapidOcrNet 自己的 CPU 侧预处理/后处理吃掉了它检测阶段约 9 成时间，另外 ONNX Runtime CUDA 每换一次输入 shape 还要在会话里多花 ~55 ms（REC 每个 crop 都是新宽度，逐行都付）。正确率两边接近但不逐行相同：RapidOcrNet tiny/small 的行精确略低、CER 明显更差（4.52% / 1.67% vs 2.38% / 0.40%），medium 行精确反而略高、CER 仍更差；三档的 cls 方向判定都比本库少对 25–30 行。**

## 怎么跑的

公平性安排：

- **同一套权重**。RapidOcrNet 的 `RapidOcrModelSet.PPOCRv6{Tiny,Small,Medium}` 预设提供 PP-OCRv6 的检测归一化，但四个模型路径全部用 `with { DetModelPath/ClsModelPath/RecModelPath/KeysPath }` 指到仓库自己的 `models/`（`det.onnx`、`small_det.onnx`、`medium_det.onnx` 及对应 `*_rec.onnx`、`cls.onnx`、`pp_ocr_keys/rec_keys`）。两边加载的是同一批 onnx 文件，差别只有实现。
- **同一个 harness**。新增 `--engine rapid`（CUDA EP）与 `--engine rapid-cpu`（ORT CPU，对照），实现 `IBenchEngine`，和别的引擎一样逐图吃同一份 BGR24 缓冲；`--workers N` 映射到 `RecMaxDegreeOfParallelism`（本库这边 `LineWorkerCount`）。新增 `--rapid-models`、`--rapid-det-limit`（RapidOcrNet `LimitSideLen`，min-side 语义，默认 736）、`--rapid-img-resize`（RapidOcrNet `ImgResize`，max-side 语义，默认 0=关）、`--det-limit`（本库 `LimitSideLength`，max-side 语义，默认 960）。
- **默认关闭**。`EnableRapidOcr` 为 false 时 `RapidEngine.cs` 不参与编译、也不拉 GPU 包，CI 与日常构建不受影响：

```powershell
dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -p:EnableRapidOcr=true
test/Sdcb.SimdPaddleOCR.GpuBench/tools/rapid-cuda-runtime.ps1      # 见下
test/Sdcb.SimdPaddleOCR.GpuBench/tools/rapid-ab.ps1 -tag rapid-3080ti -rounds 3
```

- **CUDA 运行库**：ORT ≥ 1.23 的 `Microsoft.ML.OnnxRuntime.Gpu` 不再随包发 CUDA/cuDNN，而 `onnxruntime_providers_cuda.dll` 会按名字去加载 `cudnn64_9.dll`（缺了会报 `cuDNN is unavailable or disabled for CUDA Execution Provider: LoadLibrary failed for cudnn64_9.dll`，此时 CUDA EP 会静默退化成一行都识别不出来）。`tools/rapid-cuda-runtime.ps1` 把 `cudart64_12 / cublas64_12 / cublasLt64_12 / cudnn*64_9` 凑到 `bench-out/rapid/cuda-runtime/`（优先用本机已有的副本，缺的从 PyPI 的 NVIDIA wheel 取），runner 把该目录前置到 `PATH`。
- **已知的一处不对称**：harness 给引擎的是 BGR24，RapidOcrNet 只接受 `SKBitmap`，所以 rapid 侧多一次 BGR24→BGRA8888 拷贝，单图 2.0–2.9 ms，作为 stage `input_convert` 单独记账、也计入总时间（对 tiny 的 22 ms 基线约 1 成，对 rapid 的 400 ms 可忽略）。
- 其余按各自库的推荐配置：RapidOcrNet 用 `RapidOcrOptions.PPOCRv6`（= `PythonCompat`：无白边、短边自适应 736、box_score 0.5/box 0.3/unclip 1.6、cls_thresh 0.9）；本库用 harness 既有选项（`LineWorkerCount=4`、`AdaptiveWidth`+`TargetWidth=320`、session cache 32、GPU REC 批 16）。两者的检测输入几何本身不同（下述），所以额外做了一组对齐实验。

## 端到端（median ms/图，越低越好，4 workers）

三轮 median 都列出，加粗是三轮的中位数。倍数 = 慢 ÷ 快。

| 模型   |            Vulkan（本库 GPU） |                  RapidOcrNet CUDA |      倍数 | sharp（本库 CPU） | RapidOcrNet CPU |
| ------ | ----------------------------: | --------------------------------: | --------: | ----------------: | --------------: |
| tiny   | 22.8 / 21.9 / 18.7 → **21.9** | 408.1 / 394.5 / 401.9 → **401.9** | **18.4×** |              50.7 |           312.6 |
| small  | 28.5 / 30.8 / 29.5 → **29.5** | 487.9 / 476.0 / 482.8 → **482.8** | **16.4×** |             171.5 |           464.1 |
| medium | 41.0 / 42.3 / 39.5 → **41.0** | 545.9 / 546.2 / 545.6 → **545.9** | **13.3×** |             568.0 |          1403.5 |

同一批数据的三轮 mean、p95、吞吐（mean 取三轮 mean 的均值）：

| 模型   | 列                |   mean |    p95 |    img/s |
| ------ | ----------------- | -----: | -----: | -------: |
| tiny   | Vulkan            |   24.2 |   34.7 | **41.4** |
| tiny   | RapidOcrNet CUDA  |  411.9 |  566.1 |     2.43 |
| tiny   | sharp（本库 CPU） |   56.0 |   87.6 |     17.9 |
| tiny   | RapidOcrNet CPU   |  316.9 |  435.7 |     3.16 |
| small  | Vulkan            |   32.6 |   48.6 | **30.7** |
| small  | RapidOcrNet CUDA  |  494.2 |  711.8 |     2.02 |
| small  | sharp（本库 CPU） |  170.9 |  225.0 |      5.9 |
| small  | RapidOcrNet CPU   |  458.6 |  630.7 |      2.2 |
| medium | Vulkan            |   43.9 |   64.1 | **22.8** |
| medium | RapidOcrNet CUDA  |  548.9 |  719.5 |     1.82 |
| medium | sharp（本库 CPU） |  561.1 |  735.0 |      1.8 |
| medium | RapidOcrNet CPU   | 1407.8 | 1989.3 |     0.71 |

两点值得单独记：

- **RapidOcrNet 的 GPU 在 tiny / small 上还跑不过本库的 CPU**（401.9 / 482.8 vs 50.7 / 171.5），medium 才显出一点优势（545.9 vs 568.0）。
- **RapidOcrNet 自己的 CUDA 相对自己的 CPU**：tiny 401.9 vs 312.6（GPU 反而慢）、small 482.8 vs 464.1（持平）、medium 545.9 vs 1403.5（2.6×）。tiny / small 上 GPU 被会话开销和逐 shape 重规划吃掉。

### 对齐检测输入（rapid `--rapid-img-resize 960`，3 轮）

RapidOcrNet 默认按短边 736 上采样，本机 dataset 的 100 张图平均检测输入 **1.25 MP**；本库按长边 960 缩，平均 **0.61 MP**（2.05× 面积）。把 RapidOcrNet 换成同一几何后：

| 模型   |                        Vulkan |          RapidOcrNet CUDA（对齐） |      倍数 | 未对齐时 |
| ------ | ----------------------------: | --------------------------------: | --------: | -------: |
| tiny   | 20.2 / 16.2 / 18.0 → **18.0** | 313.0 / 278.6 / 261.9 → **278.6** | **15.5×** |    401.9 |
| small  | 29.6 / 29.8 / 31.7 → **29.8** | 417.4 / 349.3 / 341.0 → **349.3** | **11.7×** |    482.8 |
| medium | 42.2 / 42.3 / 43.3 → **42.3** | 461.6 / 388.2 / 389.4 → **389.4** |  **9.2×** |    545.9 |

### 换成 16 workers（各自最有利的并行度，1 轮）

| 模型   | Vulkan 16w | RapidOcrNet CUDA 16w |  倍数 |         rapid 4w → 16w |
| ------ | ---------: | -------------------: | ----: | ---------------------: |
| tiny   |       21.5 |                313.2 | 14.6× | 401.9 → 313.2（1.28×） |
| small  |       29.1 |                345.4 | 11.9× | 482.8 → 345.4（1.40×） |
| medium |       39.8 |                413.6 | 10.4× | 545.9 → 413.6（1.32×） |

### 固定尺寸对照（100 张同尺寸 1488×1248，1 轮）

把 dataset 换成同一张图复制 100 份，检测输入 shape 就不再逐图变化：

| 模型   | Vulkan | RapidOcrNet CUDA |  倍数 | 变尺寸时 rapid |
| ------ | -----: | ---------------: | ----: | -------------: |
| tiny   |   20.0 |            336.7 | 16.8× |          401.9 |
| small  |   32.8 |            430.9 | 13.1× |          482.8 |
| medium |   39.9 |            549.7 | 13.8× |          545.9 |

形状稳定只让 rapid tiny / small 快了 ~15%（401.9 → 336.7、482.8 → 430.9）：**逐图新 shape 不是主因，逐行新宽度才是**（REC 每个 crop 一个宽度，固定尺寸图里也照样一个不重复）。

## 为什么差这么多

### 阶段拆分（默认那三轮的 mean，ms/图；4 worker 下各阶段重叠，之和可以大于墙钟）

| 阶段                        | Vulkan tiny |      rapid tiny | Vulkan small |     rapid small | Vulkan medium |    rapid medium |
| --------------------------- | ----------: | --------------: | -----------: | --------------: | ------------: | --------------: |
| det_preprocess              |        3.18 |               — |         2.39 |               — |          2.69 |               — |
| det_graph                   |        7.10 |      **196.49** |         4.59 |      **185.71** |          4.70 |      **187.13** |
| det_postprocess             |        3.66 |               — |         2.45 |               — |          2.36 |               — |
| crop                        |        2.83 |               — |         2.10 |               — |          2.67 |               — |
| cls_preprocess              |        0.64 |               — |         0.49 |               — |          0.59 |               — |
| cls_graph                   |        1.38 |           27.09 |         1.12 |           22.89 |          0.97 |           21.89 |
| rec_preprocess              |        1.95 |               — |         1.64 |               — |          1.92 |               — |
| rec_graph                   |        5.99 |      **554.16** |        15.37 |      **825.45** |         24.14 |      **798.17** |
| lines_wall                  |       10.09 |               — |        18.73 |               — |         27.75 |               — |
| pipeline_wall / detect_call |           — | 400.71 / 418.59 |            — | 478.97 / 496.09 |             — | 532.06 / 549.06 |

语义提醒：右列是 RapidOcrNet 自己的秒表。`det_graph` = `OcrResult.DbNetTime`，含该库的 resize + 逐像素 float 张量填充 + 轮廓后处理；`cls_graph`/`rec_graph` = 每个 crop 的 `AngleTime`/`CrnnTime` 之和，含该 crop 的 resize。左列同名的 `det_graph` 只算图执行。所以表里的差距同时包含“谁做了什么”和“同一件事谁快”。

### 直连 ONNX Runtime 的探针（同一块 3080 Ti，同一个 `models/det.onnx` / `models/rec.onnx`）

| 场景                              | 每次 `session.Run` |
| --------------------------------- | -----------------: |
| det 960×864 连跑同 shape          |             5.7 ms |
| det 1248×1504 连跑同 shape        |             9.7 ms |
| det 在 960×864 ↔ 608×960 之间交替 |           58–60 ms |
| rec 48×320 连跑同宽度             |             1.5 ms |
| rec 在 8 个宽度间轮转             |             ~31 ms |

即：**ORT CUDA 每次换输入 shape 多花约 30–55 ms**（推断为逐 shape 的 cuDNN/内核与内存计划重做；缓存不生效）。试过且**无效**的开关：`cudnn_conv_algo_search={HEURISTIC,DEFAULT}`、`cudnn_conv_use_max_workspace=1`、`arena_extend_strategy=kSameAsRequested`、`enable_mem_pattern=0` —— 六次 Run 的均值仍在 57–60 ms（同 shape 连跑是 5.7 ms）。

把探针和 harness 对上：固定尺寸那组里，rapid 的 `det_graph` 均值 171 ms，而同 shape（1248×1504）的 ORT det 会话只要 9.7 ms —— **约 94% 的检测时间是 RapidOcrNet 自己的 CPU 侧代码**（Skia resize、逐像素张量填充、PContour 轮廓）。REC 同理：CPU 版 `rec_graph` tiny 79 ms，CUDA 版 554 ms，每个 crop 新 shape 让 GPU 版比 CPU 版还慢 7 倍。

本库这一侧：4 个 line worker + GPU 上 REC 批 16 把 10–15 行压进 6–24 ms 的 `rec_graph`，预处理是查表 + 向量化（det_preprocess 2–3 ms、det_postprocess 2–4 ms）。

## 正确率（100 张满勤，1036 行；三轮逐轮相同）

| 模型   | Vulkan cls / exact_lines / CER | RapidOcrNet CUDA cls / exact_lines / CER |                  rapid 对齐后 |
| ------ | -----------------------------: | ---------------------------------------: | ----------------------------: |
| tiny   |   1022/1022 · 741/1036 · 2.38% |              993/1022 · 743/1036 · 4.52% |   997/1028 · 745/1036 · 5.00% |
| small  |   1034/1034 · 950/1036 · 0.40% |             1008/1035 · 942/1036 · 1.67% |  1004/1034 · 963/1036 · 2.13% |
| medium |  1035/1035 · 1006/1036 · 0.14% |            1008/1034 · 1015/1036 · 0.99% | 1003/1030 · 1011/1036 · 2.00% |

（RapidOcrNet CPU 与它自己的 CUDA 逐档一致：740/942/1014 行、CER 4.72%/1.66%/0.99%、cls 992/1008/1008，说明差异来自预处理而不是后端。）

读法：**同一套权重 ≠ 同一套前后处理。** 两边的行精确互有胜负（rapid 在 tiny/medium 略高、small 略低），但 RapidOcrNet 的 CER 一致更差，而 cls（180° 判定）一致少对 25–30 行——后者的已知来源是 RapidOcrNet 的 cls 走 `ClsPreserveAspectRatio`（保持长宽比 + 中灰填充）并且有 `ClsThresh=0.9` 的门限，本库的 cls 用另一套缩放与阈值策略。要把精度也拉平，应统一 det 尺寸/阈值与 cls 预处理再比，这一层没有做。

## 内存（MB，工作集，1 轮）

| 模型   | Vulkan loaded / peak | RapidOcrNet CUDA loaded / peak | Vulkan 分配量/轮 | rapid 分配量/轮 |
| ------ | -------------------: | -----------------------------: | ---------------: | --------------: |
| tiny   |            498 / 639 |                     729 / 3012 |              147 |            2500 |
| small  |            549 / 700 |                     732 / 3082 |              201 |            2500 |
| medium |            795 / 982 |                     737 / 3283 |              257 |            2500 |

RapidOcrNet 的 GPU 版把进程工作集顶到 3 GB 左右（含 CUDA arena），每轮 GC 分配量稳定在 2.5 GB（25 MB/图，逐像素张量填充的代价），本库是 147–257 MB。显存侧 rapid 另占约 2.3 GB（`nvidia-smi`）。

## 复现

```powershell
# 1) 带 RapidOcrNet 的 harness（默认不编，CI 不受影响）
dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -p:EnableRapidOcr=true

# 2) ORT ≥1.23 需要的 CUDA 12 / cuDNN 9 运行库
test/Sdcb.SimdPaddleOCR.GpuBench/tools/rapid-cuda-runtime.ps1

# 3) A/B：默认配置 / 对齐 / 16 workers / 固定尺寸
test/Sdcb.SimdPaddleOCR.GpuBench/tools/rapid-ab.ps1 -tag rapid-3080ti -rounds 3
test/Sdcb.SimdPaddleOCR.GpuBench/tools/rapid-ab.ps1 -tag rapid-3080ti -label align -rounds 3 -extra "--rapid-img-resize 960"
test/Sdcb.SimdPaddleOCR.GpuBench/tools/rapid-ab.ps1 -tag rapid-3080ti -label w16 -rounds 1 -workers 16
test/Sdcb.SimdPaddleOCR.GpuBench/tools/rapid-ab.ps1 -tag rapid-3080ti -label cpu -rounds 1 -engines @('sharp','rapid-cpu')

# 固定尺寸对照的输入：同一张图复制 100 份（没有 metadata.json，因此只计时不评分）
mkdir -Force bench-out/tmp/fixed-shape
1..100 | % { Copy-Item dataset/img-002.jpg ('bench-out/tmp/fixed-shape/img-{0:d3}.jpg' -f $_) }
test/Sdcb.SimdPaddleOCR.GpuBench/tools/rapid-ab.ps1 -tag rapid-3080ti -label fixed -rounds 1 -input bench-out/tmp/fixed-shape
```

单次等价的命令行：

```text
Sdcb.SimdPaddleOCR.Tests --benchmark --benchmark-kind simd --engine rapid     --workers 4 --model {tiny|small|medium} --input dataset --warmup 1 --out <json>
Sdcb.SimdPaddleOCR.Tests --benchmark --benchmark-kind simd --engine vulkan    --workers 4 --model {tiny|small|medium} --input dataset --warmup 1 --out <json>
```

产物：`bench-out/rapid-3080ti/*-summarize.md`（`--summarize` 并排输出，含 e2e / 精度 / 阶段）、`run-*.log`、`bench-out/rapid/cuda-runtime/`（CUDA 运行库，bench-out 不入库）。

## 公平性自查（反方会怎么说）

先做最容易被攻击的几条，逐条给数据：

| 可能的质疑 | 实验 | 结果 |
|---|---|---|
| “你们拿自己的 onnx 权重跑别人的运行时，权重被你们优化过” | 用 RapidOcrNet **自带**的 v5 模型（det/rec/cls/dict 全用它的，本仓库文件一个不碰）跑同一批图的前 25 张 | median 421.0 / 385.6 ms（两次），同一子集用我们的 v6 权重是 412.0 ms —— 在噪声内（两次同配置差 9%） |
| “那个逐 shape 开销是你们模型导出的毛病” | 用它的 v5 det（`ch_PP-OCRv5_mobile_det.onnx`，输入同为 `-1,3,-1,-1`）做同样的 shape 交替 | 同 shape 6.8 ms ↔ 交替 44.3 ms，和我们的 det（5.7 ↔ 59.6）**同性质同量级** |
| “ORT 1.29 的 CUDA 回归” | 同探针换 `Microsoft.ML.OnnxRuntime.Gpu` **1.22.0**（更老，需要额外补 `cufft64_11.dll`） | 同 shape 6.2 ms ↔ 交替 62.8 ms，**1.22 也一样** |
| “你们没帮它调 session 选项” | `ORT_ENABLE_ALL` + `intra/inter = 8/4`、以及 `execution_mode=parallel` + `16/16` | 60–64 ms，没有改善（默认 `ORT_ENABLE_EXTENDED` 是 59.6 ms）；另外 `cudnn_conv_algo_search=HEURISTIC/DEFAULT`、`enable_mem_pattern=0`、`arena_extend_strategy` 也都试过，无效 |
| “GPU 快的那部分其实是在比 fp16（我们）对 fp32（它）” | 用本仓库 **fp32** 的 CPU 引擎（`--engine sharp`，纯 fp32 SIMD）对照它的 fp32 CUDA | tiny 50.7 vs 401.9、small 171.5 vs 482.8、medium 568.0 vs 545.9 —— 同样是 fp32，本库 CPU 在 tiny/small 上还是快 4–8 倍 |
| “你们给 rapid 的活更少” | 逐图检出行数（n=99） | rapid 1096 / 1061 / 1032 行 ≥ 本库 1079 / 1030 / 1025 行，它检得还更多一点 |
| “你们把 Vulkan 调过了” | 本库这一列用的是仓库默认（`LimitSideLength=960`、GPU REC 批 16），没有为这次实验改过参数 | 与本机已发布的 2.0 数字一致（tiny 2.0 报告为 18.5–23.0 ms median，本文 21.9 ms） |

仍然成立、需要在引用结论时一起说的几条（不是“不公平”，但要划清边界）：

- **比的是“库的端到端实现”，不是“ORT CUDA 内核速度”**。本库是一个为 OCR 写死的引擎（一次调度 DET/CLS/REC、fp16 arena、REC 批 16、4 line worker），RapidOcrNet 是三个 ORT 会话 + 通用前后处理的封装。差距的一部分来自这层“专用 vs 通用”，文中已用探针把“ORT 内核有多快”（同 shape 5.7–9.7 ms）单独拆出来。
- **精度契约不同**：本库 GPU 走 fp16，ORT 跑的是同一批 fp32 onnx。上面 fp32 CPU 那一行是这条的对照，但严格说“fp16 引擎 vs fp32 引擎”这个组合确实不同。
- **一台机器、一份数据集、单请求延迟**。没有测多并发吞吐（两边都是一次一图，服务器上并发跑会各自放大）、没有测真实照片集、没有测 ONNX Runtime 的 DirectML/TensorRT EP、也没测 fp16/量化过的 onnx 交给 ORT。
- **一处对 rapid 小不利的记账**：harness 给它的是 BGR24，它只收 `SKBitmap`，多一次 BGR→BGRA 拷贝（单图 2.0–2.9 ms，stage `input_convert`）。在它自己的 400 ms 里占 0.5%，但如果你要引用 tiny 那档的 18.4×，记得这条：扣掉也只到 17.7×。
- **`--workers 4` 这个口径对上的是“一图内并行度”**（本库 line worker / rapid `RecMaxDegreeOfParallelism`），不是服务器并发。

## 替 RapidOcrNet 找更快的路：DirectML 与 fp16（都试了）

结论：这两条路都救不了，一个用不上、一个是负收益。

| 路径 | 实测（同探针、同模型、同 shape 序列） | 能不能用在 RapidOcrNet 4.2.0 上 |
|---|---|---|
| **DirectML EP**（`Microsoft.ML.OnnxRuntime.DirectML` 1.24.4） | 同 shape 与 CUDA 持平（det 6.9 / rec 1.5 ms）；**逐 shape 开销只有 CUDA 的一半**（det ~25 ms vs 55、rec ~12 ms vs 30） | ❌ DirectML 的 NuGet 包止于 **1.24.4**，而 RapidOcrNet 4.2.0 依赖 ORT **≥ 1.29**，装不进去 |
| **fp16 onnx**（`tools/convert-fp16.py`，保持 fp32 IO） | rec/cls 转换后**精度与 fp32 逐行相同**（CER 2.77% 对 2.77%，exact 201/268 对 201/268），但**更慢**：rec 逐 shape 47.6 ms（fp32 31）、端到端 25 张 552 vs 377 ms。det 转 fp16 直接失效：0 框 | ⚠️ 能跑，负收益 |

顺带一个有意义的对照：即使把两个阶段的逐 shape 开销都按 DirectML 的实测值替换（det 55 → 25 ms、rec 每行 30 → 12 ms），rapid 的端到端也只从 402 ms 降到约 300 ms 量级，相对 Vulkan 的 21.9 ms 仍约 14×。**换 EP 或换精度都不改变这个量级**——它的时间不在算子算力上。

## 已知坑

- **ORT CUDA 某些尺寸会直接抛错**：同一个 det 模型、输入 1248×1488（长边恰好能被 16 整除）连跑两次，会报 `Shape mismatch attempting to re-use buffer. {1,64,78,93} != {1,64,78,94}`，Run 失败。本 dataset 的尺寸没踩到（都是 /32 之后的值），但换数据集时值得留意。
- **RapidOcrNet 的 det 时间分辨率是整数毫秒**（`DbNetTime` 用 `ElapsedMilliseconds`），单图看不了细节，均值仍可用。
- **ORT < 1.23 的 CUDA provider 还要 `cufft64_11.dll`**（1.29 不需要）。上面的 1.22 对照就是补了它才跑起来的；`tools/rapid-cuda-runtime.ps1` 只保证 RapidOcrNet 4.2.0 钉住的 1.29 能跑。
- **`--engine rapid` 报“needs a build with RapidOcrNet”时**，说明当前 exe 是没有 `-p:EnableRapidOcr=true` 的那次构建（IDE 后台自动构建也会覆盖它），重新带参数编一次即可。
- CUDA EP 缺 cuDNN 时不是硬失败：会打一行 `[E:onnxruntime:] cuDNN is unavailable...`，然后逐图返回 0 行（本次第一版就是这样，精度一栏会直接掉到 0%）。跑之前确认 `cudnn64_9.dll` 在 `PATH` 上。
- 本文所有绝对毫秒只在这台 5800X / 3080 Ti / 驱动 581.80 上可比；跨机请只看倍数。
