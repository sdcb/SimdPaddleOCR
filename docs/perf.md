# Sdcb.SimdPaddleOCR 性能基准

## 怎么读

- **墙钟**：去掉首张 warmup 后的 **median ms/图**（`--warmup 1`）。本机 5800X 表额外报 mean，和 median 几乎重合。
- **准确率、Δ WS**：初始化后满勤（100 张 / smoke 20 张），**不再跳过首张**。CI tiny bench 是 **767/1032、CER 2.36%**（cls 1020/1020）；本机 5800X 是 **742/1036、2.37%**。两边都是 `windows` 生成机、字体齐，但是各自出的图，行数就是 **1032** 和 **1036**，不要对绝对 exact。1.4.2 与 2.0 的行精确 / CER 逐行相同，质量差异只出现在 GPU fp16 边界（见本机表）。
- **倍数口径**：统一按「加速比 = 慢 ÷ 快」，**>1 表示更快**（与 [vulkan-b580.md](vulkan-b580.md) 一致）。同 replica 比是单 VM 内对基线 case 的倍数。
- **replica**：每台 GitHub-hosted VM 一份。先在单 replica 内算比值，再只汇总 **同一 CPU**。
- GitHub `windows-2025` 会随机分到 EPYC 7763 / 9V74 / 9V45 / Xeon。**7763 没有 AVX-512**；9V74 / 9V45 / Xeon 有时走 AVX-512。这几类绝对时间不可比。`ubuntu-24.04-arm` 全部是 Neoverse N2（`CPU part 0xd49`），最稳。
- 4 worker 下算子会并行重叠，**之和可以大于墙钟**，只适合看结构。
- **20 张 smoke 不能和 100 张 bench 比快慢**；osx-arm64 / osx-x64 绝对毫秒不能做门禁。
- 1.4.2 之前的旧基线（1.2 / 1.3 / 1.4）已删；相对数字见 git 历史里的当时文档。

## 2.0 改了什么

主打 **GPU 后端**：纯 C# **Vulkan**（DET/CLS/REC 全图调度，win / linux / 安卓）与纯 C# **Metal**（macOS），`OcrBackend.{Vulkan,Metal,Auto}`，**net10.0 专属**；`netstandard2.0` 只留 CPU（取舍见 [vulkan-b580.md](vulkan-b580.md)）。GPU 内部细节（coopmat GEMM、显存 arena、回落路径、已知缺口）也在该文档；各设备实测见[文末](#其他-gpu-设备)。

CPU 侧相对 **1.4.2**（`68a009a`，与 `v1.4.2` 标签同源码；仓库 onnx 自 v1.4.2 起没改过，差值都是代码不是权重）：

| 改动 | 提交 | 可见效果 |
| ---- | ---- | ---- |
| REC 并行 CTC ArgMax + 动态 intra-op 预算 | `085219c` | rec_graph 时间 **0.83–0.93×**（加速约 1.1–1.2×）；要空闲核，CI 4 vCPU 上持平 |
| 流式 REC 批 CTC ArgMax 按列拆分 | `990994b` | 同上，AVX-512 机器额外受益 |
| ARM64 AdvSIMD MatMul + fused ArgMax | `a21dc99` | N2 `tiny-1w` / `tiny-4w` 加速 **1.25× / 1.11×** |
| Det flood-fill 游程扫描向量化（`DbPostprocess`） | `22c2ce1` | det_postprocess 加速 **~2.5×**（5.4 → 2.2 ms/图） |
| Shape-plan 缓存 + 权重 pack 去重 | `8cbc299` | rec_reshape 略降；首图 / 冷启开销变小 |
| Det 残差 Add 吸收顺序、GPU 回落输入布局等修复 | 见 [vulkan-b580.md](vulkan-b580.md) | 正确性修复；CPU 输出逐行不变 |

结论（细节在后面几节）：

- **正确率**：与 1.4.2 逐行相同——CI sharp **767/1032、CER 2.36%**（cls 1020/1020）、c **764/1032、3.03%**（cls 998/1019）；本机 742 / 950 / 1004 行、CER 2.37% / 0.41% / 0.14%。
- **墙钟（CPU）**：AdvSIMD 最大（N2 1w / 4w 加速 **1.25× / 1.11×**）；x64 桌面（16 逻辑核）本机 tiny / small / medium 加速 **1.11× / 1.15× / 1.04×**；CI win-x64（4 vCPU）持平；ns2 / scalar 路径不变。
- **墙钟（GPU）**：本机 3080 Ti 相对当前 CPU 加速 **2.2× / 6.1× / 14.4×**（tiny / small / medium），相对已发布的 1.4.2 是 **2.5× / 7.0× / 15.0×**。
- **内存**：CPU 两版基本持平（本机 medium peak 1208 → 1170 MB）；Vulkan 显存 arena 按生命周期复用，medium GPU peak 反而最低（984 MB）。

## CI：1.4.2 → 2.0

两次口径：**1.4.2** = [35513018083](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35513018083)（`68a009a`）；**2.0** = [36727088539](https://github.com/sdcb/SimdPaddleOCR/actions/runs/36727088539)（`f8b1197`）。回归只看这两条：**win-x64 EPYC 7763** 的 SIMD / 引擎套件，以及 **linux-arm64 Neoverse N2**（`CPU part 0xd49`，6 replica 几乎一条直线）。osx-arm64 是 3 核 / 7 GB 虚拟 M1，只看同 replica 比值。CI VM 只有 **4 vCPU**——2.0 的并行 CTC ArgMax 收益需要空闲核，所以 win-x64 CI 上 CPU 墙钟基本持平，收益要看 ARM 内核与本机 5800X。

| 判定 | 尺子 |
| ---- | ---- |
| x64 | 引擎套件（warm 例）`tiny-4w` 中位大约 **137–152 ms**；SIMD 套件首例含冷启更高（148–169）。`tiny-4w-ns2` 同 replica 比值大约 **1.4–1.6**。不要用一份 replica 喊回归 |
| ARM64 | `tiny-4w` **160–167 ms**；1w 比 **1.42**、ns2 比 **1.79–1.81**、scalar 比 **5.13–5.26**。比 Windows 更适合做自动阈值 |
| 引擎 | 只用 win-x64 引擎套件、同一 replica。c/sharp 4w 大约 **1.75–1.82**（c 钉 `20d0de6` DLL，两版对照） |
| 不要 | 把 9V74 / 9V45 / Xeon 的 AVX-512 和 7763 的 AVX2 写成「优化了 xx%」；用 osx 绝对时间做门禁；拿 20 张 smoke 和 100 张 bench 比 |

win-x64 SIMD 先丢一次 25 张 tiny-4w 烤 VM（不上传），再跑默认 / noavx512 / ns2。noavx2 / noavx / scalar 只在 `smoke-win-x64-isa` 跑 20 张。**tiny-4w 是套件第一例，绝对值含冷启开销**（2.0 的 plan 缓存让首例开销变小，跨版本比较首例要小心）。

数据集、模型、预解码 BGR、ISA 开关、runner：`.github/workflows/test.yml`，`dataset/` 固定种子合成 100 张 JPG。库 TFM 默认 `net10.0`；`tiny-4w-ns2` 把库编成 `netstandard2.0`，仍跑在 .NET 10 上。

正确率（100 张满勤，两版逐行相同）：sharp **767/1032、CER 2.36%**（cls 1020/1020；含 scalar、ns2、全部 ISA）；c **764/1032、3.03%**（cls 998/1019）。smoke 20 张满勤：tiny **160/218**（CER 1.90%），small **198/218**（0.40%），medium **210/218**（0.13%）。

### linux-arm64 N2（最稳，6 replica）

同一 `ubuntu-24.04-arm`、同一 `0xd49`，两版各 6 份 replica；中位是各 replica median 的中位数。

| 用例             | 1.4.2 中位 | 2.0 中位 |      加速 | 1.4.2 同 replica 比 | 2.0 同 replica 比 |
| ---------------- | -------: | -------: | --------: | ----------------: | ----------------: |
| `tiny-4w`        |  **180** |  **163** | **1.11×** |              1.00 |              1.00 |
| `tiny-1w`        |      287 |  **231** | **1.25×** |              1.60 |              1.42 |
| `tiny-4w-ns2`    |      296 |      293 |     1.01× |              1.65 |              1.80 |
| `tiny-4w-scalar` |      859 |      856 |     1.00× |              4.77 |              5.26 |

- 收益全在 net10 AdvSIMD：`a21dc99` 的 AdvSIMD MatMul + fused ArgMax，1w / 4w 加速 **1.25× / 1.11×**。范围无重叠（4w 177–185 → 160–167，1w 286–292 → 227–235）。
- ns2（ARM 上仍是 `Vector` 128-bit）与 scalar 不吃这次内核收益，持平。4w 自己变快后，相对倍数被拉开：ns2 1.65 → **1.80**，scalar 4.77 → **5.26**。
- 内存两版持平（4w peak 约 570 MB、Δ WS 约 160 MB）。

### win-x64 SIMD（只报 EPYC 7763）

两版的 7763 份数都少（1.4.2 3 份、2.0 2 份，其余 replica 落到 9V74 / 9V45 / Xeon）。中位是各 replica median 的中位数。**tiny-4w 是套件第一例，绝对值含冷启；noavx512 / ns2 是其后的 warm 例。**

| 用例               | 有效 ISA      | 1.4.2 中位（范围） | 2.0 中位（范围） |             加速 | 1.4.2 同 replica 比 | 2.0 同 replica 比 |
| ------------------ | ------------- | -----------------: | ---------------: | ---------------: | ------------------: | ----------------: |
| `tiny-4w`          | AVX2          |     169（167–174） |  152（148–155）  | 1.12×（含冷启差） |                1.00 |              1.00 |
| `tiny-4w-noavx512` | AVX2          |     142（137–145） |  140（139–141）  |            1.01× |                0.84 |              0.92 |
| `tiny-4w-ns2`      | Vector / NHWC |     232（229–243） |  232（227–236）  |            1.00× |                1.37 |              1.53 |

- **warm 例（noavx512、ns2）与 1.4.2 持平。** win-x64 AVX2 墙钟没有变快：CI VM 只有 4 vCPU，`085219c` 的并行 CTC ArgMax 没有空闲核可用；本机 5800X / 5950X（8C16T）上同一改动让 rec_graph 降到 **0.83×**（见本机表）。
- tiny-4w 的 1.12× 里有一部分是「首例冷启」开销在 2.0 变小（`8cbc299` plan 缓存 + pack 去重），不要当纯内核收益；更不要拿一份 replica 喊回归（7763 单次 183–221 都见过）。
- `noavx512` 在 7763 上本来就没有 AVX-512，和 `tiny-4w` 同 ISA，两者差值是跑序 / 冷启。`noavx2` / `noavx` / `scalar` 只在 20 张 `smoke-win-x64-isa` 里跑（见平台 smoke）。

### win-x64 引擎套件（只报 EPYC 7763）

和 SIMD job 不是同一台 VM，绝对毫秒不要和上一张表硬接。c 钉 [`lw_ppocr_c.20260914.20d0de6.dll`](https://cv-public.sdcb.ai/2026/lw_ppocr_c.20260914.20d0de6.dll)，两版都没换，是对照组；1.4.2 5 份 7763、2.0 4 份。

| 用例            | 1.4.2 中位 | 2.0 中位 |   加速 | 1.4.2 vs sharp 4w | 2.0 vs sharp 4w |
| --------------- | -------: | -------: | -----: | ----------------: | --------------: |
| sharp `tiny-4w` |      140 |      139 |  1.00× |              1.00 |            1.00 |
| sharp `tiny-1w` |      198 |      189 |  1.05× |              1.42 |            1.35 |
| c `tiny-4w`     |      245 |      254 |  0.96× |              1.75 |            1.82 |
| c `tiny-1w`     |      353 |      356 |  0.99× |              2.53 |            2.55 |

- sharp 4w 持平，1w 加速 **1.05×**；和 SIMD 套件 warm 例一致——4 vCPU 上 2.0 的 CPU 墙钟没有大动。c 对照组也在噪声内（c-4w 的 2.0 有一份 327 ms 离群 replica，中位 254 里含它）。
- c/sharp 4w 约 **1.8×**，与 1.4.2 的 1.75× 同级。c 仍然略省内存（peak ~536 MB vs sharp ~511 MB）。

### osx-arm64（高噪声，只看比值）

虚拟 M1、3 逻辑核、7 GB。绝对毫秒 replica 之间可以差一倍。

| 用例             | 1.4.2 同 replica 比 | 2.0 同 replica 比 |
| ---------------- | ------------------: | ----------------: |
| `tiny-4w`        |                1.00 |              1.00 |
| `tiny-1w`        |    1.74（1.34–2.07） |  1.60（0.96–1.62） |
| `tiny-4w-ns2`    |    1.52（1.31–1.56） |  1.45（1.25–2.07） |

2.0 新增 **Metal** smoke（同一虚拟 M1）：`metal` 中位 **100 ms** vs 同 RID `sharp` CPU **280 ms**，约 **2.6–2.8×**（跨 VM 只看量级），行精确 161/218（CPU 160/218）。本套只用来确认 AdvSIMD / Metal 路径能跑。

### 平台 smoke（20 张，只看覆盖）

行精确按 **20 张满勤**，两版逐行相同：tiny **160/218**（CER 1.90%），small **198/218**（0.40%），medium **210/218**（0.13%）。中位 ms / CPU 各取本版 run，只看覆盖——**同一 RID 两次分到的 CPU 经常不同，标注 ✗ 的行不要跨版本比快慢**。

| 用例              | 1.4.2 CPU / 中位 ms | 2.0 CPU / 中位 ms | 可比             |
| ----------------- | ------------------: | ----------------: | ---------------- |
| linux-arm64       |          N2 / 224   |        N2 / 208   | ✓ 同 CPU，1.08× |
| linux-x64 tiny    |       7763 / 319    |     7763 / 344    | 同 CPU，20 张噪声 |
| linux-x64 small   | Xeon 6973P-C / 431  |     9V45 / 270    | ✗ 不同 CPU       |
| linux-x64 medium  | Xeon 6973P-C / 1666 |    9V45 / 1054    | ✗ 不同 CPU       |
| win-x64           |  Xeon 8573C / 192   |    7763 / 215     | ✗ 不同 CPU       |
| win-x86           |       7763 / 379    |    9V74 / 198     | ✗ 不同 CPU       |
| win-arm64         |  Cobalt 100 / 219   | Cobalt 100 / 199  | ✓ 同 CPU，1.10× |
| osx-arm64         |   M1 Virtual / 339  | M1 Virtual / 280  | 高噪声           |
| osx-arm64 metal   |                  —  | M1 Virtual / 100  | 2.0 新增         |
| osx-x64           |     i7-8700B / 244  |  i7-8700B / 465   | 高噪声           |

ISA smoke（win-x64 7763、两版同 CPU、20 张）：noavx（Vector）424 → 402，noavx2（AVX）425 → 456，scalar 1261 → 1272——噪声内持平。small 大约是 tiny 的 3 倍墙钟，medium 大约是 tiny 的 15–17 倍（同 CPU 内部比值稳定）。

### CI 工作集（tiny-4w）

| 场景                 | 1.4.2 peak | 2.0 peak | 1.4.2 Δ WS | 2.0 Δ WS |
| -------------------- | ---------: | -------: | ---------: | -------: |
| win-x64 7763 sharp   |   513 MB   |  511 MB  |   106 MB   |  102 MB  |
| linux-arm64 N2 sharp |   571 MB   |  573 MB  |   161 MB   |  163 MB  |
| osx-arm64 sharp      |   705 MB   |  704 MB  |   290 MB   |  289 MB  |
| win-x64 lw.PPOCR.C   |   536 MB   |  536 MB  |   104 MB   |  105 MB  |

1.4.2 → 2.0 的 CI 工作集持平（1.3 → 1.4.2 时代 x64 / ARM64 少的 ~300 MB 保持）。c 仍然略省，但更慢。本机 small / medium 的 CPU peak 略降（674 → 650、1208 → 1170 MB）。

## 本机 5800X / RTX 3080 Ti（2.0 发布复测）

实测机：HOME-MAIN，Windows 10.0.26200，电源方案“高性能”，Ryzen 7 5800X（8C/16T，AVX2，无 AVX-512），64 GB，RTX 3080 Ti（驱动 581.80）。运行时 .NET 10.0.11。`test/Sdcb.SimdPaddleOCR.Tests`，`--workers 4 --benchmark-kind simd --warmup 1`，n=99，同一 `dataset/` 100 张变尺寸图（对 GPU 最不利的逐图新 shape）。先用当前树 tiny 25 张烤机，再按 tiny → small → medium、每档 **1.4.2 CPU → 当前 CPU → Vulkan** 交替跑 3 轮。本表 `a75a0e9`。

两条 CPU 用同一套选项（`LineWorkerCount=4`、`AdaptiveWidth`、`TargetWidth=320`、session cache 32、REC 批大小保持默认 1）。1.4.2 是 NuGet `Sdcb.SimdPaddleOCR` **1.4.2** 加模型包 **1.0.0**（harness 临时改成 `UseNuget142=true` 另编一份）。当前树是同一 harness 的源码引用，`--engine sharp` 钉死 CPU，`--engine vulkan` 走 GPU；GPU 在批大小未显式设置时用默认 16。仓库里的 onnx 自 `v1.4.2` 起没有改过，CPU 差值是代码，不是权重。不要用这里的绝对毫秒卡 GitHub runner（CI VM 只有 4 vCPU）。

**结论：Vulkan 三档都快于两份 CPU。相对当前 CPU，端到端中位是 tiny 2.2× / small 6.1× / medium 14.4×；相对已发布的 1.4.2 是 2.5× / 7.0× / 15.0×。当前 CPU 比 1.4.2 快（1.11× / 1.15× / 1.04×），行精确和 CER 与 1.4.2 逐行相同。Vulkan 正确率与当前 CPU 持平（tiny 少 1 行，medium 多对 2 行，CER 持平）。**

### 端到端（4 workers，median ms/图，越低越好）

三轮 median 都列出来，加粗的是三轮的中位数。加速是较慢一列的中位毫秒除以较快一列（大于 1 表示更快）。

| 模型 | 1.4.2 CPU | 当前 CPU | Vulkan | 当前 CPU / 1.4.2 | Vulkan / 当前 CPU | Vulkan / 1.4.2 |
|---|---:|---:|---:|---:|---:|---:|
| tiny | **55.1** / 50.9 / 56.3 | **49.6** / 43.7 / 49.7 | 18.5 / 23.0 / **22.1** | 1.11× | **2.2×** | 2.5× |
| small | **191.1** / 189.2 / 194.7 | 172.4 / **166.8** / 153.9 | **27.4** / 28.1 / 25.9 | 1.15× | **6.1×** | 7.0× |
| medium | 542.6 / **545.8** / 558.2 | 521.8 / 528.7 / **523.2** | **36.4** / 36.6 / 35.9 | 1.04× | **14.4×** | 15.0× |

三轮 mean 的均值，以及由此得到的 img/s（1000 / mean）。p95 是三轮 p95 的中位数。工作集峰值是三轮里的最大值。

| 模型 | 列 | mean | p95 | img/s | WS peak |
|---|---|---:|---:|---:|---:|
| tiny | 1.4.2 CPU | 59.9 | 92.7 | 16.7 | 516 MB |
| tiny | 当前 CPU | 52.2 | 83.8 | 19.2 | 516 MB |
| tiny | Vulkan | 23.7 | 33.5 | **42.2** | 642 MB |
| small | 1.4.2 CPU | 190.1 | 247.5 | 5.3 | 674 MB |
| small | 当前 CPU | 166.4 | 231.3 | 6.0 | 650 MB |
| small | Vulkan | 30.9 | 59.8 | **32.3** | 702 MB |
| medium | 1.4.2 CPU | 544.4 | 709.6 | 1.8 | 1208 MB |
| medium | 当前 CPU | 518.0 | 666.6 | 1.9 | 1170 MB |
| medium | Vulkan | 40.0 | 59.8 | **25.0** | 984 MB |

### 分阶段耗时（三轮 stage mean 再平均，ms/图）

4 worker 下各阶段重叠，加总可以大于墙钟。倍数是 Vulkan 相对当前 CPU。

| 模型 | 阶段 | 1.4.2 CPU | 当前 CPU | Vulkan | 相对当前 CPU |
|---|---|---:|---:|---:|---:|
| tiny | det_preprocess | 1.9 | 1.9 | 2.8 | 0.7× |
| tiny | det_graph | 18.4 | 18.5 | 6.1 | 3.0× |
| tiny | det_postprocess | 5.4 | 2.2 | 2.8 | 0.8× |
| tiny | crop | 3.5 | 3.0 | 2.7 | 1.1× |
| tiny | cls_preprocess | 3.5 | 3.3 | 0.6 | 5.5× |
| tiny | cls_graph | 16.0 | 14.8 | 1.3 | 11× |
| tiny | rec_preprocess | 7.9 | 7.3 | 1.9 | 3.8× |
| tiny | rec_graph | 83.5 | 69.6 | 5.3 | 13× |
| tiny | lines_wall | 30.0 | 25.9 | 9.2 | 2.8× |
| small | det_preprocess | 1.5 | 1.6 | 2.7 | 0.6× |
| small | det_graph | 69.6 | 67.3 | 5.3 | 13× |
| small | det_postprocess | 8.7 | 2.6 | 2.8 | 0.9× |
| small | crop | 4.3 | 3.5 | 3.1 | 1.1× |
| small | cls_preprocess | 3.7 | 3.6 | 0.6 | 6.0× |
| small | cls_graph | 19.2 | 19.1 | 1.0 | 19× |
| small | rec_preprocess | 8.6 | 9.1 | 1.9 | 4.8× |
| small | rec_graph | 358.3 | 302.4 | 13.2 | 23× |
| small | lines_wall | 105.6 | 91.0 | 16.8 | 5.4× |
| medium | det_preprocess | 1.3 | 1.3 | 1.9 | 0.7× |
| medium | det_graph | 181.1 | 177.9 | 8.8 | 20× |
| medium | det_postprocess | 3.4 | 1.1 | 2.1 | 0.5× |
| medium | crop | 1.8 | 1.8 | 2.5 | 0.7× |
| medium | cls_preprocess | 2.6 | 2.6 | 0.5 | 5.2× |
| medium | cls_graph | 15.4 | 15.7 | 0.8 | 20× |
| medium | rec_preprocess | 6.5 | 6.7 | 1.7 | 3.9× |
| medium | rec_graph | 1288.3 | 1197.5 | 21.5 | 56× |
| medium | lines_wall | 356.4 | 335.5 | 24.6 | 14× |

当前 CPU 相对 1.4.2 的墙钟差几乎全在 `rec_graph`（83.5 → 69.6，0.83×；small 358 → 302，medium 1288 → 1198）和 `det_postprocess`（5.4 → 2.2，约 2.5×）；`det_graph` 三档都在 1.4.2 的 ±3% 里。`lines_wall`（1.4.2 → 当前 CPU → Vulkan）：tiny 30.0 → 25.9 → **9.2**，small 106 → 91 → **16.8**，medium 356 → 336 → **24.6**。

### 正确率（100 张满勤，1036 行；三轮逐轮相同）

| 模型 | 1.4.2 exact_lines / CER / exact_img | 当前 CPU | Vulkan |
|---|---:|---:|---:|
| tiny | 742 / 2.37% / 5 | 742 / 2.37% / 5 | 741 / 2.38% / 5 |
| small | 950 / 0.41% / 44 | 950 / 0.41% / 44 | 950 / 0.40% / 44 |
| medium | 1004 / 0.14% / 71 | 1004 / 0.14% / 71 | 1006 / 0.14% / 73 |

cls 在各自检出的行上全对：三列都是 1022/1022、1034/1034、1035/1035。small 与两份 CPU 逐行持平。medium 多对的 2 行、tiny 少的 1 行是 fp16 边界（同款抖动在 det 上已证实无害，定位见 [vulkan-b580.md](vulkan-b580.md) 已知缺口）。

### 内存（MB）

loaded / peak 是三轮范围，Δ WS 是三轮（last − loaded）的均值。

| 模型 | 1.4.2 loaded | 1.4.2 peak | 1.4.2 Δ | 当前 CPU loaded | 当前 CPU peak | 当前 CPU Δ | Vulkan loaded | Vulkan peak | Vulkan Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny | 401 | 515–516 | 114 | 401–402 | 515–516 | 114 | 498–499 | 640–642 | 142 |
| small | 452–453 | 673–674 | 217 | 453 | 648–650 | 189 | 549 | 701–702 | 152 |
| medium | 699–700 | 1208 | 508 | 699 | 1169–1170 | 470 | 796 | 983–984 | 170 |

两份 CPU 的 loaded 几乎一样，当前 CPU 的 peak 略低于 1.4.2（差在跑图 Δ WS）。Vulkan 的 loaded 更高（图和 arena 在加载时就占上）：tiny peak 高于 CPU，small 与 CPU 同级，medium 反而最低。三轮之后不再爬。

### 本机备注

- **rec_graph 收益要有空闲核**：`085219c` 的并行 CTC ArgMax + 动态 intra-op 预算在 16 逻辑核上把 rec_graph 降到 0.83×，但 CI win-x64（4 vCPU）持平。ARM64 的收益是另一条路（`a21dc99` AdvSIMD MatMul + fused ArgMax，跟核数无关）。
- **det_postprocess 约 2.5×** 来自 `22c2ce1` 里 `DbPostprocess` 的 flood-fill 重写：pending / blocked 状态折叠成单字节，`NextNonZero` / `NextZero` 向量化游程扫描。
- 3080 Ti 上 Vulkan 走 sg32 coopmat 档（大 GEMM 25–28 TFLOPS）；GPU 内部数字（纯 GPU 分段、显存 arena、~32-shape 崩坏验证）见 [vulkan-b580.md](vulkan-b580.md)。medium `rec_graph` 21.5 ms 里约一半是 CPU 上的 CTC 投影 + ArgMax（词表 18710 列，契约要求留在 CPU）。
- mean 明显高于 median 主要来自前 ~40 张的 .NET 分层 JIT 预热（`DOTNET_TieredCompilation=0` 下消失），不是 GPU。
- 显存：sg32 上 arena 按生命周期复用，进程工作集有界，peak 后不再爬升。
- JSON：`bench-out/v20-3080ti/{v142,cpu,vk}-{tiny,small,medium}-r{1,2,3}.json`。复现：`--engine sharp|vulkan --workers 4 --model {tiny,small,medium} --input dataset --warmup 1`。

### lw.PPOCR.C 4w（`20d0de6`，对照）

同机同尺子，1.4.2 口径复测时所测；c DLL 与 harness `--engine c` 路径在 2.0 没有变化，CI 引擎套件的 c 对照两版持平（见上文）。`--engine c`、`--c-assets bench-out/c-runtime`，DLL [`lw_ppocr_c.20260914.20d0de6.dll`](https://cv-public.sdcb.ai/2026/lw_ppocr_c.20260914.20d0de6.dll)。C 没有 stage / operator 剖析。

| 模型   |   mean | median |    p95 | img/s |  loaded |        peak |    Δ WS |   exact_lines |   exact_img |       CER |
| ------ | -----: | -----: | -----: | ----: | ------: | ----------: | ------: | ------------: | ----------: | --------: |
| tiny   |  201.9 |  202.7 |  264.4 |   5.0 | 396 MB |     541 MB | 102 MB |      752/1036 |        7/100 |     4.09% |
| small  |  541.7 |  538.5 |  692.5 |   1.8 | 434 MB |     760 MB | 271 MB |      936/1036 |       39/100 |     1.08% |
| medium | 2319.5 | 2362.3 | 2924.0 |   0.4 | 585 MB |    1479 MB | 774 MB | **1014/1036** |       80/100 | **0.24%** |

JSON：`bench-out/local-5800x-c-{tiny,small,medium}-4w.json`。

## 其他 GPU 设备

5800X / 3080 Ti 之外的 Vulkan / Metal 实测（各设备自己的尺子，绝对毫秒不可跨机比）：

| 设备 | 文档 |
| ---- | ---- |
| Ryzen 9 5950X / Intel Arc B580（2.0 发布复测，同款 1.4.2 → 当前 → Vulkan 三列对比） | [vulkan-b580.md](vulkan-b580.md) |
| 锐龙 AI 9 H365 / Radeon 880M 核显 | [vulkan-880m.md](vulkan-880m.md) |
| 骁龙 8 Gen 3 / Adreno 750（安卓，无协作矩阵档） | [vulkan-8gen3.md](vulkan-8gen3.md) |
| Intel UHD 770 核显（无协作矩阵，`Auto` 走 CPU） | [vulkan-uhd770.md](vulkan-uhd770.md) |
| Apple M1 / M4（Metal） | [metal-m4.md](metal-m4.md)、[perf-devin-m4.md](perf-devin-m4.md) |

## 与其它 C# OCR 库对照

同机同尺子、同一套 PP-OCRv6 权重，把别的 ONNX Runtime GPU 实现塞进同一个 harness 的对照实验：

| 对照对象 | 结论 | 文档 |
| ---- | ---- | ---- |
| RapidOcrNet 4.2.0（ONNX Runtime 1.29 + CUDA EP，`--engine rapid`） | 本机 3080 Ti 上 Vulkan 快 **13–18×**；对齐检测输入并给到 16 workers 后仍快 **9–15×** | [rapidocr-cuda.md](rapidocr-cuda.md) |

## 数据来源与复现

| 版本 | Actions | 提交 | 说明 |
| ---- | ------- | ---- | ---- |
| **1.4.2** | [35513018083](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35513018083) | `68a009a` | 基线（与 `v1.4.2` 标签同源码）。准度：sharp **767/1032、CER 2.36%**；c **764/1032、3.03%** |
| **2.0** | [36727088539](https://github.com/sdcb/SimdPaddleOCR/actions/runs/36727088539) | `f8b1197` | 当前 CI 口径；准度与 1.4.2 逐行相同 |
| 本机 5800X / 3080 Ti | — | `a75a0e9` | 1.4.2（NuGet）→ 当前 → Vulkan 同轮交替 3 轮 4w，tiny/small/medium；`bench-out/v20-3080ti/` |
| 5950X / B580 | — | `f8b1197` | 同款三列对比，见 [vulkan-b580.md](vulkan-b580.md) |
| 880M Vulkan | — | `2520c77` | 锐龙 AI 9 H365 + Radeon 880M 同轮 sharp/vulkan 4w，见 [vulkan-880m.md](vulkan-880m.md) |
| 8 Gen 3 Vulkan | — | `vulkan-android-8gen3` 分支 | 真我 GT5 Pro，`test/Sdcb.SimdPaddleOCR.AndroidBench`，本机 CPU / Vulkan 交替 3 轮 4w，见 [vulkan-8gen3.md](vulkan-8gen3.md) |

推送或手动触发 [`.github/workflows/test.yml`](../.github/workflows/test.yml)，下载 `perf-report` artifact。本地同一套数据：

```text
dotnet run --project test/Sdcb.SimdPaddleOCR.TestData -c Release -- --out dataset
dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -o artifacts/net10
artifacts/net10/Sdcb.SimdPaddleOCR.Tests --benchmark --engine sharp --workers 4 --model tiny --input dataset --out bench-out/tiny-4w.json
```

C 另加 `--engine c --c-assets bench-out/c-runtime`。Vulkan 另加 `--engine vulkan`。RapidOcrNet 需要另编一份（`dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -p:EnableRapidOcr=true`）再用 `--engine rapid|rapid-cpu`，运行库与 A/B 脚本见 [rapidocr-cuda.md](rapidocr-cuda.md)。`--summarize` 可并排多份 JSON。本表的 1.4.2 A/B 用的是 harness 临时开关 `UseNuget142=true`（`dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -p:UseNuget142=true`，产出在 `bin/Release/nuget142/`，引用 NuGet `Sdcb.SimdPaddleOCR` 1.4.2 + 模型包 1.0.0），复测后已从树里还原。
