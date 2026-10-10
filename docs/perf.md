# Sdcb.SimdPaddleOCR 性能基准

## 怎么读

- **墙钟**：去掉首张 warmup 后的 **median ms/图**（`--warmup 1`）。本机 5800X 表报 **6 轮中位（范围）**，并附 mean / p95 / img/s。
- **准确率、Δ WS**：初始化后满勤（100 张 / smoke 20 张），**不再跳过首张**。CI tiny bench 是 **767/1032、CER 2.36%**（cls 1020/1020）；本机 5800X 是 **742/1036、2.37%**。两边都是 `windows` 生成机、字体齐，但是各自出的图，行数就是 **1032** 和 **1036**，不要对绝对 exact。2.0 与 main 的行精确 / CER 逐行相同，质量差异只出现在 GPU fp16 边界（见本机表）。
- **倍数口径**：统一按「加速比 = 慢 ÷ 快」，**>1 表示更快**（与 [vulkan-b580.md](vulkan-b580.md) 一致）。同 replica 比是单 VM 内对基线 case 的倍数（>1 表示比基线慢）。
- **replica**：每台 GitHub-hosted VM 一份。先在单 replica 内算比值，再只汇总 **同一 CPU**。
- GitHub `windows-2025` 会随机分到 EPYC 7763 / 9V74 / 9V45 / Xeon。**7763 没有 AVX-512**；9V74 / 9V45 / Xeon 有时走 AVX-512。这几类绝对时间不可比。`ubuntu-24.04-arm` 全部是 Neoverse N2（`CPU part 0xd49`），最稳。
- 4 worker 下算子会并行重叠，**之和可以大于墙钟**，只适合看结构。
- **20 张 smoke 不能和 100 张 bench 比快慢**；osx-arm64 / osx-x64 绝对毫秒不能做门禁。
- 本机复测期间桌面有突发负载（浏览器 / VS Code / 其他项目测试），因此从 3 轮加到 **6 轮**用中位抗噪；mean / p95 含干扰长尾，**版本倍数一律以中位为准**。
- 1.4.2 基线（含 1.4.2 → 2.0 的对比）已删；相对数字见 git 历史里的当时文档。

## main 相对 2.0 改了什么

2.0 发布后合入的改动（不含纯文档）：

| 改动 | PR / 提交 | 可见效果 |
| ---- | ---- | ---- |
| Vulkan 映射窗口 OOM 时重试兼容 coherent 内存（不再整会话退 CPU） | #24 `0051503` | 只在 `vkAllocateMemory` 失败的故障场景生效；正常路径零变化 |
| GPU 会话形状 / CTC 投影元数据缓存（128 项 FIFO） | #25 `9a1ff04` | 主机端重复图解析消失；重复宽度 / 大模型受益更明显 |
| AVX2 融合 MatMul+ArgMax 暂存区复用（栈 + 池） | #26 `3474475` | 减少每行批量的短命分配与 GC 压力；输出逐位不变 |
| GPU 会话 CPU 线程预算（`GpuCtcIntraOpThreads` / `PreprocessWorkerCount`） | #27 `e4e2aab` | 纯可调性 API，默认 `0` 保持原自动预算 |
| shape plan 空可修复 | `2b90dbc` | 正确性修复 |

结论（细节在后面几节）：

- **正确率**：与 2.0 逐行相同——CI sharp **767/1032、CER 2.36%**（cls 1020/1020）、c **764/1032、3.03%**（cls 998/1019）；本机 742 / 950 / 1004 行、CER 2.37% / 0.41% / 0.14%（Vulkan 741 / 950 / 1006），全部六个维度（两引擎 × 三模型）逐轮一致。
- **墙钟**：默认路径**持平**——本机 3080 Ti 三档 CPU **1.01–1.02×**、Vulkan **1.02–1.03×**（small 一档有离群干扰，见表注），CI 各套件也全部在 replica 噪声内。与代码等价性分析一致：#25 / #27 默认路径逐输入等价，#26 输出 hash 逐位一致，#24 不触发时零变化。
- **稳健性 / 可调性**（墙钟看不出的部分）：#24 让映射窗口不足的机器继续走 GPU（该故障场景实测可差 2.6×）；#26 降低长时间服务的 GC 压力；#27 让调用方在进程 CPU 占用 / 吞吐 / 延迟间自行取舍（README 有取舍说明）。
- **内存**：持平（loaded / peak 差 <2%；medium GPU peak 984 → 996 MB，差在 #25 的计划缓存元数据）。
- Vulkan 相对 CPU 的加速不变：本机 3080 Ti tiny / small / medium **2.6× / 6.5× / 15.0×**。

## CI：2.0 → main

两次口径：**2.0** = [36727088539](https://github.com/sdcb/SimdPaddleOCR/actions/runs/36727088539)（`f8b1197`）；**main** = [38039751248](https://github.com/sdcb/SimdPaddleOCR/actions/runs/38039751248)（`b7c7afe`，含上表全部改动）。回归只看这两条：**win-x64 EPYC 7763** 的 SIMD / 引擎套件，以及 **linux-arm64 Neoverse N2**（`CPU part 0xd49`，6 replica 几乎一条直线）。osx-arm64 是 3 核 / 7 GB 虚拟 M1，只看同 replica 比值。CI VM 只有 **4 vCPU**，两次口径的 CPU 墙钟没有预期差异。

| 判定 | 尺子 |
| ---- | ---- |
| x64 | 引擎套件（warm 例）`tiny-4w` 中位大约 **136–152 ms**（单 replica 离群到 181 都见过）；`tiny-4w-ns2` 同 replica 比值大约 **1.5–1.7**。不要用一份 replica 喊回归 |
| ARM64 | `tiny-4w` **156–171 ms**；1w 比 **1.39–1.42**、ns2 比 **1.78–1.81**、scalar 比 **5.13–5.30**。比 Windows 更适合做自动阈值 |
| 引擎 | 只用 win-x64 引擎套件、同一 replica。c/sharp 4w 大约 **1.76–1.84**（c 钉 `20d0de6` DLL，两版对照） |
| 不要 | 把 9V74 / 9V45 / Xeon 的 AVX-512 和 7763 的 AVX2 写成「优化了 xx%」；用 osx 绝对时间做门禁；拿 20 张 smoke 和 100 张 bench 比 |

win-x64 SIMD 先丢一次 25 张 tiny-4w 烤 VM（不上传），再跑默认 / noavx512 / ns2。noavx2 / noavx / scalar 只在 `smoke-win-x64-isa` 跑 20 张。**tiny-4w 是套件第一例，绝对值含冷启开销**（2.0 的 plan 缓存让首例开销变小，跨版本比较首例要小心）。

数据集、模型、预解码 BGR、ISA 开关、runner：`.github/workflows/test.yml`，`dataset/` 固定种子合成 100 张 JPG。库 TFM 默认 `net10.0`；`tiny-4w-ns2` 把库编成 `netstandard2.0`，仍跑在 .NET 10 上。

正确率（100 张满勤，两版逐行相同）：sharp **767/1032、CER 2.36%**（cls 1020/1020；含 scalar、ns2、全部 ISA）；c **764/1032、3.03%**（cls 998/1019）。smoke 20 张满勤：tiny **160/218**（CER 1.90%），small **198/218**（0.40%），medium **210/218**（0.13%）。

### linux-arm64 N2（最稳，6 replica）

同一 `ubuntu-24.04-arm`、同一 `0xd49`，两版各 6 份 replica；中位是各 replica median 的中位数。

| 用例             | 2.0 中位 | main 中位 |    加速 | 2.0 同 replica 比 | main 同 replica 比 |
| ---------------- | -------: | -------: | ------: | ----------------: | -----------------: |
| `tiny-4w`        |      163 |      166 |   1.02× |              1.00 |               1.00 |
| `tiny-1w`        |      231 |      229 |   1.01× |              1.42 |               1.39 |
| `tiny-4w-ns2`    |      293 |      293 |   1.00× |              1.81 |               1.78 |
| `tiny-4w-scalar` |      856 |      854 |   1.00× |              5.30 |               5.13 |

- 四档全部持平（1.00–1.02×），两版范围互相覆盖（4w 160–167 → 156–170）。main 相对 2.0 没有内核级改动，N2 上也不该有收益预期。
- 同 replica 比两版一致：1w **1.42 → 1.39**、ns2 **1.81 → 1.78**、scalar **5.30 → 5.13**。
- 内存两版逐 MB 持平（4w peak 约 574 MB、Δ WS 约 164 MB）。准度两版逐行相同：**767/1032、CER 2.36%**。

### win-x64 SIMD（只报 EPYC 7763）

两版的 7763 份数不一（2.0 2 份、main 4 份，其余 replica 落到 9V74 / 9V45 / Xeon）。中位是各 replica median 的中位数。**tiny-4w 是套件第一例，绝对值含冷启；noavx512 / ns2 是其后的 warm 例。**

| 用例               | 有效 ISA      | 2.0 中位（范围） | main 中位（范围） |   加速 | 2.0 同 replica 比 | main 同 replica 比 |
| ------------------ | ------------- | ---------------: | ----------------: | -----: | ----------------: | -----------------: |
| `tiny-4w`          | AVX2          |  152（148–155）  |   149（143–150）  |  1.02× |              1.00 |               1.00 |
| `tiny-4w-noavx512` | AVX2          |  140（139–141）  |   140（139–143）  |  1.00× |              0.92 |               0.95 |
| `tiny-4w-ns2`      | Vector / NHWC |  232（227–236）  |   240（231–284）  |  1.00× |              1.53 |               1.65 |

- 三档全部持平；ns2 的 main 侧有一份 284 ms 离群 replica，中位未被带偏。同 replica 比两版一致。
- `noavx512` 在 7763 上本来就没有 AVX-512，和 `tiny-4w` 同 ISA，两者差值是跑序 / 冷启。`noavx2` / `noavx` / `scalar` 只在 20 张 `smoke-win-x64-isa` 里跑（见平台 smoke）。
- 准度两版逐行相同：**767/1032、CER 2.36%**。

### win-x64 引擎套件（只报 EPYC 7763）

和 SIMD job 不是同一台 VM，绝对毫秒不要和上一张表硬接。c 钉 [`lw_ppocr_c.20260914.20d0de6.dll`](https://cv-public.sdcb.ai/2026/lw_ppocr_c.20260914.20d0de6.dll)，两版都没换，是对照组；两版各 4 份 7763。

| 用例            | 2.0 中位 | main 中位 |   加速 | 2.0 vs sharp 4w | main vs sharp 4w |
| --------------- | -------: | -------: | -----: | --------------: | ---------------: |
| sharp `tiny-4w` |      139 |      139 |  1.00× |            1.00 |             1.00 |
| sharp `tiny-1w` |      189 |      209 |  1.00× |            1.36 |             1.41 |
| c `tiny-4w`     |      254 |      245 |  1.04× |            1.84 |             1.76 |
| c `tiny-1w`     |      356 |      357 |  1.00× |            2.54 |             2.55 |

- sharp 4w 中位几乎重合（139.4 vs 139.3）。sharp 1w 的 main 侧有一份 234 ms 离群（范围 189–234），中位被抬高，不能当回归。c 对照组是同一份 DLL，差异纯属跑序（2.0 的 c-4w 也有一份 327 ms 离群）。
- c/sharp 4w 两版 **1.76–1.84×** 同级。c 略省内存（peak ~536 MB vs sharp ~511 MB），两版一致。
- 准度两版逐行相同：sharp **767/1032**、c **764/1032**。

### osx-arm64（高噪声，只看比值）

虚拟 M1、3 逻辑核、7 GB。绝对毫秒 replica 之间可以差一倍（本口径两版的 `tiny-4w` 中位 140 vs 205，纯 VM 抖动，不做门禁）。

| 用例             | 2.0 同 replica 比 | main 同 replica 比 |
| ---------------- | ----------------: | -----------------: |
| `tiny-4w`        |              1.00 |               1.00 |
| `tiny-1w`        |  1.59（1.12–1.74） |   1.64（1.52–1.72） |
| `tiny-4w-ns2`    |  1.61（1.43–1.94） |   1.59（1.45–1.72） |

比值两版一致（1w 1.59 → 1.64、ns2 1.61 → 1.59）。**Metal** smoke（同一虚拟 M1）：2.0 `metal` 中位 **100 ms** vs main **118 ms**，同 RID `sharp` CPU 280 / 560 ms——绝对值受 VM 抖动影响，只看量级（metal ≈ sharp 的 2.6–2.8×）。行精确 161/218（CPU 160/218），两版一致。本套只用来确认 AdvSIMD / Metal 路径能跑。

### 平台 smoke（20 张，只看覆盖）

行精确按 **20 张满勤**，两版逐行相同：tiny **160/218**（CER 1.90%），small **198/218**（0.40%），medium **210/218**（0.13%）。中位 ms / CPU 各取本版 run，只看覆盖——**同一 RID 两次分到的 CPU 经常不同，标注 ✗ 的行不要跨版本比快慢**。

| 用例              | 2.0 CPU / 中位 ms | main CPU / 中位 ms | 可比                            |
| ----------------- | ----------------: | -----------------: | ------------------------------- |
| linux-arm64       |        N2 / 208   |        N2 / 206    | ✓ 同 CPU，1.01×                |
| linux-x64 tiny    |     9V45 / 132    |     9V45 / 104     | 同 CPU，20 张噪声              |
| linux-x64 small   |     9V45 / 270    |     9V45 / 259     | 同 CPU，20 张噪声              |
| linux-x64 medium  |     9V45 / 1054   |     9V45 / 1078    | 同 CPU，20 张噪声              |
| win-x64           |    7763 / 215     | Xeon 8573C / 216   | ✗ 不同 CPU                     |
| win-x86           |    9V74 / 198     |    7763 / 365      | ✗ 不同 CPU                     |
| win-arm64         | Cobalt 100 / 199  | Cobalt 100 / 208   | ✓ 同 CPU，1.05×（20 张噪声）   |
| osx-arm64         | M1 Virtual / 280  | M1 Virtual / 560   | 高噪声                          |
| osx-arm64 metal   | M1 Virtual / 100  | M1 Virtual / 118   | 高噪声                          |
| osx-x64           |  i7-8700B / 465   |  i7-8700B / 390    | 高噪声                          |

ISA smoke 两版分到的 CPU 不同（2.0 7763 vs main 8573C，✗ 不可比）：noavx（Vector）402 → 339、noavx2（AVX）456 → 366、scalar 1272 → 1101。small 大约是 tiny 的 2–3 倍墙钟，medium 大约是 tiny 的 8–10 倍（同 CPU 内部比值才可参考）。

### CI 工作集（tiny-4w）

| 场景                 | 2.0 peak | main peak | 2.0 Δ WS | main Δ WS |
| -------------------- | -------: | --------: | -------: | --------: |
| win-x64 7763 sharp   |  511 MB  |  511 MB   |  103 MB  |  105 MB   |
| linux-arm64 N2 sharp |  574 MB  |  574 MB   |  164 MB  |  164 MB   |
| osx-arm64 sharp      |  705 MB  |  703 MB   |  290 MB  |  287 MB   |
| win-x64 lw.PPOCR.C   |  536 MB  |  536 MB   |  104 MB  |  105 MB   |

2.0 → main 的 CI 工作集逐 MB 持平。c 仍然略省，但更慢。

## 本机 5800X / RTX 3080 Ti（main 复测）

实测机：HOME-MAIN，Windows 10.0.26200，电源方案“高性能”，Ryzen 7 5800X（8C/16T，AVX2，无 AVX-512），64 GB，RTX 3080 Ti（驱动 581.80）。运行时 .NET 10.0.11。`test/Sdcb.SimdPaddleOCR.Tests`，`--workers 4 --benchmark-kind simd --warmup 1`，n=99，同一 `dataset/` 100 张变尺寸图（对 GPU 最不利的逐图新 shape）。先用当前树 tiny 25 张烤机，再按 tiny → small → medium、每档 **2.0 CPU → main CPU → 2.0 Vulkan → main Vulkan** 交替跑 **6 轮**。本表 `f81b3ae`。

两版用同一套选项（`LineWorkerCount=4`、`AdaptiveWidth`、`TargetWidth=320`、session cache 32、REC 批大小保持默认 1）。**2.0** 是 NuGet `Sdcb.SimdPaddleOCR` **2.0.0** 加模型包 **1.0.0**（harness 临时改成 `UseNuget20=true` 另编一份）；**main** 是同一 harness 的源码引用。`--engine sharp` 钉死 CPU，`--engine vulkan` 走 GPU；GPU 在批大小未显式设置时用默认 16。仓库里的 onnx 两版相同，任何差值都是代码，不是权重。复测期间桌面有突发负载（浏览器 / VS Code / 其他项目测试进程），因此加到 6 轮取中位，离群值列在范围里。不要用这里的绝对毫秒卡 GitHub runner（CI VM 只有 4 vCPU）。

**结论：默认路径持平。六轮中位上 main 相对 2.0：三档 CPU 1.01× / 1.02× / 1.01×、三档 Vulkan 1.02× / 1.14× / 1.03×（small 的 1.14× 由 2.0 侧一次离群抬高中位所致，范围重叠）——全部在噪声内，与 #25 / #27 逐输入等价、#26 输出 hash 逐位一致相符。行精确和 CER 两版逐行相同。Vulkan 相对 CPU 的加速不变：tiny 2.6× / small 6.5× / medium 15.0×。**

### 端到端（4 workers，median ms/图，越低越好）

括号里是 6 轮范围。加速是较慢一列的中位毫秒除以较快一列（大于 1 表示更快）。

| 模型 | 2.0 CPU | main CPU | 2.0 Vulkan | main Vulkan | main/2.0 CPU | main/2.0 Vulkan | Vulkan / CPU（main） |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 53.3（44.5–75.5） | 53.8（44.3–60.7） | 20.3（17.0–23.8） | 20.7（17.9–26.1） | 1.01× | 1.02× | **2.6×** |
| small | 186.2（174.0–197.5） | 190.2（184.2–214.3） | 33.3（28.6–72.6） | 29.2（28.1–48.4） | 1.02× | 1.14×* | **6.5×** |
| medium | 665.4（649.4–705.7） | 673.4（619.4–865.9） | 43.7（41.9–45.6） | 44.8（41.3–135.6） | 1.01× | 1.03× | **15.0×** |

\* small Vulkan 的 1.14× 由 2.0 侧一次 72.6 ms 离群抬高中位所致，两版范围重叠，不能当收益；medium 的 main 侧 865.9 / 135.6 两个离群同理来自桌面突发。

六轮 mean 的均值，以及由此得到的 img/s（1000 / mean）。p95 是六轮 p95 的中位数。工作集峰值是六轮里的最大值。

| 模型 | 列 | mean | p95 | img/s | WS peak |
|---|---|---:|---:|---:|---:|
| tiny | 2.0 CPU | 68.6 | 99.5 | 14.6 | 517 MB |
| tiny | main CPU | 95.5 | 178.2 | 10.5 | 515 MB |
| tiny | 2.0 Vulkan | 27.2 | 61.4 | **36.7** | 649 MB |
| tiny | main Vulkan | 30.7 | 69.1 | **32.6** | 643 MB |
| small | 2.0 CPU | 227.8 | 398.9 | 4.4 | 650 MB |
| small | main CPU | 264.0 | 666.1 | 3.8 | 650 MB |
| small | 2.0 Vulkan | 46.5 | 83.5 | **21.5** | 709 MB |
| small | main Vulkan | 43.4 | 64.0 | **23.1** | 716 MB |
| medium | 2.0 CPU | 788.6 | 1576.9 | 1.3 | 1171 MB |
| medium | main CPU | 794.9 | 1668.9 | 1.3 | 1170 MB |
| medium | 2.0 Vulkan | 48.9 | 80.0 | **20.5** | 985 MB |
| medium | main Vulkan | 66.4 | 117.9 | 15.1 | 1003 MB |

mean / p95 含桌面突发干扰的长尾（个别单张被拖到秒级），**版本倍数一律以中位表为准**。按中位换算的 img/s（1000 / 中位）：tiny 18.8 / 18.6 / **49.3** / **48.3**、small 5.4 / 5.3 / **30.0** / **34.2**、medium 1.5 / 1.5 / **22.9** / **22.3**（列序同上表）。

### 分阶段耗时（六轮 stage mean 再平均，ms/图）

4 worker 下各阶段重叠，加总可以大于墙钟。倍数是 main Vulkan 相对 main CPU。**阶段数也受干扰（CPU 侧个别档位有 30–50% 抖动），只看结构与倍数。**

| 模型 | 阶段 | 2.0 CPU | main CPU | 2.0 Vulkan | main Vulkan | 相对 main CPU |
|---|---|---:|---:|---:|---:|---:|
| tiny | det_preprocess | 2.8 | 3.1 | 3.2 | 3.7 | 0.84× |
| tiny | det_graph | 29.0 | 50.1 | 6.7 | 7.3 | 6.9× |
| tiny | det_postprocess | 2.2 | 1.8 | 2.7 | 2.9 | 0.62× |
| tiny | crop | 2.8 | 2.8 | 2.9 | 3.2 | 0.89× |
| tiny | cls_preprocess | 3.1 | 2.7 | 0.8 | 1.0 | 2.8× |
| tiny | cls_graph | 14.6 | 14.2 | 1.5 | 1.9 | 7.6× |
| tiny | rec_preprocess | 6.6 | 5.8 | 2.0 | 2.2 | 2.7× |
| tiny | rec_graph | 87.9 | 99.9 | 7.1 | 8.1 | 12.4× |
| tiny | lines_wall | 31.0 | 37.1 | 11.6 | 13.6 | 2.7× |
| small | det_preprocess | 2.5 | 2.8 | 3.8 | 3.6 | 0.77× |
| small | det_graph | 99.6 | 117.5 | 7.9 | 7.9 | 14.9× |
| small | det_postprocess | 2.6 | 2.5 | 2.7 | 2.8 | 0.92× |
| small | crop | 3.3 | 3.6 | 3.7 | 3.4 | 1.05× |
| small | cls_preprocess | 2.9 | 3.0 | 1.1 | 1.1 | 2.9× |
| small | cls_graph | 16.4 | 17.3 | 2.3 | 1.8 | 9.7× |
| small | rec_preprocess | 6.9 | 7.3 | 2.4 | 2.3 | 3.1× |
| small | rec_graph | 413.6 | 468.0 | 22.0 | 19.8 | 23.6× |
| small | lines_wall | 119.4 | 137.0 | 28.4 | 25.6 | 5.4× |
| medium | det_preprocess | 2.4 | 2.5 | 2.5 | 4.2 | 0.58× |
| medium | det_graph | 313.2 | 328.1 | 9.7 | 12.0 | 27.3× |
| medium | det_postprocess | 1.3 | 1.3 | 2.4 | 2.3 | 0.57× |
| medium | crop | 2.5 | 2.8 | 2.6 | 4.3 | 0.65× |
| medium | cls_preprocess | 2.6 | 2.6 | 0.6 | 1.8 | 1.45× |
| medium | cls_graph | 16.1 | 16.1 | 1.0 | 1.9 | 8.4× |
| medium | rec_preprocess | 6.5 | 6.5 | 1.8 | 2.9 | 2.2× |
| medium | rec_graph | 1708.1 | 1671.3 | 27.9 | 35.6 | 47× |
| medium | lines_wall | 468.8 | 459.7 | 31.5 | 43.5 | 10.6× |

两版阶段结构一致，`lines_wall`（2.0 → main → main Vulkan）：tiny 31.0 → 37.1 → **13.6**，small 119 → 137 → **25.6**，medium 469 → 460 → **43.5**。medium `rec_graph` 35.6 ms 里约一半是 CPU 上的 CTC 投影 + ArgMax（词表 18710 列，契约要求留在 CPU；#27 的 `GpuCtcIntraOpThreads` 可调这段的线程预算）。

### 正确率（100 张满勤，1036 行；六轮逐轮相同）

| 模型 | 2.0 CPU | main CPU | 2.0 Vulkan | main Vulkan |
|---|---:|---:|---:|---:|
| tiny | 742 / 2.37% / 5 | 742 / 2.37% / 5 | 741 / 2.38% / 5 | 741 / 2.38% / 5 |
| small | 950 / 0.41% / 44 | 950 / 0.41% / 44 | 950 / 0.40% / 44 | 950 / 0.40% / 44 |
| medium | 1004 / 0.14% / 71 | 1004 / 0.14% / 71 | 1006 / 0.14% / 73 | 1006 / 0.14% / 73 |

四列两两逐行相同（列内六轮也逐轮相同）。cls 在各自检出的行上全对：四列都是 1022/1022、1034/1034、1035/1035。medium 多对的 2 行、tiny 少的 1 行是 GPU fp16 边界（与 2.0 完全一致，定位见 [vulkan-b580.md](vulkan-b580.md) 已知缺口）。

### 内存（MB）

loaded / peak 是六轮范围，Δ WS 是六轮（last − loaded）的均值。

| 模型 | 列 | loaded | peak | Δ WS |
|---|---|---:|---:|---:|
| tiny | 2.0 CPU | 401 | 512–517 | 113 |
| tiny | main CPU | 401 | 511–515 | 110 |
| tiny | 2.0 Vulkan | 497–498 | 639–649 | 142 |
| tiny | main Vulkan | 497–498 | 634–643 | 138 |
| small | 2.0 CPU | 452–453 | 646–650 | 190 |
| small | main CPU | 452–453 | 646–650 | 190 |
| small | 2.0 Vulkan | 548–549 | 700–709 | 151 |
| small | main Vulkan | 548 | 715–716 | 167 |
| medium | 2.0 CPU | 699 | 1165–1171 | 467 |
| medium | main CPU | 699 | 1167–1170 | 468 |
| medium | 2.0 Vulkan | 795 | 982–985 | 170 |
| medium | main Vulkan | 795 | 996–1003 | 193 |

两版逐行接近：CPU peak 差 <1%，Vulkan peak 差 <2%（medium 984 → 1003，差在 #25 计划缓存与 #26 池驻留）。Vulkan 的 loaded 依旧更高（图和 arena 在加载时就占上），medium GPU peak 仍然最低。六轮之后不再爬。

### 本机备注

- **默认路径没有墙钟变化是预期结果**：#25 / #27 默认路径逐输入等价、#26 输出 hash 逐位一致、#24 不触发时零改动——本表把这四条合起来又验证了一遍（1.01–1.03× 全在噪声内）。
- **收益在墙钟之外**：#26 少了每行批量的短命数组（长时间服务 GC 压力更小）；#24 在映射窗口不足的机器上避免整会话退 CPU（故障场景 5.1 → 13.3 张/秒见 #24 描述）；#27 提供线程预算调节（`GpuCtcIntraOpThreads` / `PreprocessWorkerCount`，默认不变）。
- 3080 Ti 上 Vulkan 走 sg32 coopmat 档（大 GEMM 25–28 TFLOPS）；GPU 内部数字（纯 GPU 分段、显存 arena、~32-shape 崩坏验证）见 [vulkan-b580.md](vulkan-b580.md)。
- mean 明显高于 median 有两个来源：前 ~40 张的 .NET 分层 JIT 预热（`DOTNET_TieredCompilation=0` 下消失），以及复测期间的桌面突发负载（后者把 p95 也拖长了）。
- 显存：sg32 上 arena 按生命周期复用，进程工作集有界，peak 后不再爬升。
- JSON：`bench-out/main-3080ti/{v20,main}-{sharp,vulkan}-{tiny,small,medium}-r{1..6}.json`。复现：`--engine sharp|vulkan --workers 4 --model {tiny,small,medium} --input dataset --warmup 1`。

## 其他 GPU 设备

5800X / 3080 Ti 之外的 Vulkan / Metal 实测（各设备自己的尺子，绝对毫秒不可跨机比）：

| 设备 | 文档 |
| ---- | ---- |
| Ryzen 9 5950X / Intel Arc B580（2.0 发布复测，同款 1.4.2 → 当前 → Vulkan 三列对比） | [vulkan-b580.md](vulkan-b580.md) |
| 锐龙 AI 9 H365 / Radeon 880M 核显 | [vulkan-880m.md](vulkan-880m.md) |
| 骁龙 8 Gen 3 / Adreno 750（安卓，无协作矩阵档） | [vulkan-8gen3.md](vulkan-8gen3.md) |
| Intel UHD 770 核显（无协作矩阵，`Auto` 走 CPU） | [vulkan-uhd770.md](vulkan-uhd770.md) |
| Apple M1 / M4（Metal） | [metal-m4.md](metal-m4.md)、[perf-devin-m4.md](perf-devin-m4.md) |

## 数据来源与复现

| 版本 | Actions | 提交 | 说明 |
| ---- | ------- | ---- | ---- |
| **2.0** | [36727088539](https://github.com/sdcb/SimdPaddleOCR/actions/runs/36727088539) | `f8b1197` | 基线（已发布 NuGet 2.0.0；CI 口径 `f8b1197`）。准度：sharp **767/1032、CER 2.36%**；c **764/1032、3.03%** |
| **main** | [38039751248](https://github.com/sdcb/SimdPaddleOCR/actions/runs/38039751248) | `b7c7afe` | 本表全部改动合入后；准度与 2.0 逐行相同 |
| 本机 5800X / 3080 Ti | — | `f81b3ae` | 2.0（NuGet）↔ main 同轮交替 6 轮 4w，tiny/small/medium × sharp/vulkan；`bench-out/main-3080ti/` |
| 5950X / B580 | — | `f8b1197` | 2.0 口径的 1.4.2 / 当前 / Vulkan 三列对比（历史口径），见 [vulkan-b580.md](vulkan-b580.md) |
| 880M Vulkan | — | `2520c77` | 锐龙 AI 9 H365 + Radeon 880M 同轮 sharp/vulkan 4w，见 [vulkan-880m.md](vulkan-880m.md) |
| 8 Gen 3 Vulkan | — | `vulkan-android-8gen3` 分支 | 真我 GT5 Pro，`test/Sdcb.SimdPaddleOCR.AndroidBench`，本机 CPU / Vulkan 交替 3 轮 4w，见 [vulkan-8gen3.md](vulkan-8gen3.md) |

推送或手动触发 [`.github/workflows/test.yml`](../.github/workflows/test.yml)，下载 `perf-report` artifact。本地同一套数据：

```text
dotnet run --project test/Sdcb.SimdPaddleOCR.TestData -c Release -- --out dataset
dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -o artifacts/net10
artifacts/net10/Sdcb.SimdPaddleOCR.Tests --benchmark --engine sharp --workers 4 --model tiny --input dataset --out bench-out/tiny-4w.json
```

C 另加 `--engine c --c-assets bench-out/c-runtime`。Vulkan 另加 `--engine vulkan`。`--summarize` 可并排多份 JSON。本表的 2.0 A/B 用的是 harness 临时开关 `UseNuget20=true`（`dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -p:UseNuget20=true`，产出在 `bin/Release/nuget20/`，引用 NuGet `Sdcb.SimdPaddleOCR` 2.0.0 + 模型包 1.0.0），复测后已从树里还原。
