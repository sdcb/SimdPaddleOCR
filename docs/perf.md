# Sdcb.SimdPaddleOCR 性能基准（2026-09）

本文是当前仓库的 **CI 综合基线**，汇总 4 次 GitHub Actions `test` 工作流、共 **484** 份 JSON 报告。
之后对比性能时，请按「同一 CPU 型号 + 同一有效 ISA + 同一套件」对照，不要把不同 runner 的绝对毫秒数混在一起。

## 怎么读

- **墙钟**是去掉首张 warmup 后、99 张图（bench）或 19 张图（smoke）的 **median ms/图**。
- 每个 replica 跑在 **一台独立的 GitHub-hosted VM** 上。先在单 replica 内算比值，再对比值或同 CPU 的 median 做汇总。
- 表中「中位」是各 replica median 的中位数；「IQR」是这些 replica median 的 25%–75%；「范围」是最小–最大；`n` 是 replica 数。
- GitHub `windows-2025` 会随机分到 AMD EPYC 7763 / 9V74 或 Intel Xeon。**7763 没有 AVX-512**，9V74 / Xeon 有时暴露 AVX-512、有时只走到 AVX2。这两类绝对时间不可比。
- 算子耗时（Conv / Conv1x1 / MatMul 等）是每张图该算子的均值。4 worker 下算子会并行重叠，**之和可以大于墙钟**，只适合看结构，不适合当吞吐。

## 数据来源

| Actions run | 提交 | 时间（UTC） | 说明 |
| --- | --- | --- | --- |
| [34239354155](https://github.com/sdcb/SimdPaddleOCR/actions/runs/34239354155) | `6812806` | 2026-09-08 14:36 | 内核与注释已落地；osx-arm64 当时 6 replica |
| [34239766768](https://github.com/sdcb/SimdPaddleOCR/actions/runs/34239766768) | `32cb921` | 2026-09-08 14:40 | osx-arm64 改为 5 replica（GitHub Pro macOS 并发上限） |
| [34291744223](https://github.com/sdcb/SimdPaddleOCR/actions/runs/34291744223) | `32cb921` | 2026-09-08 23:41 | 同提交复跑 |
| [34293010903](https://github.com/sdcb/SimdPaddleOCR/actions/runs/34293010903) | `32cb921` | 2026-09-08 23:58 | 同提交复跑 |

`6812806` 与 `32cb921` 的 SIMD 内核相同，只差 macOS replica 数量，因此一并纳入。
两次复跑之间 runner 上的 .NET 补丁从 **10.0.11** 升到 **10.0.12**，同 CPU 墙钟没有可分辨的台阶。

Replica 配置（`32cb921`）：

| Job | Runner | Replica |
| --- | --- | --- |
| `bench-win-x64-simd` | `windows-2025` | 6 |
| `bench-win-x64-engines` | `windows-2025` | 6 |
| `bench-linux-arm64` | `ubuntu-24.04-arm` | 6 |
| `bench-osx-arm64` | `macos-26` | 5 |
| `smoke-platform` / `smoke-models` | 见下表 | 每平台 1 |

4 次合计：win-x64 SIMD 24 replica、win-x64 引擎 24 replica、linux-arm64 24 replica、osx-arm64 21 replica（第一次多 1 个）。

## 测试方法

- **数据集**：`test/Sdcb.SimdPaddleOCR.TestData` 用固定种子 `20260830` 生成的 100 张合成 JPG（中英混合、多字号、旋转与背景变化）。Bench 跑全量，第 1 张 warmup；smoke 只取前 20 张。
- **模型**：默认 PP-OCRv6 **tiny** + TextLineOrientation CLS；`smoke-models` 另跑 small / medium。
- **输入**：8-bit BGR 紧排内存，不计 JPEG 解码。
- **运行时**：.NET 10，库 TFM 默认 `net10.0`；`tiny-4w-ns2` 把库编成 `netstandard2.0`，仍跑在 .NET 10 上（走 `System.Numerics.Vector`）。
- **ISA 开关**：`DOTNET_EnableAVX512` / `AVX2` / `AVX` / `EnableHWIntrinsic`。
- **引擎**（后两者只在 win-x64 引擎套件里比；**没有**和 PaddleOCR 官方实现做对比）：
  - `sharp` = 本库 Sdcb.SimdPaddleOCR
  - `c` = [lw.PPOCR.C](https://github.com/lxw112190/lw.PPOCR.C) 的 **2026 年 8 月底**一份构建（`lw_ppocr_c.dll`），**不代表**该项目最新版本
  - `openvino` = 同作者的 [OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET)（CI 里是 `Sdcb.OpenVINO.PaddleOCR` 0.8.1 + `Sdcb.OpenVINO.runtime.win-x64` 2026.2.0）

## 环境

所有 GitHub-hosted runner 都是 **4 逻辑核**（osx-arm64 为 **3**），内存约 16 GB（osx-arm64 约 7 GB）。这是共享云主机，不是安静实验室。

| RID | Actions runner | 实测 OS | 出现过的 CPU | 有效 ISA | 内存 |
| --- | --- | --- | --- | --- | --- |
| win-x64 | `windows-2025` | Windows 10.0.26100 | **AMD EPYC 7763**（主基线） | AVX2（无 AVX-512） | 16 GB |
| win-x64 | 同上 | 同上 | AMD EPYC 9V74 | AVX2 或 AVX-512 | 16 GB |
| win-x64 | 同上 | 同上 | Xeon Platinum 8370C / 8573C、Xeon 6973P-C | 多为 AVX-512 | 16 GB |
| win-x86 | `windows-2025`（x86 SDK） | 同上 | 7763 或 9V74 | AVX2 / AVX-512 | 16 GB |
| win-arm64 | `windows-11-arm` | Windows 10.0.26200 | Cobalt 100 | AdvSimd | 16 GB |
| linux-x64 | `ubuntu-24.04` | Ubuntu 24.04.4/5 | 7763 或 9V74 | AVX2 / AVX-512 | ~16 GB |
| linux-arm64 | `ubuntu-24.04-arm` | Ubuntu 24.04.4/5 | **Neoverse N2**（`CPU part 0xd49`） | AdvSimd | ~16 GB |
| osx-arm64 | `macos-26` | macOS 26.6.2 | **Apple M1 (Virtual)** | AdvSimd | 7 GB |
| osx-x64 | `macos-26-intel` | macOS 26.6.1 | i7-8700B @ 3.20 GHz | AVX2 | 14 GB |

主基线选 **win-x64 + EPYC 7763 + AVX2**：样本最多（SIMD 17、引擎 13），IQR 窄，且不会和 AVX-512 缠在一起。

## 正确率

同一套合成数据上，本库跨 OS / CPU / ISA（含 scalar、ns2）的识别结果一致。

| 套件 | 引擎 | 模型 | 行精确 | CER | 说明 |
| --- | --- | --- | --- | --- | --- |
| bench（99 张） | sharp（本库） | tiny | **757/1022** | **3.53%** | 全部 4 次、全部平台相同 |
| bench | lw.PPOCR.C | tiny | 759/1022 | 4.18% | 行精确略高，CER 略差 |
| bench | OpenVINO.NET | tiny | 698/1022（常见） | 3.63% | 个别 Xeon 主机落到 677–680/1022 |
| smoke（19 张） | sharp | tiny | 152/208 | 3.67% | 全部平台相同 |
| smoke | sharp | small | 188/208 | 1.57% | linux-x64 |
| smoke | sharp | medium | 200/208 | 0.34% | linux-x64 |

## 主基线：win-x64 AVX2（EPYC 7763）

### SIMD / ISA 套件

同一 replica 内以 `tiny-4w`（net10 + AVX2）为 1.00x。下表绝对时间只含 **7763**。

| 用例 | 有效 ISA | 中位 ms/图 | IQR | 范围 | n | 同 replica 比值 |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny-4w` | AVX2 | **237.6** | 233.9–241.9 | 230.0–285.3 | 17 | 1.00 |
| `tiny-4w-noavx512` | AVX2 | 223.8 | 221.6–232.8 | 216.1–268.8 | 17 | **0.95**（7763 无 AVX-512，与基线同 ISA） |
| `tiny-4w-ns2` | AVX2（Vector） | **373.4** | 366.4–376.0 | 355.7–408.0 | 17 | **1.56** |
| `tiny-4w-noavx2` | AVX | 360.2 | 357.5–369.7 | 351.7–454.8 | 17 | **1.53** |
| `tiny-4w-noavx` | Vector128 | 493.5 | 490.3–509.9 | 481.2–548.3 | 17 | **2.10** |
| `tiny-4w-scalar` | scalar | 1536.5 | 1502–1563 | 1467–1729 | 17 | **6.51** |

读数：

- 默认 AVX2 大约 **4.2 图/秒**（4 worker，含 DET+CLS+REC）。
- `noavx512` 比 `tiny-4w` 快约 5%，是同 ISA 的噪声地板，**不是** AVX-512 收益。
- ns2 比 net10 AVX2 慢 **56%**，和关掉 AVX2、只留 AVX 几乎同级。
- 关掉全部硬件加速大约慢 **6.5 倍**。

EPYC 9V74 上同一套件、有效 ISA 仍为 AVX2 时，`tiny-4w` 中位 **240.9 ms**（n=5），与 7763 相差约 1%，可作交叉验证。
这 4 次 SIMD 套件里，9V74 的默认用例都没有走到 AVX-512。

### 引擎对比套件（必须同机）

引擎 job 与 SIMD job 是不同 VM，绝对毫秒不要和上一张表硬接。下面只报 **7763** 上、以同 replica 的 `tiny-4w` sharp 为 1.00x。

| 用例 | 引擎 | worker | 中位 ms/图 | IQR | n | 相对 sharp 4w |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny-4w` | sharp | 4 | **224.2** | 222.3–229.0 | 13 | 1.00 |
| `tiny-1w` | sharp | 1 | 275.8 | 274.8–280.1 | 13 | 1.23 |
| `tiny-c-4w` | lw.PPOCR.C | 4 | 280.5 | 278.9–284.4 | 13 | **1.26** |
| `tiny-c-1w` | lw.PPOCR.C | 1 | 485.4 | 481.8–495.8 | 13 | 2.18 |
| `tiny-openvino` | OpenVINO.NET | 4 | 213.7 | 209.8–216.8 | 13 | **0.95** |

结论（7763 / AVX2）：

- 本库 4 worker 比 lw.PPOCR.C 4 worker **快约 26%**（对方是 1.26x 墙钟）。
- 本库 1→4 worker 大约再快 23%；lw.PPOCR.C 的 4 worker 收益更大（485→281），但绝对时间仍慢于本库。
- OpenVINO.NET 墙钟略快（约 5%），但工作集约 **2600 MB**，本库约 **790 MB**（约 1/3）。OpenVINO.NET 有一个 279 ms 的离群 replica，P95 可到 752 ms。

AVX-512 主机样本少，只作参考：9V74 AVX-512 上 sharp 4w 中位 160 ms（n=3），OpenVINO.NET 130 ms，lw.PPOCR.C 231 ms；Xeon 6973P-C 上 sharp 151 ms、OpenVINO.NET 129 ms（各 n=2）。数量不够当基线。

### 算子结构（7763 / SIMD / tiny-4w）

| 项 | 中位（ms/图，算子均值） | IQR | n |
| --- | --- | --- | --- |
| Conv（合计） | 358 | 351–365 | 17 |
| Conv1x1 | 226 | 222–231 | 17 |
| Conv3x3Stride2 | 35 | 34–37 | 17 |
| Depthwise3x3 | 26 | 26–27 | 17 |
| MatMul | 30 | 29–31 | 17 |
| Erf | 36 | 35–36 | 17 |
| WS peak | 789 MB | 779–793 | 17 |

Conv1x1 仍是最大单项。ns2 上 Conv1x1 升到约 235 ms，但 **Stride2 从 35 ms 升到 75 ms**、MatMul/Erf 大约翻倍，墙钟差距主要来自这些 net10 专化路径而不是 Conv1x1 单独崩掉。

## linux-arm64 AdvSimd（Neoverse N2）

`ubuntu-24.04-arm` 在这 4 次里 **全部**是 `CPU part 0xd49`，24 replica 的墙钟几乎一条直线，是最稳的平台。

| 用例 | 有效 ISA | 中位 ms/图 | IQR | 范围 | n | 同 replica 比值 |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny-4w` | AdvSimd | **272.8** | 271.9–273.7 | 265.4–281.0 | 24 | 1.00 |
| `tiny-1w` | AdvSimd | 398.5 | 394.4–401.0 | 390.3–407.5 | 24 | **1.46** |
| `tiny-4w-ns2` | AdvSimd | 420.2 | 418.9–425.5 | 414.2–428.4 | 24 | **1.54** |
| `tiny-4w-scalar` | scalar | 1159 | 1158–1163 | 1150–1178 | 24 | **4.25** |

4 worker 相对 1 worker 稳稳快 46%。ns2 慢 54%，和 win-x64 ns2 的 56% 同量级。
scalar 只慢 4.25 倍（win-x64 是 6.5 倍），因为 ARM 上 `Vector<float>` 本身就是 128-bit，和 x64 AVX2→scalar 的落差不同。

算子（`tiny-4w`）：Conv1x1 176 ms，Stride2 19 ms，MatMul 71 ms，Erf 61 ms。相对 x64，MatMul/Erf 更突出。

## osx-arm64 AdvSimd（Apple M1 Virtual）

`macos-26` 是 3 逻辑核、7 GB 的虚拟 M1，**replica 之间可以差一倍**，不能拿单次墙钟下结论。

| 用例 | 中位 ms/图 | IQR | 范围 | n |
| --- | --- | --- | --- | --- |
| `tiny-4w` | 336 | 302–384 | 223–484 | 21 |
| `tiny-1w` | 534 | 499–646 | 359–756 | 21 |
| `tiny-4w-ns2` | 502 | 405–544 | 303–683 | 21 |
| `tiny-4w-scalar` | 1396 | 1230–1621 | 1021–1814 | 21 |

同 replica 比值（相对 `tiny-4w`）比绝对时间稳一些：ns2 大约 1.3–1.5x，scalar 大约 3–5x，但离散仍明显大于 N2 / 7763。
这份基线把 osx-arm64 标成 **高噪声平台**，只用于确认 AdvSimd 路径能跑、正确率一致，不用于回归判定。

## 平台 smoke（20 张，仅供覆盖）

smoke 图集是 bench 的前 20 张，median **不能**和 100 张 bench 直接比。每平台 4 次各 1 条，CPU 还会变。

| RID | CPU（本次出现） | ISA | 中位 ms/图 | 行精确 |
| --- | --- | --- | --- | --- |
| win-x64 | EPYC 7763 | AVX2 | 272（n=3） | 152/208 |
| win-x64 | Xeon Platinum | AVX-512 | 295（n=1） | 152/208 |
| win-x86 | 7763 / 9V74 | AVX2 | 418–443（各 n=1） | 152/208 |
| win-x86 | 9V74 | AVX-512 | 390（n=2） | 152/208 |
| win-arm64 | Cobalt 100 | AdvSimd | 315（n=4） | 152/208 |
| linux-x64 | EPYC 7763 | AVX2 | 277–286 | 152/208 |
| linux-x64 | EPYC 9V74 | AVX-512 | 192（n=2） | 152/208 |
| linux-arm64 | Neoverse N2 | AdvSimd | 323（n=4） | 152/208 |
| osx-arm64 | Apple M1 | AdvSimd | 369（n=4，334–400） | 152/208 |
| osx-x64 | i7-8700B | AVX2 | 883（n=4，252–1081，噪声极大） | 152/208 |

linux-x64 模型档位（9V74 AVX2 与 7763 AVX2 分开；**不要**和 AVX-512 tiny smoke 的 192 ms 比模型差距）：

| 模型 | CPU | ISA | 中位 ms/图 | 行精确 | CER |
| --- | --- | --- | --- | --- | --- |
| tiny | 7763 | AVX2 | 277 | 152/208 | 3.67% |
| small | 7763 | AVX2 | 836 | 188/208 | 1.57% |
| medium | 7763 | AVX2 | 4724 | 200/208 | 0.34% |
| tiny | 9V74 | AVX2 | 281 | 152/208 | 3.67% |
| small | 9V74 | AVX2 | 853 | 188/208 | 1.57% |
| medium | 9V74 | AVX2 | 5162 | 200/208 | 0.34% |

small 大约是 tiny 的 3 倍墙钟，medium 大约是 tiny 的 17 倍；正确率按 CER 从 3.7% → 1.6% → 0.34%。

## 工作集

| 场景 | WS peak 中位 | 备注 |
| --- | --- | --- |
| win-x64 sharp tiny 4w（bench） | ~790 MB | 7763 / 9V74 接近 |
| win-x64 sharp tiny 1w | ~730 MB | |
| win-x64 lw.PPOCR.C tiny 4w | ~586 MB | 更省，但更慢 |
| win-x64 OpenVINO.NET tiny | ~2600 MB | 约 3 倍于本库 |
| linux-arm64 sharp tiny 4w | ~847 MB | |
| osx-arm64 sharp tiny 4w | ~726 MB | 机器内存只有 7 GB |
| 各平台 tiny smoke（20 张） | 330–390 MB | |

## 以后怎么用这份基线

1. **回归判定（x64）**：只看 win-x64 **EPYC 7763** 的 SIMD 套件。`tiny-4w` 中位应落在约 **234–242 ms**（IQR）；`tiny-4w-ns2` 同 replica 比值应在 **1.51–1.61**。单次 230–285 都出现过，不要用单 replica 绝对时间喊回归。
2. **回归判定（ARM64）**：只看 linux-arm64 N2。`tiny-4w` 应在 **272–274 ms**（IQR）；ns2 比值 **1.53–1.56**。这个平台比 Windows 更适合做自动阈值。
3. **引擎对比**：只用 win-x64 引擎套件、同一 replica。sharp 4w vs lw.PPOCR.C 4w 应在 **1.25–1.27**；vs OpenVINO.NET 应在 **0.94–0.98**（忽略个别 OpenVINO.NET 长尾）。
4. **不要**：把 9V74/Xeon 的 AVX-512 160 ms 和 7763 的 238 ms 写成「优化了 30%」；不要用 osx-arm64 / osx-x64 的绝对时间做 CI 门禁；不要拿 20 张 smoke 和 100 张 bench 比快慢。

复现：推送或手动触发 [`.github/workflows/test.yml`](../.github/workflows/test.yml)，下载 `perf-report` artifact。本地同一套数据可用 `test/Sdcb.SimdPaddleOCR.Tests` 的 `--benchmark` / `--summarize`。
