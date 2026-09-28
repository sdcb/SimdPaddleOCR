# Sdcb.SimdPaddleOCR 性能基准

## 怎么读

- **墙钟**：去掉首张 warmup 后的 **median ms/图**（`--warmup 1`）。本机 5800X 表额外报 mean，和 median 几乎重合。
- **准确率、Δ WS**：初始化后满勤（100 张 / smoke 20 张），**不再跳过首张**。CI tiny bench 是 **767/1032、CER 2.36%**（cls 1020/1020）；本机 5800X 是 **742/1036、2.37%**。两边都是 `windows` 生成机、字体齐，但是各自出的图，行数就是 **1032** 和 **1036**，不要对绝对 exact。1.3 CI 仍是跳过首张的 **757/1022、3.53%**；和 767/1032 对不上涨幅（99 张 vs 100 张）。质量变化看本机同尺子：行精确没动，CER 因左 1:4 CLS 下降（tiny 2.78% → **2.37%**，small 0.60% → **0.41%**，medium 0.67% → **0.14%**）。
- **replica**：每台 GitHub-hosted VM 一份。先在单 replica 内算比值，再只汇总 **同一 CPU**。
- GitHub `windows-2025` 会随机分到 EPYC 7763 / 9V74 / Xeon。**7763 没有 AVX-512**；9V74 / Xeon 有时走 AVX-512。这两类绝对时间不可比。
- 4 worker 下算子会并行重叠，**之和可以大于墙钟**，只适合看结构。
- **20 张 smoke 不能和 100 张 bench 比快慢**；osx-arm64 / osx-x64 绝对毫秒不能做门禁。
- 1.2 旧基线已删。相对 1.2 的数字见当时的 1.3 说明：AVX2 tiny 约 0.69×，本机 medium 约 0.43×。

## 1.4.2 改了什么

相对 **1.3.0**（`37fe1fd`），图级 NHWC 从「仅 AVX2+FMA」扩到：

| 路径                                        | 1.3                    | 1.4.2                                                                              |
| ------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| net10 AVX2+FMA                              | 图级 NHWC              | 同左；预处理直接写 NHWC，少一次输入 `LayoutConvert`                                |
| `netstandard2.0`（`tiny-4w-ns2`，`Vector`） | NCHW                   | **NHWC**                                                                           |
| x64 `DOTNET_EnableHWIntrinsic=0`（scalar）  | NCHW + 软件模拟 Vector | **专用 NHWC 标量 tile**（不再转 `*Vec`）                                           |
| net10 AdvSIMD（linux-arm64 / osx-arm64）    | NCHW                   | **手写 NHWC NEON tile**                                                            |
| 公开 `InferenceSession.Run`                 | 逻辑 NCHW              | 仍收逻辑 NCHW；图输入已标 NHWC 时入口自动转置                                      |
| 像素格式                                    | 只认紧排 BGR           | 默认仍 `Bgr24`；RGB24 / BGRA32 / RGBA32 在 resize / warp 就地 gather，不摊中间 BGR |
| CLS 预处理                                  | PaddleX 拉伸 160×80 + ImageNet RGB | `ClsResizeImg` 保比例 + RecNorm BGR；宽高比 >4:1 只采左边 4×height（整行挤进 160 对 0/180 没意义） |

`PPOCR_NHWC=0` 仍可整图关回 NCHW。CI 不再跑 OpenVINO.NET。像素格式不增加整图缓冲：gather 写的是已经要做的双线性 / cubic scratch。

结论（细节在后面两节）：

- **正确率**：CI tiny bench sharp **767/1032、CER 2.36%**（cls 1020/1020；各 ISA / ns2 / scalar 相同）；c **764/1032、3.03%**（cls 998/1019）。相对 1.3 的 757/1022、3.53% 是满勤口径，不能当 CI tiny 涨了。本机行精确没动（742 / 950 / 1004）；CER 因左 1:4 CLS 下降（2.78% → **2.37%**，0.60% → **0.41%**，0.67% → **0.14%**）。
- **墙钟**：CI 上收益在 ns2 / noavx / scalar / ARM AdvSIMD；x64 AVX2 默认路径和 1.3 持平（噪声）。本机 tiny net10 **0.73×**，ns2 三个模型 **0.48–0.70×**。
- **内存**：tiny-4w 工作集峰值大约少 **300 MB**（7763 817→515，N2 840→572）。主要是 NHWC workspace 别名，以及预处理直写 NHWC、不再为输入 `LayoutConvert` 留第二份缓冲。

## CI 基线

回归只看这两条：**win-x64 EPYC 7763** 的 SIMD 套件，以及 **linux-arm64 Neoverse N2**（`CPU part 0xd49`，6 replica 几乎一条直线）。osx-arm64 是 3 核 / 7 GB 虚拟 M1，只看同 replica 比值。

| 判定 | 尺子 |
| ---- | ---- |
| x64 | `tiny-4w` 中位大约 **180–210 ms**（单次 180–221 都见过，不要用一份 replica 喊回归）；`tiny-4w-ns2` 同 replica 比值大约 **1.02–1.30** |
| ARM64 | `tiny-4w` **175–185 ms**；ns2 比值 **1.63–1.68**。比 Windows 更适合做自动阈值 |
| 引擎 | 只用 win-x64 引擎套件、同一 replica。c/sharp 4w 大约 **1.67–1.80**（`20d0de6`）。不要再和 OpenVINO.NET 比新数 |
| 不要 | 把 9V74/Xeon 的 AVX-512 和 7763 的 AVX2 写成「优化了 30%」；用 osx 绝对时间做门禁；拿 20 张 smoke 和 100 张 bench 比 |

win-x64 SIMD 先丢一次 25 张 tiny-4w 烤 VM（不上传），再跑默认 / noavx512 / ns2。noavx2 / noavx / scalar 只在 `smoke-win-x64-isa` 跑 20 张。

数据集、模型、预解码 BGR、ISA 开关、runner：`.github/workflows/test.yml`，`dataset/` 固定种子合成 100 张 JPG。库 TFM 默认 `net10.0`；`tiny-4w-ns2` 把库编成 `netstandard2.0`，仍跑在 .NET 10 上。

正确率（100 张满勤）：sharp **767/1032、CER 2.36%**（cls 1020/1020；含 scalar、ns2、全部 ISA）；c **764/1032、3.03%**（cls 998/1019）。smoke 20 张满勤：tiny **160/218**（CER 1.90%），small **198/218**（0.40%），medium **210/218**（0.13%）。

### linux-arm64 N2（最稳，6 replica）

1.3 是 NCHW AdvSIMD；1.4.2 是手写 NHWC NEON。同一 `ubuntu-24.04-arm`、同一 `0xd49`。

| 用例             | 1.3 中位 | 1.4.2 中位 |      相对 | 1.3 同 replica 比 | 1.4.2 同 replica 比 |
| ---------------- | -------: | -------: | --------: | ----------------: | ----------------: |
| `tiny-4w`        |  **241** |  **180** | **0.75×** |              1.00 |              1.00 |
| `tiny-1w`        |      390 |  **291** | **0.75×** |              1.63 |              1.65 |
| `tiny-4w-ns2`    |      374 |  **295** | **0.79×** |              1.55 |              1.66 |
| `tiny-4w-scalar` |      984 |  **856** | **0.87×** |              4.16 |              4.75 |

- net10 AdvSIMD 是 1.4.2 最大的平台级收益：4w / 1w 都大约快 **25%**。
- ns2 在 ARM 上仍走 `Vector`（128-bit），没有手写 tile，但也吃到预处理直写 NHWC，大约快 **21%**。
- scalar 有专用 NHWC tile，大约快 **13%**。相对 4w 的倍数从 4.16 升到 4.75，是因为 4w 自己更快了。
- 1.3 曾用 NCHW AdvSIMD vs NHWC `Vector` Count==4（约 230 vs 286）决定默认关闸；手写 NEON 之后已经翻过来。

### win-x64 SIMD（只报 EPYC 7763）

1.3 这次 run 有 3 份 7763；1.4.2 墙钟底（当时对外叫 1.4）也是 3 份（另 3 份落到 Xeon / 9V74）。中位是各 replica median 的中位数。

| 用例               | 有效 ISA      | 1.3 中位 |    1.4.2 中位（范围） |          相对 | 1.3 同 replica 比 | 1.4.2 同 replica 比 |
| ------------------ | ------------- | -------- | --------------------: | ------------: | ----------------: | ----------------: |
| `tiny-4w`          | AVX2          | **167**  |    **184**（183–221） | ~1.1×（噪声） |              1.00 |              1.00 |
| `tiny-4w-noavx512` | AVX2          | 152      |    **140**（136–156） |         0.92× |              0.89 |              0.76 |
| `tiny-4w-noavx2`   | AVX           | 307      |    **328**（327–331） |         1.07× |              1.75 |              1.75 |
| `tiny-4w-noavx`    | Vector        | 481      |    **380**（376–390） |     **0.79×** |              2.76 |              2.00 |
| `tiny-4w-ns2`      | Vector / NHWC | **343**  |    **228**（220–229） |     **0.66×** |          **2.02** |          **1.24** |
| `tiny-4w-scalar`   | scalar        | 1368     | **1220**（1199–1278） |     **0.89×** |               7.7 |               6.3 |

- **AVX2 默认路径和 1.3 持平。** SIMD job 连续跑 6 个 case，7763 单次 183–221 都见过；引擎套件同机 4w 反而从 154 降到 142。不要用一份 221 喊回归。
- **ns2 是 x64 上 1.4.2 最大的收益**：343 → 228（约 **1.5×** 吞吐）。1.3 的 ns2 仍是 NCHW Vector，比值 ~2.0；1.4.2 走到 NHWC，比值 **1.24**。
- **noavx**（只留 `Vector`）481 → 380（**0.79×**）。**scalar** 换成专用 16 路寄存器累加 tile：1368 → 1220（**0.89×**）。
- `noavx512` 在 7763 上本来就没有 AVX-512，和 `tiny-4w` 同 ISA，差值是噪声。`noavx2`（只留 AVX）没有单独的 NHWC AVX tile，和 1.3 重叠。

并入 [35217602432](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35217602432) 的 4 份 7763 之后，1.4.2 `tiny-4w` 中位约 **191**（n=7，范围 183–221），ns2 仍是 221–241。结论不变。

### win-x64 引擎套件（只报 EPYC 7763）

和 SIMD job 不是同一台 VM，绝对毫秒不要和上一张表硬接。1.3 CI 的 c 还不是 `20d0de6`；1.4.2 已钉到 [`lw_ppocr_c.20260914.20d0de6.dll`](https://cv-public.sdcb.ai/2026/lw_ppocr_c.20260914.20d0de6.dll)。

| 用例            | 1.3 中位 | 1.4.2 中位 |         相对 | 1.3 vs sharp 4w | 1.4.2 vs sharp 4w |
| --------------- | -------: | -------: | -----------: | --------------: | --------------: |
| sharp `tiny-4w` |      154 |  **142** |        0.92× |            1.00 |            1.00 |
| sharp `tiny-1w` |      217 |  **200** |        0.92× |            1.41 |            1.41 |
| c `tiny-4w`     |      288 |      254 | （c 换 DLL） |            1.82 |        **1.76** |
| c `tiny-1w`     |      496 |      360 | （c 换 DLL） |            3.10 |            2.43 |

1.4.2 c/sharp 约 **1.7×**，c 更省内存（peak ~535 MB vs sharp ~513 MB）。OpenVINO.NET 已从 CI 去掉；1.3 同 replica 是 sharp 的 1.38×（本库反超），只解释历史。

### osx-arm64（高噪声，只看比值）

虚拟 M1、3 逻辑核、7 GB。绝对毫秒 replica 之间可以差一倍。

| 用例             | 1.3 同 replica 比 | 1.4.2 同 replica 比 |
| ---------------- | ----------------: | ----------------: |
| `tiny-4w`        |              1.00 |              1.00 |
| `tiny-1w`        |              ~1.9 |              ~2.1 |
| `tiny-4w-ns2`    |              1.20 |          **1.67** |
| `tiny-4w-scalar` |              4.08 |              4.39 |

1.4.2 的 4w 自己变快之后，ns2（仍是 Vector）相对倍数会被拉开，和 linux-arm64 同一现象。只用来确认 AdvSIMD 路径能跑。

### 平台 smoke（20 张，只看覆盖）

CPU 每次都会变。行精确按 **20 张满勤**（[35361330684](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35361330684)）；中位 ms / CPU 仍用 [35241030966](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35241030966)，只看覆盖。旧口径 152/208 是跳过首张，满勤是 **160/218**。

| RID              | 1.4.2 这次 CPU | ISA     | 中位 ms |  行精确 |
| ---------------- | ------------ | ------- | ------: | ------: |
| linux-arm64      | N2           | AdvSimd |     227 | 160/218 |
| linux-x64 tiny   | 7763         | AVX2    | 324–352 | 160/218 |
| linux-x64 small  | 7763         | AVX2    |     958 | 198/218 |
| linux-x64 medium | 7763         | AVX2    |    5200 | 210/218 |
| win-x64          | 9V45         | AVX-512 |     138 | 160/218 |
| win-x86          | 9V74         | AVX2    |     406 | 160/218 |
| win-arm64        | Cobalt 100   | AdvSimd |     213 | 160/218 |
| osx-arm64        | M1 Virtual   | AdvSimd |     324 | 160/218 |
| osx-x64          | i7-8700B     | AVX2    |     217 | 160/218 |

small 大约是 tiny 的 3 倍墙钟，medium 大约是 tiny 的 15–17 倍；CER 从 3.7% → 1.6% → 0.34%。

### CI 工作集（tiny-4w）

| 场景                    |              peak |            Δ WS |
| ----------------------- | ----------------: | --------------: |
| win-x64 7763 sharp      | 817 → **515 MB** | 398 → **107 MB** |
| linux-arm64 N2 sharp    | 840 → **572 MB** | 418 → **162 MB** |
| osx-arm64 sharp         | 715 → 704 MB     |        1.4.2：289 MB |
| win-x64 lw.PPOCR.C      | 586 → 535 MB     |        1.4.2：105 MB |

x64 / ARM64 本库 peak 大约少 **300 MB**。c 仍然略省，但更慢。OpenVINO.NET 1.3 时 tiny 已近 2.6 GB，不再新测。

## 本机 5800X

发布前复测（墙钟/内存 `97c1448`，2026-09-19；CER `68a009a` 左 1:4 同机复测）：HOME-MAIN，Ryzen 7 5800X / 16 逻辑核 / 64 GB / AVX2（无 AVX-512），`.NET 10.0.11`。1.4.2 走当前树；1.3 走 NuGet `Sdcb.SimdPaddleOCR` 1.3.0。同一 `dataset/`，先 25 张 tiny 烤机，再各 100 张 `--warmup 1`。墙钟 n=99；准确率与 Δ WS 满勤 **1036** 行。箭头均为 **1.3 → 1.4.2**。不要用这里的绝对毫秒卡 GitHub runner。

| 模型   | 引擎            |                mean |         loaded |              peak |            Δ WS |     exact_lines |           CER |
| ------ | --------------- | ------------------: | -------------: | ----------------: | --------------: | --------------: | ------------: |
| tiny   | 本库 net10 AVX2 |  86.0 → **63.1**（0.73×） | 415 → 401 MB | 804 → **515 MB** | 389 → **113 MB** |        742/1036 | 2.78% → **2.37%** |
| tiny   | 本库 ns2        | 203.1 → **96.5**（0.48×） | 417 → 404 MB | 766 → 522 MB | 342 → 117 MB |        742/1036 | 2.78% → **2.37%** |
| small  | 本库 net10 AVX2 | 222.1 → **200.0**（0.90×） | 512 → 452 MB | 1195 → **674 MB** | 677 → **217 MB** |        950/1036 | 0.60% → **0.41%** |
| small  | 本库 ns2        | 432.1 → **303.0**（0.70×） | 525 → 461 MB | 1277 → 684 MB | 751 → 218 MB |        950/1036 | 0.60% → **0.41%** |
| medium | 本库 net10 AVX2 |   628 → **585**（0.93×） | 959 → 699 MB | 2489 → **1206 MB** | 1528 → **505 MB** | 1002 → **1004**/1036 | 0.67% → **0.14%** |
| medium | 本库 ns2        |  1606 → **874**（0.54×） | 986 → 726 MB | 2663 → 1218 MB | 1677 → 490 MB | 1002 → **1004**/1036 | 0.67% → **0.14%** |

- net10：**tiny 0.73×**（预处理直写 NHWC），small / medium **0.90× / 0.93×**，墙钟接近，收益主要在内存。
- **ns2 是本机最大墙钟收益**（1.3 仍是 NCHW `Vector`）：tiny **0.48×**、small **0.70×**、medium **0.54×**。1.4.2 的 ns2 / net10 收成 **1.50–1.53×**。
- **工作集大约腰斩**：tiny peak 804 → **515 MB**（和 7763 CI 同一把尺子），Δ WS 389 → 113；medium peak 2489 → **1206**。loaded 两边接近，差在跑图 Δ WS。
- 行精确在 1.3 / 1.4.2、net10 / ns2 之间相同（tiny / small 没动；medium 1002 → **1004** 是更早的 `ClsResizeImg`）。CER 再降一截是左 1:4 CLS：倒长行会先转正再进 REC。ns2 与 net10 共用 `Cls()`，CER 不必另测。

1.4.2 JSON：`bench-out/local-5800x-{tiny,small,medium}-4w.json`、`bench-out/local-5800x-ns2-{tiny,small,medium}-4w.json`。1.3：`bench-out/local-5800x-v13-{net10,ns2}-{tiny,small,medium}-4w.json`。

### Vulkan GPU（RTX 3080 Ti，`8d62a35`）

同机同轮：HOME-MAIN 5800X + RTX 3080 Ti（驱动 581.80），`.NET 10.0.11`，`--engine sharp|vulkan --workers 4 --benchmark-kind simd --warmup 1`，同一 `dataset/` 100 张变尺寸图（对 GPU 最不利的逐图新 shape）。墙钟 n=99。**CPU 列是同轮重测**，不要拿上表 1.4.2 的 `97c1448` mean 硬接（那次 tiny 63.1 / small 200 / medium 585）。B580 另一台机、另一份尺子，见 [vulkan-b580.md](vulkan-b580.md)；Radeon 880M 核显见 [vulkan-880m.md](vulkan-880m.md)。

端到端（median ms/图，越低越好；加速 = CPU median / Vulkan median）：

| 模型   | CPU median | Vulkan median |      加速 | CPU mean | Vulkan mean | CPU img/s | GPU img/s | CPU peak | Vulkan peak |
| ------ | ---------: | ------------: | --------: | -------: | ----------: | --------: | --------: | -------: | ----------: |
| tiny   |       50.3 |      **19.8** |  **2.5×** |     54.9 |        23.3 |     18.22 |     42.91 |  520 MB |     643 MB |
| small  |      173.1 |      **28.4** |  **6.1×** |    173.1 |        31.5 |      5.78 |     31.72 |  675 MB |     703 MB |
| medium |      562.6 |      **38.1** | **14.8×** |    555.5 |        42.3 |      1.80 |     23.62 | 1210 MB |     985 MB |

相对上一版 `26ad4c3`（`d6c260c`），同一时段交替 A/B 各 3 轮（Vulkan median ms/图）：tiny 21.9 / 18.9 / 23.5 → 19.8 / 20.1 / 18.5（持平），small 36.0 / 35.8 / 34.3 → 28.4 / 27.8 / 29.1，medium 129.8 / 126.3 / 128.3 → **38.1 / 39.5 / 39.0（3.3×）**。纯 GPU（`SIMD_OCR_GPU_PROF` 最小值）：medium DET 960×960 55.5 → 9.1 ms，medium REC 8×480 36.6 → 7.7 ms。

分阶段 mean ms/图（4w 下算子重叠，之和可以大于墙钟）：

| 模型   | 阶段      |    CPU | Vulkan | 加速 |
| ------ | --------- | -----: | -----: | ---: |
| tiny   | det_graph |   20.1 |    4.5 | 4.5× |
| tiny   | cls_graph |   16.2 |    1.1 |  15× |
| tiny   | rec_graph |   70.3 |    5.7 |  12× |
| small  | det_graph |   71.3 |    6.5 |  11× |
| small  | cls_graph |   18.3 |    1.2 |  16× |
| small  | rec_graph |  314.4 |   14.2 |  22× |
| medium | det_graph |  199.2 |    9.2 |  22× |
| medium | cls_graph |   15.9 |    0.9 |  18× |
| medium | rec_graph | 1259.6 |   22.0 |  57× |

正确率（100 张满勤，1036 行）：

| 模型   | CPU exact_lines | Vulkan exact_lines | CPU CER | Vulkan CER | CPU exact_img | Vulkan exact_img |
| ------ | --------------: | -----------------: | ------: | ---------: | ------------: | ---------------: |
| tiny   |        742/1036 |           741/1036 |   2.37% |      2.38% |         5/100 |            5/100 |
| small  |        950/1036 |           950/1036 |   0.41% |      0.40% |        44/100 |           44/100 |
| medium |       1004/1036 |          1006/1036 |   0.14% |      0.14% |        71/100 |           73/100 |

- 3080 Ti 上 `26ad4c3` 的瓶颈几乎全在 sg32 coopmat GEMM：它直接从全局内存 `coopMatLoad`，大 GEMM 只有 ~3.5 TFLOPS，medium DET 55 ms 里占 46 ms。重写后（共享内存双缓冲暂存、16 B 读、向量化收尾、窄 tile、隐式 GEMM kxk 卷积、SE 预缩放）大 GEMM 到 25–28 TFLOPS。
- medium `rec_graph` 22 ms 里约 12 ms 是 CPU 上的 CTC 投影 + ArgMax（词表 18710 列，契约要求留在 CPU）；small 同一个词表，也基本是这部分。tiny 端到端主要是 CPU 前后处理，GPU 部分 DET ~2 ms、REC ~1 ms。
- mean 明显高于 median 主要来自前 ~40 张的 .NET 分层 JIT 预热（`DOTNET_TieredCompilation=0` 下消失），不是 GPU。
- 准确率与纯 CPU 持平，差异是 fp16 噪声：tiny 差 1 行，small 行精确一致，medium GPU 多对 2 行。cls 全对。
- 显存：sg32 上 arena 按生命周期复用，medium DET 960×960 每个 session 1233 MB → 108 MB。GpuBench `--conc` medium 8 线程不再 OOM 回退 CPU（`26ad4c3`：76.9 s；现在 1.6 s，显存峰值 4.8 GB）。进程工作集有界，peak 后不再爬升。
- JSON：`bench-out/fin-cpu-{tiny,small,medium}.json`、`bench-out/fin-{base,new}-{tiny,small,medium}-vulkan-r{1,2,3}.json`。复现：`--engine sharp|vulkan --workers 4 --model {tiny,small,medium} --input dataset --warmup 1`。

### lw.PPOCR.C 4w（`20d0de6`）

当前树 harness、`--engine c`、`--c-assets bench-out/c-runtime`，DLL [`lw_ppocr_c.20260914.20d0de6.dll`](https://cv-public.sdcb.ai/2026/lw_ppocr_c.20260914.20d0de6.dll)。同一尺子。C 没有 stage / operator 剖析。

| 模型   |   mean | median |    p95 | img/s |  loaded |        peak |    Δ WS |   exact_lines |   exact_img |       CER |
| ------ | -----: | -----: | -----: | ----: | ------: | ----------: | ------: | ------------: | ----------: | --------: |
| tiny   |  201.9 |  202.7 |  264.4 |   5.0 | 396 MB |     541 MB | 102 MB |      752/1036 |        7/100 |     4.09% |
| small  |  541.7 |  538.5 |  692.5 |   1.8 | 434 MB |     760 MB | 271 MB |      936/1036 |       39/100 |     1.08% |
| medium | 2319.5 | 2362.3 | 2924.0 |   0.4 | 585 MB |    1479 MB | 774 MB | **1014/1036** |       80/100 | **0.24%** |

JSON：`bench-out/local-5800x-c-{tiny,small,medium}-4w.json`。

## 数据来源与复现

| 版本       | Actions                                                                       | 提交      | 说明                                          |
| ---------- | ----------------------------------------------------------------------------- | --------- | --------------------------------------------- |
| **1.3.0**  | [34818949921](https://github.com/sdcb/SimdPaddleOCR/actions/runs/34818949921) | `37fe1fd` | 只 AVX2 NHWC；ns2 / ARM / scalar 仍 NCHW      |
| 1.4 墙钟底 | [35241030966](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35241030966) | `b784d28` | 当时对外叫 1.4；下文 7763 / N2 **墙钟**默认指这次 |
| 1.4 复核   | [35361330684](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35361330684) | `97c1448` | 准确率改满勤 100 张（当时 766/1032）；墙钟与 1.3 比意思不变。SIMD 7763 只有 1 份，不重写中位表 |
| 1.4 前一次 | [35217602432](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35217602432) | `67cf1fa` | 内核已是后来的 1.4.2 墙钟；用来给 win-x64 7763 补样本 |
| **1.4.2**  | [35513018083](https://github.com/sdcb/SimdPaddleOCR/actions/runs/35513018083) | `68a009a` | 当前口径。准度：sharp **767/1032、CER 2.36%**；c **764/1032、3.03%**。墙钟中位表不重写 |
| 本机 5800X | —                                                                             | `97c1448` / `68a009a` | 墙钟/内存 `97c1448`；CER 左 1:4 后同机复测（1.4.2） |
| 本机 Vulkan | —                                                                            | `8d62a35` | 5800X + 3080 Ti 同轮 sharp/vulkan 4w；tiny/small/medium；A/B 基线 `d6c260c` |
| 880M Vulkan | —                                                                            | `2520c77` | 锐龙 AI 9 H365 + Radeon 880M 同轮 sharp/vulkan 4w；A/B 基线 `713fc03` / `d6c260c` |

推送或手动触发 [`.github/workflows/test.yml`](../.github/workflows/test.yml)，下载 `perf-report` artifact。本地同一套数据：

```text
dotnet run --project test/Sdcb.SimdPaddleOCR.TestData -c Release -- --out dataset
dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -o artifacts/net10
artifacts/net10/Sdcb.SimdPaddleOCR.Tests --benchmark --engine sharp --workers 4 --model tiny --input dataset --out bench-out/tiny-4w.json
```

C 另加 `--engine c --c-assets bench-out/c-runtime`。Vulkan 另加 `--engine vulkan`。`--summarize` 可并排多份 JSON。
