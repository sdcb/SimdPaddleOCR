# SimdPaddleOCR — Radeon 880M（RDNA 3.5 核显）Vulkan

实测机：SDCB-PC，锐龙 AI 9 H365（4× Zen 5 + 6× Zen 5c / 20 线程，AVX-512），Radeon 880M 核显（12 CU），32 GB LPDDR5X 与 GPU 共享，Windows 11 26200、电源方案“高性能”，.NET SDK 10.0.302。`test/Sdcb.SimdPaddleOCR.Tests` bench（`--workers 4 --benchmark-kind simd --warmup 1`，n=99，同一 `dataset/` 100 张变尺寸图）。本表 `2520c77`。

**结论：sg32 路径在 AMD 上第一次跑就是对的，但按 NVIDIA 调的内核只用到了核显的一小部分。按能力门控的一套轻量 GEMM 变体加上几项访存优化之后，medium 188 → 119 ms（1.58×），small 61 → 47 ms，tiny 在噪声内（CPU 前后处理为主）；准确率与 CPU 持平，NVIDIA / Arc 的管线和 spv 不变。**

## 设备能力

| 项 | 值 |
|---|---|
| subgroup | 默认 64，`sgRange` 32–64；`requiredSubgroupSizeStages` = 0xf0（含 compute） |
| sg32 管线实际宽度 | 探针 shader（同一 `NewPipeline` 路径）：`requiredSubgroupSize=32` 时 `gl_SubgroupSize`=32、每 256 线程 8 个 subgroup；不指定时 64 / 4 |
| cooperative matrix | 16×16×16：f16→f32、f16→f16、int8/uint8→int32，全部 subgroup scope |
| 共享内存 | `maxComputeSharedMemorySize` = **32 KB**（原 sg32 GEMM 用 48 KB，n64 34 KB，靠 RDNA 每 WG 实有 64 KB 才没出错） |
| 峰值 | 纯寄存器 WMMA 实测 16.7 TFLOPS（≈2.7 GHz 满速） |
| 内存 | heap1 12 GB DEVICE_LOCAL（含 DL\|HV\|HC 类型），heap0 6 GB 系统内存（含 HV\|HC\|CACHED）；输入缓冲落在 DL\|HV\|HC，回读落在 HOST_CACHED |
| 队列 | family 1（compute-only，8 个）被选中；`globalPriority`=512（HIGH，非管理员也被授予）；`tsBits`=64，周期 10 ns |
| push descriptor | 有 |

## 端到端（4 workers，median ms/图）

| 模型 | CPU（sharp，同轮） | `d6c260c` | `713fc03` | **`2520c77`** | 对 CPU |
|---|---:|---:|---:|---:|---:|
| tiny   |  51.1 | 32.0 / 32.3 / 31.5 | 31.6 / 32.6 / 33.3 | **29.9 / 29.8 / 29.2** | 1.7× |
| small  | 159.4 | 56.2 / 56.6 / 55.5 | 61.0 / 61.0 / 60.5 | **46.7 / 47.1 / 47.7** | 3.4× |
| medium | 516.1 | 188.5 / 186.4 / 184.1 | 188.8 / 187.5 / 188.1 | **120.7 / 117.8 / 118.1** | 4.3× |

- 三列 Vulkan 都是同一时段与 `2520c77` 交替跑的 3 轮（先对 `713fc03`，再对 `d6c260c`，`2520c77` 取对 `713fc03` 那组）。
- CPU 列是在 30 分钟 GPU 负载之后测的，机器偏热；同一天冷机时 sharp 是 39.0 / 129.6 / 484.7。
- `--workers` 是单请求内的行并行度，请求本身串行：这里的数字是**单请求延迟**，GPU 在 CPU 阶段必然空闲（medium 时计算引擎利用率约 73%）。
- `d6c260c` → `713fc03`（3080 Ti 上的 GEMM 重写）在这台机器上 medium 持平、small/tiny 反而慢 5～7%：DET 变快（im2col 消失、大部分 GEMM 更快），REC 变慢（大 K 的 GEMM 比旧的直接全局读还慢）。

分阶段 mean ms/图（`2520c77`）：

| 模型 | 阶段 | CPU | Vulkan | `713fc03` Vulkan |
|---|---|---:|---:|---:|
| tiny | det_graph | 23.2 | 9.6 | 10.0 |
| tiny | rec_graph | 72.5 | 9.3 | 10.7 |
| small | det_graph | 69.0 | 12.4 | 14.8 |
| small | rec_graph | 287.6 | 23.4 | 33.6 |
| medium | det_graph | 190.7 | 37.0 | 52.8 |
| medium | rec_graph | 1149.4 | 68.9 | 120.7 |

纯 GPU（`SIMD_OCR_GPU_PROF=1`，10 次取最小，ms）：

| 用例 | `713fc03` | `2520c77` |
|---|---:|---:|
| DET medium 960×960 | 64.9 | 41.8 |
| DET medium 640×960 | 43.4 | 29.4 |
| REC medium 8×480 | 51.9 | 28.0 |
| REC medium 1×320 | 5.24 | 2.01 |
| DET small 960×960 | 14.2 | 11.0 |
| REC small 8×480 | 12.6 | 7.95 |
| DET tiny 960×960 | 7.7 | 6.3 |
| REC tiny 8×480 | 3.36 | 2.14 |

## 正确率（100 张满勤，1036 行）

| 模型 | CPU exact_lines / CER | Vulkan exact_lines / CER |
|---|---:|---:|
| tiny   | 742 / 2.37% | 740 / 2.39% |
| small  | 950 / 0.41% | 951 / 0.39% |
| medium | 1004 / 0.14% | 1003 / 0.15% |

cls 全对。tiny 的 2 行差来自 `436aa5c` 把几层小 1×1 卷积从 split-K 点积换到 coopmat（fp16 求和顺序噪声，含已知的 img-014）；其余提交逐图一致。`SIMD_OCR_VK_NOPUSH=1`、`SIMD_OCR_VK_QUEUE=gfx`、`SIMD_OCR_NOSK=1` 三档都与默认逐图一致；GpuBench `--conc` 4/8 线程零差异、无 fallback（medium 8 线程设备内存峰值 2.2 GB）；CPU 路径与 `d6c260c` 三档逐图一致。

工作集 peak（MB，CPU / Vulkan）：tiny 511 / 603，small 667 / 664，medium 1203 / 913。

## 改了什么

| 提交 | 内容 | 门控 |
|---|---|---|
| `767efe1` | 建 sg32 管线前检查能否把 compute subgroup 固定为 32 lane、有没有 16×16×16 f16/f32 coopmat；不满足就抛 `NotSupportedException`，会话在运行前回退 CPU（以前请求会被静默跳过，wave64 设备会静默算错） | sg32 |
| `46de4d1` | sg32l（`LITE`）：单缓冲暂存 28 KB、标量收尾、按 4 个 N 列分组光栅化，覆盖全部 9 个 conv1x1 / prescale / convk 变体；sg32d（`DIRECT`）：普通 1×1 与 MatMul GEMM（M≥16、K%16==0）直接从全局内存读 A/B 片段。K 仍按 16 顺序累加、收尾运算不变，结果与原内核逐位一致 | `SubgroupMax >= 64` 或 共享内存 < 48 KB |
| `a1dad0a` | k≤3 的 depthwise 走平铺内核 `conv_dw4a`（通道 quad 连续读，带 addps 吸收读），k9 仍走 tile 版 | sg32l |
| `9ed87b6` | SE 池化任意尺寸都走两段式合并读（原来 hw<4096 的 REC 走每 WG 一通道、跨步 2 字节读） | sg32l |
| `436aa5c` | K≤128 的 1×1 卷积改走 coopmat 的门槛从 M×cout > 2^18 降到 2^16 | sg32l |
| `2520c77` | 流式 REC 的 CTC ArgMax 在 AVX-512 机器上按列切分给线程，一次调用只读一遍 14.4 MB 词表矩阵（原来 T%8==4 的单元逐个调用，16 个线程各读一遍，和核显抢带宽） | AVX-512，仅 GPU 批量路径 |

几个关键观察：

- GEMM 在这块核显上不是算力瓶颈：纯寄存器 WMMA 16.7 TFLOPS，原 GEMM 只有 2.3–3.4 TFLOPS。最大的单项因素是 16 B 向量化收尾（换成标量收尾，13 个 REC/DET 形状的合成耗时和 17.0 → 13.5 ms），其次是光栅化分组（L2 只有 2 MB、没有 Infinity Cache），再次是去掉 LDS 暂存（→ 8.4 ms）。
- CTC ArgMax 单线程已接近 Zen 5 移动版（256 位数据通路）峰值，单独做 AVX2 式的列块化没有收益；问题在调用方式（逐单元 + 按行切分）导致的内存流量。

## 试过但没有保留

- 每 subgroup 64×64（4 或 8 个 subgroup）、BK=64、光栅化分组 2/8/16：都更慢。
- 仅单缓冲暂存（不换收尾）：更慢 5%。
- 直接读 GEMM 加寄存器双缓冲预取：持平。
- 64×256 tile（每 subgroup 32×64）：与 128×128 + 分组光栅化持平，不值得多一套 tile。
- AVX-512 ArgMax 行内核的列块化：隔离测试差异 3–5%（噪声内），pipeline 内无变化。
- `convt2s2` 单输出通道的“每 lane 一个输入像素”变体：读取不合并，medium DET 的这层 2.8 → 5.5 ms。
- `StreamCuts`：扫了 6 组（含完全不流式、更细的切分），默认值仍最好；完全不流式 medium 慢约 4%。
- split-K 门槛（1024–65536 个 tile）：全部在噪声内。
- 统一内存：输入写入 0.3–0.5 ms、录制+提交 0.3–0.6 ms，相对 20–80 ms 的 GPU 时间不到 2%，没有可省的拷贝。

## 需要在 B580 / 3080 Ti 上复测

- `767efe1` 的前置检查对所有 sg32 设备生效：3080 Ti 报 32–32 固定宽度、有 16×16×16 f16/f32，应当通过，未在该机实测。
- 其余 GPU 改动都只在 sg32l 设备上生效（NVIDIA 32–32 / 48 KB、Arc sg16 不满足门控）；新 spv 都是独立文件，原有 48 个 spv 重编后逐字节一致。
- `2520c77` 只在 AVX-512 机器上生效（5800X / 5950X 走原路径）。

## 剩余瓶颈

- medium REC：直接读 GEMM 仍占纯 GPU 的约 80%（8×480 里 22.4 / 28.0 ms，约 6–7 TFLOPS，峰值的 40%），其后是 `conv_dw4a` 1.7 ms、`elem4` 1.1 ms。
- medium DET：直接读 GEMM 17.2 ms，隐式 GEMM kxk 卷积（sg32l LDS 版）8.2 ms，9×9 depthwise 3.3 ms，SE 预缩放 GEMM 2.4 ms，最后一层 ConvTranspose 64→1 2.1 ms。
- CPU：每图 CTC ArgMax 约 13–16 ms（前面的 wave 与 GPU 重叠，最后一个 wave 暴露在关键路径上），加上前后处理、裁剪约 10 ms；tiny 基本是这部分。
