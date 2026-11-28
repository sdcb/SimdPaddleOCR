# Apple M1（macOS ARM64 CI VM，CPU + Metal）

* **平台**：GitHub Actions `macos-26`（Apple M1 (Virtual)，3 vCPU，7 GB RAM，macOS 26.7.2，Metal 设备 `Apple Paravirtual device`），.NET 10.0.15
* **测试口径**：[`bench-macos-ab.yml`](../../.github/workflows/bench-macos-ab.yml)（CPU：run `46111171748`；Metal：修复后 run `46127900662`）。同一台 VM 内分别构建 `v2.0.0` 与 `main` 的 `test/Sdcb.SimdPaddleOCR.Tests`，`--benchmark --workers 4 --count 100 --warmup 1`，CPU 与 Metal 各 2 轮 ABBA 交替，中位延时取两轮中位的均值
* **数据集**：CI 在 `windows-2025` 上用 `test/Sdcb.SimdPaddleOCR.TestData` 现场生成。该机字体与本地不同，生成结果与本地 `dataset/` 不是同一份（1032 行），所以本页精度不能与其他平台直接比较，只用于两个版本之间的对比。
* **对比版本**：`v2.0.0` (`6972083`) vs `main`（CPU 行 `76f210c`；Metal 行为含 Metal 修复的 `f99df67`，CPU 路径与 `76f210c` 相同）
* **测试日期**：2026-11-28

## 端到端性能、内存与精度（100 张）

| 后端 | 模型 | 样本数 | `v2.0.0` 中位 (ms) | **main 中位 (ms)** | 速度比 | main 吞吐 (img/s) | 峰值 WS `v2.0 → main` (MB) | 加载后 WS `v2.0 → main` (MB) | 行精确匹配 `v2.0 → main` | CER `v2.0 → main` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU (NEON, 4w) | `tiny` | 100 | 216.3 | **171.0** | **1.26×** | 5.02 | 702 → 711 | 416 → 404 | 767 → 756 / 1032 | 3.04% → 3.10% |
| CPU (NEON, 4w) | `small` | 100 | 608.2 | **496.5** | **1.22×** | 1.99 | 913 → 970 | 466 → 453 | 939 → 956 / 1032 | 0.49% → 0.40% |
| CPU (NEON, 4w) | `medium` | 100 | 2082.2 | **1960.9** | **1.06×** | 0.48 | 1557 → 1444 | 707 → 669 | 988 → 1008 / 1032 | 0.25% → 0.15% |
| Metal (Paravirtual) | `tiny` | 100 | 95.1 | **100.3** | 0.95× | 9.75 | 597 → 612 | 418 → 407 | 767 → 754 / 1032 | 3.05% → 3.11% |
| Metal (Paravirtual) | `small` | 100 | 144.1 | **179.8** | 0.80× | 5.45 | 648 → 692 | 469 → 456 | 939 → 959 / 1032 | 0.49% → 0.40% |
| Metal (Paravirtual) | `medium` | 100 | 436.9 | **470.2** | 0.93× | 2.13 | 952 → 1061 | 710 → 673 | 988 → 1008 / 1032 | 0.25% → 0.15% |

* **WS** 为进程工作集：「加载后」为模型加载完成、处理第一张图之前；「峰值」为整轮最大值。精度按 `scripts/evaluate.py` 口径，对照该次 CI 生成的 `metadata.json`。
* CI VM 为共享虚拟机，轮间波动较大（CPU `small` `v2.0.0` 两轮 651.6 / 564.8 ms）。

## Metal 阶段拆解（`tiny`，ms/图均值，各 worker 合计）

| 阶段 | `v2.0.0` | main 修复前 (`76f210c`) | **main 修复后** |
|---|---:|---:|---:|
| `det_graph` | 24.5 | 57.1 | **29.2** |
| `cls_graph` | 6.4 | 14.0 | **9.5** |
| `rec_graph` | 43.6 | 71.9 | **39.3** |
| `lines_wall` | 61.5 | 99.4 | **55.6** |

## 小结

* **CPU (NEON)**：全档快于 `v2.0.0`，`tiny` **1.26×**、`small` **1.22×**、`medium` 1.06×。
* **Metal 回退已基本修复**（`f99df67`，引擎 SimdOnnx `23fab7d`）：修复前 `main` 为 `v2.0.0` 的 0.50× / 0.60× / 0.72×，峰值工作集 1329 / 1562 / 2027 MB；修复后为 **0.95× / 0.80× / 0.93×**，峰值 **612 / 692 / 1061 MB**，与 `v2.0.0` 相当。原因与修复：
  1. SE（squeeze-excitation）只用一个 threadgroup 池化整张图，GPU 大部分空闲 → 改为 `gap_part` + `se_gate_part` 两遍并行归约（与 Vulkan 一致），通道广播乘法与通道 concat 改为 vec4。
  2. REC 每个单元单独提交并等待 → `MtlBatch`：一张图的所有单元按层交错录进 concurrent encoder，每层一个 barrier，分 wave 提交。
  3. batch 形式的 CLS / REC 本来就逐张图派发空间 kernel → 改为交错的单图单元。
  4. 每个新尺寸都新建设备 arena 和主机输入输出数组（托管堆每张图约增长 10 MB）→ 每个消费者共用一个 grow-only `MtlArena`。
* 剩余差距主要在 `small` 的 DET：`v2.0.0` 把 SE 乘法融合进了其他 kernel，这里要融合需要改共享的计划编译器，暂未做。
* CI VM 轮间波动约 ±15%（例如 `v2.0.0` Metal `tiny` 两轮 107.3 / 82.8 ms）。
* **精度**：`small` / `medium` 因默认 `UnclipRatio` 1.6 提升（959 / 1008 行，CER 0.40% / 0.15%）。`tiny` 少 11–13 行，来自测试壳解码器（ImageSharp → `Sdcb.SimdCodecs`）在这份数据上的波动。Metal 修复前后输出一致。

## 复现

```powershell
gh workflow run bench-macos-ab.yml -f baseline=v2.0.0 -f count=100 -f models=tiny,small,medium -f engines=sharp,metal

# 逐 kernel 剖面（v2.0.0 vs 当前，同一 VM）：推送到 perf/** 分支即触发 .github/workflows/metal-prof.yml
```
