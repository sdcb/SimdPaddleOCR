# Apple M1（macOS ARM64 CI VM，CPU + Metal）

* **平台**：GitHub Actions `macos-26`（Apple M1 (Virtual)，3 vCPU，7 GB RAM，macOS 26.7.2，Metal 设备 `Apple Paravirtual device`），.NET 10.0.15
* **测试口径**：[`bench-macos-ab.yml`](../../.github/workflows/bench-macos-ab.yml)（run `46111171748`）。同一台 VM 内分别构建 `v2.0.0` 与 `main` 的 `test/Sdcb.SimdPaddleOCR.Tests`，`--benchmark --workers 4 --count 100 --warmup 1`，CPU 与 Metal 各 2 轮 ABBA 交替，中位延时取两轮中位的均值
* **数据集**：CI 在 `windows-2025` 上用 `test/Sdcb.SimdPaddleOCR.TestData` 现场生成。该机字体与本地不同，生成结果与本地 `dataset/` 不是同一份（1032 行），所以本页精度不能与其他平台直接比较，只用于两个版本之间的对比。
* **对比版本**：`v2.0.0` (`6972083`) vs `main` (`76f210c`)
* **测试日期**：2026-11-28

## 端到端性能、内存与精度（100 张）

| 后端 | 模型 | 样本数 | `v2.0.0` 中位 (ms) | **main 中位 (ms)** | 速度比 | main 吞吐 (img/s) | 峰值 WS `v2.0 → main` (MB) | 加载后 WS `v2.0 → main` (MB) | 行精确匹配 `v2.0 → main` | CER `v2.0 → main` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU (NEON, 4w) | `tiny` | 100 | 216.3 | **171.0** | **1.26×** | 5.02 | 702 → 711 | 416 → 404 | 767 → 756 / 1032 | 3.04% → 3.10% |
| CPU (NEON, 4w) | `small` | 100 | 608.2 | **496.5** | **1.22×** | 1.99 | 913 → 970 | 466 → 453 | 939 → 956 / 1032 | 0.49% → 0.40% |
| CPU (NEON, 4w) | `medium` | 100 | 2082.2 | **1960.9** | **1.06×** | 0.48 | 1557 → 1444 | 707 → 669 | 988 → 1008 / 1032 | 0.25% → 0.15% |
| Metal (Paravirtual) | `tiny` | 100 | 81.6 | **163.2** | 0.50× | 5.87 | 597 → 1329 | 418 → 407 | 767 → 754 / 1032 | 3.05% → 3.11% |
| Metal (Paravirtual) | `small` | 100 | 148.7 | **246.3** | 0.60× | 3.93 | 646 → 1562 | 469 → 457 | 939 → 959 / 1032 | 0.49% → 0.40% |
| Metal (Paravirtual) | `medium` | 100 | 432.2 | **600.2** | 0.72× | 1.68 | 951 → 2027 | 709 → 674 | 988 → 1008 / 1032 | 0.25% → 0.15% |

* **WS** 为进程工作集：「加载后」为模型加载完成、处理第一张图之前；「峰值」为整轮最大值。精度按 `scripts/evaluate.py` 口径，对照该次 CI 生成的 `metadata.json`。
* CI VM 为共享虚拟机，轮间波动较大（CPU `small` `v2.0.0` 两轮 651.6 / 564.8 ms）。

## Metal 阶段拆解（`tiny`，ms/图均值）

| 阶段 | `v2.0.0` | main |
|---|---:|---:|
| `det_graph` | 22.4 | 57.1 |
| `cls_graph` | 6.2 | 14.0 |
| `rec_graph`（各 worker 合计） | 36.2 | 71.9 |
| `lines_wall` | 49.8 | 99.4 |

## 小结

* **CPU (NEON)**：全档快于 `v2.0.0`，`tiny` **1.26×**、`small` **1.22×**、`medium` 1.06×；`medium` 峰值 1557 → 1444 MB。
* **Metal 明显回退**：100 张完整跑下 `main` 为 `v2.0.0` 的 **0.50× / 0.60× / 0.72×**，峰值工作集翻倍以上（`tiny` 597 → 1329 MB，`medium` 951 → 2027 MB）。det / cls / rec 三个图都慢约 2×，回退位于共享引擎 `external/SimdOnnx` 的 Metal 后端（上一版报告已指出它尚缺 `MtlBufferPool` 缓冲池与批调度；具体原因本轮未剖析）。上一版报告只跑了 20 张 smoke（128.2 vs 100.0 ms），低估了这个差距。在 Metal 后端补齐之前，macOS 上 `tiny` 用 CPU 已经和 Metal 一样快（171.0 vs 163.2 ms），内存只有约一半。
* **精度**：`small` / `medium` 因默认 `UnclipRatio` 1.6 提升（956 / 1008 行，CER 0.40% / 0.15%）。`tiny` 少 11–13 行来自测试壳解码器（ImageSharp → `Sdcb.SimdCodecs`）在这份数据上的波动。

## 复现

```powershell
gh workflow run bench-macos-ab.yml -f baseline=v2.0.0 -f count=100 -f models=tiny,small,medium -f engines=sharp,metal
```
