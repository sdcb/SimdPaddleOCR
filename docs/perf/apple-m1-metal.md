# Apple M1（macOS 15 ARM64，CPU + Metal）

* **平台**：GitHub Actions `macos-15`（Apple M1，macOS ARM64，Paravirt Metal GPU），.NET 10
* **测试口径**：GitHub Actions `test.yml`（run `45918847207`，commit `e92df9d`，已启用 `Sdcb.SimdCodecs 2.1.0` 流式解码与 Metal 缓存上限）；CPU 跑 100 张（3 轮），Metal 跑 20 张 smoke
* **测试日期**：2026-11-27

## 端到端性能与内存

| 后端 | 模型 | 样本数 | `v2.0.0` 中位 (ms) | **当前中位 (ms)** | `v2.0.0` 峰值 WS | **当前峰值 WS** |
|---|---|---:|---:|---:|---:|---:|
| CPU (NEON, 4w) | `tiny` | 100 (3轮中位) | 139.9 | **162.4** (104–194 波动) | 704 MB | **708 MB** |
| CPU (NEON, 4w) | `tiny` smoke | 20 | 280.3 | **185.3** | 302 MB | **305 MB** |
| Metal (GPU) | `tiny` smoke | 20 | 100.0 | **128.2** | 282 MB | **389 MB**（末张稳定 344 MB） |

## 阶段拆解（Metal `tiny` 20 张 smoke）

| 阶段 | `v2.0.0` (ms) | 当前 `e92df9d` (ms) |
|---|---:|---:|
| `det_graph` | 24.7 | 50.0 |
| `cls_graph` | 6.6 | 11.3 |
| `rec_graph` | 32.3 | 49.8 |
| **总中位** | **100.0** | **128.2** |

## 说明

* 相比共享引擎迁移初期（`52f805b`：186.9 ms / 562.8 MB），收紧 Metal `GpuOcrSession` 缓存上限并升级流式解码后，**中位延时下降 31%，峰值 WS 下降 174 MB**，且逐图内存已收敛（第 15 张 347.3 MB → 第 20 张 344.4 MB）。
* 剩余与 `v2.0.0` 的约 28 ms / 100 MB 差距来自 `external/SimdOnnx` 的 Metal 后端尚未加入 `MtlBufferPool` 与批调度/1×1 专用算子，后续将在 `SimdOnnx` 上游持续优化。
