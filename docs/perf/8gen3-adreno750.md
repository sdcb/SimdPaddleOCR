# Qualcomm Snapdragon 8 Gen 3 + Adreno 750（Android 16 ARM64）

* **平台**：realme RMX3888，高通骁龙 8 Gen 3（`SM8650`，`arm64-v8a`），Adreno 750（Vulkan），16 GB RAM，Android 16
* **测试口径**：`test/Sdcb.SimdPaddleOCR.AndroidBench`（`net10.0-android`，CoreCLR RyuJIT，`--e2e --model tiny|small|medium --backend cpu|vulkan`），`dataset/` 100 张（共 1036 行），交替 A/B（`main` vs `v2.0.0`）
* **测试日期**：2026-11-27

## 端到端性能与精度

| 后端 | 模型 | `v2.0.0` 中位 (后50) | **当前中位 (后50)** | 速度比 | 正确行数 | CER | 温控状态 |
|---|---|---:|---:|---:|---:|---:|---|
| CPU (NEON) | `tiny` | 182.4 (177.4) ms | **150.9 (154.7) ms** | **1.21×** | 753 / 1036 | 2.23% | 0 / 0 |
| CPU (NEON) | `small` | 585.3 (587.5) ms | **511.2 (517.4) ms** | **1.15×** | 934 / 1036 | 0.48% | 0 / 0 |
| CPU (NEON) | `medium` | 2594.9 (2685.9) ms | **2686.5 (2697.5) ms** | **1.00× (后50)**† | 998 / 1036 | 0.23% | 3 / 3 |
| Vulkan (Adreno 750) | `tiny` | 123.6 (119.2) ms | **117.4 (113.6) ms** | **1.05×** | 751 / 1036 | 2.26% | 0 / 0 |
| Vulkan (Adreno 750) | `small` | 226.1 (217.9) ms | **205.0 (196.4) ms** | **1.10×** | 935 / 1036 | 0.48% | 0 / 0 |
| Vulkan (Adreno 750) | `medium` | 849.4 (850.1) ms | **652.2 (680.9) ms** | **1.30×** | 999 / 1036 | 0.22% | 3 / 2 |

† `medium CPU`：`v2.0.0` 冷机（43.1℃）开跑并将机身拉至 54.5℃（`thermal=3`），`main` 接棒在 54.5℃ 开跑；热平衡后后 50 张中位几乎一致（2685.9 vs 2697.5 ms）。

## 内存（升级 `Sdcb.SimdCodecs 2.1.0` `OpenRowReader` 后实测）

| 配置（`tiny`，100 图） | `v2.0.0` 峰值 WS | **当前峰值 WS** |
|---|---:|---:|
| CPU | 744 MB | **778 MB** |
| Vulkan | 776 MB | **766 MB**（低于 `v2.0.0` 10 MB） |

## 复现方式

```powershell
# 构建、安装并推送模型与数据集
.\test\Sdcb.SimdPaddleOCR.AndroidBench\run.ps1 -Build -Install -Push -Bench "--e2e --model tiny --backend vulkan"
```
