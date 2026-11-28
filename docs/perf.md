# Sdcb.SimdPaddleOCR 性能基准目录

统一口径：100 张多尺寸合成图（共 1036 行，含方向分类，1 张预热后取中位延时 ms/图），测试壳使用 `Sdcb.SimdCodecs 2.1.0` 流式解码。自 2026-11-28 起，各平台均为 `main` 与 `v2.0.0` 在同机 ABBA 交替实测，并记录进程工作集（加载后 / 峰值）。

> **精度口径（2026-11-28 起）**：行精确匹配与 CER 均与数据集官方 `scripts/evaluate.py` 一致——一条预测行被精确匹配消耗后，不能再计入另一条 GT 行，CER 只在剩余预测行中取最小编辑距离，按 Unicode 码点计数。
> 在此之前，各平台报告里的 **CER 用的是旧口径**（在全部预测行中取最小编辑距离），数值偏低。以 `v2.1.0` 的 x64 CPU 输出为例：`tiny` 2.23% → **2.62%**，`small` 0.48% → **0.68%**，`medium` 0.23% → **0.38%**。行精确匹配数不受影响。
> `5800x-rtx3080ti.md` 与 `wasm-5800x.md` 中原先的 781 / 954 / 1010 不是精度回退前的数据：它们是用"预测行可被重复匹配"的错误计分得出的，相同输出按官方口径为 753 / 934 / 998。2026-11-28 起所有平台报告均已按官方口径重测。

## 各平台报告

`main` (`76f210c`) 中位延时，4 Workers；括号内为同机 ABBA 交替实测的 `v2.0.0 → main` 速度比，峰值 WS 为 `tiny` 整轮进程工作集峰值（WASM 为托管堆）。

| 平台 | CPU / GPU | 后端 | `tiny` 中位 | `small` 中位 | `medium` 中位 | `tiny` 峰值 WS（CPU / GPU） | 详细报告 |
|---|---|---|---:|---:|---:|---:|---|
| Windows x64（独显） | Ryzen 7 5800X + RTX 3080 Ti | CPU (AVX2) / Vulkan | 43.5 / **16.3 ms**（1.09× / 1.13×） | 113.3 / **30.5 ms**（1.28× / 1.17×） | 498.0 / **42.0 ms**（1.07× / 1.08×） | 525 / 623 MB | [`docs/perf/5800x-rtx3080ti.md`](perf/5800x-rtx3080ti.md) |
| Windows x64（独显） | Ryzen 9 5950X + Intel Arc B580 | CPU (AVX2) / Vulkan | 68.2 / **23.3 ms**（0.98× / 0.97×） | 151.0 / **35.3 ms**（1.42× / 0.99×） | 482.4 / **51.3 ms**（1.12× / 1.10×） | 533 / 571 MB | [`docs/perf/5950x-b580.md`](perf/5950x-b580.md) |
| Windows x64（笔记本核显） | Ryzen AI 9 H 365 (Zen 5) + Radeon 880M | CPU (AVX-512) / Vulkan | 61.7 / **32.1 ms**（1.04× / 0.94×） | 174.0 / **51.2 ms**（1.23× / 0.97×） | 764.4 / **117.9 ms**（1.05× / 1.07×） | 517 / 832 MB | [`docs/perf/ai9-h365-880m.md`](perf/ai9-h365-880m.md) |
| Windows x64（无 AVX / 入门独显） | Pentium Gold G5500 (SSE4.2) + GeForce MX150 | CPU (SSE4.2) / Vulkan | 384.9 / **273.3 ms**（1.18× / 2.61×） | 1388.2 / **747.4 ms**（1.27× / 4.21×）¹ | 6421.3 / **2757.0 ms**（1.09× / 2.76×）¹ | 492 / 616 MB | [`docs/perf/g5500-mx150.md`](perf/g5500-mx150.md) |
| Android 16 ARM64（手机） | Snapdragon 8 Gen 3 + Adreno 750 | CPU (NEON) / Vulkan | 150.9 / **117.4 ms** | 511.2 / **205.0 ms** | 2686.5 / **652.2 ms** | 778 / 766 MB | [`docs/perf/8gen3-adreno750.md`](perf/8gen3-adreno750.md)（2026-11-27，待重测） |
| Linux ARM64（开发板） | Rockchip RK3399（Orange Pi 4 LTS） | CPU (NEON, 4w) | **1039.2 ms**（1.11×） | **3720.5 ms**（1.38×）¹ | **15243.6 ms**（1.00×）¹ | 524 MB | [`docs/perf/rk3399-arm64.md`](perf/rk3399-arm64.md) |
| macOS ARM64（CI VM） | Apple M1（Paravirt Metal） | CPU (NEON) / Metal | 171.0 / 163.2 ms（1.26× / **0.50×**） | 496.5 / 246.3 ms（1.22× / **0.60×**） | 1960.9 / 600.2 ms（1.06× / **0.72×**） | 711 / 1329 MB | [`docs/perf/apple-m1-metal.md`](perf/apple-m1-metal.md) |
| WebAssembly（浏览器） | Ryzen 7 5800X（Edge） | WASM SIMD128 (`PackedSimd`) + AOT | **92.9 ms**（1.26×） | **274.8 ms**（1.61×） | **1323.7 ms**（1.31×） | 85 MB（托管堆） | [`docs/perf/wasm-5800x.md`](perf/wasm-5800x.md) |

¹ `small` 25 张、`medium` 6 张。

* **精度**（官方 `evaluate.py` 口径，本地 `dataset/` 100 张 / 1036 行，`main` CPU 输出在 AVX2 / AVX-512 / SSE4.2 / NEON / WASM 上逐字一致）：`tiny` 753（CER 2.62%）、`small` **954**（0.58%）、`medium` **1014**（0.31%）。`small` / `medium` 自本版起默认 `UnclipRatio` 1.6（`tiny` 保持 1.4），相对 1.4 多对 20 / 16 行，且延时下降 2–5%。
* **Metal 回退**：macOS Metal 在 `main` 上明显慢于 `v2.0.0`，峰值内存约翻倍，详见 M1 报告。

## 历史归档（`perf/legacy/`）

* [`docs/perf/legacy/vulkan-uhd770.md`](perf/legacy/vulkan-uhd770.md) — Intel UHD Graphics 770（无协作矩阵核显，v2.0 时期实测归档）

## 复现方式

```bash
# 桌面 CPU / Vulkan（Windows / Linux / macOS）
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- \
  --benchmark --engine sharp --workers 4 --model tiny --input dataset --count 100 --warmup 1

dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- \
  --benchmark --engine vulkan --workers 4 --model tiny --input dataset --count 100 --warmup 1
```

