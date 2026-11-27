# Sdcb.SimdPaddleOCR 性能基准目录

统一口径：100 张多尺寸合成图（共 1036 行，含方向分类，1 张预热后取中位延时 ms/图），测试壳使用 `Sdcb.SimdCodecs 2.1.0` 流式解码。

## 各平台报告

| 平台 | CPU / GPU | 后端 | `tiny` 中位 | `small` 中位 | `medium` 中位 | 详细报告 |
|---|---|---|---:|---:|---:|---|
| Windows x64（独显） | Ryzen 7 5800X + RTX 3080 Ti | CPU (AVX2) / Vulkan | 49.4 / **17.7 ms** | 121.0 / **29.9 ms** | 541.7 / **49.0 ms** | [`docs/perf/5800x-rtx3080ti.md`](perf/5800x-rtx3080ti.md) |
| Windows x64（独显） | Ryzen 9 5950X + Intel Arc B580 | CPU (AVX2) / Vulkan | 59.4 / **21.7 ms** | 136.6 / **32.3 ms** | 520.1 / **53.3 ms** | [`docs/perf/5950x-b580.md`](perf/5950x-b580.md) |
| Windows x64（笔记本核显） | Ryzen AI 9 H 365 (Zen 5) + Radeon 880M | CPU (AVX-512) / Vulkan | 58.0 / **32.2 ms** | 169.9 / **54.9 ms** | 752.6 / **121.3 ms** | [`docs/perf/ai9-h365-880m.md`](perf/ai9-h365-880m.md) |
| Android 16 ARM64（手机） | Snapdragon 8 Gen 3 + Adreno 750 | CPU (NEON) / Vulkan | 150.9 / **117.4 ms** | 511.2 / **205.0 ms** | 2686.5 / **652.2 ms** | [`docs/perf/8gen3-adreno750.md`](perf/8gen3-adreno750.md) |
| Linux ARM64（开发板） | Rockchip RK3399（Orange Pi 4 LTS） | CPU (NEON, 4w) | **1040.3 ms** | **3970.9 ms** | **15769.1 ms** | [`docs/perf/rk3399-arm64.md`](perf/rk3399-arm64.md) |
| macOS 15 ARM64（CI VM） | Apple M1（Paravirt Metal） | CPU (NEON) / Metal | 162.4 / **128.2 ms** | — | — | [`docs/perf/apple-m1-metal.md`](perf/apple-m1-metal.md) |
| WebAssembly（浏览器） | Ryzen 7 5800X（Edge 158） | WASM SIMD128 (`PackedSimd`) + AOT | **95.2 ms** | **313.9 ms** | **1482.9 ms** | [`docs/perf/wasm-5800x.md`](perf/wasm-5800x.md) |

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

