# WebAssembly（browser-wasm, Ryzen 7 5800X）：main vs v2.0.0 基准报告

- **测试日期**：2026-11-28
- **宿主硬件**：AMD Ryzen 7 5800X（8C/16T），Windows 11
- **浏览器引擎**：Microsoft Edge（Chromium Headless）
- **运行时与编译**：.NET 10 (`browser-wasm`，`wasm-tools-net10`)，LLVM AOT (`RunAOTCompilation=true`)，Web Workers + `SharedArrayBuffer`（`--workers 4`）
- **SIMD 路径**：`v2.0.0` 走 `Vector<float>`（`wasm-simd128`）；`main` 走专用 `PackedSimd`（WASM SIMD128）内核
- **测试集**：`dataset/` 100 张（共 1036 行，第 1 张为预热，延时统计 99 张，精度计全部 100 张），同机 2 轮 ABBA 交替（`v2.0.0 → main → main → v2.0.0`），中位延时取两轮中位的均值
- **对比版本**：`v2.0.0` (`6972083`) vs `main` (`76f210c`)

## 1. 端到端性能、内存与精度（4 Workers, 100 张）

| 后端 | 模型 | 样本数 | `v2.0.0` 中位 (ms) | **main 中位 (ms)** | 速度比 | main 吞吐 (img/s) | 托管堆峰值 `v2.0 → main` (MB) | 托管堆首图 `v2.0 → main` (MB) | 行精确匹配 `v2.0 → main` | CER `v2.0 → main` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WASM SIMD128 (4 Workers) | `tiny` | 100 | 116.7 | **92.9** | **1.26×** | 10.82 | 72 → 85 | 44 → 47 | 742 → 753 / 1036 | 2.78% → 2.62% |
| WASM SIMD128 (4 Workers) | `small` | 100 | 441.8 | **274.8** | **1.61×** | 3.73 | 165 → 180 | 108 → 111 | 950 → 954 / 1036 | 0.60% → 0.58% |
| WASM SIMD128 (4 Workers) | `medium` | 100 | 1728.7 | **1323.7** | **1.31×** | 0.77 | 456 → 471 | 334 → 333 | 1004 → 1014 / 1036 | 0.26% → 0.31% |

- 浏览器内没有进程工作集，内存列为 .NET 托管堆（`GC.GetTotalMemory`）：「首图」为处理完第一张图后，「峰值」为整轮最大值。全模型峰值都远低于 WASM 的 2 GB / 4 GB 寻址上限。
- 两轮中位：`tiny` `v2.0.0` 121.2 / 112.2 ms，`main` **94.1 / 91.8 ms**；`small` 476.3 / 407.3 vs **273.9 / 275.7 ms**。
- `main` 的 WASM 输出与桌面 CPU（AVX2 / AVX-512 / NEON）**100/100 逐字一致**。
- 同机桌面原生 AVX2 `tiny 4w` 为 43.5 ms，浏览器 WASM（92.9 ms）约为其 **2.1×**。

## 2. 分阶段耗时（Mean ms/图，两轮均值）

| 阶段 | tiny (v2.0 → main) | small (v2.0 → main) | medium (v2.0 → main) |
|---|---:|---:|---:|
| `det_graph` | 53.5 → **34.2** (-36%) | 179.0 → **77.7** (-57%) | 526.0 → **386.6** (-27%) |
| `cls_graph` | 28.2 → **25.7** | 31.6 → **26.9** | 32.6 → **29.0** |
| `rec_graph`（各 worker 合计） | 162.4 → **141.5** (-13%) | 885.2 → **637.0** (-28%) | 4270.8 → **3352.5** (-22%) |
| `lines_wall`（4w 墙钟） | 54.4 → **47.8** (-12%) | 252.3 → **180.5** (-28%) | 1166.7 → **905.7** (-22%) |
| `rec_reshape` | 2.4 → **1.8** | 5.6 → **3.0** | 7.1 → **4.5** |

## 3. 精度说明

`v2.0.0` 的 `WasmBench` 测试壳用 `SixLabors.ImageSharp` 解码 JPEG，`main` 用 `Sdcb.SimdCodecs`。数据集 JPEG 为 4:2:0 色度子采样，两者差异主要在色度上采样（平均 |ΔCb/Cr| 0.23、|ΔY| 0.04，约 1.7% 像素差 ≥ 8，最大 78）；`Sdcb.SimdCodecs` 与 libjpeg-turbo（PIL / OpenCV，即上游 PaddleOCR 的解码器）几乎逐像素一致。`small` / `medium` 另有默认 `UnclipRatio` 1.4 → 1.6 的提升，详见 [`5800x-rtx3080ti.md`](5800x-rtx3080ti.md#小结)。

> 2026-11-27 版本中的 781 / 954 / 1010 与 769 / 968 / 1016 是用"已被精确匹配消耗的预测行仍可再匹配另一条重复 GT 行"的错误计分得出的，已作废，以本页为准。

## 4. 运行要点与复现命令

- **COOP/COEP 响应头**：`<WasmEnableThreads>true</WasmEnableThreads>` 依赖浏览器 `SharedArrayBuffer`，服务端必须返回 `Cross-Origin-Opener-Policy: same-origin` 与 `Cross-Origin-Embedder-Policy: require-corp`（`server.mjs` 已内置）。
- **必须开启 LLVM AOT**：生产部署需保持 `<RunAOTCompilation>true</RunAOTCompilation>`（需安装 `dotnet workload install wasm-tools-net10`），未开启 AOT 时性能会慢 8–30×。

```powershell
# 编译发布 WASM AOT
dotnet publish test/Sdcb.SimdPaddleOCR.WasmBench -c Release

# 运行基准测试（Headless Edge/Chrome）
node test/Sdcb.SimdPaddleOCR.WasmBench/server.mjs --workers 4 --model tiny --count 100 --out main-tiny-wasm-r1.json
node test/Sdcb.SimdPaddleOCR.WasmBench/server.mjs --workers 4 --model small --count 100
node test/Sdcb.SimdPaddleOCR.WasmBench/server.mjs --workers 4 --model medium --count 100
```
