# WebAssembly（browser-wasm, Ryzen 7 5800X）：main vs v2.0.0 基准报告

- **测试日期**：2026-11-27
- **宿主硬件**：AMD Ryzen 7 5800X（8C/16T），Windows 11
- **浏览器引擎**：Microsoft Edge 158（Chromium Headless）
- **运行时与编译**：.NET 10.0 (`browser-wasm`，`wasm-tools-net10` / Emscripten 3.1.56)，LLVM AOT (`RunAOTCompilation=true`)，Web Workers + `SharedArrayBuffer`（`--workers 4`）
- **SIMD 路径**：`v2.0.0` 走 `Vector<float>`（`wasm-simd128`）；`main` 走专用 `PackedSimd`（WASM SIMD128）内核
- **测试集**：`dataset/` 100 张（共 1036 行，1 张预热 + 99 张实测），同机交替运行（`tiny` 为 2 轮 ABBA 交替均值，`small`/`medium` 为单轮交替）
- **对比版本**：`v2.0.0` (`6972083`) vs `main` (`33f7bc7`)

## 1. 端到端性能（4 Workers, 100 张）

| 模型 | v2.0.0 Mean (ms) | main Mean (ms) | v2.0.0 Median (ms) | main Median (ms) | main P95 (ms) | 吞吐 (img/s) | 加速比 (Median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tiny` | 127.2 | **95.5** | 127.2 | **95.2** | 121.1 | **10.5** | **1.34×** (-25.2%) |
| `small` | 416.0 | **309.5** | 414.6 | **313.9** | 396.0 | **3.23** | **1.32×** (-24.3%) |
| `medium` | 1729.0 | **1446.7** | 1753.7 | **1482.9** | 1853.8 | **0.69** | **1.18×** (-15.4%) |

- `tiny` 两轮 ABBA 交替中位：`v2.0.0` 123.6 / 130.8 ms，`main` **96.0 / 94.4 ms**。
- 在同机 Ryzen 7 5800X 上，浏览器 WASM `tiny 4w`（95.2 ms）已达到桌面原生 AVX2（50.9 ms）的约 **1.87×**（相比 v2.0.0 的 2.5× 大幅拉近）。

## 2. 分阶段耗时（Mean ms/图）

| 阶段 | tiny (v2.0 → main) | small (v2.0 → main) | medium (v2.0 → main) |
|---|---:|---:|---:|
| `det_graph` | 59.0 → **34.4** (-41.7%) | 165.4 → **82.6** (-50.1%) | 517.5 → **392.1** (-24.2%) |
| `cls_graph` | 28.4 → **25.9** | 32.1 → 35.0 | 35.9 → 36.8 |
| `rec_graph`（总和） | 172.0 → **143.6** (-16.5%) | 843.3 → **744.9** (-11.7%) | 4363.7 → **3789.8** (-13.2%) |
| `lines_wall`（4w 墙钟） | 56.8 → **48.6** (-14.4%) | 238.8 → **215.8** (-9.6%) | 1202.5 → **1042.8** (-13.3%) |
| `rec_reshape` | 2.4 → **2.0** | 5.6 → **3.2** | 7.3 → **4.0** |

## 3. 托管堆内存（Managed Heap）

| 模型 | v2.0.0 平均 / 峰值 (MB) | main 平均 / 峰值 (MB) |
|---|---:|---:|
| `tiny` | 59.9 / 73.3 | 75.0 / 91.5 |
| `small` | 129.0 / 160.9 | 150.5 / 179.3 |
| `medium` | 384.4 / 460.8 | 389.6 / 474.3 |

全模型峰值均远低于 WASM 2 GB / 4 GB 寻址上限。

## 4. 准确率（100 张 / 1036 行，`dataset/metadata.json`）

| 模型 | v2.0.0 字符准确率 / CER | main 字符准确率 / CER | v2.0.0 行精确匹配 | main 行精确匹配 |
|---|---:|---:|---:|---:|
| `tiny` | 97.63% / 2.37% | **97.77% / 2.23%** | 769 / 1036 (74.23%) | **781 / 1036 (75.39%)** |
| `small` | 99.59% / 0.41% | 99.52% / 0.48% | 968 / 1036 (93.44%) | 954 / 1036 (92.08%) |
| `medium` | 99.86% / 0.14% | 99.77% / 0.23% | 1016 / 1036 (98.07%) | 1010 / 1036 (97.49%) |

> 注：`v2.0.0` 的 `WasmBench` 测试壳使用 `SixLabors.ImageSharp` 解码 JPEG，`main` 切换为 `Sdcb.SimdCodecs 2.1.0`，两套 JPEG IDCT 存在 ±1 LSB 像素差异，属于正常解码器差异，三档模型 CER 均保持同水平（`tiny` 略升）。

## 5. 运行要点与复现命令

- **COOP/COEP 响应头**：`<WasmEnableThreads>true</WasmEnableThreads>` 依赖浏览器 `SharedArrayBuffer`，服务端必须返回 `Cross-Origin-Opener-Policy: same-origin` 与 `Cross-Origin-Embedder-Policy: require-corp`（`server.mjs` 已内置）。
- **必须开启 LLVM AOT**：生产部署需保持 `<RunAOTCompilation>true</RunAOTCompilation>`（需安装 `dotnet workload install wasm-tools-net10`），未开启 AOT 时性能会慢 8–30×。

```powershell
# 编译发布 WASM AOT
dotnet publish test/Sdcb.SimdPaddleOCR.WasmBench -c Release

# 运行基准测试（Headless Edge/Chrome）
node test/Sdcb.SimdPaddleOCR.WasmBench/server.mjs --workers 4 --model tiny --count 100
node test/Sdcb.SimdPaddleOCR.WasmBench/server.mjs --workers 4 --model small --count 100
node test/Sdcb.SimdPaddleOCR.WasmBench/server.mjs --workers 4 --model medium --count 100
```
