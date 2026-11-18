# Sdcb.SimdPaddleOCR 性能基准

## 本机 5800X / RTX 3080 Ti：引擎改为源码内嵌（2026-11-18）

SimdOnnx 从 NuGet 包（`Sdcb.SimdOnnx` 0.1.0-preview.8）改为 git submodule 源码内嵌、类型全部 internal 后的同机 A/B。
旧 = `9b82a85`（引用 preview.8 包），新 = `45ce328`（`external/SimdOnnx` @ `0672ea8`）。两边引擎代码相同，只是类型可见性
（public → internal）和 XML 注释不同，因此预期墙钟不变。

- 本地 CPU / Vulkan / ns2.0：每轮新旧各跑一遍，**4 轮**，每轮换先后（奇数轮先旧、偶数轮先新）；100 张，`--warmup 1`。
- WASM：`test/Sdcb.SimdPaddleOCR.WasmBench`（AOT + 多线程，Edge headless），tiny 4w，100 张，**3 轮**同样交替。
- 「新/旧」是**逐轮**新旧 median 之比的几何平均（<1 表示新版更快）。这一节逐轮配对，不用跨轮汇总的 median 直接相除：
  机器越跑越热，两版一起漂移（CPU tiny 4w 从第 1 轮约 40 ms 涨到第 4 轮约 52 ms），漂移远大于版本差。

| case | 旧 median ms | 新 median ms | 新/旧 | 逐轮新/旧 | 峰值 WS 旧 → 新 (MB) |
| ---- | -----------: | -----------: | ----: | --------- | -------------------: |
| CPU tiny 4w | 49.27 | 50.91 | 1.018 | 1.009 / 1.083 / 0.997 / 0.987 | 654 → 654 |
| CPU tiny 1w | 79.63 | 78.57 | 0.984 | 0.995 / 1.007 / 0.969 / 0.966 | 648 → 648 |
| CPU small 4w | 129.27 | 129.59 | 1.021 | 1.058 / 1.005 / 1.000 / 1.024 | 816 → 804 |
| CPU medium 4w | 619.53 | 602.16 | 1.000 | 1.035 / 0.908 / 1.036 / 1.028 | 1202 → 1208 |
| Vulkan tiny 4w | 18.82 | 18.65 | 0.978 | 0.943 / 0.993 / 0.988 / 0.989 | 769 → 769 |
| Vulkan small 4w | 28.78 | 29.23 | 0.986 | 0.912 / 1.004 / 0.982 / 1.050 | 815 → 817 |
| Vulkan medium 4w | 43.18 | 44.58 | 1.015 | 0.970 / 1.035 / 1.030 / 1.026 | 989 → 989 |
| ns2.0 tiny 4w（net10 宿主） | 76.26 | 77.21 | 0.999 | 0.961 / 1.038 / 0.971 / 1.027 | 666 → 679 |
| WASM tiny 4w | 97.3 / 98.1 / 97.2 | 97.4 / 97.7 / 96.9 | 0.998 | 1.001 / 0.996 / 0.998 | — |

（「旧 / 新 median」一栏是 4 轮 median 的中位数；WASM 一栏列出 3 轮各自的 median。）

- **结论：没有回退。** 几何平均全在 ±2.2% 以内，而且有快有慢；单轮比值在 0.91–1.08 之间跳动，说明 ±2% 是噪声量级。WASM 最稳，3 轮比值是 1.001 / 0.996 / 0.998。
- **输出逐张一致**：每个 case 新旧各轮的行 hash 全部相同。峰值工作集持平。
- JSON 没有入库。复现方法：在两个 worktree 里各执行
  `dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release -r win-x64 --no-self-contained -o <dir>`（ns2.0 另加 `-p:UseNs20Library=true`），
  再交替运行 `--benchmark --engine sharp|vulkan --workers 4|1 --model tiny|small|medium --input dataset --count 100 --warmup 1`；
  WASM 跑 `dotnet publish test/Sdcb.SimdPaddleOCR.WasmBench -c Release` 和 `node test/Sdcb.SimdPaddleOCR.WasmBench/server.mjs --workers 4 --model tiny --count 100`。
