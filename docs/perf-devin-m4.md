# SimdPaddleOCR 性能基准 — Apple M4 (osx-arm64)

> 口径与 CI 守门见 `perf.md`。本文是**单台 Apple Silicon 的实测基线**，主要用途：
> 给将来的 Metal GPU 后端提供同机 CPU 对照（det/rec 图级耗时、算子分布、带宽与内存水位）。

## 环境

| 项 | 值 |
| --- | --- |
| CPU | Apple M4 (Virtual)，8 vCPU |
| 内存 | 16 GB |
| OS | macOS 26.5.2 |
| 运行时 | .NET 10.0.12，`advsimd` 档（`Vector<float>.Count=4`） |
| 引擎 | sharp（纯托管 ONNX 解释器），默认 NHWC 图 |
| 数据 | `test/Sdcb.SimdPaddleOCR.TestData` 固定种子合成 **100 张**（~1200–1800px 宽，均 ≈10.2 行/图） |
| 选项 | `AdaptiveWidth=true, TargetWidth=320`，`--warmup 1` |
| commit | `8025058`（含 AdvSimd MatMul + plan/pack 去重） |

注：M4 (Virtual) 为虚拟化核数，`1w` 数字接近物理单线程，`4w` 受调度影响。

## 端到端（median ms/图，越小越好）

| 模型 | 1w | 4w |
| --- | ---: | ---: |
| tiny | 111.9 | 80.6 |
| small | 306.3 | 257.8 |
| medium | 1030.6 | 907.5 |

img/s（mean）：tiny 8.8→12.2，small 3.4→3.8，medium 0.98→1.10。
**medium 在 4w 已饱和**（加到 8w 反而略退）。

## 准确率（满勤 100 张）

| 模型 | exact_lines | CER | cls |
| --- | --- | --- | --- |
| tiny | 744/1024 | 1.61% | 1020/1020 |
| small | 811/1024 | 1.40% | 1018/1019 |
| medium | 894/1024 | **0.68%** | 1023/1023 |

## 结构（4w，stage / operator ms 均值——多线程重叠，仅供看占比）

| stage | tiny | small | medium |
| --- | ---: | ---: | ---: |
| det_graph | 39.2 | 102.1 | 305.2 |
| cls_graph | 11.3 | 12.0 | 11.6 |
| rec_graph | 112.0 | 572.1 | 2220.6 |
| lines_wall（实际行墙钟） | 34.5 | 157.0 | 594.6 |

| operator（medium-4w） | ms |
| --- | ---: |
| Conv | 2388（≈Conv1x1 为主） |
| MatMul | 84 |
| Erf / Concat / ReduceMean | 各 ≤12 |

要点：**Conv 占绝对大头且集中在 Conv1x1（本质 GEMM）**，REC 图耗时 ≈ DET 的 7×。
MatMul 已有 AdvSimd 直写 kernel（较旧 `Vector<T>` 档 ~2.5–3×），占 op 总量 <4%。
对 Metal 后端的含义：GPU 版首先要赢过 `medium-4w` 的 **rec_graph 2221ms / lines_wall 595ms**
这一格；算子预算全部押在 Conv(1x1) 与 MatMul/ArgMax（CTC 投影）上。

## 内存 / GC

| 模型(4w) | 每图分配 | WS 峰值 |
| --- | ---: | ---: |
| tiny | 1.6 MB | ~689 MB |
| small | 2.6 MB | ~830 MB |
| medium | 4.2 MB | ~1457 MB |

首图一次性权重 pack 为 model 级留存（Lazy 去重后只算一次），稳态每图 ~1–4 MB 零星分配。
各模型 `rec_reshape` 已 ≈0（shape plan 缓存命中即回放）。

## 复现

```bash
dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release
./test/Sdcb.SimdPaddleOCR.Tests/bin/Release/net10.0/Sdcb.SimdPaddleOCR.Tests \
  --model {tiny|small|medium} --input dataset --out out.json --workers {1|4} --warmup 1
# 数据集：dotnet run --project test/Sdcb.SimdPaddleOCR.TestData -c Release -- --out dataset
#         （需要 CJK 字体：PPOCR_FONTS_DIR=<dir-of-otf/ttf>）
```
