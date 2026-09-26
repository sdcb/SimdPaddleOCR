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
| commit | `8cbc299`（本分支） |

注：M4 (Virtual) 为虚拟化核数，`1w` 数字接近物理单线程，`4w`/`8w` 受调度影响。

## 端到端（median ms/图，越小越好）

| 模型 | 1w | 4w | 8w |
| --- | ---: | ---: | ---: |
| tiny | 110.1 | 80.5 | 75.8 |
| small | 330.9 | 272.4 | 264.3 |
| medium | 1033.3 | 1016.0 | 1060.3 |

img/s（mean）：tiny 8.7→12.7，small 3.0→3.8，medium 0.97→0.99→0.91。
**medium 在 4w 已饱和**（8w 反而略退）；tiny/small 4w→8w 只剩 <5%。

## 准确率（满勤 100 张）

| 模型 | exact_lines | CER | cls |
| --- | --- | --- | --- |
| tiny | 744/1024 | 1.61% | 1020/1020 |
| small | 811/1024 | 1.40% | 1018/1019 |
| medium | 894/1024 | **0.68%** | 1023/1023 |

## 结构（4w，stage / operator ms 均值——多线程重叠，仅供看占比）

| stage | tiny | small | medium |
| --- | ---: | ---: | ---: |
| det_graph | 34.5 | 91.1 | 322.2 |
| cls_graph | 11.0 | 11.2 | 12.4 |
| rec_graph | 125.7 | 624.7 | 2549.1 |
| lines_wall（实际行墙钟） | 37.9 | 170.5 | 682.6 |

| operator（medium-4w） | ms |
| --- | ---: |
| Conv | 2546（≈Conv1x1 为主） |
| MatMul | 267 |
| Erf / Concat / ReduceMean / MaxPool | 各 ≤12 |

要点：**Conv 占绝对大头且集中在 Conv1x1（本质 GEMM）**，REC 图耗时 ≈ DET 的 8×。
对 Metal 后端的含义：GPU 版首先要赢过 `medium-4w` 的 **rec_graph 2549ms / lines_wall 683ms**
这一格；算子预算全部押在 Conv(1x1) 与 MatMul/ArgMax（CTC 投影）上。

## 内存 / GC

| 模型(4w) | 每图分配 | WS 峰值 |
| --- | ---: | ---: |
| tiny | 1.6 MB | ~687 MB |
| small | 2.6 MB | ~941 MB |
| medium | 4.2 MB | ~1413 MB |

本分支修复后已无 pack 惊群与逐行重规划：剩余分配是首图一次性权重 pack（model 级缓存，留存非垃圾）+ 每图 ~1–4 MB 稳态零星分配。各模型 `rec_reshape` 已 ≈0。

## 复现

```bash
dotnet build test/Sdcb.SimdPaddleOCR.Tests -c Release
./test/Sdcb.SimdPaddleOCR.Tests/bin/Release/net10.0/Sdcb.SimdPaddleOCR.Tests \
  --model {tiny|small|medium} --input dataset --out out.json --workers {1|4|8} --warmup 1
# 数据集：dotnet run --project test/Sdcb.SimdPaddleOCR.TestData -c Release -- --out dataset
#         （需要 CJK 字体：PPOCR_FONTS_DIR=<dir-of-otf/ttf>）
```
