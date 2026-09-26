# SimdPaddleOCR — Ryzen 9 5950X CPU vs Intel Arc B580 Vulkan

实测机:Windows Server,Ryzen 9 5950X(16C/32T),Intel Arc B580 12GB(ReBAR on),SDK .NET 10.0.401,`test/Sdcb.SimdPaddleOCR.Tests` bench(`--workers 4 --benchmark-kind simd`,warmup=1,n=99,dataset/ 100 张固定图)。

**结论:GPU 路径在此 harness 下目前是负收益且正确率有缺陷——仅作现状记录,不可作为合并依据。**

## 端到端(4 workers,median ms/图,越低越好)

| 模型 | CPU(sharp) | Vulkan | 比值 | CPU img/s | GPU img/s |
|---|---:|---:|---:|---:|---:|
| tiny    | **68.7**  | 276.4  | 0.25× | 13.67 | 3.89 |
| small   | **246.5** | 438.1  | 0.56× | 4.04  | 2.23 |
| medium  | **555.2** | 2208.2 | 0.25× | 1.82  | 0.45 |

## 分阶段耗时(mean ms/图)

| 模型 | 阶段 | CPU | Vulkan |
|---|---|---:|---:|
| tiny | det_graph | 30.9 | 33.1 |
| tiny | cls_graph | 19.9 | 60.8 |
| tiny | rec_graph | 80.4 | 292.3 |
| small | det_graph | 105.6 | 85.1 |
| small | cls_graph | 29.8 | 3.9 |
| small | rec_graph | 381.0 | 340.4 |
| medium | det_graph | 236.3 | 409.1 |
| medium | cls_graph | 23.0 | 55.6 |
| medium | rec_graph | 1078.4 | 4253.7 |

GPU 慢的根因(已定位,非硬件能力不足):**变 shape 时每个新 (N,H,W) 都触发一次 GpuGraph.BuildPlan**(录 CB + 分配 arena + pipeline 建链),100 张图的 det 输入几乎张张不同 → 每张图摊上一次数百 ms 的计划构建;且 plan/arena 无上限缓存不释放 → 内存单调爬升。

## 正确率

| 模型 | 指标 | CPU | Vulkan |
|---|---|---:|---:|
| tiny | cls | 1022/1022 | 408/420* |
| tiny | exact_lines | 742/1036 | 256/1036 |
| tiny | CER | 2.37% | **53.98%** |
| small | cls | 1034/1034 | 1034/1034 |
| small | exact_lines | 950/1036 | **0/1036** |
| small | CER | 0.41% | **84.84%** |
| medium | cls | 1035/1035 | 468/485* |
| medium | exact_lines | 1004/1036 | **0/1036** |
| medium | CER | 0.14% | **85.36%** |

*cls_total 不同 = DET 检出的框数已不同(tiny/medium 的 det 在变 shape 下产出不一致;small 的 det 完全一致)。REC 文本是"形状对、内容错"的近乱码——GPU 静默算错而非回落 CPU。

初步定位:
- **medium_rec/small_rec 的算子面远宽于 rec.onnx(tiny)**:medium_rec 含 MaxPool/Split 风格结构(13×MatMul、9×Transpose、8×Slice/Reshape、Sub/Pow/Sqrt 等);`GpuDetGraph.BuildPlan` 对 `n>1` 的 MaxPool 直接抛 `NotSupportedException`(单图 --pipe 可复现),走 `_gpuDead` 回落应给出正确结果——但实际输出仍错,说明图中有算子在批处理/变 shape 下被静默算错(或部分 dispatch 已执行后回落路径复用了脏状态)。
- **tiny rec  GPU 单图 --pipe 结果与 CPU 一致**(行数/文本同),但 bench 中 CER 0.54 —— 差异大概率在批处理路径(`RecognizeBatch` 宽度分桶)或并发复用,而非单图 kernel。

## 内存

| 模型 | 指标 | CPU | Vulkan |
|---|---|---:|---:|
| tiny | WS peak / Δ | 521MB / 118MB | 3897MB / **3468MB** |
| small | WS peak / Δ | 675MB / 218MB | 5781MB / **5254MB** |
| medium | WS peak / Δ | 1216MB / 516MB | 24133MB / **23406MB** |

20 张图连续运行时 WS 单调爬升(738→6570MB)——plan cache 无界 + arena 不回收。GPU 常驻内存(权重上传、plan arena)预期高于 CPU,但当前是无界增长。

## 修复方向(按收益排序)

1. **正确性**:medium_rec 的 MaxPool n>1 及宽算子面(Graph 里 Slice/Sub/Pow/Reshape/Transpose)逐个对拍;tiny 路径查批处理/并发下 rec 输出污染。
2. **变 shape 开销**:plan 构建与 arena 分离——同 pipeline 下 CB 参数化重录(或录一次 CB、push constants 走 shape),plan 缓存加 LRU 上限;arena 按 shape 家族复用而非每 plan 独占。
3. **回落粒度**:不支持算子应提前在 `Reshape` 期探出(避免提交一半再回落)。

注:以上 GPU 数字不代表 B580 上限——`--pipe` 固定 shape 下 det 曾测得 ~5× 加速、rec 批处理 5–8×;当前差距几乎全部来自变 shape 下重复 BuildPlan 的固定成本。
