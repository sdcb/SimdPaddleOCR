# small 1w/4w：sharp vs lw.PPOCR.C 内存（Δ WS）分析

给后续实现用的交接文档。约束：**不要牺牲本库（sharp）的墙钟和正确率**，只评估/借鉴 C 侧为什么 100 张之后的 Working Set 涨得少。

本文数字是本机 Ryzen 7 5800X，不要和 CI replica 毫秒硬接。

---

## 1. 问题

small 模型、仓库 `dataset/` 100 张合成图。加载后两边进程内存接近（约 430–480 MB）；**跑完 100 张后的 Δ WS（`last - loaded`）差距很大**：

| workers | sharp Δ WS | c Δ WS | sharp 多涨 |
| ---: | ---: | ---: | ---: |
| 1 | +447 MB | +117 MB | **+330 MB** |
| 4 | +618 MB | +274 MB | **+344 MB** |

4w 相对 1w 的增量两边接近（sharp +171、c +157）。所以 **1w 上那 330 MB 不是 worker 乘出来的**。

sharp 的 peak ≈ last（涨上去不掉）。c 的 peak > last（中间冲一波再回落）。

---

## 2. 怎么跑

在仓库根目录。Release、同一会话内 **先 sharp 再 c**，不要并行，避免抢同一颗 CPU。

### 2.1 前置

| 项 | 位置 | 说明 |
| --- | --- | --- |
| 数据集 | `dataset/` | 100 张 `img-*.jpg` + `metadata.json`。gitignore。没有就 `dotnet run --project test/Sdcb.SimdPaddleOCR.TestData -c Release -- --out dataset` |
| C 资产 | `bench-out/c-runtime/` | `lw_ppocr_c.dll` + `small/{det,cls,rec}.lwm` + `small/ppocr_keys.txt`。tiny 可自动下载；**small/medium 必须本地有** |
| C 源码（对照） | `C:\Users\sdfly\source\repos\lw.PPOCR.C` | 必须是 **`20d0de66`**（2026-09-14）。本库钉的 DLL 名是 `lw_ppocr_c.20260914.20d0de6.dll` |

确认 C 仓库：

```text
git -C C:\Users\sdfly\source\repos\lw.PPOCR.C rev-parse HEAD
# 期望 20d0de66c1178df50f6f983908e6a0ab9c6ee003
```

Harness：

```text
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- --help
```

`--engine` 只能是 `sharp` / `c`。C 额外要 `--c-assets`。

统计口径：预解码 BGR，**第 1 张 warmup，n=99**。墙钟看 mean/median/p95。内存看 JSON 的四个字段（见 §3）。`--summarize` **只印时间与准确率，不印内存**，内存必须读 JSON。

### 2.2 命令（当前本机复现）

```powershell
# sharp small 4w
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release -- `
  --workers 4 --model small --engine sharp `
  --benchmark --benchmark-kind engine `
  --input dataset --out bench-out/now-small-4w.json --case-id small-4w

# c small 4w
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release --no-build -- `
  --workers 4 --model small --engine c `
  --benchmark --benchmark-kind engine `
  --input dataset --out bench-out/now-c-small-4w.json --case-id small-c-4w `
  --c-assets bench-out/c-runtime

# sharp small 1w
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release --no-build -- `
  --workers 1 --model small --engine sharp `
  --benchmark --benchmark-kind engine `
  --input dataset --out bench-out/now-small-1w.json --case-id small-1w

# c small 1w
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release --no-build -- `
  --workers 1 --model small --engine c `
  --benchmark --benchmark-kind engine `
  --input dataset --out bench-out/now-c-small-1w.json --case-id small-c-1w `
  --c-assets bench-out/c-runtime

# 只汇总墙钟/准确率（内存仍要自己读 JSON）
dotnet run --project test/Sdcb.SimdPaddleOCR.Tests -c Release --no-build -- `
  --summarize `
  bench-out/now-small-1w.json bench-out/now-c-small-1w.json `
  bench-out/now-small-4w.json bench-out/now-c-small-4w.json `
  --input dataset
```

1w 大约 2 分钟，4w 大约 1.5 分钟（含加载）。必须串行。

Harness 语义（不要改了还当同一基准）：

| 引擎 | 关键选项 |
| --- | --- |
| sharp | `LineWorkerCount = workers`；REC `AdaptiveWidth=true`、`TargetWidth=320`（**无 320 上限**，32px 一档）；DET `MaxSessionCacheEntries=32`（**已废弃，不生效**）；`MaxPooledSessions` 默认 `ProcessorCount`（本机 16） |
| c | `NativeOcr` 把 C 的 `Recognizer.TargetWidth` 设成 **960**（`LongTextTargetWidth`）；worker 数传入 `lw_ocr_create` |

两边 **REC 宽度策略本来就不同**。准确率数字不能假设“同一套预处理”。

---

## 3. 内存字段

每份 JSON 的 `meta` / `summary`：

| 字段 | 含义 | 何时采样 |
| --- | --- | --- |
| `working_set_mb_loaded` | 引擎加载完、还没跑图 | `PaddleOcrAll.Load` / `lw_ocr_create` 之后 |
| `working_set_mb_last` | 最后一张之后 | 第 100 张 `Run` 返回后 |
| `working_set_mb_peak` | 整段峰值 | 每张图之后取 max |
| `working_set_mb_delta` | `last - loaded` | 本文最关心的量 |

实现：`test/Sdcb.SimdPaddleOCR.Tests/Program.cs` 的 `WorkingSetMb()`（Windows `Process.WorkingSet64`）。

CI `TestReport` 印 peak 和 Δ WS，**不印 loaded**。本机 `--summarize` 三者都不印。

预解码的 100 张 BGR 在进程里常驻，**两引擎都有**，对比 Δ WS 时可忽略。

---

## 4. 当前基准

机器：Windows，AMD Ryzen 7 5800X（8C/16T），AVX2，无 AVX-512，内存 96 GB。机器名 `HOME-MAIN`。数据集：仓库 `dataset/` 100 张，n=99。

### 4.1 本轮（`b98db40`，2026-09-17）

JSON：`bench-out/now-small-{1,4}w.json`、`now-c-small-{1,4}w.json`。c DLL：`bench-out/c-runtime/lw_ppocr_c.dll`（`20d0de6`）。

| 引擎 | w | mean (median / p95) ms | img/s | vs sharp | exact_lines | CER | loaded | last | **Δ WS** | peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **sharp** | 1 | **307.8** (307.8 / 410.2) | 3.25 | 1.00 | **940/1026** | **0.61%** | 476.8 | 924.1 | **+447.3** | 927.3 |
| c | 1 | 916.8 (941.4 / 1160.5) | 1.09 | 2.98 | 926/1026 | 1.09% | 428.2 | 545.4 | **+117.1** | 591.4 |
| **sharp** | 4 | **225.9** (225.4 / 310.8) | 4.43 | 1.00 | **940/1026** | **0.61%** | 476.8 | 1094.4 | **+617.7** | 1094.4 |
| c | 4 | 474.7 (486.9 / 604.2) | 2.11 | 2.10 | 926/1026 | 1.09% | 432.2 | 706.1 | **+273.9** | 758.8 |

sharp 4w 阶段（mean ms，c 无 profiler）：

| stage | 1w | 4w |
| --- | ---: | ---: |
| `det_graph` | 75.4 | 96.0 |
| `cls_graph` | 12.0 | 18.0 |
| `rec_graph` | 203.8 | 407.0 |
| `lines_wall` | 223.9 | 117.8 |

`rec_graph` 4w 更大是算子并行重叠，**之和可以大于墙钟**，只看结构。

读法：

- 加载只差约 45–50 MB。
- 1w Δ WS：sharp 比 c 多涨 **330 MB**（约 3.8×）。
- 4w 相对 1w：sharp +170、c +157。多 3 个 worker **解释不了** 4w 上 344 MB 的 Δ WS 差距。
- 速度：1w c/sharp = 2.98×，4w = 2.10×。c 的 1→4 worker 加速比更大，绝对时间仍慢。
- 准确率两边都和 1.3.0 本机 small 表一致（与 worker 无关）。

### 4.2 1.3.0 本机 small 4w（`37fe1fd`，2026-09-14）

JSON：`bench-out/csharp-small-4w.json`、`c-small-4w.json`。

| 引擎 | mean (median / p95) | exact_lines | CER | WS peak |
| --- | ---: | ---: | ---: | ---: |
| sharp | 237.5 (239.6 / 313) | 940/1026 | 0.61% | 1211 MB |
| c | 541.6 (540.6 / 688) | 926/1026 | 1.09% | 759 MB |

本轮 sharp 225.9 / c 474.7，两边都快一些。c 没换 DLL，绝对毫秒差更像本机负载；**比值和 Δ WS 形状比单次墙钟稳**。1.3.0 那次没记 1w，也没单独报 loaded / Δ WS。

验收任何内存改动时，至少复跑 §2.2 四条，对照 §4.1：

1. sharp exact_lines 仍是 **940/1026**，CER **0.61%**。
2. sharp 1w mean 不要明显差于 ~308 ms，4w 不要明显差于 ~226 ms（同机、同负载）。
3. 若降 Δ WS，优先看 **1w**（那才是 330 MB 鸿沟）。

---

## 5. C 侧为什么 Δ WS 小

对照源码：`C:\Users\sdfly\source\repos\lw.PPOCR.C` @ `20d0de66`。本机 `out/ninja-rel`：`LW_EXPERIMENTAL_AVX2_FMA_DISPATCH=ON`，`LW_REC_RESIDENT_WIDTHS=OFF`。

Planner 本身不是差距：`src/runtime/memory.c` 的 `lw_plan_workspace()` 和 C# `InferenceSession.PlanWorkspace()` 一类——中间张量按生命周期复用，**一次推理零分配**。省的是 **跨图、跨 shape 的驻留**。

### 5.1 REC：最多 2 个具体宽度

`src/ppocr/recognizer.c`：adaptive 桶 **`{192,320,480,640,960}`**。默认模式每 worker 只留 active + cached；切到第 3 档前 `release_cached_session()`。扫过五档之后，稳态只剩最后两档，**更大宽度的 session/workspace 可以丢掉**。

可选 `lw_recognizer_enable_resident_widths()` 会预建 5 套（CMake `LW_REC_RESIDENT_WIDTHS`，本机 **OFF**）。

Harness 给 C 的上限是 960，不是 C# 的 320。

### 5.2 DET：换尺寸就拆 session

`src/ppocr/detector.c` `ensure_session()`：`(W,H)` 变了就 `lw_session_free` 旧 session 和 input/prob，再 malloc。当前图小，工作区按小图来，**不 ratchet 到历史最大**。

### 5.3 DB 后处理：每图 malloc/free

`src/ppocr/db_postprocess.c`：每张图 bitmap / visited / queue / points / hull，结束全部 `free`。这是 **peak > last** 的主要来源之一。

### 5.4 CTC：不常驻 `[T×词表]`

官方 REC 末端 Softmax 可走 greedy：只分配每 timestep 的 `best_indices` + `best_probabilities`（2 个值），不要完整概率平面。

### 5.5 权重

LWM 常量零拷贝指向 `model->bytes`（`src/runtime/executor.c`）。SIMD packed weights 在 `lw_session_create` 时进独立 aligned 块；clone worker 用 `lw_session_share_prepared_constants()` refcount 共享。session 释放则 pack 一起释放。路径是 **NCHW**，没有 NHWC 第二套 pack。

### 5.6 分配器

Windows `_aligned_malloc` / CRT `malloc`/`free`。没有 grow-only managed `float[]`。`free` 后页更容易离开 Working Set，所以 last 能低于 peak。

### 5.7 4w 多出来的 ~157 MB

每额外 worker：一份 CLS session + 一份 REC session（workspace + I/O）。模型 bytes 和 packed weights **不** ×N。DET、OCR 级 scratch、`ocr->crop` 共享（crop 只增不减）。+157 / 3 ≈ 52 MB/worker，和 small REC@960 + CLS 量级相符。

---

## 6. C# 侧为什么 Δ WS 大

### 6.1 先澄清：`MaxSessionCacheEntries` 无效

`SharpEngine` 设了 32，但 `PaddleOcrDetectorOptions.MaxSessionCacheEntries` 注释写明 **Ignored**。没有 per-shape session 表。真正的上限是 **`MaxPooledSessions`（默认 `Environment.ProcessorCount`）**。

1w/4w 并发建不出 16 个 REC session，**这个默认值解释不了 1w +447 MB**。

### 6.2 主因（按可能性）

**1. REC/DET `InferenceSession` grow-only workspace（最高）**

`src/Sdcb.SimdPaddleOCR/OnnxSharp/InferenceSession.cs` `PlanWorkspace()`：`_workspace` 只涨不缩。注释写明一缩就要 realloc+zero，曾经把 `rec_reshape` 打到 70–250 ms，tiny 1w / small 4w 回归。

REC 池（`PaddleOcrRecognizer`）按 `HighWaterInputVolume` best-fit。每个 session 记住见过的最大 `batch×3×48×width`。100 张之后工作区 ≈ **见过的最大 shape**，不回落 → peak ≈ last。

**2. Adaptive 宽度：32px、无上限（高）**

`PaddleOcrRecognizer.SelectTargetWidth()`：按 48×宽高比，向上取 32 的倍数，**故意没有 320 帽**（长行不压进固定 tensor）。100 张多行可能见到几十个 distinct width。C 最多 5 档。

**3. `Model` 级 lazy packed weights（高，与 worker 无关）**

`OnnxSharp/Model.cs` `_packedWeights`：key = `(WeightIndex, PackKind)`，11 种 pack（含 `PackNhwcDense` / `PackMatMulNhwc` 等）。**与激活 H/W 无关**，首次打到对应 kernel 时 `GetOrAdd`，**永不淘汰**。

加载时只有原始 ONNX float；跑图过程中 NHWC/AVX 路径再分配整图 packed 副本。这直接解释 **loaded 接近、Δ WS 大**。这是为速度付的副本，不是泄漏。

**4. NHWC 图变换（中）**

AVX2+FMA 上连续卷积走图级 NHWC（1.3.0）。C 没有。激活计划里可能多 layout 槽；pack kind 也可能多一套。关掉会回到 1.2 量级的墙钟，**不能作为省内存手段**。

**5. DET grow-only vs C 的 replace-on-resize（中）**

DET 在 line worker 之前串行，通常 1 个 session，但会 ratchet 到数据集最大 DET 输入。C 换 `(H,W)` 就释放。

**6. 4w 额外 +171 MB（中）**

`PaddleOcrAll` 用 `LineWorkerCount` 并行 CLS/REC，共享一个 REC 池、一个 CLS 池。4 个并发 `RentSession` 可同时持有 4 个已膨胀 session。和 C 的 +157 MB 同量级，**不是 1w 鸿沟**。

### 6.3 次要 / 可忽略

| 项 | 判断 |
| --- | --- |
| `DbPostprocess.Workspace` | grow-only 到 `LimitSideLength²`（960²），通常 1 份、几 MB |
| `PaddleOcrAll` crop 池 | grow-only，cap ≈ lineWorkers；通常 < 几十 MB |
| `ArrayPool` / `PooledArrays` | ≥64 KiB 的 Return 直接丢，防 LOH 桶泄漏；不是 447 MB 主因 |
| 预解码 100 张 BGR | 两引擎共享 |
| JIT / LOH | 次要；解释 sharp peak≈last，解释不了 330 MB |
| `ResizeWorkspace` | 每 session 一份，相对 activation 小 |

---

## 7. 移植评估（正确率、墙钟都不能掉）

C 省 Δ WS，主要不是漏掉的实现细节，而是它愿意用 **重建 session / 更粗的 REC 桶 / 不用 NHWC** 换内存。C 也更慢（1w 2.98×，4w 2.10×）。

两边 REC 宽度策略不同，exact_lines / CER 已经不同（940 / 0.61% vs 926 / 1.09%），里面有 LWM 转换，也有宽度策略。**不要把 C 的五档或 960 帽原样搬过来当“对齐”。**

| C 的做法 | 搬到 C#？ | 原因 |
| --- | --- | --- |
| workspace planner | 已有 | 不是差距来源 |
| packed weights 跨 worker 共享 | 已有（`Model` 级） | 已共享 |
| `MaxPooledSessions = LineWorkerCount` | 可做，**1w 几乎无收益** | 1w 本来 1 个 REC session；4w 增量已经接近 |
| 启动时 eager pack | 只改口径 | peak 不降，Δ WS 挪到 `loaded` |
| CTC 不保留完整 `[T×V]` | 可试，省得不多 | 词表再大也是十几 MB，换不来 330 MB；必须和现 greedy 逐位一致 |
| REC 最多 2 宽、多了就扔 | **会伤速度** | 回切常见宽度要重建 session；grow-only 就是为了躲 reshape |
| DET 换尺寸就拆 session | **会伤速度** | 这 100 张尺寸很散，等于每张 `CreateRequest` + `PlanWorkspace` |
| DB 后处理每图 free | 几乎不值 | 现在池化到 960²，只有几 MB |
| 改成五档 / 加 960 帽 | **会改正确率** | 和当前 adaptive（长行不压）相反 |
| 关 NHWC / 丢 pack 缓存 | **会伤速度** | 1.3 快的来源 |

若以后做 **有界折中**（不是无损）：REC 仍用现在的 32px 宽度和 grow-only，但加 **按宽度的 LRU，只在本张图结束且该宽度冷却后丢**。能做成 C 那种 last < peak，混合宽度的下一批图会重新付 session 创建。那是明确的速度换内存，验收必须同时看 §4.1 的墙钟和 940/1026。

不建议一上来就改的：

- 每次 smaller shape 都 shrink `_workspace`（已量过回归）。
- 每行 `new`/`Dispose` `InferenceSession`。
- 把 `MaxPooledSessions` 设成 1 却保留 4 个 line worker。

---

## 8. 关键文件

### 本库

| 路径 | 看什么 |
| --- | --- |
| `test/Sdcb.SimdPaddleOCR.Tests/Program.cs` | bench 循环、WS 采样、`--summarize` |
| `test/Sdcb.SimdPaddleOCR.Tests/SharpEngine.cs` | sharp 选项（320 + AdaptiveWidth） |
| `test/Sdcb.SimdPaddleOCR.Tests/CEngine.cs` / `CNative/NativeOcr.cs` | c 选项（TargetWidth=960） |
| `test/Sdcb.SimdPaddleOCR.Tests/CAssets.cs` | small 不自动下载 |
| `src/Sdcb.SimdPaddleOCR/OnnxSharp/InferenceSession.cs` | grow-only `_workspace` |
| `src/Sdcb.SimdPaddleOCR/OnnxSharp/Model.cs` | `_packedWeights` lazy 缓存 |
| `src/Sdcb.SimdPaddleOCR/PaddleOcrRecognizer.cs` | 32px adaptive、best-fit 池 |
| `src/Sdcb.SimdPaddleOCR/PaddleOcrDetector.cs` | DET session 池 |
| `src/Sdcb.SimdPaddleOCR/PaddleOcrAll.cs` | line workers、crop 池 |
| `src/Sdcb.SimdPaddleOCR/PPOCRTypes.cs` | `MaxSessionCacheEntries` 已废弃；`MaxPooledSessions` |
| `src/Sdcb.SimdPaddleOCR/DbPostprocess.cs` | grow-only 后处理 scratch |

### lw.PPOCR.C @ `20d0de66`

| 路径 | 看什么 |
| --- | --- |
| `src/runtime/memory.c` | workspace planner |
| `src/runtime/session.c` | packed weights、`share_prepared_constants` |
| `src/runtime/executor.c` | 常量零拷贝、推理零分配 |
| `src/ppocr/recognizer.c` | 五档桶、双 session、greedy CTC |
| `src/ppocr/detector.c` | replace-on-resize |
| `src/ppocr/db_postprocess.c` | 每图 malloc/free |
| `src/ppocr/ocr.c` | worker clone、共享 crop |

---

## 9. 给实现者的验收清单

改完至少在本机串行重跑 §2.2，并填：

| 项 | 改前（§4.1） | 改后（§10，2026-09-17 同机同日 A/B） |
| --- | --- | --- |
| sharp 1w mean ms | 307.8 | 300.9 / 309.0（同日 stash 基线 325.7 / 322.8，**−4~8%**） |
| sharp 4w mean ms | 225.9 | 192.7 / 190.6（同日 stash 基线 207.9 / 210.8，**−7~10%**） |
| sharp exact_lines / CER | 940/1026 / 0.61% | 940/1026 / 0.61%；100 行 texts+hash 逐行相同 |
| sharp 1w Δ WS | +447.3 | **+142**（loaded 453.6 → last 595.3） |
| sharp 4w Δ WS | +617.7 | **+216**（loaded 453.9 → last 670.1） |
| sharp loaded | 476.8 | **453.6**（常量不再按 CompiledModel 复制一份） |
| sharp 1w/4w peak vs last | 几乎相等 | peak 略高于 last（600.3 vs 595.3），workspace 换代时旧块立即归还 |

文本输出（每行字和 `hash`）若动了 REC 宽度或 CTC 融合，必须和改前 JSON **逐行对比**，不能只看 CER。本轮用 `bench-out/tools/cmp_texts.ps1 <new.json> <old.json>` 对 small 1w/4w、tiny 4w、medium 4w 各 100 行做了逐行比对，全部 0 差异。

---

## 10. 已实施（2026-09-17）

不改 REC 宽度策略、不改 CTC 数值、不做 shrink/重建 session。所有改动只影响“活值放在哪”，不影响算什么。

| 改动 | 文件 | 省的是什么 |
| --- | --- | --- |
| **融合幽灵张量不占槽**：Conv+Relu/HardSwish/BiasAdd/ResidualAdd/GELU/LayerNorm 等融合组里，只有 sink 真被写；中间张量（`PhantomSink`）绑定到 sink 窗口，不再单独分配 | `CompiledModel.MarkPhantoms`、`InferenceSession.PlanWorkspace` | 旧 planner 给每个融合中间量都留了整块（DET 960² 一个 Conv 输出就是 21 MB） |
| **planner 改为离线区间贪心（greedy-by-size）**，替换按节点顺序的 first-fit | `InferenceSession.PlanWorkspace` | first-fit 比活值下界多 35–40%（DET 108 vs 79 MB） |
| **别名组**：NCHW 通道 Concat 输入直接写进输出分片（`Concat` 跳过拷贝）；GELU/HardSwish 原地；独立 Add/Sub/Mul/Div 复用同形状、当节点死亡的操作数 | 同上 + `Concat` | DET stem 的 Concat 三份 → 一份；FPN 的 Add 少一份 |
| **REC 只规划到 CTC 投影之前**（`PlanForCtcProjection`）：`TryArgMax` 路径根本不写 `[T×18710]` 的 logits/Add/Softmax 三张平面 | `InferenceSession.Reshape/EnsureFullPlan`、`PaddleOcrRecognizer.RentSession` | REC@1600 35 MB → 14 MB；若真走全图 `Run`，先透明补全规划 |
| **workspace 改为非托管内存**（`NativeWorkspace`，64B 对齐，尾部 64 float 护栏） | `NativeWorkspace.cs`、`TensorValue` | grow-only 换代时旧 `float[]` 是 LOH 垃圾，dump 里 112–120 MB “Free” 常驻；`AlignedFree` 立即还给 OS，last 才能低于 peak |
| **DB 后处理 contour/hull 按需增长**，不再预分配 `pixels` / `2×pixels` 个 Point | `DbPostprocess.Workspace.AddBoundary` | 22 MB Point[]；上限仍是 pixels，语义不变 |
| **常量不再按 CompiledModel 复制成 float[]**，`TensorMeta.Constant` 直接引用 `Model` 的 byte[] | `Model.GetTensorArray`、`CompiledModel.TensorMeta`、`TensorValue` | 只影响 loaded：small −26 MB，medium −127 MB |

Δ WS 现在的构成（small 1w，约 142 MB）：DET workspace ≈ 72 MB（960² 一次到位、grow-only）、REC ≈ 14 MB、CLS 很小；其余是 lazy packed weights（速度换的副本）、DB 位图、GC gen0 预算。DET 的 72 MB 已接近该图的活值下界（stem `Concat` 邻接 + FPN 三张 5 MB 平面），再降只能动 NHWC Concat 的分片写入（kernel 级改动）。

其他模型同口径（texts 逐行相同）：tiny 4w Δ WS 383 → 114；medium 4w loaded 827 → 700、Δ WS 1169 → 514（medium DET stem 的 Concat 是 NHWC，无法别名，202 MB 即活值下界）。

诊断手段（已留在代码里）：`PPOCR_DUMP_PLAN=1` 逐次 reshape 打印 bump / 活值下界 / 最大张量；harness 的 `PPOCR_BENCH_HOLD=1` 跑完后挂起等回车，方便 `dotnet-dump collect`；JSON `meta` 新增 `gc_*` 字段与 `working_set_mb_after_gc`。
