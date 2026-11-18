# 官方 PP-OCRv5 识别图：水有多深

Simd 对外承诺的是 **PP-OCRv6**。识别热路径、模型包、CI 和合成数据集都钉在 `ChineseV6Tiny / Small / Medium` 上。

「顺手支持 V5 rec」看起来便宜：官方中英 mobile 包能下载，图上没有未知算子，CTC 字典编号也对得上。实测结论是另一回事——**能加载，不能推理**；而且就算把形状推断补上，**现有 harness 也接不住这张图**。没有回归网，这条支持会变成每次改 `ReshapeShape` 都要人肉赌 V6 没被带崩。

本文记录 2026-09-18 对官方 paddlex `paddle3.0.0` ONNX 的拆图，以及「要支持的话该怎么改、有哪些坑」。**不是实施清单。** 先把测试网补上，再谈改核心。

## 先把「V5 rec」拆开

至少有两条完全不同的导出路径，不能互相当通行证。

| 来源                                                                                | 图                                                                                                             | 现在 Simd 上的状态                             |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| 官方 `PP-OCRv5_mobile_rec` / `en_PP-OCRv5_mobile_rec`（paddlex `*_onnx_infer.tar`） | **opset 7**，Squeeze / Unsqueeze 的 axes 在属性里；颈段 flatten 和尾巴 reshape 是动态 `Shape → Slice → Concat` | `Model.Load` 成功；`Recognize` 在形状推断里崩  |
| 公式微调（PP-OCRv5_mobile_rec 再训）                                                | **opset 13**，Squeeze axes 走第二个输入张量                                                                    | 加载期把常量 axes 折进属性后能跑；那是另一张图 |

官方包地址仍在同一套 PaddleX 前缀下：

`https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/`

| 包                                      |                                          体积（约） |
| --------------------------------------- | --------------------------------------------------: |
| `PP-OCRv5_mobile_rec_onnx_infer.tar`    |                                             16.7 MB |
| `en_PP-OCRv5_mobile_rec_onnx_infer.tar` |                                              7.9 MB |
| `PP-OCRv5_server_rec_onnx_infer.tar`    | 84.7 MB（本文没跑，图更大，不能默认和 mobile 同构） |

预处理和字典不是障碍。`inference.yml` 仍是 BGR、`RecResizeImg` `[3,48,320]`、`(x/255-0.5)/0.5`、CTC。中文 `character_dict` 18383 行（含全角空格），英文 436 行；Simd 的 `ClassCount = labels + 2` 分别等于输出最后一维 **18385 / 438**。崩在形状，不在字典。

## 实测：加载过了，第一枪就挂

中英 mobile 是 **同构 1019 节点**。DCE 掉 `Constant` / `Identity` / 非常量 shape 链之后，执行计划大约 **413 节点，`Unknown = 0`**。

随后 `Recognize`（`AdaptiveWidth=true`，官方 demo `general_ocr_rec_001.png`）在 `CompiledModel.ResolveShapesFor` → `TransposeShape` 越界。补了一句诊断后，失败是：

```text
Transpose perm axis 2 exceeds rank 1 [6240]
```

`6240 = 120 × 52`。颈段特征是 `[1,120,1,52]`，被收成了 rank-1。下一手 `Transpose perm=[0,2,1]` 要的是 rank-3 `[N,C,W]`。

英文包还没跑到 `Recognize`——同构图，会在同一处挂。没必要用第二张图再证明一次形状推断坏了。

## 图上到底缺什么

缺的不是算子。V6 rec 已经为 SVTR 注意力付过一次钱：通用 `Transpose` / `Slice`（rank ≤ 8）、带 batch 的 `MatMul`、任意轴 `Softmax`、LayerNorm 那条 `ReduceMean` 链。LayoutPlanner 只把 rank-4 卷积收进 NHWC；后面的注意力本就走通用路径。

官方 V5 只是 **PaddleX 把 shape 链导得更「动态」，value_info 更空**。执行计划里不再有整数 `Shape`，`Reshape` 只剩一个欠定模板。`ReshapeShape` 必须在不跑整数张量的前提下猜回真实形状。

现在能看清的有三处。

### 1. 颈段 flatten（已经撞上）

图上是：

```text
Shape(swish) → Slice[:2] → Concat([-1]) → Reshape → Transpose[0,2,1]
```

语义是 `[N,C,H,W] → [N,C,-1]`。H 已经被骨干收成 1，所以是 `[1,120,1,52] → [1,120,52]`。

`Model.Load` 会丢掉非常量 shape 输入，只留 `value_info` 模板。官方这张图的模板几乎是空的，退化成 rank-1 `[-1]`。现有启发式里：

- `[N,C,1,1]` + rank-1 模板 → `[N,C]`（给 PP-LCNet 方向分类，必须保住）
- `[N,C,1,W] → [N,C,W]` 写在 `TryInferReshape` 的注释里，但 **只在输出模板本身是 rank-3 时生效**
- rank-1 `[-1]` + 四维输入 → 把四个轴全乘进去，得到 `[6240]`

V6 rec 的 flatten 能过，是因为导出还留了够用的模板。官方 V5 把模板也拿掉了。同一条启发式，两张图答案相反。

### 2. 中间 5D QKV（大概率不是新算子）

QKV 打包是常量：

```text
Reshape [0, -1, 3, 8, 15] → Transpose[2, 0, 3, 1, 4]
  → Slice 出 Q / K / V → Squeeze → 标准注意力 MatMul / Softmax
```

`linear_0.w` 是 `(120, 360)`，`360 = 3 × 8 × 15`。`[0,-1,3,8,15]` 是 `Constant` 节点，load 时应当折成 initializer，走已有的「常量 shape 输入」分支。前面 rank 对了，这里多半不用新算子。两块注意力重复同一套路。

V6 rec 的注意力是 4 维（`[N,L,8,15] → [N,L,120]`，再插单轴）。V5 官方包先排成 5 维再 `Slice`。内核吃得下 rank-5；**前提是 flatten 没把前面收错**。

### 3. 尾巴再一次动态 stack

接近输出还有：

```text
四个 Shape 片 → Unsqueeze → Concat → Reshape → Transpose[0,3,1,2]
```

又是非常量 shape 链，又会被 DCE。补完第 1 处之后，十有八九会在这里再撞一次：模板不够，`TryInferReshape` 再猜错，后面的 4 轴 `Transpose` 再越界，或者更糟——形状「能算」，字是错的。

最后 `Squeeze(axes=2) → Transpose[0,2,1] → MatMul → Softmax` 是常规 CTC 头，不是新问题。

## 估计要怎么改

如果只谈核心，水到膝盖，不是没顶。方向也清楚：**继续当通用 opset 形状恢复，不要开 V5 开关，不要加 per-session context。**

建议的改法，按顺序：

1. **收窄 `ReshapeShape`，不要看下一手算子也能写对的那种。**  
   rank-1 符号模板 + `[N,C,1,W]` 且 `W>1` → `[N,C,W]`。  
   `[N,C,1,1]` 继续走现在的 `[N,C]`。  
   这两条必须并排锁死，不能互相覆盖。

2. **常量 5D reshape 保持现状。**  
   第一枪过了再看 `Transpose[2,0,3,1,4]` 的入边是不是 `[N,L,3,8,15]`。不是再查 Identity 有没有把 `full_int_array` 解丢。

3. **尾巴那次 stack。**  
   优先看 `value_info` 够不够喂给现有 `TryInferReshape`；不够再加一条「3 维 → 4 维、单轴插在固定位置」的窄规则。V6 已经有一条 `[N,L,C] → [N,1,L,C]`，不要写第二条抢同一输入的规则。

4. **失败要响。**  
   `Transpose` perm 轴超出 rank，应抛带节点名和形状的 `InvalidDataException`，不要 `IndexOutOfRange`。这和 Squeeze axes 折失败就响是同一条原则。成功路径上的静默回退（把 4 维收成 `[6240]` 再碰运气）比崩掉更难查。

5. **中英 mobile 只算一张图。**  
   server rec、多语言、第三方微调各自是回归项，不是「再跑一遍英文 demo」就能覆盖的。

不该做的：

- 不要 `if (v5)`。公式微调已经证明「V5」不是一张图。
- 不要为了这张图加运行时属性或 session 上下文。load 期折常量、推断期用模板，是现在这套解释器的边界。
- 不要为 5 维注意力手写 NHWC。卷积骨干已经在 NHWC 里；QKV 张量小，走通用 `StridedCopy` / batched `MatMul` 即可。
- 不要假设公式微调图能代替官方包。那张图过了，只说明 opset 13 的 Squeeze 折对了。

工作量本身不大：形状推断一两处，加上 fail-loud，再加 V6 4w 回归。带上官方 demo 人工看一眼，大概小半天到一天。**贵的是后面每次有人动 `ReshapeShape`。**

## 坑

### 和 V6 抢形状

这是唯一真正伤热路径的坑。`ReshapeShape` 是全局的。官方 V5 要的 `[N,C,1,W] → [N,C,W]`，和方向分类的 `[N,C,1,1] → [N,C]`、以及 V6 rec 的「不要把单轴插错位置」，共用同一个函数。

猜错的失败模式有两种，第二种更危险：

- 形状非法，`Transpose` / `MatMul` 立刻炸（现在这样）。
- 形状合法但轴序错，attention residual 广播成 `width × width`，输出是能解码的错字。V6 注释里已经写过后一种。没有逐行对照，CI 的 exact match 掉几个点，看起来会像「数据噪声」。

### 导出路径会分叉

PaddleX 换一次导出，opset、axes 在属性还是输入、`value_info` 留不留，都会变。我们已经看见三种：

- 官方 V5 mobile：opset 7，axes 在属性，模板空；
- 公式微调：opset 13，axes 在输入；
- V6 rec：同样动态链，但模板够 `TryInferReshape` 用。

「支持 V5」若写成一句产品话，用户会拿任意 `PP-OCRv5_*_rec*.onnx` 过来。细调、旧 `paddle2onnx`、自己导的动态轴，都不会先打招呼。

### 静默折错比响更麻烦

PR #13 那类「折不了就当没这根输入」会把 Squeeze 收成「去掉所有 1 轴」，下一手 Conv 再变成难追的 rank 错。官方 V5 的 flatten 是镜像问题：模板不够时，现在的回退是「整段压成一维」。看起来像成功推断，其实已经错了。

### server / 多语言 / 微调不是免费附赠

mobile 中英同构，修一次两个都能加载。`PP-OCRv5_server_rec` 没拆过，不能默认同一套启发式够用。多语言 rec 字典不同、图未必同构。第三方微调更是一张新图——公式模型已经演示过这一点。

### 预处理会冒充「模型坏了」

官方 yml 写死宽 320。Simd 默认 `AdaptiveWidth=true`。公式数据集上，ImageSharp JPEG 和 OpenCV 在固定 320 时就能差出几张 exact match；自适应宽度反而把那几张拉回来。没有「官方预处理 + 固定 320 + 自适应宽」三列对照，准确率争议没法判是图的问题还是 JPEG 解码的问题。

### 没有第二套对照

仓库里没有 V5 的 C 对照、没有 V5 模型包、没有 V5 的 `exact_lines` 基线。改完之后，没有第二套引擎能说「这张图就该是这串字」。

## 真正麻烦的是 harness，不是那两行启发式

当前网只认识 V6。

| 已有                | 覆盖                                                                       |
| ------------------- | -------------------------------------------------------------------------- |
| `test.yml` unittest | 内核、NHWC、像素格式、契约。**没有 `ReshapeShape` 用例**，也没有官方 V5 图 |
| smoke / bench       | `--model tiny\|small\|medium`，全部是 `ChineseV6*` 整管线                  |
| 合成 `dataset/`     | 为 V6 DET+REC 生成                                                         |
| C 引擎对照          | 同一套 V6 资源                                                             |
| 模型 NuGet          | 只有中文 V6 + 方向 CLS                                                     |

所以：核心里加上 `[N,C,1,W] → [N,C,W]`，本地对一张公寓招牌出字，**并不能**保证下周改 V6 注意力启发式时官方 V5 还活着，也不能保证这条新规则没把 tiny/small/medium 的 exact match 啃掉。没有人会在 PR 里跑那两个 16 MB / 8 MB 的 tar——它们不在树里，CI 也不拉。

公式模型那次是人肉：本机路径、临时字典、一批标注图、另写一份 Python CTC。这种活只能做一次，不能当回归。

若没有下面这张网就宣称支持官方 V5 rec，之后每次动 `OnnxSharp` 形状推断都要靠记忆。漏测的代价是用户侧「升级后中文 V5 能 load、字全错」或「V6 CER 无声涨了 0.3%」。

## 若要支持，harness 最小集

核心改动之前，至少要有这些，否则不要把 V5 写进 README。

1. **形状推断单测，不拉大模型。**  
   锁死：  
   - `[1,120,1,52]` + rank-1 `[-1]` → `[1,120,52]`  
   - `[1,C,1,1]` + rank-1 `[-1]` → `[1,C]`（V6 / CLS）  
   - V6 rec 现有的 `[N,L,C] → [N,1,L,C]`  
   这三条是防回归的钉子。没有它们，启发式就是注释。

2. **官方 mobile 中英 rec 作为测试资产。**  
   不必做成 NuGet。CI 缓存 tar、从 `inference.yml` 抽字典即可。没有包，就没有稳定的 `ClassCount` 和加载契约。

3. **只测 rec 的 smoke，和整管线分开。**  
   官方 V5 这次只谈识别图。用固定作物、固定宽 320 和 `AdaptiveWidth` 两列跑：  
   - 官方 demo（中文那张公寓招牌）  
   - 一张干净的英文行  
   断言「能跑完」和「gold 字符串」。gold 必须来自同一预处理下的对照实现（PaddleX / ONNX Runtime），不要用「看起来像」当期望。

4. **V6 4w 基线不能动。**  
   现有 tiny / small / medium 的 exact match 和 CER 是准入门槛。V5 形状规则若让 V6 掉一行，就回滚。

5. **明确不覆盖。**  
   server rec、其它语言、第三方微调，在有专属资产之前都写「未支持」。公式图继续当 opset 13 Squeeze 的回归，不当官方 V5 的代理。

没有第 1 条，改了也会在下一次 `ReshapeShape` 调整里烂掉。没有第 3 条，只能证明「这台机器上这张 PNG 出过字」。没有第 4 条，就是用 V5 的名义去赌 V6。

## 现在的结论

- 官方 V5 中英 mobile rec **不是**「折一下 Squeeze 就能跑」的图；卡在动态 flatten 的形状恢复。
- 改核心不深，也用不到新算子或 V5 开关。
- **不深不等于该做。** 缺的是模型资产、rec-only gold、形状单测和继续钉住 V6 的网。
- 在这张网落地之前，官方 V5 rec 保持现状：调用方可自行 `Model.Load`，但仓库不宣称支持，也不往 `ReshapeShape` 里堆没有对测的启发式。
