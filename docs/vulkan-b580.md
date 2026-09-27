# SimdPaddleOCR — Ryzen 9 5950X CPU vs Intel Arc B580 Vulkan

实测机:Windows Server,Ryzen 9 5950X(16C/32T),Intel Arc B580 12GB(ReBAR on),SDK .NET 10.0.401,`test/Sdcb.SimdPaddleOCR.Tests` bench(`--workers 4 --benchmark-kind simd --engine sharp|vulkan`,warmup=1,n=99,dataset/ 100 张固定图、张张不同尺寸)。

**结论(第二轮,正确性+内存修复后):small/medium GPU 净胜 CPU ~18-20% 且正确率持平;tiny 基本持平略负(rec 在 GPU 上的 fp16 噪声翻转 + 单行宽批反而失去并行度)。内存已有界(plan LRU=16 + 共享 grow-only buffer)。**

## 端到端(4 workers,median ms/图,越低越好)

| 模型 | CPU(sharp) | Vulkan | 比值 | CPU img/s | GPU img/s |
|---|---:|---:|---:|---:|---:|
| tiny    | **71.0**  | 80.4   | 0.88× | 13.59 | 12.01 |
| small   | 226.2 | **180.4** | 1.25× | 4.39  | 5.45  |
| medium  | 557.1 | **458.5** | 1.21× | 1.81  | 2.11  |

注:CPU 基线列与 GPU 同轮重测;dataset 的 100 张图尺寸几乎两两不同,是最不利 GPU 的变 shape 场景。

## 分阶段耗时(mean ms/图)

| 模型 | 阶段 | CPU | Vulkan | 说明 |
|---|---|---:|---:|---|
| tiny | det_graph | 31.0 | 20.0 | GPU det 全 shape IoU≥0.9996 |
| tiny | cls_graph | ~20 | ~4 | cls 固定 shape 全图单批 |
| tiny | rec_graph | 81.1 | 35.9 | tiny_rec 在 GPU 上真跑(CRNN 结构全覆盖) |
| small | det_graph | 98.2 | 34.9 | 2.8× |
| small | cls_graph | ~30 | ~3.9 | ~7× |
| small | rec_graph | 353.1 | 344.4 | SVTR 尾部(5-D Transpose/Slice/ReduceMean[-1]/MaxPool n>1)未 emit → 整段回落 CPU |
| medium | det_graph | 232.4 | 82.6 | 2.8× |
| medium | cls_graph | 23.0 | 3.8 | 6× |
| medium | rec_graph | 1111.7 | 1016.2 | 同上,回落后并行度由 GpuAlive 反馈恢复(lines_wall 902→384ms) |

GPU 收益集中在 det(2.8×)+ cls(6×);rec 仅 tiny 真走 GPU。SVTR rec 全量 emit 是下一个里程碑。

## 正确率

| 模型 | 指标 | CPU | Vulkan |
|---|---|---:|---:|
| tiny | exact_lines | 742/1036 | 678/1036 |
| tiny | CER | 2.37% | 2.91% |
| small | exact_lines | 950/1036 | 948/1036 |
| small | CER | 0.41% | 0.41% |
| medium | exact_lines | 1004/1036 | 1001/1036 |
| medium | CER | 0.14% | 0.15% |
| 全模型 | cls | 1022-1035 | 一致 |

small/medium 的 rec 回落 CPU 后输出与纯 CPU 逐行一致,残余 ~0.3% 差距全部来自 det fp16 概率图阈值化引起的框微移(合并/丢低置信行)。tiny 的 rec 真在 GPU 上跑,差异是 fp16 特征噪声在低置信行上的 argmax 近平局翻转(--reciso 实测 1/11 行差 1 字符,均在 e≤10 的噪声行)。

## 内存(本轮修复目标)

| 模型 | CPU WS peak | Vulkan WS peak(修复前→修复后) |
|---|---:|---:|
| tiny | 521MB | 3897MB → 964MB |
| small | 678MB | 5781MB → 1036MB |
| medium | 1218MB | **24133MB(无界爬升) → 1457MB(稳定)** |

修复内容(commit 512bcd2):arena/inF32/outF32 改为 graph 级共享 grow-only buffer(扩容即失效全部 plan 重绑);plan 缓存 LRU,evict 时真正释放 cmd buffer/descriptor set/query pool(VkDevice 补 vkFreeCommandBuffers/vkFreeDescriptorSets,desc pool 开 FREE_DESCRIPTOR_SET_BIT);VkBuffer.Free() 落地。

**~32-shape 崩坏验证(应 3080Ti reviewer 要求):** 临时 `MaxPlans`→64 跑全 100 图(每张都是不同 det shape,plan 数必然越过旧崩坏点):零 mass-corruption,与 CPU 行数差只剩 img-014 一例(见下)。→ root cause 确证是资源耗尽/陈旧绑定,已被真释放+invalidate 杀死,不是被 LRU 上限掩盖。WS 峰值 981MB 仍有界(plan 共享 arena,不独占显存)。随后把上限定为 64:shape 剧烈变化的负载下重建更少(实测 12.0→14.0 img/s)。

## 已修的正确性问题

- **det 残差 Add 吸收顺序**:relu(conv+res) vs relu(conv)+res —— `act==0` 门控修复,det 全 shape IoU 0.9996+。
- **回落路径输入布局**:模型图输入被 TensorNhwc 标记而 GpuSession 报 InputIsNhwc=false → 回落 CPU 时 NCHW 被当 NHWC 读 → 确定性乱码。`CpuInput()` 转置修复,--reciso 对拍 diff=0/11。
- **回落后的批形状**:`GpuAlive` 反馈让 recognizer 在首个 rec plan 失败后切回 CPU 风格分组,避免单批串行+最大宽 padding 双重损失。

## 已知缺口(按收益排序)

1. **SVTR rec 的 GPU emit**(small/medium 主线机会):需覆盖 MaxPool batch、rank-3 ReduceMean、5-D Transpose、Slice、Concat(axis=0) 等 ~10 种算子;工程量大,单独立项。
2. tiny 的 GPU rec 精度:fp16 特征噪声在低置信行翻转 argmax —— 要么 fp32 段,要么接受(det 同款抖动已证实无害)。
3. **det fp16 边界丢框(img-014,已定位):** CPU 16 框 / GPU 15 框 —— GPU 把右侧两个竖排条带合并。根因:单个桥接像素 (918,71) 概率 cpu=0.1892 vs gpu=0.2020 跨过 0.2 bitmap 阈值;全图 400K 像素仅 15 个阈值翻转且全部落在 [0.189,0.205] 窄带,输出 map maxAbs=0.029 即 ~50 节点 fp16 存储的累计噪声(最大的是 FPN concat t399 elemMax=0.41,张量量程 ±160 时 fp16 ulp≈0.125)。**非 kernel bug**:同一输入用 GDI+ 解码跑纯 CPU 也合并(15 行),本来就是临界输入;唯一实质性修法是 fp32 arena,代价是带宽翻倍。暂不修,记为已知限制。
4. per-dispatch ~55µs 固定开销 × 78 dispatch ≈ 4.3ms/Run 下限 —— 继续融合或 push-descriptor 直录可压。
5. `_dev.Sync` 全局锁串行 det/rec —— 双流 overlap 未做。

## 目标框架:GPU 仅 net10.0,netstandard2.0 不走 Vulkan

`Backends/Vulkan/**` 在 ns2.0 下整目录 `Compile Remove`,`OcrSessionFactory` 在该 TFM 上恒返回 CPU session。**这不是能力限制而是刻意的维护决策**——实测过一遍 API 差距,全部可移植但没有一处是免费的:

| ns2.0 缺的 API | 用量 | 移植代价 |
|---|---|---|
| `LibraryImport`(net7+ source-gen P/Invoke) | 59 个入口 | 退回 `DllImport` 或自写委托加载——失去 source-gen marshal 的可维护性收益,这是有意保留的现代写法 |
| `NativeLibrary.SetDllImportResolver`(net5+) | 1 处 | 用于定位 vulkan-1.dll/libvulkan.so.1;ns2.0 只能 kernel32 `LoadLibrary`/libdl `dlopen` 手写 loader,又要一套平台分叉 |
| `System.Half`(net5+) | ~40 处 | fp16 权重转换;可换成 ushort+位运算 helper,但多一条自编码路径要维护 |
| `BitConverter.SingleToUInt32Bits`(ns2.1+) | 1 处 | 一行 unsafe 转换,小事 |
| `MathF`/`Span`/`MemoryMarshal`/`ArrayPool`/`stackalloc`/`delegate*` | 多处 | 已有 BCL 包(`System.Memory`/`Unsafe`/`Microsoft.Bcl.Numerics`)或语言特性,均可直接用 |

结论:移植是纯体力活(~半天),但会在 P/Invoke 层和 loader 层各长出第二套实现。ns2.0 的定位本来就是 best-effort 兼容旧消费方,而 GPU 场景的用户天然在 modern .NET 上;为不让两份 Vulkan 互操作代码同步腐烂,决定 **ns2.0 只留 CPU**。若未来真有需求(如 .NET Framework 应用要吃 GPU),再按上表做一次性移植即可。
