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

修复内容(commit 512bcd2):arena/inF32/outF32 改为 graph 级共享 grow-only buffer(扩容即失效全部 plan 重绑);plan 缓存 LRU=16,evict 时真正释放 cmd buffer/descriptor set/query pool(VkDevice 补 vkFreeCommandBuffers/vkFreeDescriptorSets,desc pool 开 FREE_DESCRIPTOR_SET_BIT);VkBuffer.Free() 落地。3080Ti 侧报告的"~32 个不同 det shape 后全局错乱"在此上限下不可复现(100 图 bench 无 mass-corruption 特征)。

## 已修的正确性问题

- **det 残差 Add 吸收顺序**:relu(conv+res) vs relu(conv)+res —— `act==0` 门控修复,det 全 shape IoU 0.9996+。
- **回落路径输入布局**:模型图输入被 TensorNhwc 标记而 GpuSession 报 InputIsNhwc=false → 回落 CPU 时 NCHW 被当 NHWC 读 → 确定性乱码。`CpuInput()` 转置修复,--reciso 对拍 diff=0/11。
- **回落后的批形状**:`GpuAlive` 反馈让 recognizer 在首个 rec plan 失败后切回 CPU 风格分组,避免单批串行+最大宽 padding 双重损失。

## 已知缺口(按收益排序)

1. **SVTR rec 的 GPU emit**(small/medium 主线机会):需覆盖 MaxPool batch、rank-3 ReduceMean、5-D Transpose、Slice、Concat(axis=0) 等 ~10 种算子;工程量大,单独立项。
2. tiny 的 GPU rec 精度:fp16 特征噪声在低置信行翻转 argmax —— 要么 fp32 段,要么接受(det 同款抖动已证实无害)。
3. per-dispatch ~55µs 固定开销 × 78 dispatch ≈ 4.3ms/Run 下限 —— 继续融合或 push-descriptor 直录可压。
4. `_dev.Sync` 全局锁串行 det/rec —— 双流 overlap 未做。
