> **历史归档（v2.0 实测）**：本报告测于 v2.0 时期，因当前暂不方便复测该 UHD 770 机器，作为无协作矩阵 Intel 核显的历史参考保留。

# SimdPaddleOCR — Intel UHD Graphics 770（无协作矩阵）Vulkan

实测机：Intel UHD Graphics 770（Xe-LP，32 EU，核显最大动态频率 1.55 GHz），与 CPU 共享内存。驱动 101.7079（Vulkan API 1.4.323）。Windows 11 26200，电源计划「高性能」。.NET SDK 11.0.100-rc.1 编译 `net10.0`。`test/Sdcb.SimdPaddleOCR.Tests`，`--workers 4 --benchmark-kind simd --warmup 1`，同一 `dataset/` 100 张，墙钟 n=99。本表 `2ee12a1`。

**结论：没有 16×16×16 fp16 协作矩阵。三档端到端慢于同机 sharp（约 1.8×–2.0×）。所以 `Auto` 在这种设备上走 CPU，只有显式指定 `OcrBackend.Vulkan`（或 `Auto` + `SIMD_OCR_BACKEND=vulkan`）才走这条 GPU 路径。两种情况都不编译协作矩阵 shader（Intel 编译器会把进程直接打掉）。sg16 / sg32 / sg32l 的 shader、spv 和路由没改；有 coopmat 的设备按它实际要用的形状判定（sg32 要 16×16×16，sg16 要 8×16×16），B580 仍走 sg16。**

## 设备能力

| 项 | 值 |
|---|---|
| 设备 | Intel(R) UHD Graphics 770，`vendor=0x8086`，`INTEGRATED_GPU`。枚举里没有独显 |
| 队列 | family 0：`flags=0xf` count=1（图形+计算，被选中）；family 1：`flags=0x20` count=2（视频解码） |
| subgroup | 默认 32，`sgRange` 8–32，`requiredSubgroupSizeStages` 含 compute，可指定 8 / 16 / 32 |
| cooperative matrix | 无 `VK_KHR_cooperative_matrix` |
| fp16 | `shaderFloat16` 和 16-bit storage 都有。加速的整数点积只有 int8 / DP4A，16-bit 点积没有 |
| 共享内存 | `maxComputeSharedMemorySize` = **32 KB**。`maxComputeWorkGroupInvocations` = 1024 |
| 内存 | 一个 heap，65343 MB，`DEVICE_LOCAL`。三种类型全带 device-local：`0x1`、`0x7`（再加 host-visible + coherent）、`0xf`（再加 host-cached）。没有「非 device-local 的主机堆」 |
| push descriptor | 有。`globalPriority`=512（HIGH） |
| 峰值 | fp32 约 0.8 TFLOPS（32 EU × 16 flop/clk × 1.55 GHz），fp16 打包约 1.6 TFLOPS。GEMM 必须 fp32 累加，所以实际上限按 0.8 算 |

## 走了哪一档

`SubgroupMin` 是 8，进不了 sg32。也没有 coopmat 扩展（`CoopGemm` 为假），所以不建任何 `conv1x1_cm*` / `convk_cm*` 管线。大 GEMM（1×1 里没被 dot 接住的、im2col 之后的 GEMM、窄 M 接不住的 MatMul）走 `gemm_nc`：fp16 加载、fp32 按 K 正序累加、fp16 写回，绑定和 16 字节 push constant 与 `conv1x1_cm` 相同，不用 subgroup 内置量。

当前 tile：工作组 64 线程，输出块 64×64，每线程 8×8 个 fp32 累加器，K 方向 16。A / B 块在写入共享内存时就转成 fp32，并按 k-major 存放，内层每步是一次 A 广播读、一次 B 连续读，没有 bank 冲突，也不再在每次 FMA 前做 fp16→fp32 转换。下一块的全局读在本块 FMA 之前发出（寄存器预取）。

两个要点，都是 Intel 编译器（IGC）的脾气，不是算法问题：

- **管线锁定 8 lane**（`requiredSubgroupSize = 8`，仅当设备允许在 compute 上指定且范围含 8）。默认让驱动挑时它选 SIMD16，64 个累加器放不下寄存器，同一个内核慢 4 倍以上。32×64 / 4×4 内核锁 8 lane 时为 15.1 ms（5760×1024×512）。
- **尾部拆成两份二进制**。64 个累加器各自展开 gelu / sigmoid 会让编译器把主循环挤进溢出，慢 5–20 倍，而且只删掉其中一种激活反而更糟。`gemm_nc` 的尾部把每行 8 个累加值暂存到已空闲的共享内存，再用不展开的循环算激活，支持全部激活和双输出。`gemm_nc_s`（`-DSIMPLE`）只做无激活 / relu / hardswish 且无双输出，全展开。host 按 flags 选（`NcTail`）。两份放进同一个二进制时，IGC 又会把主循环打成溢出，所以只能分开编。

路由上的两处门槛（只对无矩阵档生效）：

- 1×1 卷积 K≤128 原本走 `conv1x1_dot`。现在 M×cout > 2¹⁶ 且不是 SE prescale、输入不是 addps 吸收源时改走 `gemm_nc`。dot 在大 M 上只有约 150 GFLOPS。
- im2col 之后 Kp≤128 的 `im2c_dot` 用同一个门槛。

`PackSlabs` 只在这条档上额外打开，`_sg32` 的条件没动。kxk 仍是 `convk_dot` / im2col，门槛没改（见下文放弃项）。

能力判定是 `VkDevice.CoopGemm`：设备会选的那套 cm shader 所需的 fp16→fp32 subgroup 形状是否在 coopmat 列表里。sg32（`SubgroupMin > 16`）要 16×16×16；sg16 的 `conv1x1_cm` 是 A 8×16、B 16×16、累加 8×16，要 8×16×16。B580 只报 8×16×16，所以不能只认 16×16×16，否则它会从 sg16 cm 掉进这条档（本机实测 medium 56 → 775 ms）。

选路在 `GpuBackend.UsesGpu`：`CoopGemm` 为假时，只有显式选了 Vulkan 才建 GPU 会话，`Auto` 直接给 CPU 会话。`OcrSessionFactory.IsGpuBackend` 用同一个谓词，所以 `Auto` 在这里也按 CPU 的方式分批（之前它只看"设备探测成功"，会给 CPU 会话配上 GPU 的 16 行识别批）。`GpuGraphModel` 只看能力：`CoopGemm` 为假就建 `gemm_nc` / `gemm_nc_s`，不碰任何 cm 管线。

## 端到端（4 workers，median ms/图）

同一时段交替测试 3 轮：Vulkan（`--engine vulkan`）与 sharp。

| 模型 | sharp（3 轮） | Vulkan（3 轮） | Vulkan / sharp |
|---|---|---|---:|
| tiny | 50.2 / 44.2 / 43.2 | 83.6 / 83.3 / 84.9 | 1.89× |
| small | 139.0 / 130.5 / 136.1 | 244.4 / 264.7 / 249.5 | 1.83× |
| medium | 537.2 / 624.0 / 656.7 | 1104.2 / 1061.8 / 1096.7 | 1.76×（首轮 2.04×） |

精度：Vulkan 为 tiny exact_lines 740 / CER 2.37%，small 950 / 0.40%，medium 1006 / 0.14%。CPU 是 742 / 950 / 1004。

逐图对拍（`cmp.ps1` 的 `hash` / `detected` / `texts`）：

- Vulkan 对 sharp：tiny 91/100，small 90/100，medium 98/100。差异都是 fp16 噪声下个别字的识别变化，精度表基本不变。
- Vulkan 轮次之间 100/100。`SIMD_OCR_VK_NOPUSH=1` 对默认三档 100/100。
- `--conc` 4 线程和 8 线程（tiny、medium，24 张 × 3）`mismatches=0`，stderr 没有 `fallback`。

分阶段 mean ms/图（第 2 轮；CPU 的 `rec_graph` 是各线程累加，`lines_wall` 才是墙钟）：

| 模型 | 阶段 | Vulkan | sharp |
|---|---|---:|---:|
| tiny | det_graph | 31.3 | 20.2 |
| tiny | lines_wall | 46.2 | 20.3 |
| small | det_graph | 82.1 | 52.7 |
| small | lines_wall | 176.4 | 70.4 |
| medium | det_graph | 374.5 | 232.8 |
| medium | lines_wall | 664.3 | 385.9 |

tiny 的检测已经和 CPU 持平，差距全在识别。识别在 CPU 上按 4 个 worker 重叠，这块核显只有一条计算队列，`lines_wall` ≈ `rec_graph`。即使纯 GPU 的 REC 再快一倍，medium 的 `lines_wall` 也只是接近 CPU。

## 纯 GPU

`gprof.ps1`，次数取最小（ms）：

| 用例 | 耗时 (ms) |
|---|---:|
| medium DET 960×960 | 537.9 |
| medium DET 640×960 | 360.0 |
| medium REC 8×480 | 328.1 |
| medium REC 1×320 | 33.0 |
| small DET 960×960 | 102.5 |
| small REC 8×480 | 79.9 |
| tiny DET 960×960 | 41.5 |
| tiny REC 8×480 | 18.4 |

热点分布：medium DET 是 `gemm_nc_s` 205、`convk_dot` 86、`gemm_nc` 75、`conv1x1_dot` 48、`conv_dw4` 45、`im2col` 35。medium REC 是 `gemm_nc_s` 144 + `gemm_nc` 135（合计约 83%），其余是 `conv_dw4t` 22、`convk_dot` 19。small DET 的第一名是 `convk_dot` 28。

相对峰值：GEMM 在 5760×1024×512 上 12.7 ms，约 474 GFLOPS（fp32 峰值的 59%）。区间来自 GPU 频率状态：同一内核在一次会话里会整体慢 10–30%，而且与 CPU 负载无关。fp32 累加是红线，再往上只能靠 fp16 累加，没有做。

## 试过但没留下的

GEMM（`--rawbench`，SIMD8，GEMM 形状集总和；同一时段内比较）：

| 变体 | 结果 |
|---|---|
| 32×64 / 4×4、fp16 进共享内存，驱动默认宽度 | 196.5 ms（基线） |
| 同上，锁 8 lane | 133.4 ms |
| 64×64 / 8×8，fp32 k-major 共享内存（保留） | 约 100–118 ms，随 GPU 状态 |
| 同上，共享内存按 vec4 读 | 与标量读相同（编译器已向量化） |
| 同上，行 padding 1 / 4 | 慢 48–61% |
| 同上，去掉寄存器预取 | 慢 3–11% |
| BK 4 | 某一频率状态下快 5%，另一状态下慢 13%；BK 16 更稳，保留 16 |
| BK 8 / 32 | 慢 75–100% |
| 128×64、64×128、128×128 工作组 | 慢 3–20% |
| 32 线程工作组（32×64、64×32，8×8） | 慢一倍以上 |
| SIMD16 下的 8×8 | 寄存器溢出，慢 6.7 倍 |
| 8 lane subgroup，A 用 `subgroupShuffle` / 常量 `subgroupBroadcast` 广播，B 直接全局读，不用共享内存 | 慢 2.4–7 倍（几种 tile 都是），瓶颈在分散的全局读 |
| 尾部按 `f16vec4` 写回 | 无收益，删掉 |
| 一份二进制里同时放展开尾部和暂存尾部 | 主循环溢出，慢约 20 倍 |
| gelu 全展开的单独变体 | 慢约 24 倍，gelu 必须走暂存尾部 |

路由：

- dot→GEMM 门槛扫了 2¹² / 2¹⁴ / 2¹⁶ / 2¹⁸ / 2²⁰，medium 几乎一样，tiny 在 2¹⁴–2¹⁶ 最好，取 2¹⁶。
- `SIMD_OCR_CONVD_KMAX`（kxk 何时从 `convk_dot` 改走 im2col+GEMM）试了 512 / 256 / 128 / 48：medium DET 最多快 2%，small 慢 4–13%，tiny 慢 15–35%（cout 只有 24 / 32 时 64 宽的 tile 浪费一半，还丢了 concat 吸收）。保持 1024。

没有动 depthwise / SE / convT：还不在前两名。

## 默认路径

三档都没有快过同机 sharp，所以 `Auto` 不走这条路径。显式 `Vulkan` 走：结果和 CPU 精度一致，只是慢 1.8–2.0 倍。tiny 上实测：`--engine vulkan` 83.6 ms，`--engine auto` 44.2 ms（与 sharp 逐图 100/100），`auto` + `SIMD_OCR_BACKEND=vulkan` 84.9 ms（与 vulkan 100/100）。

没有 coopmat、subgroup 固定为 32 的设备（例如不带协作矩阵的旧 NVIDIA）以前在 sg32 分支里抛异常回 CPU，现在显式 `Vulkan` 也会进这条无矩阵档。内核不依赖 subgroup 宽度，结果应该正确，但没有在这类设备上测过；锁不了 8 lane 时由驱动选宽度，可能溢出而很慢。

统一内存上的 `preferHost` 修正是一直生效的：先仍选非 device-local 的主机缓存类型（独显走这里），只有选不中时才接受 device-local + host-cached。

没有改 sg16 / sg32 / sg32l 的 shader、spv 或选择条件，新增的判断都挂在 `_nocm` 上（`CoopGemm` 为假时置位），`UsesGpu` 对 `CoopGemm` 为真的设备与原来的条件相同。全量重编后其余 `.spv` 逐字节不变。B580 已按 `CoopGemm` 测试，仍走 sg16，端到端保持稳定；3080 Ti、880M 报 16×16×16，判定结果与之前相同。
