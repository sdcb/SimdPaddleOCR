# 本机 CPU / GPU 耗时与资源使用率

2026-10-13（Asia/Shanghai），i7-12700 / 20 逻辑处理器，32 GB 内存，RTX 3060 / 12 GiB 显存，驱动 610.88，.NET 10 Release。

使用 `examples/sample.jpg`（500×500，16 行），预先顺时针旋转 90°；单请求连续运行。模型为 `PP-LCNet_x1_0_doc_ori` 与 `ChineseV6TinyModels.Default`。四个模式各运行两轮，每轮预热至少 3 秒且不少于 10 次调用，再计时 20 秒。第二轮反向执行各模式，避免只用一个顺序作比较。

计时包含预处理；完整 OCR 包含纸张方向、整页旋转、DET、文本行 CLS、REC 和后处理。模型加载、首次 shader 编译、文件读取与 JPEG 解码不计时；未请求 CTC 对齐。方向单独测试只运行 Classify，不旋转整页。所有 GPU 阶段在预热后及测试结束均检查实际 Vulkan 会话和 GpuAlive，CPU 模式则检查全部为 InferenceSession。GPU 模式保留 CPU 预处理、后处理与 CTC 投影。

固定预算：方向 4 线程、DET 8 线程、行 worker 4、预处理 worker 4、GPU CTC 4；请求并发为 1。

## 耗时

平均耗时按调用次数汇总两轮；没有将两轮 P95 直接平均。

| 模式 | 合计平均 ms | 两轮平均 ms | 两轮 P95 ms | 40 秒调用数 | 吞吐 次/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 纸张方向识别 / CPU | 2.82 | 2.35 / 3.52 | 2.94 / 8.81 | 14192 | 354.8 |
| 纸张方向识别 / Vulkan | 1.02 | 1.03 / 1.02 | 1.21 / 1.18 | 38988 | 974.7 |
| 完整 OCR（含纸张方向） / CPU | 36.93 | 36.59 / 37.27 | 39.62 / 43.70 | 1084 | 27.1 |
| 完整 OCR（含纸张方向） / Vulkan | 15.32 | 15.20 / 15.44 | 16.32 / 16.69 | 2612 | 65.3 |

完整 OCR 的 GPU 约快 **2.41 倍**，两轮结果稳定。CPU 方向测试第二轮长尾增多（P95 8.81 ms），因此保留两轮结果并按所有调用汇总，不只报告较快的一轮。

## CPU / GPU 使用率

CPU 进程占用按整机 20 个逻辑处理器归一化，100% 表示全部逻辑处理器满载；一个逻辑处理器满载约 5%。CPU 与 GPU 均给出平均 / 采样峰值。GPU 是整卡 NVML 读数，包含桌面与其他应用；基线仅供参考，没有直接从测量值中扣除。

| 模式 | 进程 CPU 平均 / 峰值 % | 整机 CPU 平均 / 峰值 % | 整卡 GPU 平均 / 峰值 % | 测前 GPU 基线 % |
| --- | ---: | ---: | ---: | ---: |
| 纸张方向识别 / CPU | 12.3 / 20.0 | 16.4 / 36.7 | 7.8 / 84.0 | 6.7 |
| 纸张方向识别 / Vulkan | 3.0 / 4.7 | 5.8 / 18.7 | 38.9 / 72.0 | 14.3 |
| 完整 OCR（含纸张方向） / CPU | 40.2 / 55.3 | 44.5 / 70.5 | 11.5 / 63.0 | 3.8 |
| 完整 OCR（含纸张方向） / Vulkan | 9.9 / 16.0 | 12.5 / 26.4 | 44.1 / 69.0 | 2.3 |

CPU 模式没有执行 GPU 推理；表中对应的 GPU 占用来自整卡后台活动。

## 内存

进程内存是物理工作集（Working Set），使用率分母为系统报告的总物理内存 32509.4 MiB；私有提交内存单独列出。整机 RAM 与整卡显存包含其他进程，显存分母为 12288 MiB。GPU 内存这里不是 OCR 进程独占显存。采样间隔 250 ms，峰值为采样峰值。

| 模式 | 工作集平均 / 峰值 MiB | 工作集占物理内存 % | 私有提交平均 / 峰值 MiB | 整机 RAM 平均 % | 整卡显存平均 / 峰值 MiB | 整卡显存平均 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 纸张方向识别 / CPU | 108.8 / 114.9 | 0.33 | 74.6 / 80.7 | 43.4 | 2204.5 / 2663.8 | 17.9 |
| 纸张方向识别 / Vulkan | 242.4 / 247.8 | 0.75 | 212.8 / 219.6 | 44.0 | 2135.1 / 2265.6 | 17.4 |
| 完整 OCR（含纸张方向） / CPU | 239.7 / 285.2 | 0.74 | 202.3 / 247.5 | 44.1 | 2083.5 / 2243.8 | 17.0 |
| 完整 OCR（含纸张方向） / Vulkan | 332.4 / 382.3 | 1.02 | 399.9 / 451.0 | 44.0 | 2170.8 / 2264.8 | 17.7 |

## 正确性与复现

四次完整 OCR 测试均识别 16 行，文本 hash 为 `46d99468540b5eb7`；四次方向独立测试与完整 OCR 的方向标签均为 90°。全部会话的实际后端与请求后端一致，Vulkan 未回落。

[数据快照](resource-performance-validation.json) 保留各轮测量值、硬件、预算、平均/峰值、基线和会话检查。完整逐点使用率/内存采样位于 `bench-out/resource-*.json`，该输出目录不进入 Git。

```powershell
dotnet build test/Sdcb.SimdPaddleOCR.GpuBench -c Release
$perfDll = 'test/Sdcb.SimdPaddleOCR.GpuBench/bin/Release/net10.0/Sdcb.SimdPaddleOCR.GpuBench.dll'
dotnet $perfDll --resource-perf --backend cpu --scope orientation --seconds 20 --out bench-out/resource-orientation-cpu.json
dotnet $perfDll --resource-perf --backend vulkan --scope orientation --seconds 20 --out bench-out/resource-orientation-vulkan.json
dotnet $perfDll --resource-perf --backend cpu --scope pipeline --seconds 20 --out bench-out/resource-pipeline-cpu.json
dotnet $perfDll --resource-perf --backend vulkan --scope pipeline --seconds 20 --out bench-out/resource-pipeline-vulkan.json
```

资源采样仅在 Windows/NVIDIA 开发测试工具中使用 Win32 与 NVML，未增加生产库依赖。
