# Qualcomm Snapdragon 8 Gen 3 + Adreno 750（Android 16 ARM64）

* **平台**：realme RMX3888，高通骁龙 8 Gen 3（`SM8650`，`arm64-v8a`），Adreno 750（Vulkan），16 GB RAM，Android 16（API 36），.NET 10.0.15
* **测试口径**：`test/Sdcb.SimdPaddleOCR.AndroidBench`（`net10.0-android`，CoreCLR RyuJIT，`--e2e --model tiny|small|medium --backend cpu|vulkan`），`dataset/` 100 张（共 1036 行），2 轮 ABBA 交替（`v2.0.0 → main → main → v2.0.0`）。两个版本包名相同，每次运行前重装对应 APK；两次运行之间冷却 30 秒；中位延时取两轮中位的均值。USB 连接供电。
* **对比版本**：`v2.0.0` (`6972083`) vs `main` (`76f210c`)
* **测试日期**：2026-11-28

## 端到端性能、内存与精度（4 Workers, 100 张）

| 后端 | 模型 | 样本数 | `v2.0.0` 中位 (ms) | **main 中位 (ms)** | 速度比 | main 吞吐 (img/s) | 峰值 WS `v2.0 → main` (MB) | 加载后 WS `v2.0 → main` (MB) | 行精确匹配 `v2.0 → main` | CER `v2.0 → main` | 温控状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CPU (NEON, 4w) | `tiny` | 100 | 193.0 | **160.6** | **1.20×** | 5.97 | 922 → 919 | 771 → 780 | 742 → 753 / 1036 | 2.78% → 2.62% | 0 |
| CPU (NEON, 4w) | `small` | 100 | 640.6 | **525.4** | **1.22×** | 1.94 | 1009 → 1047 | 819 → 809 | 950 → 954 / 1036 | 0.60% → 0.58% | 1–2 |
| CPU (NEON, 4w) | `medium` | 100 | 2962.7 | **2839.6** | **1.04×** | 0.35 | 1515 → 1546 | 957 → 923 | 1004 → 1014 / 1036 | 0.26% → 0.31% | 3 |
| Vulkan (Adreno 750) | `tiny` | 100 | 120.8 | **105.3** | **1.15×** | 9.36 | 880 → 814 | 791 → 753 | 743 → 751 / 1036 | 2.78% → 2.62% | 0 |
| Vulkan (Adreno 750) | `small` | 100 | 226.6 | **189.8** | **1.19×** | 5.30 | 892 → 947 | 821 → 814 | 949 → 957 / 1036 | 0.62% → 0.56% | 1–2 |
| Vulkan (Adreno 750) | `medium` | 100 | 1099.5 | **812.0** | 1.35×† | 1.30 | 989 → 1048 | 955 → 932 | 1003 → 1012 / 1036 | 0.27% → 0.32% | 3 |

* **WS** 为进程工作集：「加载后」为模型加载完成、处理第一张图之前；「峰值」为整轮最大值。精度按 `scripts/evaluate.py` 口径。
* **温控状态** 为 Android `PowerManager` thermal status（0 = 无，3 = 严重降频）。

† `medium` 跑在 thermal=3（降频）下，轮间波动很大：Vulkan 两轮中位 `v2.0.0` 908.6 / 1290.3 ms，`main` 973.9 / 650.1 ms；后 50 张中位为 979.8 / 998.6 vs 969.6 / 681.8 ms。所以 1.35× 只表示 `main` 不慢于 `v2.0.0`，不宜当作精确倍率。CPU `medium` 两轮比较稳定（`main` 2832.5 / 2846.6 ms，`v2.0.0` 2825.5 / 3099.9 ms）。

## 小结

* **CPU (NEON)**：`tiny` **1.20×**、`small` **1.22×**、`medium` 1.04×。
* **Adreno 750 Vulkan**：`tiny` **105.3 ms（1.15×）**、`small` **189.8 ms（1.19×）**；`medium` 降频严重，但仍不慢于 `v2.0.0`。相对本机 CPU 加速 **1.5× / 2.8× / 3.5×**。
* **内存**：`tiny` Vulkan 峰值 880 → **814 MB**、加载后 791 → 753 MB；`small` / `medium` 的峰值比 `v2.0.0` 高 30–60 MB，加载后工作集全档下降。
* **精度**：CPU 输出与桌面 x64 / ARM64 一致（753 / 954 / 1014）。差异来源（解码器 + `small` / `medium` 默认 `UnclipRatio` 1.6）见 [`5800x-rtx3080ti.md`](5800x-rtx3080ti.md#小结)。

## 复现方式

```powershell
# 构建、安装并推送模型与数据集
.\test\Sdcb.SimdPaddleOCR.AndroidBench\run.ps1 -Build -Install -Push -Bench "--e2e --model tiny --backend vulkan"
```
