# ImageSharp 3.2 升级评估（暂缓）

日期：2026-10-11

## 背景

本仓库的示例/测试/基准项目目前引用 `SixLabors.ImageSharp` 3.1.11（核心库 `Sdcb.SimdPaddleOCR` 本身**不**依赖 ImageSharp）：

- `test/Sdcb.SimdPaddleOCR.Tests`
- `test/Sdcb.SimdPaddleOCR.GpuBench`
- `test/Sdcb.SimdPaddleOCR.WasmBench`
- `test/Sdcb.SimdPaddleOCR.AndroidBench`
- `README.md` / `README_EN.md` 中的安装命令

## 3.1.x 的安全公告

以下公告只在 3.2.0（或 4.1.x）中修复，3.1.12 **未**修复：

| GHSA | 严重性 | 概要 | 修复版本 |
|---|---|---|---|
| GHSA-jj3q-cwqj-842r | high | CCITT T4/MH 解码堆越界写 | 3.2.0, 4.1.1 |
| GHSA-v76p-62qx-wwq2 | high | Tiled fax TIFF 堆越界写 | 3.2.0, 4.1.1 |
| GHSA-j3p4-wp97-rph4 | high | HistogramEqualization 未校验索引 | 3.2.0, 4.1.2 |
| GHSA-jjfr-hcj7-qf5w | high | TIFF CCITT T6 编码越界写 | 3.2.0, 4.1.2 |
| GHSA-j9gm-c75j-xc9q | high | TIFF CCITT T4 编码越界写 | 3.2.0, 4.1.2 |
| GHSA-gwg2-r3hj-4w44 | medium | ICC CLUT 解析未校验分配 | 3.2.0, 4.1.2 |
| GHSA-wmxv-xphr-5c9g | medium | BigTIFF IFD 死循环 | 3.2.0, 4.1.2 |

本仓库只用 ImageSharp 解码 JPEG/PNG 测试图片，且只在示例和测试中使用，实际暴露面较小。

## 为什么暂缓

ImageSharp 3.2.0 新增了**构建期许可证校验**（`build/SixLabors.ImageSharp.targets` 中的 `SixLabors_ValidateLicense`，在 `CoreCompile` 之前执行）：

- 未提供 `SixLaborsLicenseKey` 属性或 `sixlabors.lic` 文件时：
  - `Release` 构建**直接失败**：`error : Please obtain a license from https://sixlabors.com/pricing/`
  - `Debug` 构建仅警告（`ContinueOnError`）
- CI（`.github/workflows/test.yml`）以 `-c Release` 构建 `test/Sdcb.SimdPaddleOCR.Tests`，升级后 CI 会全部失败。

本地已验证：将上述项目改为 3.2.0 后，`dotnet build -c Release` 在 Tests、AspNetCore 示例、GpuBench 上均因上述错误失败。

## 后续可选方案

1. 申请 Six Labors 许可证密钥，作为仓库 secret（如 `SIXLABORS_LICENSE_KEY`）在 CI 中通过 `-p:SixLaborsLicenseKey=...` 传入；本地开发者同样需要配置。
2. 在示例/测试中把图片解码换成不需要许可证的库（如 SkiaSharp、`System.Drawing` 仅限 Windows，或自带的最小 PNG/JPEG 解码器）。
3. 保持 3.1.11，并接受上述公告的风险（仅影响示例/测试，且输入为受控测试图片）。

目前选择方案 3，后续再做决定。
