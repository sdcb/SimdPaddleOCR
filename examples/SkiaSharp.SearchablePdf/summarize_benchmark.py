"""Combine sequential Auto and CPU single-PDF benchmark runs into a Chinese report.

Usage: python summarize_benchmark.py output/pdf/single-benchmark
"""
import argparse
import csv
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path


def memory_mean(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    weights = []
    for previous, current in zip(rows, rows[1:]):
        if current["phase"].startswith("warm-batch") and current["phase"] == previous["phase"]:
            weights.append((float(current["elapsed_ms"]) - float(previous["elapsed_ms"]), current))
    total = sum(w for w, _ in weights)
    return {key: sum(w * float(row[key]) for w, row in weights) / total
            for key in ["working_set_mib", "private_mib"]}


def build_report(directory):
    reports = [json.loads((directory / name / "benchmark.json").read_text(encoding="utf-8"))
               for name in ["auto", "cpu"]]
    auto, cpu = reports
    warm = [r["WarmSummary"] for r in reports]
    assert all(r["Errors"] == [] for r in reports)
    assert all(r["ImageCount"] == 53 and r["Repeat"] == 3 for r in reports)
    assert cpu["GpuPolicySelected"] is False, "CPU control selected a GPU"
    assert all(len(r["Images"]) == r["ImageCount"] * r["Repeat"] for r in reports)
    assert all(r["Model"] == auto["Model"] and r["DetectorSize"] == auto["DetectorSize"] for r in reports)
    ram = auto["Hardware"]["PhysicalMemoryBytes"] / 2**30
    created = datetime.fromisoformat(cpu["CreatedUtc"]).astimezone(timezone(timedelta(hours=8)))
    out = ["# 每张图片单独一个双层 PDF：性能报告", "",
           f"实测时间：{created:%Y-%m-%d %H:%M}（Asia/Shanghai）。", "",
           f"同一进程复用一份 OCR 模型，依次处理 53 张图片，每张写成一个独立 PDF。"
           f"预热后的 Auto 平均每张 {warm[0]['PerImageMeanMs']:.1f} ms，"
           f"纯 CPU 平均每张 {warm[1]['PerImageMeanMs']:.1f} ms。", "",
           "## 环境和测试方式", "",
           f"- 输入：`{auto['SourceDirectory']}`，53 张 JPEG，合计 {auto['InputBytes']/2**20:.2f} MiB。",
           f"- CPU：{auto['Hardware']['Cpu']}，{auto['LogicalProcessors']} 个逻辑处理器。",
           f"- 内存：{ram:.2f} GiB；GPU：{auto['ProbedGpu']}。",
           f"- 系统：{auto['Hardware']['Os']}；{auto['Hardware']['Runtime']}，{auto['Hardware']['Architecture']}；Release。",
           f"- 模型：PP-OCRv6 {auto['Model']}；检测器长边上限 {auto['DetectorSize']}；输出 DPI {auto['Dpi']}；方向识别开启；CTC 逐字对齐开启。",
           f"- 图片串行；每次 OCR 内部使用 {auto['LineWorkers']} 个行处理 worker，其余线程预算沿用库默认设置。",
           f"- Auto GPU 选择策略：{auto['GpuPolicySelected']}；纯 CPU GPU 选择策略：{cpu['GpuPolicySelected']}。Auto 选择 Vulkan GPU，个别运算可能回退 CPU。",
           "- Auto 和 CPU 使用不同进程先后执行，各跑 3 轮；每轮重新解码、重新 OCR，写出并关闭 53 个 PDF。第一轮为冷批次，后两轮作为预热样本，共 106 次转换。",
           "- 输入和驱动缓存没有清空；测试期间使用正常桌面环境。GPU 组在 CPU 组开始前退出。", "",
           "## 预热性能", "",
           "| 指标 | Auto / GPU | 纯 CPU |", "| --- | ---: | ---: |"]
    metrics = [
        ("53 张完整批次平均时间", lambda r, w: f"{w['BatchMeanMs']/1000:.3f} s"),
        ("逐张平均时间", lambda r, w: f"{w['PerImageMeanMs']:.1f} ms"),
        ("逐张中位数 P50", lambda r, w: f"{w['MedianMs']:.1f} ms"),
        ("逐张 P95", lambda r, w: f"{w['P95Ms']:.1f} ms"),
        ("最快 / 最慢", lambda r, w: f"{w['MinMs']:.1f} / {w['MaxMs']:.1f} ms"),
        ("吞吐量", lambda r, w: f"{w['ImagesPerSecond']:.3f} 张/s"),
        ("进程平均 CPU 占整机比例", lambda r, w: f"{w['CpuPercent']:.2f}%"),
        ("平均使用的逻辑核心数", lambda r, w: f"{w['AverageLogicalCores']:.2f} / {r['LogicalProcessors']}"),
        ("50 ms 采样 CPU 峰值", lambda r, w: f"{w['PeakSampleCpuPercent']:.2f}%"),
        ("预热期工作集峰值", lambda r, w: f"{w['PeakWorkingSetBytes']/2**20:.1f} MiB"),
        ("预热期私有提交内存峰值", lambda r, w: f"{w['PeakPrivateBytes']/2**20:.1f} MiB"),
        ("全程 OS 工作集峰值", lambda r, w: f"{r['PeakWorkingSetBytes']/2**20:.1f} MiB"),
        ("最终 53 个 PDF 合计大小", lambda r, w: f"{r['OutputPdfBytes']/2**20:.2f} MiB"),
    ]
    for label, formatter in metrics:
        out.append(f"| {label} | {formatter(auto,warm[0])} | {formatter(cpu,warm[1])} |")
    means = [memory_mean(directory / name / "resources.csv") for name in ["auto", "cpu"]]
    out.extend([
        f"| 预热期平均工作集（时间加权） | {means[0]['working_set_mib']:.1f} MiB | {means[1]['working_set_mib']:.1f} MiB |",
        f"| 预热期平均私有提交内存（时间加权） | {means[0]['private_mib']:.1f} MiB | {means[1]['private_mib']:.1f} MiB |",
        "", f"此配置的 GPU 预热吞吐量约为纯 CPU 的 **{warm[0]['ImagesPerSecond']/warm[1]['ImagesPerSecond']:.2f} 倍**。"
        "这是本机、本模型和这组图片的结果。", "",
        "## 启动和每轮结果", "",
        "| 指标 | Auto / GPU | 纯 CPU |", "| --- | ---: | ---: |",
        f"| 模型加载前的进程启动与参数准备 | {auto['StartupBeforeModelMs']/1000:.3f} s | {cpu['StartupBeforeModelMs']/1000:.3f} s |",
        f"| 模型加载 | {auto['ModelLoadMs']/1000:.3f} s | {cpu['ModelLoadMs']/1000:.3f} s |",
        f"| 首张图片转换（不含模型加载） | {auto['ColdFirstImageMs']/1000:.3f} s | {cpu['ColdFirstImageMs']/1000:.3f} s |",
        f"| 启动至首轮全部 PDF 完成 | {auto['ColdEndToEndMs']/1000:.3f} s | {cpu['ColdEndToEndMs']/1000:.3f} s |"])
    for i in range(3):
        out.append(f"| 第 {i+1} 轮 53 张批次 | {auto['Batches'][i]['ElapsedMs']/1000:.3f} s | {cpu['Batches'][i]['ElapsedMs']/1000:.3f} s |")
    out.extend(["", "首张时间包含首次推理、JIT 和后端初始化等开销；不是后续每张的耗时。"
                "冷启动仅指新进程，未重启机器、清空系统文件缓存或驱动缓存。", "",
                "## 单张预热平均阶段耗时", "",
                "| 阶段 | Auto / GPU | 纯 CPU |", "| --- | ---: | ---: |"])
    for label, key in [("读取文件", "ReadMeanMs"), ("SkiaSharp 解码及白底合成", "DecodeMeanMs"),
                       ("完整 OCR（含方向、检测、分类、识别）", "OcrMeanMs"),
                       ("文字对齐、PDF 写入关闭及重命名", "PdfMeanMs")]:
        out.append(f"| {label} | {warm[0][key]:.3f} ms | {warm[1][key]:.3f} ms |")
    out.extend(["", "阶段间还有少量计时与统计开销，因此阶段之和可能略小于总耗时。", "",
                "## 53 张图片明细（后两轮平均）", "",
                "| 图片 | 像素尺寸 | Auto 平均 ms | CPU 平均 ms | Auto CPU 占比 | CPU 组 CPU 占比 |",
                "| --- | --- | ---: | ---: | ---: | ---: |"])
    for image in [x for x in auto["Images"] if x["Pass"] == 3]:
        per = [[x for x in r["Images"] if x["Pass"] > 1 and x["File"] == image["File"]] for r in reports]
        latency = [statistics.mean(x["ElapsedMs"] for x in group) for group in per]
        usage = [100 * sum(x["CpuSeconds"] for x in group) / (sum(x["ElapsedMs"] for x in group)/1000) / r["LogicalProcessors"]
                 for group, r in zip(per, reports)]
        out.append(f"| {image['File']} | {image['Width']} × {image['Height']} | {latency[0]:.1f} | {latency[1]:.1f} | {usage[0]:.1f}% | {usage[1]:.1f}% |")
    out.extend(["", "## 指标口径", "",
                "- CPU 占比 = 进程累计 CPU 秒数 ÷ 实际经过秒数 ÷ 逻辑处理器数 × 100%。例如 20 个逻辑处理器下 65% 表示平均约使用 13 个逻辑核心。",
                "- 单张 CPU 秒数可大于单张耗时，因为多个线程同时计算；平均占比使用累计 CPU 时间计算，不是采样百分比的简单平均。",
                "- 工作集是 Windows 报告的进程驻留内存，包含托管、原生、映射和共享驻留页；私有提交内存是私有内存承诺量，可能高于工作集。两者不等同于独立显存占用。",
                "- 资源每 50 ms 采样一次，短瞬间峰值可能漏采；全程工作集峰值另外使用操作系统峰值计数器。CPU 采样峰值受调度和累计计时精度影响，应优先看完整批次平均值。",
                "- PDF 每张独立创建和关闭，JPEG 原样嵌入；逐张耗时包含文件读取、解码、OCR、文字定位、PDF 完成、位图释放和重命名。",
                "- 批次还包含日志和采样开销。文本、CSV、JSON 报告在计时结束后输出。文件写入使用系统缓冲，不强制刷新到物理磁盘。",
                "- 此报告采用持续复用 OCR 引擎的方式，没有每张图片重新启动进程和加载模型。", "",
                "## 原始数据和复现", "",
                "- [Auto 逐张数据](auto/images.csv)、[CPU 逐张数据](cpu/images.csv)。",
                "- [Auto 资源时间序列](auto/resources.csv)、[CPU 资源时间序列](cpu/resources.csv)。",
                "- [Auto 原始报告](auto/benchmark.json)、[CPU 原始报告](cpu/benchmark.json)。",
                "- [Auto PDF 检查](auto/verification.json)、[CPU PDF 检查](cpu/verification.json)。", "",
                "在 `SimdPaddleOCR1` 目录下分别运行，两组不要并发：", "", "```powershell",
                'dotnet build examples/SkiaSharp.SearchablePdf -c Release',
                'dotnet examples/SkiaSharp.SearchablePdf/bin/Release/net10.0/SkiaSharp.SearchablePdf.dll "E:\\temp\\22\\33\\17122502" "output\\pdf\\single-benchmark\\auto" --separate --repeat 3 --backend auto',
                'dotnet examples/SkiaSharp.SearchablePdf/bin/Release/net10.0/SkiaSharp.SearchablePdf.dll "E:\\temp\\22\\33\\17122502" "output\\pdf\\single-benchmark\\cpu" --separate --repeat 3 --backend cpu',
                'python examples/SkiaSharp.SearchablePdf/summarize_benchmark.py output/pdf/single-benchmark',
                "```", ""])
    target = directory / "performance-report.md"
    target.write_text("\n".join(out), encoding="utf-8")
    print(target)
    print(json.dumps({name: r["WarmSummary"] for name, r in zip(["auto", "cpu"], reports)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build_report(parser.parse_args().directory)
