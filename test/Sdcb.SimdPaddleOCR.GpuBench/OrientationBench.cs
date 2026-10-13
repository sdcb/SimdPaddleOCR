using System.Text.Json;
using Sdcb.SimdPaddleOCR.Validation;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>方向模型的桌面数值验证、耗时测试和 Vulkan 调度分析入口。</summary>
internal static class OrientationBench
{
    /// <summary>执行方向测试或 GPU 逐调度计时，并将验证报告写入 JSON。</summary>
    /// <param name="args">命令行参数：--cpu 选择 CPU，--no-pipeline 跳过完整 OCR，--out 指定输出，--profile 分析 GPU。</param>
    /// <returns>成功返回 0，验证失败时抛出异常。</returns>
    internal static int Run(string[] args)
    {
        if (args.Contains("--profile"))
        {
            using Stream stream = Models.DocumentOrientation.DocumentOrientationModel.OpenRead();
            using OnnxSharp.Model model = OnnxSharp.Model.Load(stream);
            var compiled = new OnnxSharp.CompiledModel(model, [1, 3, 224, 224], 1);
            try
            {
                using var graph = new Backends.Vulkan.GpuDetGraph(Backends.Vulkan.GpuBackend.TryGetDevice()!, compiled);
                float[] input = new float[3 * 224 * 224];
                // 先预热 shader 和缓冲区，再统计每次 GPU 调度的耗时。
                for (int i = 0; i < 8; i++) graph.Run([1, 3, 224, 224], input);
                graph.DumpProfile([1, 3, 224, 224]);
            }
            finally { compiled.Dispose(); }
            return 0;
        }
        OcrBackend backend = args.Contains("--cpu") ? OcrBackend.Cpu : OcrBackend.Vulkan;
        DocumentOrientationReport result = DocumentOrientationValidation.Run(backend, pipeline: !args.Contains("--no-pipeline"));
        string output = args.SkipWhile(a => a != "--out").Skip(1).FirstOrDefault() ?? $"bench-out/orientation-{backend}.json";
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        File.WriteAllText(output, JsonSerializer.Serialize(result, OrientationJsonContext.Default.DocumentOrientationReport));
        return 0;
    }
}
