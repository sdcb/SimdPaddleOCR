using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Sdcb.SimdPaddleOCR.Validation;

/// <summary>桌面与浏览器 WASM 共用的四方向数值、坐标和流水线验证。</summary>
internal static class DocumentOrientationValidation
{
    /// <summary>比较独立 Paddle 参考分数，并按后端测试方向推理及可选的完整 OCR。</summary>
    /// <param name="backend">待验证的显式 CPU 或 Vulkan 后端；Vulkan 必须保持 GPU 会话有效。</param>
    /// <param name="iterations">每个线程预算下计时的迭代次数，默认 20，调用方应传入正数。</param>
    /// <param name="pipeline">是否同时验证四种方向的文本、原图行框及字符框一致性。</param>
    /// <returns>包含图推理、预处理耗时、数值误差和流水线结果的报告；验证失败时抛出异常。</returns>
    // 参考分数由 build/convert-orientation.py 独立运行 Paddle 得到，不使用本项目输出生成期望值。
    internal static DocumentOrientationReport Run(OcrBackend backend, int iterations = 20, bool pipeline = true)
    {
        Assembly assembly = typeof(DocumentOrientationValidation).Assembly;
        using Stream sampleStream = assembly.GetManifestResourceStream("orientation.sample.jpg")!;
        using Image<Bgr24> image = Image.Load<Bgr24>(sampleStream);
        byte[] source = new byte[image.Width * image.Height * 3];
        image.CopyPixelDataTo(source);
        using Stream manifestStream = assembly.GetManifestResourceStream("orientation.manifest.json")!;
        using JsonDocument manifest = JsonDocument.Parse(manifestStream);
        JsonElement[] expectedCases = manifest.RootElement.GetProperty("models").GetProperty("PP-LCNet_x1_0_doc_ori")
            .GetProperty("comparisons").EnumerateArray().Where(c => c.GetProperty("case").GetString()!.StartsWith("sample-")).ToArray();
        using Stream modelStream = DocumentOrientationModel.OpenRead();
        using Model model = Model.Load(modelStream);
        List<DocumentOrientationMeasurement> measurements = [];
        // CPU 比较多个线程预算；GPU 图内并行不由此预算决定，使用一个 CPU 预算即可。
        int[] budgets = backend == OcrBackend.Cpu ? [1, 2, 4, 8] : [1];
        double tolerance = backend == OcrBackend.Cpu ? 2e-5 : 0.005;
        foreach (int budget in budgets)
        {
            using CompiledScope scope = new(model, budget);
            IOcrSession session = scope.Session;
            if (backend == OcrBackend.Vulkan)
            {
                // 用实际 GPU 会话替换默认 CPU 请求，随后检查未发生后端回落。
                session.Dispose();
                scope.Session = session = OcrSessionFactory.Create(scope.Compiled, backend);
                session.Reshape([1, 3, 224, 224]);
                if (!session.GpuAlive || session is InferenceSession) throw new Exception("Vulkan did not engage.");
            }
            double maxError = 0;
            for (int angle = 0; angle < 360; angle += 90)
            {
                // 构造顺时针 angle 的测试图；Rotate 接受逆时针角度，因此使用其补角。
                int correction = (360 - angle) % 360;
                byte[] rotated = new byte[source.Length];
                DocumentRotation.Rotate(source, image.Width, image.Height, image.Width * 3,
                    ImagePixelFormat.Bgr24, correction, rotated);
                var size = DocumentRotation.Size(image.Width, image.Height, correction);
                PPOCRPreprocess.DocumentOrientation(rotated, size.Width, size.Height, size.Width * 3,
                    session.InputData, session.ResizeWorkspace, session.InputIsNhwc);
                float[] output = session.RunInternal(session.InputData).ToArray();
                float[] expected = expectedCases[angle / 90].GetProperty("scores")[0].EnumerateArray().Select(v => v.GetSingle()).ToArray();
                for (int c = 0; c < 4; c++) maxError = Math.Max(maxError, Math.Abs(output[c] - expected[c]));
                int label = Array.IndexOf(output, output.Max());
                if (label != angle / 90 || maxError > tolerance) throw new Exception($"{backend} angle={angle} label={label} max_abs={maxError:G6}");
                if (backend == OcrBackend.Vulkan && !session.GpuAlive) throw new Exception("Vulkan fell back during orientation.");
            }
            PPOCRPreprocess.DocumentOrientation(source, image.Width, image.Height, image.Width * 3,
                session.InputData, session.ResizeWorkspace, session.InputIsNhwc);
            for (int i = 0; i < 3; i++) session.RunInternal(session.InputData);
            double[] graphMs = new double[iterations], preprocessMs = new double[iterations];
            for (int i = 0; i < iterations; i++)
            {
                long start = Stopwatch.GetTimestamp();
                PPOCRPreprocess.DocumentOrientation(source, image.Width, image.Height, image.Width * 3,
                    session.InputData, session.ResizeWorkspace, session.InputIsNhwc);
                preprocessMs[i] = (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
                start = Stopwatch.GetTimestamp();
                session.RunInternal(session.InputData);
                graphMs[i] = (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
            }
            // 图推理与预处理分别取中位数；模型加载和预热均不计入这些耗时。
            Array.Sort(graphMs); Array.Sort(preprocessMs);
            measurements.Add(new(budget, graphMs[iterations / 2], preprocessMs[iterations / 2], maxError,
                session is not InferenceSession && session.GpuAlive));
            Console.WriteLine($"orientation {backend} threads={budget} graph={graphMs[iterations / 2]:F3} ms preprocess={preprocessMs[iterations / 2]:F3} ms max_abs={maxError:G6}");
        }
        List<DocumentOrientationPipelineCase> pipelineCases = [];
        if (pipeline)
        {
            using PaddleOcrAll ocr = PaddleOcrAll.Load(ChineseV6TinyModels.Default, DocumentOrientationModel.Provider,
                new PaddleOcrOptions { DocumentOrientation = new() { Backend = backend },
                    Detector = new() { Backend = backend }, Classifier = new() { Backend = backend },
                    Recognizer = new() { Backend = backend } });
            PaddleOcrResult? baseline = null;
            for (int angle = 0; angle < 360; angle += 90)
            {
                int rotation = (360 - angle) % 360;
                byte[] rotated = new byte[source.Length];
                DocumentRotation.Rotate(source, image.Width, image.Height, image.Width * 3, ImagePixelFormat.Bgr24, rotation, rotated);
                var size = DocumentRotation.Size(image.Width, image.Height, rotation);
                Stopwatch watch = Stopwatch.StartNew();
                PaddleOcrResult result = ocr.Run(rotated, size.Width, size.Height, returnCtcAlignment: true);
                if (result.DocumentOrientation?.OrientationDegrees != angle || result.AppliedDocumentRotationDegrees != angle)
                    throw new Exception($"Pipeline orientation failed for {angle}");
                baseline ??= result;
                if (result.Text != baseline.Text || result.Lines.Length == 0) throw new Exception($"Pipeline text differs for {angle}");
                for (int i = 0; i < result.Lines.Length; i++)
                {
                    // 再逆变换测试图上的结果，与零度原图逐点比较，覆盖行框及 CTC 字符框。
                    PaddleOcrDetectionBox restored = DocumentRotation.ToSource(result.Lines[i].Box, image.Width, image.Height, rotation);
                    for (int p = 0; p < 4; p++)
                        if (System.Numerics.Vector2.Distance(restored[p], baseline.Lines[i].Box[p]) > 0.01f)
                            throw new Exception($"Pipeline coordinates differ for {angle}");
                    PaddleOcrCharacterBox[] chars = result.Lines[i].EstimateCharacterBoxes();
                    PaddleOcrCharacterBox[] originalChars = baseline.Lines[i].EstimateCharacterBoxes();
                    if (chars.Length == 0 || chars.Length != originalChars.Length) throw new Exception("Missing character boxes.");
                    for (int c = 0; c < chars.Length; c++)
                    {
                        var ch = chars[c];
                        var mapped = DocumentRotation.ToSource(new PaddleOcrDetectionBox(ch.X1, ch.Y1, ch.X2, ch.Y2,
                            ch.X3, ch.Y3, ch.X4, ch.Y4, ch.Score), image.Width, image.Height, rotation);
                        var reference = originalChars[c];
                        float[] actualPoints = [mapped.X1, mapped.Y1, mapped.X2, mapped.Y2, mapped.X3, mapped.Y3, mapped.X4, mapped.Y4];
                        float[] expectedPoints = [reference.X1, reference.Y1, reference.X2, reference.Y2, reference.X3, reference.Y3, reference.X4, reference.Y4];
                        for (int p = 0; p < 8; p++)
                            if (Math.Abs(actualPoints[p] - expectedPoints[p]) > 0.01f) throw new Exception($"Character coordinates differ for {angle}");
                    }
                }
                pipelineCases.Add(new(angle, result.Lines.Length, result.PackedTextHash, watch.Elapsed.TotalMilliseconds));
                Console.WriteLine($"orientation pipeline {backend} angle={angle} lines={result.Lines.Length} hash={result.PackedTextHash:x16}");
            }
        }
        return new(backend.ToString(), iterations, measurements.ToArray(), pipelineCases.ToArray(), true);
    }

    /// <summary>管理一组编译模型和可替换推理会话的生命周期。</summary>
    private sealed class CompiledScope : IDisposable
    {
        internal readonly CompiledModel Compiled;
        internal IOcrSession Session;
        /// <summary>固定四方向输入形状并创建初始 CPU 请求。</summary>
        /// <param name="model">借用的方向模型，释放本作用域时不会释放模型本身。</param>
        /// <param name="threads">CPU 算子线程预算。</param>
        internal CompiledScope(Model model, int threads)
        {
            Compiled = new CompiledModel(model, [1, 3, 224, 224], threads);
            Session = Compiled.CreateRequest();
        }
        /// <summary>先释放实际推理会话，再释放编译资源。</summary>
        public void Dispose() { Session.Dispose(); Compiled.Dispose(); }
    }
}

/// <summary>可用于桌面和浏览器的方向测试汇总报告。</summary>
/// <param name="Backend">实际请求的后端名称。</param>
/// <param name="Iterations">每个线程预算下的计时迭代数。</param>
/// <param name="Measurements">不同线程预算的推理和预处理测量。</param>
/// <param name="Pipeline">完整 OCR 四方向测试记录；关闭流水线验证时为空。</param>
/// <param name="Passed">全部验证是否通过；遇到失败时不会生成成功报告。</param>
internal record DocumentOrientationReport(string Backend, int Iterations, DocumentOrientationMeasurement[] Measurements,
    DocumentOrientationPipelineCase[] Pipeline, bool Passed);
/// <summary>单个线程预算的方向图测量。</summary>
/// <param name="Threads">编译模型使用的 CPU 算子线程数。</param>
/// <param name="GraphMedianMs">图推理中位耗时，单位毫秒。</param>
/// <param name="PreprocessMedianMs">预处理中位耗时，单位毫秒。</param>
/// <param name="MaxAbs">四个方向预测相对 Paddle 参考概率的最大绝对误差。</param>
/// <param name="GpuAlive">是否实际使用 GPU 会话并且未回落到 CPU。</param>
internal record DocumentOrientationMeasurement(int Threads, double GraphMedianMs, double PreprocessMedianMs, double MaxAbs, bool GpuAlive);
/// <summary>单个纸张方向的完整 OCR 验证记录。</summary>
/// <param name="Angle">输入纸张的顺时针旋转角度。</param>
/// <param name="Lines">识别出的非空文本行数。</param>
/// <param name="Hash">按固定 UTF-8 格式计算的文本哈希，用于检查方向之间的输出一致性。</param>
/// <param name="ElapsedMs">该次完整调用的耗时，单位毫秒；并非稳态性能统计。</param>
internal record DocumentOrientationPipelineCase(int Angle, int Lines, ulong Hash, double ElapsedMs);

/// <summary>预生成报告序列化代码，避免浏览器 AOT 和裁剪场景依赖反射。</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
[JsonSerializable(typeof(DocumentOrientationReport))]
internal partial class OrientationJsonContext : JsonSerializerContext { }
