using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Numerics;
using System.Reflection;
using System.Runtime.Intrinsics.Wasm;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.ModelProvider;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Medium;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Small;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Sdcb.SimdPaddleOCR.WasmBench;

public static class Program
{
    /// <summary>运行浏览器性能测试；包含 --orientation 时改为验证纸张四方向模型及完整 OCR。</summary>
    /// <param name="args">页面传入的命令行参数；--base-url 指向本机测试服务，--out 指定报告文件。</param>
    /// <returns>成功时返回退出码 0，运行或报告保存失败时抛出异常。</returns>
    public static async Task<int> Main(string[] args)
    {
        if (args.Contains("--orientation"))
        {
            // 浏览器复用桌面的四方向数值、文本及坐标验证；WASM 使用 CPU/SIMD 算子路径。
            Validation.DocumentOrientationReport result = Validation.DocumentOrientationValidation.Run(OcrBackend.Cpu);
            string baseAddress = args.SkipWhile(a => a != "--base-url").Skip(1).First();
            string outFile = args.SkipWhile(a => a != "--out").Skip(1).FirstOrDefault() ?? "bench-out/orientation-wasm.json";
            using HttpClient http = new() { BaseAddress = new Uri(baseAddress) };
            // 使用预生成 JSON 上下文兼容浏览器 AOT/裁剪，由本机测试服务保存完整报告。
            using HttpResponseMessage response = await http.PostAsJsonAsync("/api/save-benchmark?out=" + Uri.EscapeDataString(outFile), result,
                Validation.OrientationJsonContext.Default.DocumentOrientationReport);
            response.EnsureSuccessStatusCode();
            return 0;
        }
        Console.WriteLine("[WASM] SimdPaddleOCR WebAssembly Benchmark Starting...");

        int workers = 4;
        string modelType = "tiny";
        int countLimit = 0;
        int warmupCount = 1;
        string outPath = "bench-out/wasm-tiny-4w.json";
        string? caseId = null;
        string? baseUrl = null;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--base-url" && i + 1 < args.Length)
            {
                baseUrl = args[i + 1];
                i++;
            }
            else if (args[i] == "--workers" && i + 1 < args.Length && int.TryParse(args[i + 1], out int w))
            {
                workers = w;
                i++;
            }
            else if (args[i] == "--model" && i + 1 < args.Length)
            {
                modelType = args[i + 1].ToLowerInvariant();
                i++;
            }
            else if (args[i] == "--count" && i + 1 < args.Length && int.TryParse(args[i + 1], out int c))
            {
                countLimit = c;
                i++;
            }
            else if (args[i] == "--warmup" && i + 1 < args.Length && int.TryParse(args[i + 1], out int wm))
            {
                warmupCount = wm;
                i++;
            }
            else if (args[i] == "--out" && i + 1 < args.Length)
            {
                outPath = args[i + 1];
                i++;
            }
            else if (args[i] == "--case-id" && i + 1 < args.Length)
            {
                caseId = args[i + 1];
                i++;
            }
        }

        caseId ??= $"wasm-{modelType}-{workers}w";

        Console.WriteLine($"[WASM] Config: model={modelType}, requested_workers={workers}, count={countLimit}, out={outPath}");
        Console.WriteLine($"[WASM] Environment.ProcessorCount: {Environment.ProcessorCount}");
        Console.WriteLine($"[WASM] PackedSimd.IsSupported: {PackedSimd.IsSupported}, PPOCR_DET_THREADS: {Environment.GetEnvironmentVariable("PPOCR_DET_THREADS") ?? "auto"}, Vector.IsHardwareAccelerated: {Vector.IsHardwareAccelerated}, Vector<float>.Count: {Vector<float>.Count}");

        PaddleOcrModelBundle bundle = modelType switch
        {
            "tiny" => ChineseV6TinyModels.Default,
            "small" => ChineseV6SmallModels.Default,
            "medium" => ChineseV6MediumModels.Default,
            _ => throw new ArgumentException($"Unsupported model '{modelType}'. Supported: tiny, small, medium")
        };

        Stopwatch sw = Stopwatch.StartNew();
        using PaddleOcrAll ocr = PaddleOcrAll.Load(bundle, new PaddleOcrOptions
        {
            LineWorkerCount = workers,
            Detector = new PaddleOcrDetectorOptions { MaxSessionCacheEntries = 32, Backend = OcrBackend.Cpu },
            Recognizer = new PaddleOcrRecognizerOptions { AdaptiveWidth = true, TargetWidth = 320, Backend = OcrBackend.Cpu },
            Classifier = new PaddleOcrClassifierOptions { Backend = OcrBackend.Cpu },
        });
        sw.Stop();
        Console.WriteLine($"[WASM] Model loaded in {sw.ElapsedMilliseconds} ms. Effective LineWorkers: {ocr.EffectiveLineWorkerCount}");

        PipelineProfiler.Enable(true);
        InferenceSession.EnableProfiling(true);

        using HttpClient httpClient = new HttpClient();
        if (!string.IsNullOrEmpty(baseUrl))
        {
            httpClient.BaseAddress = new Uri(baseUrl);
        }

        string[] files = [];
        string? metadataJson = null;

        try
        {
            string filesJson = await httpClient.GetStringAsync("/api/dataset-files");
            if (JsonNode.Parse(filesJson) is JsonArray array && array.Count > 0)
            {
                files = array.Select(x => x!.GetValue<string>()).ToArray();
                Console.WriteLine($"[WASM] Found {files.Length} images from dataset endpoint.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WASM] Could not load dataset list via /api/dataset-files: {ex.Message}");
        }

        if (files.Length > 0)
        {
            try
            {
                metadataJson = await httpClient.GetStringAsync("/dataset/metadata.json");
                Console.WriteLine("[WASM] metadata.json successfully loaded.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WASM] Note: metadata.json not available: {ex.Message}");
            }
        }

        if (files.Length == 0)
        {
            Console.WriteLine("[WASM] No dataset files found via HTTP. Using embedded sample.jpg for smoke benchmark.");
            files = ["sample.jpg"];
        }

        if (countLimit > 0 && files.Length > countLimit)
        {
            files = files.Take(countLimit).ToArray();
        }

        List<BenchmarkRow> rows = new List<BenchmarkRow>();
        var prevStages = PipelineProfiler.Snapshot();
        var prevOp = InferenceSession.ProfileSnapshot();
        var prevConv = InferenceSession.ConvClassProfileSnapshot();

        for (int i = 0; i < files.Length; i++)
        {
            string file = files[i];
            byte[] imageBytes;

            if (file == "sample.jpg")
            {
                using Stream? resStream = typeof(Program).Assembly.GetManifestResourceStream("orientation.sample.jpg");
                if (resStream is null) throw new InvalidOperationException("sample.jpg not found in embedded resources");
                using MemoryStream ms = new MemoryStream();
                await resStream.CopyToAsync(ms);
                imageBytes = ms.ToArray();
            }
            else
            {
                imageBytes = await httpClient.GetByteArrayAsync("/dataset/" + file);
            }

            using Image<Rgb24> image = Image.Load<Rgb24>(imageBytes);
            byte[] pixels = new byte[checked(image.Width * image.Height * 3)];
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Rgb24 p = image[x, y];
                    int o = (y * image.Width + x) * 3;
                    pixels[o] = p.B;
                    pixels[o + 1] = p.G;
                    pixels[o + 2] = p.R;
                }
            }

            bool isWarmup = i < warmupCount && files.Length > 1;

            sw.Restart();
            PaddleOcrResult result = ocr.Run(pixels, image.Width, image.Height, image.Width * 3);
            sw.Stop();

            var curStages = PipelineProfiler.Snapshot();
            var curOp = InferenceSession.ProfileSnapshot();
            var curConv = InferenceSession.ConvClassProfileSnapshot();

            var stageMs = new Dictionary<string, double>();
            var stageCalls = new Dictionary<string, long>();
            for (int s = 0; s < PipelineProfiler.StageCount; s++)
            {
                stageMs[PipelineProfiler.StageNames[s]] = curStages[s].Milliseconds - prevStages[s].Milliseconds;
                stageCalls[PipelineProfiler.StageNames[s]] = curStages[s].Calls - prevStages[s].Calls;
            }

            var ops = new Dictionary<string, BenchmarkMetric>();
            string[] opNames = Enum.GetNames<OperatorId>();
            for (int op = 1; op < opNames.Length; op++)
            {
                double ms = (curOp[op].Ticks - prevOp[op].Ticks) * 1000.0 / Stopwatch.Frequency;
                long calls = curOp[op].Calls - prevOp[op].Calls;
                if (calls > 0)
                    ops[opNames[op]] = new BenchmarkMetric { Ms = ms, Calls = calls };
            }

            var conv = new Dictionary<string, BenchmarkMetric>();
            string[] convNames = ["Conv1x1", "Conv3x3", "Depthwise3x3", "Stride2Conv3x3", "OtherConv"];
            for (int c = 0; c < 5; c++)
            {
                double ms = (curConv[c].Ticks - prevConv[c].Ticks) * 1000.0 / Stopwatch.Frequency;
                long calls = curConv[c].Calls - prevConv[c].Calls;
                if (calls > 0)
                    conv[convNames[c]] = new BenchmarkMetric { Ms = ms, Calls = calls };
            }

            prevStages = curStages;
            prevOp = curOp;
            prevConv = curConv;

            BenchmarkRow row = new BenchmarkRow
            {
                File = file,
                Width = image.Width,
                Height = image.Height,
                Warmup = isWarmup,
                TotalMs = sw.Elapsed.TotalMilliseconds,
                StageMs = stageMs,
                StageCalls = stageCalls,
                OperatorMs = ops,
                ConvClassMs = conv,
                Detected = result.DetectedCount,
                Lines = result.Lines.Length,
                Hash = $"{result.PackedTextHash:x16}",
                Texts = result.Lines.Select(l => l.Text).ToArray(),
                Rotations = result.Lines.Select(l => l.AppliedRotationDegrees).ToArray(),
                WorkingSetMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0),
            };

            rows.Add(row);

            string warmupTag = isWarmup ? " [WARMUP]" : "";
            Console.WriteLine($"[WASM][{i + 1}/{files.Length}] {file}: {row.TotalMs:F1} ms, {row.Lines} lines, hash={row.Hash}{warmupTag}");
        }

        JsonObject meta = new JsonObject
        {
            ["caseId"] = caseId,
            ["mode"] = "sharp",
            ["model"] = modelType,
            ["workers"] = workers,
            ["effectiveWorkers"] = ocr.EffectiveLineWorkerCount,
            ["processorCount"] = Environment.ProcessorCount,
            ["rid"] = "browser-wasm",
            ["platform"] = "wasm",
            ["vectorHardwareAccelerated"] = Vector.IsHardwareAccelerated,
            ["vectorCount"] = Vector<float>.Count,
            ["packedSimdSupported"] = PackedSimd.IsSupported,
            ["effectiveIsa"] = PackedSimd.IsSupported ? "packedsimd" : Vector.IsHardwareAccelerated ? "vector128" : "scalar",
            ["detThreads"] = Environment.GetEnvironmentVariable("PPOCR_DET_THREADS") ?? "auto",
            ["timestamp"] = DateTimeOffset.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"),
            ["sampleCount"] = rows.Count(r => !r.Warmup),
        };

        BenchmarkAccuracy? accuracy = WasmBenchSummary.ComputeAccuracy(rows, metadataJson);
        if (accuracy is not null)
        {
            meta["accuracy"] = WasmBenchSummary.AccuracyNode(accuracy);
        }

        JsonObject doc = WasmBenchSummary.WrapWithMeta(rows, meta);
        WasmBenchSummary.Print(doc);

        try
        {
            using var content = new StringContent(doc.ToJsonString(), System.Text.Encoding.UTF8, "application/json");
            HttpResponseMessage resp = await httpClient.PostAsync("/api/save-benchmark?out=" + Uri.EscapeDataString(outPath), content);
            if (resp.IsSuccessStatusCode)
            {
                Console.WriteLine($"[WASM] Successfully saved benchmark report to {outPath}");
            }
            else
            {
                Console.WriteLine($"[WASM] Server returned error saving benchmark: {resp.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WASM] Could not post result to /api/save-benchmark: {ex.Message}");
        }

        return 0;
    }
}
