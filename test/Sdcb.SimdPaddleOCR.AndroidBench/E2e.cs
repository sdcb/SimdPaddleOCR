using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Tests;

namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>
/// End-to-end dataset run with the desktop Tests harness's options and JSON
/// layout (rows: file / total_ms / hash / detected / texts / stage_ms ...),
/// so cmp.ps1, steady.ps1 and Tests --summarize work on the pulled file.
/// </summary>
static class E2e
{
    internal static int Run(string[] args)
    {
        string model = "tiny", backendName = "cpu";
        int workers = 4, warmup = 1;
        int? count = null;
        string? outName = null;
        for (int i = 1; i < args.Length; i++)
        {
            string Next() => args[++i];
            switch (args[i])
            {
                case "--model": model = Next(); break;
                case "--backend": backendName = Next(); break;
                case "--workers": workers = int.Parse(Next()); break;
                case "--count": count = int.Parse(Next()); break;
                case "--warmup": warmup = int.Parse(Next()); break;
                case "--out": outName = Next(); break;
                default: throw new ArgumentException($"unknown argument {args[i]}");
            }
        }
        OcrBackend backend = Paths.ParseBackend(backendName);
        string metadataPath = Path.Combine(Paths.Dataset, "metadata.json");
        string[] files = Paths.DatasetFiles();
        if (files.Length != 100) throw new InvalidOperationException($"expected 100 images, got {files.Length}");
        if (count is { } c) files = files.Take(c).ToArray();

        var decoded = files.Select(f => (Name: Path.GetFileName(f), Img: Paths.LoadBgr(f))).ToArray();

        const int CacheEntries = 32;
        using var ocr = PaddleOcrAll.Load(Paths.ModelPath(model, "det"), Paths.ModelPath(model, "cls"),
            Paths.ModelPath(model, "rec"), Paths.DictPath(model), new PaddleOcrOptions
            {
                LineWorkerCount = workers,
                Detector = new PaddleOcrDetectorOptions { MaxSessionCacheEntries = CacheEntries, Backend = backend },
                Recognizer = new PaddleOcrRecognizerOptions { AdaptiveWidth = true, TargetWidth = 320, Backend = backend },
                Classifier = new PaddleOcrClassifierOptions { Backend = backend },
            });
        PipelineProfiler.Enable(true);
        var prev = PipelineProfiler.Snapshot();
        bool gpu = OnnxSharp.OcrSessionFactory.IsGpuBackend(backend);
        double wsLoaded = WorkingSetMb();
        Console.WriteLine($"loaded model={model} backend={backendName} usesGpu={gpu} workers={ocr.EffectiveLineWorkerCount}/{workers} ws={wsLoaded:F1}MB");

        int warmupCount = Math.Min(warmup, Math.Max(0, decoded.Length - 1));
        var rows = new List<BenchmarkRow>();
        var boxes = new List<float[][]>();
        double wsPeak = wsLoaded;
        for (int index = 0; index < decoded.Length; index++)
        {
            var (name, (bgr, w, h)) = decoded[index];
            var sw = Stopwatch.StartNew();
            PaddleOcrResult r = ocr.Run(bgr, w, h, w * 3);
            sw.Stop();
            var cur = PipelineProfiler.Snapshot();
            var stageMs = new Dictionary<string, double>();
            var stageCalls = new Dictionary<string, long>();
            for (int s = 0; s < PipelineProfiler.StageCount; s++)
            {
                stageMs[PipelineProfiler.StageNames[s]] = cur[s].Milliseconds - prev[s].Milliseconds;
                stageCalls[PipelineProfiler.StageNames[s]] = cur[s].Calls - prev[s].Calls;
            }
            prev = cur;
            double ws = WorkingSetMb();
            wsPeak = Math.Max(wsPeak, ws);
            rows.Add(new BenchmarkRow
            {
                File = name, Width = w, Height = h, Warmup = index < warmupCount,
                TotalMs = sw.Elapsed.TotalMilliseconds, StageMs = stageMs, StageCalls = stageCalls,
                Detected = r.DetectedCount, Lines = r.Lines.Length, Hash = $"{r.PackedTextHash:x16}",
                Texts = r.Lines.Select(x => x.Text).ToArray(),
                Rotations = r.Lines.Select(x => x.AppliedRotationDegrees).ToArray(),
                WorkingSetMb = ws,
            });
            boxes.Add(r.Lines.Select(x => Aabb(x.Box)).ToArray());
            Console.WriteLine($"{index + 1}/{decoded.Length} {name} total={sw.Elapsed.TotalMilliseconds:F3} det={stageMs["det_graph"]:F2} lines={r.Lines.Length} ws={ws:F1}MB");
        }

        var meta = new JsonObject
        {
            ["mode"] = backendName == "cpu" ? "sharp" : backendName,
            ["benchmark"] = true,
            ["benchmarkKind"] = "simd",
            ["model"] = model,
            ["workers"] = workers,
            ["backend"] = backendName,
            ["usesGpu"] = gpu,
            ["effectiveWorkers"] = ocr.EffectiveLineWorkerCount,
            ["cacheEntries"] = CacheEntries,
            ["rid"] = RuntimeInformation.RuntimeIdentifier,
            ["os"] = RuntimeInformation.OSDescription,
            ["arch"] = RuntimeInformation.OSArchitecture.ToString().ToLowerInvariant(),
            ["cpu"] = Environment.ProcessorCount,
            ["machine"] = $"{Android.OS.Build.Manufacturer} {Android.OS.Build.Model}",
            ["timestamp"] = DateTimeOffset.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"),
            ["pixelFormat"] = "bgr",
            ["working_set_mb_loaded"] = wsLoaded,
            ["working_set_mb_peak"] = wsPeak,
            ["working_set_mb_last"] = rows.Count > 0 ? rows[^1].WorkingSetMb : wsLoaded,
            ["advSimdSupported"] = System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported,
            ["effectiveIsa"] = System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported ? "advsimd" : "vector",
            ["sampleCount"] = decoded.Length,
            ["framework"] = RuntimeInformation.FrameworkDescription,
            ["libraryTfm"] = typeof(PaddleOcrAll).Assembly
                .GetCustomAttributes(typeof(System.Runtime.Versioning.TargetFrameworkAttribute), false)
                .OfType<System.Runtime.Versioning.TargetFrameworkAttribute>().FirstOrDefault()?.FrameworkName,
            ["thermal"] = Thermal(),
        };
        var accuracy = BenchSummary.ComputeAccuracy(rows, metadataPath, boxes);
        if (accuracy is not null) meta["accuracy"] = BenchSummary.AccuracyNode(accuracy);
        JsonObject doc = BenchSummary.WrapWithMeta(rows, meta);
        string outPath = Path.Combine(Bench.OutDir, outName ?? $"e2e-{model}-{backendName}.json");
        File.WriteAllText(outPath, doc.ToJsonString(new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        }));
        Console.WriteLine($"saved: {outPath}");
        BenchSummary.Print(BenchSummary.Parse(outPath, null, metadataPath));
        var measured = rows.Where(r => !r.Warmup).Select(r => r.TotalMs).OrderBy(x => x).ToArray();
        var tail = rows.Where(r => !r.Warmup).TakeLast(50).Select(r => r.TotalMs).OrderBy(x => x).ToArray();
        if (measured.Length > 0 && tail.Length > 0)
            Console.WriteLine($"steady: all-med {measured[measured.Length / 2]:F1} last50-med {tail[tail.Length / 2]:F1} ws-peak {wsPeak:F0}MB thermal={Thermal()}");
        return 0;
    }

    static float[] Aabb(PaddleOcrDetectionBox b) =>
    [
        Math.Min(Math.Min(b.X1, b.X2), Math.Min(b.X3, b.X4)),
        Math.Min(Math.Min(b.Y1, b.Y2), Math.Min(b.Y3, b.Y4)),
        Math.Max(Math.Max(b.X1, b.X2), Math.Max(b.X3, b.X4)),
        Math.Max(Math.Max(b.Y1, b.Y2), Math.Max(b.Y3, b.Y4)),
    ];

    internal static double WorkingSetMb()
    {
        using Process p = Process.GetCurrentProcess();
        p.Refresh();
        return p.WorkingSet64 / (1024d * 1024d);
    }

    /// <summary>PowerManager thermal status (0 none .. 6 shutdown) or -1.</summary>
    internal static int Thermal()
    {
        try
        {
            var pm = (Android.OS.PowerManager?)Android.App.Application.Context.GetSystemService(Android.Content.Context.PowerService);
            return pm is null || !OperatingSystem.IsAndroidVersionAtLeast(29) ? -1 : (int)pm.CurrentThermalStatus;
        }
        catch { return -1; }
    }
}
