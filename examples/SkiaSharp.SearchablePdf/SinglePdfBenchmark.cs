using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.Win32;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.ModelProvider;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Pdf;
using SkiaSharp;

internal static class SinglePdfBenchmark
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public static int Run(string[] files, string directory, PaddleOcrModelBundle bundle,
        PaddleOcrOptions options, string model, OcrBackend backend, float dpi, int detSize, int repeat)
    {
        Directory.CreateDirectory(directory);
        string[] names = files.Select(f => Path.GetFileNameWithoutExtension(f) + ".pdf").ToArray();
        if (names.Distinct(StringComparer.OrdinalIgnoreCase).Count() != names.Length)
            throw new ArgumentException("Input files have duplicate basenames; output PDF names would collide.");
        using Process process = Process.GetCurrentProcess();
        double startupMs = (DateTime.UtcNow - process.StartTime.ToUniversalTime()).TotalMilliseconds;
        using ResourceSampler sampler = new();
        sampler.Phase = "model-load";
        Stopwatch timer = Stopwatch.StartNew();
        TimeSpan cpuStart = process.TotalProcessorTime;
        Console.WriteLine($"Single-image PDF benchmark: {model}/{backend}, {files.Length} images, {repeat} passes.");
        using PaddleOcrAll ocr = PaddleOcrAll.Load(bundle, DocumentOrientationModel.Provider, options);
        double loadMs = timer.Elapsed.TotalMilliseconds;
        process.Refresh();
        double loadCpuSeconds = (process.TotalProcessorTime - cpuStart).TotalSeconds;
        Console.WriteLine($"Model load: {loadMs:F1} ms; line workers={ocr.EffectiveLineWorkerCount}.");
        List<BatchMetrics> batches = [];
        List<ImageMetrics> images = [];
        Dictionary<string, string> transcripts = [];
        List<string> errors = [];
        for (int pass = 1; pass <= repeat; pass++)
        {
            sampler.Phase = pass == 1 ? "cold-batch" : $"warm-batch-{pass - 1}";
            int sampleStart = sampler.Count;
            process.Refresh();
            TimeSpan batchCpuStart = process.TotalProcessorTime;
            Stopwatch batchTimer = Stopwatch.StartNew();
            for (int i = 0; i < files.Length; i++)
            {
                string output = Path.Combine(directory, names[i]);
                process.Refresh();
                TimeSpan imageCpuStart = process.TotalProcessorTime;
                Stopwatch imageTimer = Stopwatch.StartNew();
                double readMs, decodeMs, ocrMs, pdfMs;
                int width, height, lines, characters, padding;
                string text;
                string temporary = output + "." + Guid.NewGuid().ToString("N") + ".tmp";
                try
                {
                    Stopwatch stage = Stopwatch.StartNew();
                    byte[] encoded = File.ReadAllBytes(files[i]);
                    readMs = stage.Elapsed.TotalMilliseconds;
                    stage.Restart();
                    using (SKBitmap bitmap = PdfImage.Decode(encoded))
                    {
                        width = bitmap.Width; height = bitmap.Height;
                        decodeMs = stage.Elapsed.TotalMilliseconds;
                        stage.Restart();
                        PaddleOcrResult raw = ocr.Run(bitmap.GetPixelSpan(), width, height,
                            returnCtcAlignment: true, sourceStride: bitmap.RowBytes, format: ImagePixelFormat.Bgra32);
                        ocrMs = stage.Elapsed.TotalMilliseconds;
                        stage.Restart();
                        PaddleOcrResult result = PdfTextAlignment.RemovePaddingEmissions(raw);
                        using (FileStream stream = File.Create(temporary))
                        using (PaddleOcrPdfWriter writer = new(stream, dpi))
                        {
                            writer.AddPage(bitmap, result, encoded);
                            writer.Complete();
                        }
                        lines = result.Lines.Length;
                        characters = result.Lines.Sum(l => l.Text.EnumerateRunes().Count());
                        padding = raw.Lines.Sum(l => l.CtcSpans!.Length) - result.Lines.Sum(l => l.CtcSpans!.Length);
                        text = string.Concat(result.Lines.Select(l => l.Text));
                    }
                    File.Move(temporary, output, overwrite: true);
                    pdfMs = stage.Elapsed.TotalMilliseconds;
                }
                catch (Exception ex)
                {
                    if (File.Exists(temporary)) File.Delete(temporary);
                    errors.Add($"pass {pass}, {Path.GetFileName(files[i])}: {ex.Message}");
                    throw;
                }
                imageTimer.Stop();
                process.Refresh();
                double cpuSeconds = (process.TotalProcessorTime - imageCpuStart).TotalSeconds;
                images.Add(new(pass, Path.GetFileName(files[i]), names[i], width, height, lines, characters, padding,
                    imageTimer.Elapsed.TotalMilliseconds, readMs, decodeMs, ocrMs, pdfMs, cpuSeconds,
                    100 * cpuSeconds / imageTimer.Elapsed.TotalSeconds / Environment.ProcessorCount,
                    process.WorkingSet64, process.PrivateMemorySize64, new FileInfo(output).Length));
                if (pass == repeat) transcripts[names[i]] = text;
                Console.WriteLine($"[{pass}/{repeat} {i + 1}/{files.Length}] {Path.GetFileName(files[i])}: " +
                    $"{imageTimer.Elapsed.TotalMilliseconds:F1} ms, CPU {cpuSeconds:F3}s, WS {process.WorkingSet64 / 1048576.0:F0} MiB");
            }
            batchTimer.Stop();
            process.Refresh();
            double batchCpuSeconds = (process.TotalProcessorTime - batchCpuStart).TotalSeconds;
            ResourceSample[] samples = sampler.Snapshot().Skip(sampleStart).ToArray();
            batches.Add(new(pass, pass == 1 ? "cold" : "warm", files.Length,
                batchTimer.Elapsed.TotalMilliseconds, batchCpuSeconds,
                100 * batchCpuSeconds / batchTimer.Elapsed.TotalSeconds / Environment.ProcessorCount,
                batchCpuSeconds / batchTimer.Elapsed.TotalSeconds,
                samples.Length == 0 ? process.WorkingSet64 : samples.Max(s => s.WorkingSetBytes),
                samples.Length == 0 ? process.PrivateMemorySize64 : samples.Max(s => s.PrivateBytes),
                samples.Length == 0 ? 0 : samples.Max(s => s.CpuPercent)));
            Console.WriteLine($"Pass {pass}: {batchTimer.Elapsed.TotalSeconds:F3}s, CPU {batches[^1].CpuPercent:F1}%.");
        }
        sampler.Stop();
        ResourceSample[] allSamples = sampler.Snapshot();
        process.Refresh();
        ImageMetrics[] warm = images.Where(x => x.Pass > 1).ToArray();
        double[] warmLatency = warm.Select(x => x.ElapsedMs).Order().ToArray();
        double warmMs = batches.Where(x => x.Pass > 1).Sum(x => x.ElapsedMs);
        double warmCpu = batches.Where(x => x.Pass > 1).Sum(x => x.CpuSeconds);
        object? gpuSelected = typeof(PaddleOcrAll).GetField("_recGpu", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(ocr);
        Type? gpuType = typeof(PaddleOcrAll).Assembly.GetType("Sdcb.SimdPaddleOCR.Backends.Vulkan.GpuBackend");
        object? gpuDevice = gpuType?.GetField("s_device", BindingFlags.Static | BindingFlags.NonPublic)?.GetValue(null);
        string? gpuName = gpuDevice?.GetType().GetField("DeviceName")?.GetValue(gpuDevice) as string;
        object report = new
        {
            CreatedUtc = DateTimeOffset.UtcNow, Model = model, RequestedBackend = backend.ToString(),
            GpuPolicySelected = gpuSelected, ProbedGpu = gpuName,
            EnvironmentBackendOverride = Environment.GetEnvironmentVariable("SIMD_OCR_BACKEND"),
            Dpi = dpi, DetectorSize = detSize, ImageCount = files.Length, Repeat = repeat,
            InputBytes = files.Sum(f => new FileInfo(f).Length),
            SourceDirectory = Path.GetDirectoryName(files[0]), OutputDirectory = directory,
            Hardware = HardwareInfo(), LogicalProcessors = Environment.ProcessorCount,
            LineWorkers = ocr.EffectiveLineWorkerCount, SamplingIntervalMs = ResourceSampler.IntervalMs,
            StartupBeforeModelMs = startupMs, ModelLoadMs = loadMs, ModelLoadCpuSeconds = loadCpuSeconds,
            ColdEndToEndMs = startupMs + loadMs + batches[0].ElapsedMs,
            ColdFirstImageMs = images[0].ElapsedMs,
            PeakWorkingSetBytes = process.PeakWorkingSet64,
            SampledPeakWorkingSetBytes = allSamples.Max(s => s.WorkingSetBytes),
            SampledPeakPrivateBytes = allSamples.Max(s => s.PrivateBytes),
            WarmSummary = warm.Length == 0 ? null : new
            {
                Conversions = warm.Length, BatchMeanMs = warmMs / (repeat - 1),
                PerImageMeanMs = warmLatency.Average(), MedianMs = Percentile(warmLatency, .5),
                P95Ms = Percentile(warmLatency, .95), MinMs = warmLatency[0], MaxMs = warmLatency[^1],
                ImagesPerSecond = warm.Length / (warmMs / 1000),
                CpuSeconds = warmCpu, CpuPercent = 100 * warmCpu / (warmMs / 1000) / Environment.ProcessorCount,
                AverageLogicalCores = warmCpu / (warmMs / 1000),
                PeakWorkingSetBytes = batches.Where(x => x.Pass > 1).Max(x => x.PeakWorkingSetBytes),
                PeakPrivateBytes = batches.Where(x => x.Pass > 1).Max(x => x.PeakPrivateBytes),
                PeakSampleCpuPercent = batches.Where(x => x.Pass > 1).Max(x => x.PeakSampleCpuPercent),
                ReadMeanMs = warm.Average(x => x.ReadMs), DecodeMeanMs = warm.Average(x => x.DecodeMs),
                OcrMeanMs = warm.Average(x => x.OcrMs), PdfMeanMs = warm.Average(x => x.PdfMs)
            },
            OutputPdfBytes = names.Sum(n => new FileInfo(Path.Combine(directory, n)).Length),
            Notes = new[]
            {
                "One process and one OCR engine; images processed sequentially. Every pass reruns OCR and closes one PDF per image.",
                "First pass is cold; later complete passes are warm. Input filesystem and GPU driver caches are not cleared.",
                "CPU percent = process CPU seconds / wall seconds / logical processor count * 100; 100% means the whole machine.",
                "Memory is process working set and private committed memory, not GPU VRAM. OS peak includes model loading and cold pass.",
                "Per-image latency includes read, decode, OCR, alignment, completed PDF, bitmap disposal and rename; excludes diagnostics/report serialization.",
                "Batch latency also includes console progress and instrumentation. Disk writes use normal OS buffering, not forced durable sync.",
                "GpuPolicySelected reflects this engine's backend selection policy; individual GPU operations may fall back to CPU.",
                "Runs overwrite the same PDF files. Only final-pass transcripts are saved, after timing has ended."
            },
            Batches = batches, Images = images, Errors = errors
        };
        foreach (var pair in transcripts)
            File.WriteAllText(Path.Combine(directory, Path.ChangeExtension(pair.Key, ".txt")), pair.Value, new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(directory, "benchmark.json"), JsonSerializer.Serialize(report, JsonOptions), new UTF8Encoding(false));
        StringBuilder csv = new("pass,file,width,height,lines,characters,total_ms,read_ms,decode_ms,ocr_ms,pdf_ms,cpu_s,cpu_percent,working_set_mib,private_mib,pdf_bytes\n");
        foreach (ImageMetrics m in images)
            csv.AppendLine(FormattableString.Invariant($"{m.Pass},{Csv(m.File)},{m.Width},{m.Height},{m.Lines},{m.Characters},{m.ElapsedMs:F4},{m.ReadMs:F4},{m.DecodeMs:F4},{m.OcrMs:F4},{m.PdfMs:F4},{m.CpuSeconds:F6},{m.CpuPercent:F4},{m.WorkingSetBytes / 1048576.0:F4},{m.PrivateBytes / 1048576.0:F4},{m.PdfBytes}"));
        File.WriteAllText(Path.Combine(directory, "images.csv"), csv.ToString(), new UTF8Encoding(true));
        csv = new("elapsed_ms,phase,cpu_percent,total_cpu_seconds,working_set_mib,private_mib,managed_heap_mib\n");
        foreach (ResourceSample s in allSamples)
            csv.AppendLine(FormattableString.Invariant($"{s.ElapsedMs:F3},{s.Phase},{s.CpuPercent:F4},{s.CpuSeconds:F6},{s.WorkingSetBytes / 1048576.0:F4},{s.PrivateBytes / 1048576.0:F4},{s.ManagedHeapBytes / 1048576.0:F4}"));
        File.WriteAllText(Path.Combine(directory, "resources.csv"), csv.ToString(), new UTF8Encoding(true));
        Console.WriteLine($"Saved {files.Length} independent PDFs and benchmark.json/images.csv/resources.csv in {directory}.");
        return 0;
    }

    private static string Csv(string value) => "\"" + value.Replace("\"", "\"\"") + "\"";
    private static double Percentile(double[] sorted, double p)
    {
        double position = (sorted.Length - 1) * p;
        int lower = (int)position;
        return sorted[lower] + (sorted[Math.Min(lower + 1, sorted.Length - 1)] - sorted[lower]) * (position - lower);
    }
    private static object HardwareInfo()
    {
        string? cpu = null;
        ulong? physicalMemory = null;
        if (OperatingSystem.IsWindows())
        {
            using RegistryKey? key = Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
            cpu = key?.GetValue("ProcessorNameString") as string;
            MemoryStatus status = new() { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
            if (GlobalMemoryStatusEx(ref status)) physicalMemory = status.TotalPhysical;
        }
        return new { Cpu = cpu, PhysicalMemoryBytes = physicalMemory, Os = RuntimeInformation.OSDescription,
            Runtime = RuntimeInformation.FrameworkDescription, Architecture = RuntimeInformation.ProcessArchitecture.ToString() };
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatus
    {
        public uint Length, MemoryLoad;
        public ulong TotalPhysical, AvailablePhysical, TotalPageFile, AvailablePageFile,
            TotalVirtual, AvailableVirtual, AvailableExtendedVirtual;
    }
    private sealed record ImageMetrics(int Pass, string File, string Pdf, int Width, int Height,
        int Lines, int Characters, int PaddingTokensRemoved, double ElapsedMs, double ReadMs,
        double DecodeMs, double OcrMs, double PdfMs, double CpuSeconds, double CpuPercent,
        long WorkingSetBytes, long PrivateBytes, long PdfBytes);
    private sealed record BatchMetrics(int Pass, string Kind, int Images, double ElapsedMs,
        double CpuSeconds, double CpuPercent, double AverageLogicalCores, long PeakWorkingSetBytes,
        long PeakPrivateBytes, double PeakSampleCpuPercent);
}

internal sealed record ResourceSample(double ElapsedMs, string Phase, double CpuPercent, double CpuSeconds,
    long WorkingSetBytes, long PrivateBytes, long ManagedHeapBytes);

internal sealed class ResourceSampler : IDisposable
{
    public const int IntervalMs = 50;
    private readonly List<ResourceSample> _samples = [];
    private readonly ManualResetEventSlim _stop = new(false);
    private readonly Thread _thread;
    private volatile string _phase = "startup";
    public string Phase { set => _phase = value; }
    public int Count { get { lock (_samples) return _samples.Count; } }
    public ResourceSampler()
    {
        _thread = new Thread(Sample) { IsBackground = true, Name = "PDF benchmark sampler" };
        _thread.Start();
    }
    private void Sample()
    {
        using Process process = Process.GetCurrentProcess();
        Stopwatch timer = Stopwatch.StartNew();
        double previousMs = 0, previousCpu = process.TotalProcessorTime.TotalSeconds;
        do
        {
            process.Refresh();
            double ms = timer.Elapsed.TotalMilliseconds;
            double cpu = process.TotalProcessorTime.TotalSeconds;
            double percent = ms > previousMs ? 100 * (cpu - previousCpu) / ((ms - previousMs) / 1000) / Environment.ProcessorCount : 0;
            ResourceSample sample = new(ms, _phase, percent, cpu, process.WorkingSet64,
                process.PrivateMemorySize64, GC.GetTotalMemory(false));
            lock (_samples) _samples.Add(sample);
            previousMs = ms; previousCpu = cpu;
        } while (!_stop.Wait(IntervalMs));
    }
    public ResourceSample[] Snapshot() { lock (_samples) return _samples.ToArray(); }
    public void Stop() { _stop.Set(); _thread.Join(); }
    public void Dispose() { Stop(); _stop.Dispose(); }
}
