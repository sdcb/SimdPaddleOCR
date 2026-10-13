using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.Win32;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>Windows/NVIDIA 上的持续负载测试，记录调用耗时、CPU/GPU 占用和内存/显存。</summary>
internal static class ResourceBench
{
    /// <summary>预热后连续执行单请求方向识别或完整 OCR，同时采样资源并写入 JSON 报告。</summary>
    /// <param name="args">参数含 --backend cpu|vulkan、--scope orientation|pipeline、--seconds 测量秒数及 --out 输出路径。</param>
    /// <returns>成功返回 0；后端回落、结果变化或采样失败时抛出异常。</returns>
    internal static int Run(string[] args)
    {
        if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Windows telemetry required.");
        string scope = Argument(args, "--scope", "pipeline");
        if (scope is not ("pipeline" or "orientation")) throw new ArgumentException("Unknown scope.");
        OcrBackend backend = Argument(args, "--backend", "cpu") switch
        {
            "cpu" => OcrBackend.Cpu,
            "vulkan" => OcrBackend.Vulkan,
            _ => throw new ArgumentException("Backend must be cpu or vulkan.")
        };
        double seconds = double.Parse(Argument(args, "--seconds", "20"), System.Globalization.CultureInfo.InvariantCulture);
        if (seconds < 2 || seconds > 300) throw new ArgumentOutOfRangeException(nameof(seconds));
        string output = Argument(args, "--out", $"bench-out/resource-{scope}-{backend}.json");
        using var gpu = new NvidiaTelemetry();
        using var process = Process.GetCurrentProcess();
        var baseline = new List<ResourceSample>();
        // 模型加载前记录空闲基线；GPU 是整卡数值，基线用于说明桌面及其他程序的后台负载。
        using (var sampler = new ResourceSampler(gpu, process, baseline)) Thread.Sleep(2500);

        using Stream imageStream = typeof(ResourceBench).Assembly.GetManifestResourceStream("orientation.sample.jpg")!;
        using Image<Bgr24> image = Image.Load<Bgr24>(imageStream);
        byte[] original = new byte[image.Width * image.Height * 3], pixels = new byte[original.Length];
        image.CopyPixelDataTo(original);
        // 构造顺时针 90° 的输入页，使完整流水线确实执行非零方向的整页校正。
        DocumentRotation.Rotate(original, image.Width, image.Height, image.Width * 3, ImagePixelFormat.Bgr24, 270, pixels);
        int width = image.Height, height = image.Width;
        using Stream modelStream = DocumentOrientationModel.OpenRead();
        using var orientation = scope == "orientation"
            ? new PaddleOcrDocumentOrientationClassifier(modelStream, new() { Backend = backend, IntraOpThreads = 4 }) : null;
        using var pipeline = scope == "pipeline"
            ? PaddleOcrAll.Load(ChineseV6TinyModels.Default, DocumentOrientationModel.Provider, new()
            {
                DocumentOrientation = new() { Backend = backend, IntraOpThreads = 4 },
                Detector = new() { Backend = backend },
                Classifier = new() { Backend = backend },
                Recognizer = new() { Backend = backend },
                DetIntraOpThreads = 8, LineWorkerCount = 4,
                PreprocessWorkerCount = 4, GpuCtcIntraOpThreads = 4
            }) : null;
        PaddleOcrResult? lastResult = null;
        PaddleOcrDocumentOrientationResult lastOrientation = default;
        // 执行一次所选模式，保存最终输出供测量前后校验；文件读取和图像解码不在此调用内。
        void Invoke()
        {
            if (orientation != null) lastOrientation = orientation.Classify(pixels, width, height);
            else lastResult = pipeline!.Run(pixels, width, height);
        }
        // 预热至少 3 秒且不少于 10 次，排除首次 JIT、shader 编译和缓冲区创建。
        var warmup = Stopwatch.StartNew();
        int warmupCalls = 0;
        do { Invoke(); warmupCalls++; } while (warmup.Elapsed.TotalSeconds < 3 || warmupCalls < 10);
        warmup.Stop();
        StageState[] before = CheckSessions(orientation, pipeline, backend);
        CheckResult(lastResult, lastOrientation, scope);
        ulong? expectedHash = lastResult?.PackedTextHash;
        Console.WriteLine($"Ready {scope}/{backend} {width}x{height}, warmup={warmupCalls}, sessions={string.Join(", ", before.Select(x => x.Name + ":" + x.SessionTypes))}");

        var samples = new List<ResourceSample>();
        var latencies = new List<double>();
        TimeSpan cpuStarted = process.TotalProcessorTime;
        long started = Stopwatch.GetTimestamp();
        using (var sampler = new ResourceSampler(gpu, process, samples))
        {
            do
            {
                long callStarted = Stopwatch.GetTimestamp();
                Invoke();
                latencies.Add(Stopwatch.GetElapsedTime(callStarted).TotalMilliseconds);
            } while (Stopwatch.GetElapsedTime(started).TotalSeconds < seconds);
        }
        double elapsed = Stopwatch.GetElapsedTime(started).TotalSeconds;
        double cpuSeconds = (process.TotalProcessorTime - cpuStarted).TotalSeconds;
        StageState[] after = CheckSessions(orientation, pipeline, backend);
        CheckResult(lastResult, lastOrientation, scope);
        if (expectedHash != lastResult?.PackedTextHash) throw new InvalidOperationException("OCR text changed during measurement.");
        double[] sorted = latencies.Order().ToArray();
        // 进程 CPU 时间跨线程累加，再按整机逻辑处理器数归一化；一个逻辑处理器满载约为 100/N%。
        double processCpu = cpuSeconds / elapsed / Environment.ProcessorCount * 100;
        var report = new
        {
            CapturedAt = DateTimeOffset.Now, Backend = backend.ToString(), Scope = scope,
            Hardware = new { Cpu = Registry.GetValue(@"HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString", "unknown"),
                LogicalProcessors = Environment.ProcessorCount, Gpu = gpu.Name, Driver = gpu.Driver, TotalGpuMemoryMiB = gpu.TotalMemoryMiB,
                TotalSystemMemoryMiB = Native.Memory().TotalPhys / 1048576.0 },
            Input = new { File = "examples/sample.jpg", Width = width, Height = height, ClockwiseDegrees = 90, Format = "Bgr24", Lines = lastResult?.Lines.Length,
                TextHash = lastResult?.PackedTextHash.ToString("x16"), Label = scope == "orientation" ? lastOrientation.OrientationDegrees : lastResult!.DocumentOrientation!.Value.OrientationDegrees },
            Settings = new { Concurrency = 1, OrientationThreads = 4, DetThreads = 8, LineWorkers = 4, PreprocessWorkers = 4, GpuCtcThreads = 4,
                WarmupSeconds = warmup.Elapsed.TotalSeconds, WarmupCalls = warmupCalls, SampleIntervalMs = 250, IncludesImageDecoding = false, IncludesModelLoading = false,
                IncludesPreprocessing = true, CtcAlignment = false, CpuPercentDenominator = "all logical processors", GpuUtilizationScope = "whole NVIDIA GPU, including other applications" },
            Measurement = new { Seconds = elapsed, Calls = latencies.Count, ThroughputPerSecond = latencies.Count / elapsed,
                MeanMs = latencies.Average(), MedianMs = Percentile(sorted, .5), P95Ms = Percentile(sorted, .95), MinMs = sorted[0], MaxMs = sorted[^1],
                ProcessCpuMeanPercent = processCpu, ProcessCpuCoreEquivalents = cpuSeconds / elapsed },
            Resources = Summarize(samples), Baseline = Summarize(baseline), SessionsBefore = before, SessionsAfter = after,
            Samples = samples, BaselineSamples = baseline
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true }));
        Console.WriteLine($"{scope}/{backend}: mean={latencies.Average():F3}ms p50={Percentile(sorted, .5):F3}ms p95={Percentile(sorted, .95):F3}ms calls={latencies.Count} cpu={processCpu:F1}% GPU={samples.Average(s => s.GpuPercent):F1}% RAM={samples.Average(s => s.WorkingSetMiB):F1}MiB VRAM={samples.Average(s => s.GpuMemoryMiB):F1}MiB -> {output}");
        return 0;
    }

    /// <summary>读取紧随指定命令行选项后的字符串值。</summary>
    /// <param name="args">原始命令行参数数组。</param>
    /// <param name="key">需要查找的选项名称。</param>
    /// <param name="fallback">未提供该选项时返回的默认值。</param>
    /// <returns>选项值或默认值；选项后缺少值时抛出异常。</returns>
    private static string Argument(string[] args, string key, string fallback)
    {
        int i = Array.IndexOf(args, key);
        return i < 0 ? fallback : i + 1 < args.Length ? args[i + 1] : throw new ArgumentException($"Missing {key} value.");
    }

    /// <summary>验证测试图的 90° 标签，以及完整 OCR 确实完成旋转并输出文本行。</summary>
    /// <param name="result">完整 OCR 输出；仅方向测试时为 null。</param>
    /// <param name="orientation">独立方向分类输出；完整 OCR 模式使用 result 中的方向信息。</param>
    /// <param name="scope">orientation 表示独立分类，pipeline 表示完整 OCR。</param>
    private static void CheckResult(PaddleOcrResult? result, PaddleOcrDocumentOrientationResult orientation, string scope)
    {
        if ((scope == "orientation" ? orientation.OrientationDegrees : result!.DocumentOrientation!.Value.OrientationDegrees) != 90)
            throw new InvalidOperationException("Incorrect orientation label.");
        if (scope == "pipeline" && (result!.Lines.Length == 0 || result.AppliedDocumentRotationDegrees != 90))
            throw new InvalidOperationException("OCR failed to correct the page.");
    }

    /// <summary>检查实际执行过的池化会话，确认 CPU 模式纯用 CPU，Vulkan 模式没有回落。</summary>
    /// <param name="orientation">独立分类器；完整 OCR 模式为 null。</param>
    /// <param name="pipeline">完整 OCR 引擎；独立分类模式为 null。</param>
    /// <param name="backend">要求实际使用的显式后端。</param>
    /// <returns>各阶段实际会话类型、数量及 GPU 有效状态。</returns>
    private static StageState[] CheckSessions(PaddleOcrDocumentOrientationClassifier? orientation, PaddleOcrAll? pipeline, OcrBackend backend)
    {
        List<(string Name, object Stage)> stages = [];
        if (orientation != null) stages.Add(("orientation", orientation));
        else foreach (string field in new[] { "_documentOrientation", "_detector", "_classifier", "_recognizer" })
        {
            object? stage = typeof(PaddleOcrAll).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(pipeline);
            if (stage != null) stages.Add((field.TrimStart('_'), stage));
        }
        return stages.Select(s =>
        {
            var sessions = ((IEnumerable<IOcrSession>)s.Stage.GetType().GetField("_sessions", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(s.Stage)!).ToArray();
            if (sessions.Length == 0) throw new InvalidOperationException($"No measured sessions: {s.Name}");
            bool actualGpu = sessions.All(x => x is not InferenceSession && x.GpuAlive);
            bool actualCpu = sessions.All(x => x is InferenceSession);
            if (backend == OcrBackend.Vulkan ? !actualGpu : !actualCpu) throw new InvalidOperationException($"Backend mismatch/fallback: {s.Name}");
            return new StageState(s.Name, sessions.Length, string.Join(",", sessions.Select(x => x.GetType().Name).Distinct()), actualGpu);
        }).ToArray();
    }

    /// <summary>从已排序的非空耗时数组中按向上取整的索引取得分位值。</summary>
    /// <param name="sorted">从小到大排序的非空测量数组。</param>
    /// <param name="p">分位比例，范围 0～1，例如 0.95 表示 P95。</param>
    /// <returns>数组中对应的分位测量值。</returns>
    private static double Percentile(double[] sorted, double p) => sorted[(int)Math.Ceiling((sorted.Length - 1) * p)];
    /// <summary>按全部采样点汇总资源指标的平均值、最小值和采样峰值。</summary>
    /// <param name="values">非空的资源采样列表。</param>
    /// <returns>可序列化的指标汇总对象。</returns>
    private static object Summarize(List<ResourceSample> values)
    {
        // selector 指定单个资源指标；峰值只覆盖采样点，不声称捕获两次采样间的瞬时峰值。
        object Range(Func<ResourceSample, double> selector) => new { Mean = values.Average(selector), Min = values.Min(selector), Max = values.Max(selector) };
        return new
        {
            SampleCount = values.Count, ProcessCpuPercent = Range(s => s.ProcessCpuPercent), SystemCpuPercent = Range(s => s.SystemCpuPercent),
            ProcessWorkingSetMiB = Range(s => s.WorkingSetMiB), ProcessPrivateMiB = Range(s => s.PrivateMiB), ProcessMemoryPercent = Range(s => s.ProcessMemoryPercent),
            SystemMemoryPercent = Range(s => s.SystemMemoryPercent), GpuUtilizationPercent = Range(s => s.GpuPercent),
            GpuMemoryMiB = Range(s => s.GpuMemoryMiB), GpuMemoryPercent = Range(s => s.GpuMemoryPercent),
            GpuClockMHz = Range(s => s.GpuClockMHz), GpuPowerWatts = Range(s => s.GpuPowerWatts), GpuTemperatureC = Range(s => s.GpuTemperatureC)
        };
    }

    /// <summary>单个 OCR 阶段实际使用的会话状态。</summary>
    /// <param name="Name">阶段名称。</param>
    /// <param name="Sessions">池中已创建并使用的会话数。</param>
    /// <param name="SessionTypes">实际会话实现类型名称。</param>
    /// <param name="GpuAlive">全部会话是否为有效 GPU 会话。</param>
    private record StageState(string Name, int Sessions, string SessionTypes, bool GpuAlive);
    /// <summary>单次资源采样，CPU 和 GPU 的统计范围分别由字段明确区分。</summary>
    /// <param name="Seconds">距本轮采样开始的秒数。</param>
    /// <param name="ProcessCpuPercent">测试进程 CPU 占整机逻辑处理器容量的百分比。</param>
    /// <param name="SystemCpuPercent">整机非空闲 CPU 时间的百分比。</param>
    /// <param name="WorkingSetMiB">测试进程物理工作集，单位 MiB。</param>
    /// <param name="PrivateMiB">测试进程私有提交内存，单位 MiB。</param>
    /// <param name="ProcessMemoryPercent">进程工作集占系统总物理内存的百分比。</param>
    /// <param name="SystemMemoryPercent">系统报告的整机内存使用百分比。</param>
    /// <param name="GpuPercent">整张 NVIDIA 显卡的忙碌百分比，包含其他程序。</param>
    /// <param name="GpuMemoryMiB">整卡已用显存，单位 MiB，包含其他程序。</param>
    /// <param name="GpuMemoryPercent">整卡已用显存占总显存的百分比。</param>
    /// <param name="GpuClockMHz">GPU 当前 SM 时钟频率，单位 MHz。</param>
    /// <param name="GpuPowerWatts">整卡功耗，单位瓦。</param>
    /// <param name="GpuTemperatureC">GPU 温度，单位摄氏度。</param>
    private record ResourceSample(double Seconds, double ProcessCpuPercent, double SystemCpuPercent, double WorkingSetMiB,
        double PrivateMiB, double ProcessMemoryPercent, double SystemMemoryPercent, double GpuPercent, double GpuMemoryMiB,
        double GpuMemoryPercent, double GpuClockMHz, double GpuPowerWatts, double GpuTemperatureC);

    /// <summary>使用后台线程每 250 毫秒采样一次；释放时等待线程停止并转抛采样异常。</summary>
    private sealed class ResourceSampler : IDisposable
    {
        private readonly ManualResetEventSlim _stop = new(false);
        private readonly Thread _thread;
        private Exception? _failure;
        /// <summary>开始资源采样线程。</summary>
        /// <param name="gpu">已初始化的 NVML 读取器，必须在采样线程退出后再释放。</param>
        /// <param name="process">待测进程对象，供读取 CPU 时间及内存。</param>
        /// <param name="samples">接收采样结果的列表，采样期间由此线程独占写入。</param>
        internal ResourceSampler(NvidiaTelemetry gpu, Process process, List<ResourceSample> samples)
        {
            _thread = new Thread(() =>
            {
                try
                {
                    var previousCpu = Native.CpuTimes();
                    TimeSpan previousProcess = process.TotalProcessorTime;
                    var watch = Stopwatch.StartNew();
                    double previousSeconds = 0;
                    while (!_stop.Wait(250))
                    {
                        var currentCpu = Native.CpuTimes();
                        TimeSpan currentProcess = process.TotalProcessorTime;
                        double currentSeconds = watch.Elapsed.TotalSeconds;
                        double processCpu = (currentProcess - previousProcess).TotalSeconds / (currentSeconds - previousSeconds) / Environment.ProcessorCount * 100;
                        // GetSystemTimes 的内核时间含空闲时间；从总时间减去空闲增量才是整机忙碌率。
                        ulong total = currentCpu.Total - previousCpu.Total;
                        double systemCpu = 100.0 * (total - (currentCpu.Idle - previousCpu.Idle)) / total;
                        process.Refresh();
                        var memory = Native.Memory();
                        var util = gpu.Read();
                        samples.Add(new(currentSeconds, processCpu, systemCpu, process.WorkingSet64 / 1048576.0, process.PrivateMemorySize64 / 1048576.0,
                            process.WorkingSet64 * 100.0 / memory.TotalPhys, memory.MemoryLoad, util.Gpu, util.MemoryUsed / 1048576.0,
                            util.MemoryUsed * 100.0 / util.MemoryTotal, util.Clock, util.PowerMilliwatts / 1000.0, util.Temperature));
                        previousCpu = currentCpu; previousProcess = currentProcess; previousSeconds = currentSeconds;
                    }
                }
                catch (Exception ex) { _failure = ex; }
            }) { IsBackground = true, Name = "resource-sampler" };
            _thread.Start();
        }
        /// <summary>停止并等待后台线程退出，确保结果列表可安全读取；采样异常不会被静默忽略。</summary>
        public void Dispose()
        {
            _stop.Set(); _thread.Join(); _stop.Dispose();
            if (_failure != null) throw new InvalidOperationException("Resource sampling failed.", _failure);
        }
    }

    /// <summary>仅供 Windows 开发测试使用的 CPU 时间与物理内存接口。</summary>
    private static class Native
    {
        /// <summary>对应 Win32 MEMORYSTATUSEX；Length 为结构大小，其余容量字段均为字节。</summary>
        [StructLayout(LayoutKind.Sequential)] internal struct MemoryStatus
        {
            public uint Length, MemoryLoad;
            public ulong TotalPhys, AvailPhys, TotalPageFile, AvailPageFile, TotalVirtual, AvailVirtual, AvailExtendedVirtual;
        }
        /// <summary>读取所有逻辑处理器的累计 CPU 时间，时间单位为 100 纳秒。</summary>
        /// <param name="idle">累计空闲时间。</param>
        /// <param name="kernel">累计内核时间，包含 idle。</param>
        /// <param name="user">累计用户态时间。</param>
        /// <returns>读取是否成功；失败时使用 Win32 最后错误码。</returns>
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool GetSystemTimes(out ulong idle, out ulong kernel, out ulong user);
        /// <summary>读取整机内存状态。</summary>
        /// <param name="status">调用前设置 Length，成功后接收总内存、可用内存和使用率。</param>
        /// <returns>读取是否成功。</returns>
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
        /// <summary>读取整机累计空闲时间和总 CPU 时间。</summary>
        /// <returns>单位为 100 纳秒的空闲、内核加用户态累计时间。</returns>
        internal static (ulong Idle, ulong Total) CpuTimes()
        {
            if (!GetSystemTimes(out ulong idle, out ulong kernel, out ulong user)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
            return (idle, kernel + user);
        }
        /// <summary>读取物理内存、提交容量及整机内存使用率。</summary>
        /// <returns>Win32 内存状态；内存大小字段使用字节。</returns>
        internal static MemoryStatus Memory()
        {
            var value = new MemoryStatus { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
            if (!GlobalMemoryStatusEx(ref value)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
            return value;
        }
    }

    /// <summary>通过 NVML 读取本机索引为 0 的 NVIDIA 显卡；所有读数均为整卡指标。</summary>
    private sealed class NvidiaTelemetry : IDisposable
    {
        private readonly nint _device;
        /// <summary>显卡名称。</summary>
        internal string Name { get; }
        /// <summary>NVIDIA 驱动版本。</summary>
        internal string Driver { get; }
        /// <summary>整卡总显存，单位 MiB。</summary>
        internal double TotalMemoryMiB { get; }
        /// <summary>初始化 NVML，获取首张显卡句柄及固定硬件信息。</summary>
        internal NvidiaTelemetry()
        {
            Check(nvmlInit_v2());
            Check(nvmlDeviceGetHandleByIndex_v2(0, out _device));
            var buffer = new StringBuilder(128);
            Check(nvmlDeviceGetName(_device, buffer, 128)); Name = buffer.ToString();
            buffer.Clear(); Check(nvmlSystemGetDriverVersion(buffer, 128)); Driver = buffer.ToString();
            Check(nvmlDeviceGetMemoryInfo(_device, out var memory)); TotalMemoryMiB = memory.Total / 1048576.0;
        }
        /// <summary>读取当前整卡利用率、显存、SM 时钟、功耗和温度。</summary>
        /// <returns>GPU 百分比、已用/总显存字节数、MHz、毫瓦、摄氏度组成的快照。</returns>
        internal (uint Gpu, ulong MemoryUsed, ulong MemoryTotal, uint Clock, uint PowerMilliwatts, uint Temperature) Read()
        {
            Check(nvmlDeviceGetUtilizationRates(_device, out var util));
            Check(nvmlDeviceGetMemoryInfo(_device, out var memory));
            Check(nvmlDeviceGetClockInfo(_device, 1, out uint clock));
            Check(nvmlDeviceGetPowerUsage(_device, out uint power));
            Check(nvmlDeviceGetTemperature(_device, 0, out uint temperature));
            return (util.Gpu, memory.Used, memory.Total, clock, power, temperature);
        }
        /// <summary>结束本测试使用的 NVML 会话。</summary>
        public void Dispose() => Check(nvmlShutdown());
        /// <summary>将 NVML 的非零返回码转换为显式异常，防止生成无效资源报告。</summary>
        /// <param name="result">NVML 返回码，0 表示成功。</param>
        private static void Check(int result) { if (result != 0) throw new InvalidOperationException($"NVML failed: {result}"); }
        // Gpu 为整卡忙碌率，Memory 为显存读写忙碌率，两者均为百分比。
        [StructLayout(LayoutKind.Sequential)] private struct Utilization { public uint Gpu, Memory; }
        // Total、Free、Used 分别为整卡总显存、空闲显存和已用显存，单位为字节。
        [StructLayout(LayoutKind.Sequential)] private struct MemoryInfo { public ulong Total, Free, Used; }
        /// <summary>初始化 NVML。</summary>
        /// <returns>NVML 状态码，0 为成功。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl)] private static extern int nvmlInit_v2();
        /// <summary>关闭 NVML。</summary>
        /// <returns>NVML 状态码，0 为成功。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl)] private static extern int nvmlShutdown();
        /// <summary>按设备索引取得显卡句柄。</summary>
        /// <param name="index">零起始的显卡索引。</param>
        /// <param name="device">接收显卡句柄。</param>
        /// <returns>NVML 状态码。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl)] private static extern int nvmlDeviceGetHandleByIndex_v2(uint index, out nint device);
        /// <summary>读取显卡名称。</summary>
        /// <param name="device">显卡句柄。</param>
        /// <param name="buffer">接收名称的 ANSI 缓冲区。</param>
        /// <param name="length">缓冲区容量，含字符串结尾。</param>
        /// <returns>NVML 状态码。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)] private static extern int nvmlDeviceGetName(nint device, StringBuilder buffer, uint length);
        /// <summary>读取驱动版本。</summary>
        /// <param name="buffer">接收版本字符串的 ANSI 缓冲区。</param>
        /// <param name="length">缓冲区容量，含字符串结尾。</param>
        /// <returns>NVML 状态码。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)] private static extern int nvmlSystemGetDriverVersion(StringBuilder buffer, uint length);
        /// <summary>读取 GPU 及显存读写的利用率。</summary>
        /// <param name="device">显卡句柄。</param>
        /// <param name="utilization">接收 NVML 利用率采样。</param>
        /// <returns>NVML 状态码。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl)] private static extern int nvmlDeviceGetUtilizationRates(nint device, out Utilization utilization);
        /// <summary>读取整卡显存容量和占用。</summary>
        /// <param name="device">显卡句柄。</param>
        /// <param name="memory">接收显存字节数。</param>
        /// <returns>NVML 状态码。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl)] private static extern int nvmlDeviceGetMemoryInfo(nint device, out MemoryInfo memory);
        /// <summary>读取指定时钟域的频率。</summary>
        /// <param name="device">显卡句柄。</param>
        /// <param name="type">时钟类型，1 表示 SM 时钟。</param>
        /// <param name="clock">接收频率，单位 MHz。</param>
        /// <returns>NVML 状态码。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl)] private static extern int nvmlDeviceGetClockInfo(nint device, uint type, out uint clock);
        /// <summary>读取整卡功耗。</summary>
        /// <param name="device">显卡句柄。</param>
        /// <param name="power">接收功耗，单位毫瓦。</param>
        /// <returns>NVML 状态码。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl)] private static extern int nvmlDeviceGetPowerUsage(nint device, out uint power);
        /// <summary>读取温度传感器。</summary>
        /// <param name="device">显卡句柄。</param>
        /// <param name="sensor">传感器编号，0 表示 GPU 核心。</param>
        /// <param name="temperature">接收温度，单位摄氏度。</param>
        /// <returns>NVML 状态码。</returns>
        [DllImport("nvml.dll", CallingConvention = CallingConvention.Cdecl)] private static extern int nvmlDeviceGetTemperature(nint device, uint sensor, out uint temperature);
    }
}
