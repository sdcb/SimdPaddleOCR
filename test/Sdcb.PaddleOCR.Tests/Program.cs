using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.OnnxSharp;
using Sdcb.PaddleOCR.Models.ChineseV6Medium;
using Sdcb.PaddleOCR.Models.ChineseV6Small;
using Sdcb.PaddleOCR.Models.ChineseV6Tiny;
using Sdcb.PaddleOCR.Tests;
using LwPpocrCSharp;
using ImageSharpImage = SixLabors.ImageSharp.Image;

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("usage: --workers 1..16 --model tiny|small|medium --input <dataset> --out <json> [--engine sharp|c] [--c-assets <dir>]");
    Console.WriteLine("       --summarize <file...> [--input <dataset>] [--out-md <path>]");
    return args.Length == 0 ? 2 : 0;
}

if (args[0] == "--summarize")
    return BenchSummary.Compare(args.Skip(1).ToArray());

int workers = 4;
string modelType = "tiny";
string engineName = "sharp";
string? inputDir = null;
string? outPath = null;
string cAssetsDir = "c-assets";
for (int i = 0; i < args.Length; i++)
{
    string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"missing value for {args[i]}");
    switch (args[i])
    {
        case "--workers": workers = int.Parse(Next()); break;
        case "--model": modelType = Next().ToLowerInvariant(); break;
        case "--engine": engineName = Next().ToLowerInvariant(); break;
        case "--input": inputDir = Next(); break;
        case "--out": outPath = Next(); break;
        case "--c-assets": cAssetsDir = Next(); break;
        default: throw new ArgumentException($"unknown argument: {args[i]}");
    }
}

if (workers is < 1 or > 16)
    throw new ArgumentException("--workers must be 1..16");
if (modelType is not ("tiny" or "small" or "medium"))
    throw new ArgumentException("--model must be tiny, small, or medium");
if (engineName is not ("sharp" or "c"))
    throw new ArgumentException("--engine must be sharp or c");
if (engineName == "c")
{
    if (!OperatingSystem.IsWindows())
        throw new PlatformNotSupportedException("--engine c requires Windows (lw_ppocr_c.dll)");
    if (modelType != "tiny")
        throw new ArgumentException("--engine c only supports --model tiny");
}
if (inputDir is null || outPath is null)
    throw new ArgumentException("--input and --out are required");

inputDir = Path.GetFullPath(inputDir);
outPath = Path.GetFullPath(outPath);
cAssetsDir = Path.GetFullPath(cAssetsDir);
string metadataPath = Path.Combine(inputDir, "metadata.json");
if (!File.Exists(metadataPath))
    throw new FileNotFoundException($"dataset metadata not found: {metadataPath}");

string[] files = Directory.GetFiles(inputDir, "img-*.jpg")
    .OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();
if (files.Length != 100)
    throw new InvalidOperationException($"expected 100 JPG files, got {files.Length}");

var decoded = new (string Name, byte[] Pixels, int W, int H)[files.Length];
for (int i = 0; i < files.Length; i++)
{
    using Image<Rgb24> image = ImageSharpImage.Load<Rgb24>(files[i]);
    byte[] pixels = new byte[checked(image.Width * image.Height * 3)];
    for (int y = 0; y < image.Height; y++)
        for (int x = 0; x < image.Width; x++)
        {
            Rgb24 p = image[x, y];
            int o = (y * image.Width + x) * 3;
            pixels[o] = p.B;
            pixels[o + 1] = p.G;
            pixels[o + 2] = p.R;
        }
    decoded[i] = (Path.GetFileName(files[i]), pixels, image.Width, image.Height);
}

const int cacheEntries = 32;
List<BenchmarkRow> rows;
double wsLoaded;
double wsPeak;
JsonObject extra = [];

if (engineName == "c")
{
    CAssets.EnsureAsync(cAssetsDir).GetAwaiter().GetResult();
    CAssets.CopyDll(cAssetsDir);
    string dictPath = CAssets.WriteDictionary(cAssetsDir);
    extra["cAssets"] = cAssetsDir;
    extra["cDll"] = CAssets.BaseUrl + CAssets.DllName;
    extra["cDet"] = CAssets.BaseUrl + CAssets.DetName;
    extra["cCls"] = CAssets.BaseUrl + CAssets.ClsName;
    extra["cRec"] = CAssets.BaseUrl + CAssets.RecName;
    extra["cDict"] = dictPath;

    using NativeOcr engine = new(
        CAssets.DetPath(cAssetsDir),
        CAssets.ClsPath(cAssetsDir),
        CAssets.RecPath(cAssetsDir),
        dictPath,
        useDirectionClassification: true,
        (uint)workers);
    wsLoaded = WorkingSetMb();
    wsPeak = wsLoaded;
    Console.WriteLine($"loaded working_set={wsLoaded:F1} MB engine=c");
    rows = [];
    for (int index = 0; index < decoded.Length; index++)
    {
        var d = decoded[index];
        Stopwatch sw = Stopwatch.StartNew();
        OcrResponse result = engine.RecognizeDecoded(new DecodedBgrImage
        {
            Pixels = d.Pixels,
            Width = d.W,
            Height = d.H,
            Stride = d.W * 3,
        });
        sw.Stop();
        double ws = WorkingSetMb();
        if (ws > wsPeak) wsPeak = ws;
        rows.Add(new BenchmarkRow
        {
            File = d.Name,
            Width = d.W,
            Height = d.H,
            Warmup = index == 0,
            TotalMs = sw.Elapsed.TotalMilliseconds,
            Detected = result.detected_count,
            Lines = result.result.Count,
            Texts = result.result.Select(x => x.text).ToArray(),
            Rotations = result.result.Select(x => x.rotation).ToArray(),
            WorkingSetMb = ws,
        });
        Console.WriteLine($"{index + 1}/100 {d.Name} total={sw.Elapsed.TotalMilliseconds:F3} lines={result.result.Count} ws={ws:F1}MB");
    }
}
else
{
    var bundle = modelType switch
    {
        "tiny" => ChineseV6TinyModels.Default,
        "small" => ChineseV6SmallModels.Default,
        "medium" => ChineseV6MediumModels.Default,
        _ => throw new ArgumentException(modelType),
    };

    using PaddleOcrAll engine = PaddleOcrAll.Load(bundle, new PaddleOcrOptions
    {
        LineWorkerCount = workers,
        Detector = new PaddleOcrDetectorOptions { MaxSessionCacheEntries = cacheEntries },
        Recognizer = new PaddleOcrRecognizerOptions { AdaptiveWidth = true, TargetWidth = 320 },
    });
    PipelineProfiler.Enable(true);
    InferenceSession.EnableProfiling(true);
    extra["cacheEntries"] = cacheEntries;
    extra["effectiveWorkers"] = engine.EffectiveLineWorkerCount;
    wsLoaded = WorkingSetMb();
    wsPeak = wsLoaded;
    Console.WriteLine($"loaded working_set={wsLoaded:F1} MB engine=sharp workers={engine.EffectiveLineWorkerCount}/{workers} cpu={Environment.ProcessorCount}");

    rows = [];
    var prev = PipelineProfiler.Snapshot();
    var prevOp = InferenceSession.ProfileSnapshot();
    var prevConv = InferenceSession.ConvClassProfileSnapshot();
    for (int index = 0; index < decoded.Length; index++)
    {
        var d = decoded[index];
        Stopwatch sw = Stopwatch.StartNew();
        PaddleOcrResult result = engine.Run(d.Pixels, d.W, d.H, d.W * 3);
        sw.Stop();
        double ws = WorkingSetMb();
        if (ws > wsPeak) wsPeak = ws;
        var cur = PipelineProfiler.Snapshot();
        var curOp = InferenceSession.ProfileSnapshot();
        var curConv = InferenceSession.ConvClassProfileSnapshot();
        var stageMs = new Dictionary<string, double>();
        var stageCalls = new Dictionary<string, long>();
        for (int s = 0; s < PipelineProfiler.StageCount; s++)
        {
            stageMs[PipelineProfiler.StageNames[s]] = cur[s].Milliseconds - prev[s].Milliseconds;
            stageCalls[PipelineProfiler.StageNames[s]] = cur[s].Calls - prev[s].Calls;
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
        rows.Add(new BenchmarkRow
        {
            File = d.Name,
            Width = d.W,
            Height = d.H,
            Warmup = index == 0,
            TotalMs = sw.Elapsed.TotalMilliseconds,
            StageMs = stageMs,
            StageCalls = stageCalls,
            OperatorMs = ops,
            ConvClassMs = conv,
            Detected = result.DetectedCount,
            Lines = result.Lines.Length,
            Hash = $"{result.PackedTextHash:x16}",
            Texts = result.Lines.Select(x => x.Text).ToArray(),
            Rotations = result.Lines.Select(x => x.AppliedRotationDegrees).ToArray(),
            WorkingSetMb = ws,
        });
        prev = cur;
        prevOp = curOp;
        prevConv = curConv;
        Console.WriteLine($"{index + 1}/100 {d.Name} total={sw.Elapsed.TotalMilliseconds:F3} det={stageMs["det_graph"]:F2} lines={result.Lines.Length} ws={ws:F1}MB");
    }
}

double wsLast = rows.Count > 0 ? rows[^1].WorkingSetMb : wsLoaded;
var meta = new JsonObject
{
    ["mode"] = engineName,
    ["model"] = modelType,
    ["workers"] = workers,
    ["rid"] = RuntimeInformation.RuntimeIdentifier,
    ["os"] = RuntimeInformation.OSDescription,
    ["arch"] = RuntimeInformation.OSArchitecture.ToString().ToLowerInvariant(),
    ["cpu"] = Environment.ProcessorCount,
    ["machine"] = Environment.MachineName,
    ["timestamp"] = DateTimeOffset.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"),
    ["pixelFormat"] = "bgr",
    ["working_set_mb_loaded"] = wsLoaded,
    ["working_set_mb_last"] = wsLast,
    ["working_set_mb_peak"] = wsPeak,
};
foreach ((string key, JsonNode? value) in extra)
    meta[key] = value?.DeepClone();
var (cpuName, memoryMb) = HostInfo();
if (cpuName is not null) meta["cpuName"] = cpuName;
if (memoryMb is not null) meta["memoryMb"] = memoryMb;
string? git = GitCommit();
if (git is not null) meta["gitCommit"] = git;
SetEnv(meta, "avx512", "DOTNET_EnableAVX512");
SetEnv(meta, "avx2", "DOTNET_EnableAVX2");
SetEnv(meta, "avx", "DOTNET_EnableAVX");
SetEnv(meta, "hwintrinsic", "DOTNET_EnableHWIntrinsic");

JsonObject doc = BenchSummary.WrapWithMeta(rows, meta);
BenchmarkAccuracy? accuracy = BenchSummary.ComputeAccuracy(
    doc["rows"]!.AsArray(), metadataPath);
if (accuracy is not null)
    meta["accuracy"] = BenchSummary.AccuracyNode(accuracy);

Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
File.WriteAllText(outPath, doc.ToJsonString(new JsonSerializerOptions
{
    WriteIndented = true,
    Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
}));
Console.WriteLine($"saved: {outPath}");
BenchSummary.Print(BenchSummary.Parse(outPath, null, metadataPath));
return 0;

static void SetEnv(JsonObject meta, string name, string env)
{
    string? value = Environment.GetEnvironmentVariable(env);
    if (value is not null)
        meta[name] = value;
}

static double WorkingSetMb()
{
    Process process = Process.GetCurrentProcess();
    process.Refresh();
    return process.WorkingSet64 / (1024d * 1024d);
}

static (string? CpuName, double? MemoryMb) HostInfo()
{
    try
    {
        if (OperatingSystem.IsWindows())
            return WindowsHost();
        if (OperatingSystem.IsLinux())
            return LinuxHost();
        if (OperatingSystem.IsMacOS())
            return MacHost();
    }
    catch
    {
        // host facts are optional
    }
    return (null, null);
}

static (string? CpuName, double? MemoryMb) WindowsHost()
{
    string? name = null;
    try
    {
        using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(
            @"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
        name = (key?.GetValue("ProcessorNameString") as string)?.Trim();
    }
    catch { /* ignore */ }

    double? memoryMb = HostNative.TryPhysicalMemoryMb();
    return (EmptyToNull(name), memoryMb);
}

static (string? CpuName, double? MemoryMb) LinuxHost()
{
    string? model = null, hardware = null, part = null;
    if (File.Exists("/proc/cpuinfo"))
    {
        foreach (string line in File.ReadLines("/proc/cpuinfo"))
        {
            if (KeyValue(line, "model name", ref model)) continue;
            if (KeyValue(line, "Hardware", ref hardware)) continue;
            if (KeyValue(line, "CPU part", ref part)) continue;
        }
    }
    string? name = EmptyToNull(model) ?? EmptyToNull(hardware)
        ?? (part is null ? null : "CPU part " + part);

    double? memoryMb = null;
    if (File.Exists("/proc/meminfo"))
    {
        foreach (string line in File.ReadLines("/proc/meminfo"))
        {
            if (!line.StartsWith("MemTotal:", StringComparison.Ordinal)) continue;
            string rest = line["MemTotal:".Length..].Trim();
            int space = rest.IndexOf(' ');
            if (ulong.TryParse(space < 0 ? rest : rest[..space], out ulong kb) && kb > 0)
                memoryMb = kb / 1024d;
            break;
        }
    }
    return (name, memoryMb);
}

static (string? CpuName, double? MemoryMb) MacHost()
{
    string? name = EmptyToNull(Sysctl("machdep.cpu.brand_string"));
    double? memoryMb = null;
    if (ulong.TryParse(Sysctl("hw.memsize"), out ulong bytes) && bytes > 0)
        memoryMb = bytes / (1024d * 1024d);
    return (name, memoryMb);
}

static bool KeyValue(string line, string key, ref string? into)
{
    if (into is not null) return false;
    if (!line.StartsWith(key, StringComparison.OrdinalIgnoreCase)) return false;
    int colon = line.IndexOf(':');
    if (colon < 0) return false;
    into = line[(colon + 1)..].Trim();
    return into.Length > 0;
}

static string? EmptyToNull(string? s) => string.IsNullOrWhiteSpace(s) ? null : s.Trim();

static string? Sysctl(string name)
{
    try
    {
        using var p = Process.Start(new ProcessStartInfo("sysctl", "-n " + name)
        {
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        })!;
        string s = p.StandardOutput.ReadToEnd().Trim();
        p.WaitForExit(3000);
        return p.ExitCode == 0 && s.Length > 0 ? s : null;
    }
    catch
    {
        return null;
    }
}

static string? GitCommit()
{
    try
    {
        using var p = Process.Start(new ProcessStartInfo("git", "rev-parse --short HEAD")
        {
            WorkingDirectory = Environment.CurrentDirectory,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        })!;
        string s = p.StandardOutput.ReadToEnd().Trim();
        p.WaitForExit(5000);
        return p.ExitCode == 0 && s.Length > 0 ? s : null;
    }
    catch
    {
        return null;
    }
}

static class HostNative
{
    public static double? TryPhysicalMemoryMb()
    {
        if (GetPhysicallyInstalledSystemMemory(out ulong kb) && kb > 0)
            return kb / 1024d;
        var status = new MemoryStatusEx { Length = (uint)Marshal.SizeOf<MemoryStatusEx>() };
        if (GlobalMemoryStatusEx(ref status) && status.TotalPhys > 0)
            return status.TotalPhys / (1024d * 1024d);
        return null;
    }

    [DllImport("kernel32.dll")]
    static extern bool GetPhysicallyInstalledSystemMemory(out ulong memoryInKilobytes);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx buffer);

    [StructLayout(LayoutKind.Sequential)]
    struct MemoryStatusEx
    {
        public uint Length;
        public uint MemoryLoad;
        public ulong TotalPhys;
        public ulong AvailPhys;
        public ulong TotalPageFile;
        public ulong AvailPageFile;
        public ulong TotalVirtual;
        public ulong AvailVirtual;
        public ulong AvailExtendedVirtual;
    }
}
