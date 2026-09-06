using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Tests;
using ImageSharpImage = SixLabors.ImageSharp.Image;

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("usage: --workers 1..16 --model tiny|small|medium --input <dataset> --out <json> [--engine sharp|c|openvino] [--c-assets <dir>]");
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
if (engineName is not ("sharp" or "c" or "openvino"))
    throw new ArgumentException("--engine must be sharp, c, or openvino");
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

using IBenchEngine engine = BenchEngines.Create(engineName, modelType, workers, cAssetsDir);
double wsLoaded = WorkingSetMb();
double wsPeak = wsLoaded;
Console.WriteLine(engine.LoadedMessage(wsLoaded));

List<BenchmarkRow> rows = [];
for (int index = 0; index < decoded.Length; index++)
{
    var d = decoded[index];
    Stopwatch sw = Stopwatch.StartNew();
    BenchEngineOutput result = engine.Run(d.Pixels, d.W, d.H, d.W * 3);
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
        StageMs = result.StageMs,
        StageCalls = result.StageCalls,
        OperatorMs = result.OperatorMs,
        ConvClassMs = result.ConvClassMs,
        Detected = result.Detected,
        Lines = result.Texts.Length,
        Hash = result.Hash,
        Texts = result.Texts,
        Rotations = result.Rotations,
        WorkingSetMb = ws,
    });
    string det = result.StageMs is { } stages && stages.TryGetValue("det_graph", out double detMs)
        ? $" det={detMs:F2}" : "";
    Console.WriteLine($"{index + 1}/100 {d.Name} total={sw.Elapsed.TotalMilliseconds:F3}{det} lines={result.Texts.Length} ws={ws:F1}MB");
}

JsonObject extra = engine.Extra;

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
meta["vectorCount"] = Vector<float>.Count;
meta["vectorHardwareAccelerated"] = Vector.IsHardwareAccelerated;
meta["framework"] = RuntimeInformation.FrameworkDescription;
meta["libraryTfm"] = typeof(PaddleOcrAll).Assembly
    .GetCustomAttribute<TargetFrameworkAttribute>()?.FrameworkName;

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
