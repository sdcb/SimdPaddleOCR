using System.Text.Json.Nodes;

namespace Sdcb.SimdPaddleOCR.TestReport;

sealed class Run
{
    public required string Path { get; init; }
    public required string Label { get; init; }
    public required string SortKey { get; init; }
    public string? Rid { get; init; }
    public string Engine { get; init; } = "sharp";
    public string? Os { get; init; }
    public string? Model { get; init; }
    public int? Workers { get; init; }
    public string Simd { get; init; } = "";
    public string? Git { get; init; }
    public string? Timestamp { get; init; }
    public int Cpu { get; init; }
    public string? CpuName { get; init; }
    public double? MemoryMb { get; init; }
    public int N { get; init; }
    public double Mean { get; init; }
    public double Median { get; init; }
    public double P95 { get; init; }
    public double Throughput => Mean > 0 ? 1000.0 / Mean : 0;
    public int? ExactLines { get; init; }
    public int? TotalLines { get; init; }
    public int? ExactImages { get; init; }
    public int? Images { get; init; }
    public double? Cer { get; init; }
    public double? CharAcc { get; init; }
    public double? WsLoaded { get; init; }
    public double? WsLast { get; init; }
    public double? WsPeak { get; init; }
    public Dictionary<string, double> Stages { get; init; } = [];
    public Dictionary<string, double> Operators { get; init; } = [];
    public Dictionary<string, double> Conv { get; init; } = [];

    public bool IsSharp => Engine != "c";

    public bool IsTiny4wDefault =>
        IsSharp && Model == "tiny" && Workers == 4 && Simd.Length == 0;

    public bool IsLinuxX64Tiny4w =>
        IsSharp && Rid == "linux-x64" && Model == "tiny" && Workers == 4;

    public bool IsLinuxX64Model =>
        IsSharp && Rid == "linux-x64" && Workers == 4 && Simd.Length == 0 &&
        Model is "tiny" or "small" or "medium";

    public bool IsWinX64TinyCSharpVsC =>
        Rid == "win-x64" && Model == "tiny" && Simd.Length == 0 &&
        Workers is 1 or 4;

    public int SimdRank => Simd switch
    {
        "" or "default" => 0,
        "noavx512" => 1,
        "noavx2" => 2,
        "noavx" => 3,
        "ns2" => 4,
        "scalar" or "nohw" => 5,
        _ => 6,
    };

    public int ModelRank => Model switch
    {
        "medium" => 0,
        "small" => 1,
        "tiny" => 2,
        _ => 3,
    };

    public static Run? TryLoad(string path)
    {
        JsonNode? root;
        try { root = JsonNode.Parse(File.ReadAllText(path)); }
        catch { return null; }
        if (root is null) return null;
        JsonObject? meta = root is JsonObject o ? o["meta"] as JsonObject : null;
        JsonArray? rows = root is JsonObject obj ? obj["rows"] as JsonArray : root as JsonArray;
        if (rows is null) return null;

        var totals = new List<double>();
        var stages = new Dictionary<string, List<double>>();
        var ops = new Dictionary<string, List<double>>();
        var conv = new Dictionary<string, List<double>>();
        var workingSets = new List<double>();
        foreach (JsonNode? rowNode in rows)
        {
            if (rowNode is not JsonObject row) continue;
            if (row["working_set_mb"] is JsonNode ws)
                workingSets.Add(ws.GetValue<double>());
            if (row["warmup"]?.GetValue<bool>() == true) continue;
            if (row["total_ms"] is JsonNode total)
                totals.Add(total.GetValue<double>());
            Collect(row["stage_ms"], stages);
            Collect(row["operator_ms"], ops);
            Collect(row["conv_class_ms"], conv);
        }

        var (mean, median, p95) = Stats(totals);
        string rid = meta?["rid"]?.GetValue<string>() ?? "";
        string engine = meta?["mode"]?.GetValue<string>() ?? "sharp";
        if (engine.Length == 0) engine = "sharp";
        string model = meta?["model"]?.GetValue<string>() ?? "";
        int? workers = meta?["workers"]?.GetValue<int>();
        string simd = SimdLabel(meta);
        string label = FormatLabel(rid, engine, model, workers, simd)
            ?? System.IO.Path.GetFileNameWithoutExtension(path);

        int? exactLines = null, totalLines = null, exactImg = null, images = null;
        double? cer = null, charAcc = null;
        if (meta?["accuracy"] is JsonObject a)
        {
            exactLines = a["exact_lines"]?.GetValue<int>();
            totalLines = a["total_lines"]?.GetValue<int>();
            exactImg = a["exact_img"]?.GetValue<int>();
            images = a["images"]?.GetValue<int>();
            cer = a["cer"]?.GetValue<double>();
            charAcc = a["char_acc"]?.GetValue<double>();
        }

        double? wsLoaded = Mb(meta?["working_set_mb_loaded"]);
        double? wsLast = Mb(meta?["working_set_mb_last"]) ?? (workingSets.Count > 0 ? workingSets[^1] : null);
        double? wsPeak = Mb(meta?["working_set_mb_peak"])
            ?? (workingSets.Count > 0 ? workingSets.Max() : null);

        return new Run
        {
            Path = path,
            Label = label,
            SortKey = $"{rid}|{engine}|{model}|{workers}|{simd}|{label}",
            Rid = string.IsNullOrEmpty(rid) ? null : rid,
            Engine = engine,
            Os = meta?["os"]?.GetValue<string>(),
            Model = string.IsNullOrEmpty(model) ? null : model,
            Workers = workers,
            Simd = simd,
            Git = meta?["gitCommit"]?.GetValue<string>(),
            Timestamp = meta?["timestamp"]?.GetValue<string>(),
            Cpu = meta?["cpu"]?.GetValue<int>() ?? 0,
            CpuName = meta?["cpuName"]?.GetValue<string>(),
            MemoryMb = Mb(meta?["memoryMb"]),
            N = totals.Count,
            Mean = mean,
            Median = median,
            P95 = p95,
            ExactLines = exactLines,
            TotalLines = totalLines,
            ExactImages = exactImg,
            Images = images,
            Cer = cer,
            CharAcc = charAcc,
            WsLoaded = wsLoaded,
            WsLast = wsLast,
            WsPeak = wsPeak,
            Stages = stages.ToDictionary(kv => kv.Key, kv => kv.Value.Average()),
            Operators = ops.ToDictionary(kv => kv.Key, kv => kv.Value.Average()),
            Conv = conv.ToDictionary(kv => kv.Key, kv => kv.Value.Average()),
        };
    }

    private static string? FormatLabel(string rid, string engine, string model, int? workers, string simd)
    {
        if (rid.Length == 0 && model.Length == 0) return null;
        string extra = simd.Length == 0 ? "" : " " + simd;
        return $"{rid} {engine} {model} {workers}w{extra}".Trim();
    }

    private static string SimdLabel(JsonObject? meta)
    {
        if (meta is null) return "";
        if (IsNetStandard20(meta["libraryTfm"])) return "ns2";
        if (IsOff(meta["hwintrinsic"])) return "scalar";
        if (IsOff(meta["avx"])) return "noavx";
        if (IsOff(meta["avx2"])) return "noavx2";
        if (IsOff(meta["avx512"])) return "noavx512";
        return "";
    }

    private static bool IsNetStandard20(JsonNode? node)
    {
        if (node is null) return false;
        try
        {
            string? value = node.GetValue<string>();
            return value is not null &&
                value.Contains("NETStandard", StringComparison.OrdinalIgnoreCase);
        }
        catch { return false; }
    }

    private static bool IsOff(JsonNode? node)
    {
        if (node is null) return false;
        try { return node.GetValue<string>() == "0"; }
        catch { return false; }
    }

    private static double? Mb(JsonNode? node)
    {
        if (node is null) return null;
        try { return node.GetValue<double>(); }
        catch { return null; }
    }

    private static void Collect(JsonNode? node, Dictionary<string, List<double>> into)
    {
        if (node is not JsonObject dict) return;
        foreach ((string key, JsonNode? value) in dict)
        {
            double ms = value switch
            {
                JsonObject o when o["ms"] != null => o["ms"]!.GetValue<double>(),
                JsonValue => value.GetValue<double>(),
                _ => double.NaN,
            };
            if (double.IsNaN(ms)) continue;
            if (!into.TryGetValue(key, out List<double>? list))
                into[key] = list = [];
            list.Add(ms);
        }
    }

    private static (double Mean, double Median, double P95) Stats(List<double> values)
    {
        if (values.Count == 0) return (0, 0, 0);
        double[] s = values.OrderBy(x => x).ToArray();
        int n = s.Length;
        double median = n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
        double p95 = s[Math.Min(n - 1, (int)(0.95 * (n - 1) + 0.5))];
        return (s.Average(), median, p95);
    }
}
