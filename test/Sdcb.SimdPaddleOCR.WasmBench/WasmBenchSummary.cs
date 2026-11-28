using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace Sdcb.SimdPaddleOCR.WasmBench;

public sealed class BenchmarkMetric
{
    [JsonPropertyName("ms")]
    public double Ms { get; init; }

    [JsonPropertyName("calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? Calls { get; init; }
}

public sealed class BenchmarkRow
{
    [JsonPropertyName("file")] public string File { get; init; } = "";
    [JsonPropertyName("width")] public int Width { get; init; }
    [JsonPropertyName("height")] public int Height { get; init; }
    [JsonPropertyName("warmup")] public bool Warmup { get; init; }
    [JsonPropertyName("total_ms")] public double TotalMs { get; init; }
    [JsonPropertyName("stage_ms")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, double>? StageMs { get; init; }
    [JsonPropertyName("stage_calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, long>? StageCalls { get; init; }
    [JsonPropertyName("operator_ms")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, BenchmarkMetric>? OperatorMs { get; init; }
    [JsonPropertyName("conv_class_ms")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, BenchmarkMetric>? ConvClassMs { get; init; }
    [JsonPropertyName("detected")] public int Detected { get; init; }
    [JsonPropertyName("lines")] public int Lines { get; init; }
    [JsonPropertyName("hash")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Hash { get; init; }
    [JsonPropertyName("texts")] public string[] Texts { get; init; } = [];
    [JsonPropertyName("rotations")] public int[] Rotations { get; init; } = [];
    [JsonPropertyName("working_set_mb")] public double WorkingSetMb { get; init; }
}

public sealed record BenchmarkAccuracy(
    int ExactLines,
    int TotalLines,
    int ExactImages,
    int Images,
    long Errors,
    long TotalChars,
    int? ClsCorrect = null,
    int? ClsTotal = null)
{
    public double Cer => TotalChars > 0 ? (double)Errors / TotalChars : 0;
    public double CharacterAccuracy => 1 - Cer;
}

[JsonSerializable(typeof(List<BenchmarkRow>))]
internal partial class BenchJsonContext : JsonSerializerContext
{
}

public static class WasmBenchSummary
{
    private static readonly string[] StageOrder =
    [
        "det_preprocess", "det_graph", "det_postprocess", "det_unclip", "crop",
        "cls_preprocess", "cls_graph", "cls_postprocess",
        "rec_preprocess", "rec_graph", "rec_postprocess",
        "lines_wall", "crop_setup",
        "rec_acquire", "cls_acquire", "rec_decode",
        "rec_release", "cls_release",
        "rec_cache_get", "rec_rent", "rec_pool", "rec_reshape",
    ];

    public static JsonObject WrapWithMeta(List<BenchmarkRow> rows, JsonObject meta) => new()
    {
        ["meta"] = meta,
        ["summary"] = BuildSummary(rows, meta),
        ["rows"] = JsonSerializer.SerializeToNode(rows, BenchJsonContext.Default.ListBenchmarkRow)!.AsArray(),
    };

    public static JsonObject BuildSummary(List<BenchmarkRow> rows, JsonObject meta)
    {
        List<BenchmarkRow> warmups = rows.Where(r => r.Warmup).ToList();
        List<BenchmarkRow> measured = rows.Where(r => !r.Warmup).ToList();
        double[] totals = measured.Select(r => r.TotalMs).ToArray();
        var (mean, median, p95) = Stats(totals);
        BenchmarkRow? slowest = measured.Count == 0 ? null : measured.MaxBy(r => r.TotalMs);
        int lineTotal = measured.Sum(r => r.Lines);

        var summary = new JsonObject
        {
            ["n"] = measured.Count,
            ["warmup"] = warmups.Count,
            ["total_ms"] = new JsonObject
            {
                ["mean"] = mean,
                ["median"] = median,
                ["p95"] = p95,
                ["min"] = totals.Length == 0 ? 0 : totals.Min(),
                ["max"] = totals.Length == 0 ? 0 : totals.Max(),
                ["sum"] = totals.Length == 0 ? 0 : totals.Sum(),
            },
            ["img_per_s"] = mean > 0 ? 1000.0 / mean : 0,
            ["lines"] = new JsonObject
            {
                ["mean"] = measured.Count == 0 ? 0 : (double)lineTotal / measured.Count,
                ["total"] = lineTotal,
            },
        };

        if (warmups.Count > 0)
            summary["warmup_ms"] = warmups.Average(r => r.TotalMs);
        if (slowest is not null)
        {
            summary["slowest"] = new JsonObject
            {
                ["file"] = slowest.File,
                ["total_ms"] = slowest.TotalMs,
            };
        }

        JsonObject stageMeans = MeanDict(measured.Select(r => r.StageMs));
        if (stageMeans.Count > 0)
            summary["stage_ms_mean"] = stageMeans;
        JsonObject operatorMeans = MeanMetricDict(measured.Select(r => r.OperatorMs));
        if (operatorMeans.Count > 0)
            summary["operator_ms_mean"] = operatorMeans;
        JsonObject convMeans = MeanMetricDict(measured.Select(r => r.ConvClassMs));
        if (convMeans.Count > 0)
            summary["conv_class_ms_mean"] = convMeans;
        return summary;
    }

    private static (double Mean, double Median, double P95) Stats(double[] values)
    {
        if (values.Length == 0) return (0, 0, 0);
        Array.Sort(values);
        double mean = values.Average();
        double median = values.Length % 2 == 1
            ? values[values.Length / 2]
            : (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2.0;
        int p95Index = (int)Math.Ceiling(values.Length * 0.95) - 1;
        p95Index = Math.Clamp(p95Index, 0, values.Length - 1);
        return (mean, median, values[p95Index]);
    }

    private static JsonObject MeanDict(IEnumerable<Dictionary<string, double>?> sources)
    {
        var sums = new Dictionary<string, (double Sum, int Count)>();
        foreach (Dictionary<string, double>? dict in sources)
        {
            if (dict is null) continue;
            foreach ((string key, double value) in dict)
            {
                sums.TryGetValue(key, out (double Sum, int Count) cur);
                sums[key] = (cur.Sum + value, cur.Count + 1);
            }
        }
        var result = new JsonObject();
        foreach (string key in OrderedSummaryKeys(sums.Keys.ToHashSet()))
            result[key] = sums[key].Sum / sums[key].Count;
        return result;
    }

    private static JsonObject MeanMetricDict(IEnumerable<Dictionary<string, BenchmarkMetric>?> sources)
    {
        var sums = new Dictionary<string, (double Sum, int Count)>();
        foreach (Dictionary<string, BenchmarkMetric>? dict in sources)
        {
            if (dict is null) continue;
            foreach ((string key, BenchmarkMetric metric) in dict)
            {
                sums.TryGetValue(key, out (double Sum, int Count) cur);
                sums[key] = (cur.Sum + metric.Ms, cur.Count + 1);
            }
        }
        var result = new JsonObject();
        foreach ((string key, (double Sum, int Count) value) in
            sums.OrderByDescending(kv => kv.Value.Sum / kv.Value.Count))
            result[key] = value.Sum / value.Count;
        return result;
    }

    private static IEnumerable<string> OrderedSummaryKeys(HashSet<string> rest)
    {
        foreach (string k in StageOrder)
            if (rest.Remove(k)) yield return k;
        foreach (string k in rest.OrderBy(k => k)) yield return k;
    }

    public static JsonObject AccuracyNode(BenchmarkAccuracy accuracy) => new()
    {
        ["exact_lines"] = accuracy.ExactLines,
        ["total_lines"] = accuracy.TotalLines,
        ["exact_img"] = accuracy.ExactImages,
        ["images"] = accuracy.Images,
        ["errors"] = accuracy.Errors,
        ["total_chars"] = accuracy.TotalChars,
        ["cer"] = accuracy.Cer,
        ["char_accuracy"] = accuracy.CharacterAccuracy,
        ["cls_correct"] = accuracy.ClsCorrect,
        ["cls_total"] = accuracy.ClsTotal,
    };

    private readonly record struct GtLine(string Text, int ClsDegrees);
    private readonly record struct PredLine(string Text, int Rotation);

    public static BenchmarkAccuracy? ComputeAccuracy(IReadOnlyList<BenchmarkRow> rows, string? metadataJson)
    {
        if (string.IsNullOrWhiteSpace(metadataJson)) return null;
        Dictionary<string, List<GtLine>> groundTruth = LoadGroundTruth(metadataJson);

        int exactLines = 0, totalLines = 0, exactImages = 0, images = 0;
        long errors = 0, totalChars = 0;
        int clsCorrect = 0, clsTotal = 0;

        // Warmup only drops rows from the timing summary; accuracy covers every
        // image, as in the desktop runner (100 images / 1036 lines).
        foreach (BenchmarkRow row in rows)
        {
            if (!groundTruth.TryGetValue(row.File, out List<GtLine>? gtLines))
                continue;

            var remainingText = row.Texts.ToList();
            images++;
            bool imageExact = true;
            foreach (GtLine expected in gtLines)
            {
                totalLines++;
                totalChars += CodePointLength(expected.Text);
                if (remainingText.Remove(expected.Text))
                {
                    exactLines++;
                    continue;
                }

                imageExact = false;
                int best = CodePointLength(expected.Text);
                // Like evaluate.py: only predictions not already consumed by an
                // exact match, so one predicted line cannot satisfy two GT lines.
                foreach (string actual in remainingText)
                    best = Math.Min(best, Levenshtein(expected.Text, actual));
                errors += best;
            }
            if (imageExact) exactImages++;
        }

        return new BenchmarkAccuracy(exactLines, totalLines, exactImages, images, errors, totalChars,
            clsCorrect, clsTotal);
    }

    private static Dictionary<string, List<GtLine>> LoadGroundTruth(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        var result = new Dictionary<string, List<GtLine>>(StringComparer.OrdinalIgnoreCase);
        foreach (JsonElement image in document.RootElement.GetProperty("images").EnumerateArray())
        {
            string file = image.GetProperty("file").GetString()!;
            var lines = new List<GtLine>();
            foreach (JsonElement line in image.GetProperty("lines").EnumerateArray())
            {
                string text = line.GetProperty("text").GetString() ?? "";
                int clsDegrees = 0;
                if (line.TryGetProperty("cls_degrees", out JsonElement clsEl))
                    clsDegrees = (int)Math.Round(clsEl.GetDouble());
                lines.Add(new GtLine(text, clsDegrees));
            }
            result[file] = lines;
        }
        return result;
    }

    // Matches the dataset's scripts/evaluate.py: distances and lengths count
    // Unicode code points, not UTF-16 units (the small/medium dictionaries
    // contain supplementary-plane characters).
    private static int CodePointLength(string s)
    {
        int n = 0;
        for (int i = 0; i < s.Length; i++, n++)
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])) i++;
        return n;
    }

    private static int[] CodePoints(string s)
    {
        int[] cps = new int[CodePointLength(s)];
        for (int i = 0, k = 0; i < s.Length; i++, k++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                cps[k] = char.ConvertToUtf32(s[i], s[i + 1]);
                i++;
            }
            else cps[k] = s[i];
        }
        return cps;
    }

    private static int Levenshtein(string left, string right)
    {
        if (left == right) return 0;
        int[] a = CodePoints(left), b = CodePoints(right);
        if (a.Length == 0) return b.Length;
        if (b.Length == 0) return a.Length;
        int[] previous = new int[b.Length + 1];
        int[] current = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++) previous[j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            for (int j = 1; j <= b.Length; j++)
            {
                int substitute = previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
                current[j] = Math.Min(Math.Min(previous[j] + 1, current[j - 1] + 1), substitute);
            }
            (previous, current) = (current, previous);
        }
        return previous[b.Length];
    }

    public static void Print(JsonObject doc)
    {
        JsonObject? meta = doc["meta"]?.AsObject();
        JsonObject? summary = doc["summary"]?.AsObject();
        if (summary is null) return;

        double mean = summary["total_ms"]?["mean"]?.GetValue<double>() ?? 0;
        double median = summary["total_ms"]?["median"]?.GetValue<double>() ?? 0;
        double p95 = summary["total_ms"]?["p95"]?.GetValue<double>() ?? 0;
        double imgPerSec = summary["img_per_s"]?.GetValue<double>() ?? 0;
        int n = summary["n"]?.GetValue<int>() ?? 0;

        Console.WriteLine("\n================== WASM BENCHMARK RESULT ==================");
        Console.WriteLine($"Case:         {meta?["caseId"]?.GetValue<string>() ?? "wasm-tiny"}");
        Console.WriteLine($"Workers:      {meta?["workers"]?.GetValue<int>()} (effective: {meta?["effectiveWorkers"]?.GetValue<int>()})");
        Console.WriteLine($"Hardware CPU: {meta?["processorCount"]?.GetValue<int>()} cores");
        Console.WriteLine($"WASM SIMD:    {meta?["effectiveIsa"]?.GetValue<string>()}");
        Console.WriteLine($"Images:       {n} samples");
        Console.WriteLine($"Mean:         {mean:F1} ms/img ({imgPerSec:F2} img/s)");
        Console.WriteLine($"Median:       {median:F1} ms");
        Console.WriteLine($"P95:          {p95:F1} ms");

        if (meta?["accuracy"] is JsonObject acc)
        {
            double cer = acc["cer"]?.GetValue<double>() ?? 0;
            double charAcc = acc["char_accuracy"]?.GetValue<double>() ?? 0;
            int exactLines = acc["exact_lines"]?.GetValue<int>() ?? 0;
            int totalLines = acc["total_lines"]?.GetValue<int>() ?? 0;
            Console.WriteLine($"Char Accuracy: {charAcc * 100:F2}% (CER: {cer * 100:F2}%)");
            Console.WriteLine($"Line Exact:    {exactLines}/{totalLines} ({(totalLines > 0 ? exactLines * 100.0 / totalLines : 0):F2}%)");
        }

        if (summary["stage_ms_mean"] is JsonObject stages)
        {
            Console.WriteLine("\n--- Stage Breakdown (Mean ms) ---");
            foreach (var kv in stages)
            {
                if (kv.Value is JsonValue val && val.TryGetValue<double>(out double ms) && ms > 0.05)
                {
                    Console.WriteLine($"  {kv.Key,-18}: {ms,6:F1} ms");
                }
            }
        }
        Console.WriteLine("===========================================================\n");
    }
}
