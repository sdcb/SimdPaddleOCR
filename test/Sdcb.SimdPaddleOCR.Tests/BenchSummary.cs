using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace Sdcb.SimdPaddleOCR.Tests;

sealed class BenchmarkMetric
{
    [JsonPropertyName("ms")]
    public double Ms { get; init; }

    [JsonPropertyName("calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? Calls { get; init; }
}

sealed class BenchmarkRow
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

sealed record BenchmarkAccuracy(
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

static class BenchSummary
{
    public sealed record RunResult(
        string? Label,
        JsonObject? Meta,
        List<(double Value, string? File)> Totals,
        Dictionary<string, List<double>> StageValues,
        Dictionary<string, List<double>> OperatorValues,
        Dictionary<string, List<double>> ConvValues,
        BenchmarkAccuracy? Accuracy);

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
        double? wsLoaded = meta["working_set_mb_loaded"]?.GetValue<double>();
        double? wsLast = meta["working_set_mb_last"]?.GetValue<double>();
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
        if (wsLoaded is { } loaded && wsLast is { } last)
            summary["working_set_mb_delta"] = last - loaded;

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

    public static JsonObject AccuracyNode(BenchmarkAccuracy accuracy)
    {
        var node = new JsonObject
        {
            ["exact_lines"] = accuracy.ExactLines,
            ["total_lines"] = accuracy.TotalLines,
            ["exact_img"] = accuracy.ExactImages,
            ["images"] = accuracy.Images,
            ["errors"] = accuracy.Errors,
            ["total_chars"] = accuracy.TotalChars,
            ["cer"] = accuracy.Cer,
            ["char_acc"] = accuracy.CharacterAccuracy,
        };
        if (accuracy.ClsCorrect is { } clsCorrect && accuracy.ClsTotal is { } clsTotal)
        {
            node["cls_correct"] = clsCorrect;
            node["cls_total"] = clsTotal;
        }
        return node;
    }

    public static RunResult Parse(string path, string? labelOverride, string? metadataPath = null)
    {
        JsonNode root = JsonNode.Parse(File.ReadAllText(path))!;
        string? label = labelOverride;
        JsonArray rows;
        JsonObject? meta = null;
        if (root is JsonObject obj)
        {
            meta = obj["meta"] as JsonObject;
            if (label == null && meta is not null)
                label = LabelFromMeta(meta) ?? Path.GetFileNameWithoutExtension(path);
            rows = obj["rows"]!.AsArray();
        }
        else
        {
            rows = root.AsArray();
        }
        label ??= Path.GetFileNameWithoutExtension(path);

        var totals = new List<(double, string?)>();
        var stageValues = new Dictionary<string, List<double>>();
        var operatorValues = new Dictionary<string, List<double>>();
        var convValues = new Dictionary<string, List<double>>();
        foreach (JsonNode? rowNode in rows)
        {
            if (rowNode is not JsonObject row) continue;
            if (row["warmup"]?.GetValue<bool>() == true) continue;
            if (row["total_ms"] is JsonNode total)
                totals.Add((total.GetValue<double>(), row["file"]?.GetValue<string>()));
            Collect(row["stage_ms"], stageValues);
            Collect(row["operator_ms"], operatorValues);
            Collect(row["conv_class_ms"], convValues);
        }

        BenchmarkAccuracy? accuracy = ReadAccuracy(meta)
            ?? ComputeAccuracy(rows, metadataPath ?? GuessMetadata(path));
        return new RunResult(label, meta, totals, stageValues, operatorValues, convValues, accuracy);
    }

    public static BenchmarkAccuracy? ComputeAccuracy(IReadOnlyList<BenchmarkRow> rows, string? metadataPath,
        IReadOnlyList<float[][]>? boxes = null)
        => ComputeAccuracy(rows.Select((r, i) => (r.File,
            PredLines(r, boxes is not null && i < boxes.Count ? boxes[i] : null))), metadataPath);

    public static BenchmarkAccuracy? ComputeAccuracy(JsonArray rows, string? metadataPath)
        => ComputeAccuracy(rows.OfType<JsonObject>().Select(row =>
        {
            string file = row["file"]?.GetValue<string>() ?? "";
            return (file, ReadPredictions(row));
        }), metadataPath);

    private static BenchmarkAccuracy? ComputeAccuracy(
        IEnumerable<(string File, List<PredLine> Predicted)> rows, string? metadataPath)
    {
        Dictionary<string, List<GtLine>>? groundTruth = LoadGroundTruth(metadataPath);
        if (groundTruth is null) return null;

        int exactLines = 0, totalLines = 0, exactImages = 0, images = 0;
        long errors = 0, totalChars = 0;
        int clsCorrect = 0, clsTotal = 0;
        foreach ((string file, List<PredLine> predicted) in rows)
        {
            if (!groundTruth.TryGetValue(file, out List<GtLine>? gtLines))
                continue;

            var remainingText = predicted.Select(p => p.Text).ToList();
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

            (int correct, int total) = ScoreCls(gtLines, predicted);
            clsCorrect += correct;
            clsTotal += total;
        }

        return new BenchmarkAccuracy(exactLines, totalLines, exactImages, images, errors, totalChars,
            clsCorrect, clsTotal);
    }

    private static List<PredLine> PredLines(BenchmarkRow row, float[][]? boxes)
    {
        var lines = new List<PredLine>(row.Texts.Length);
        for (int i = 0; i < row.Texts.Length; i++)
        {
            int rotation = i < row.Rotations.Length ? row.Rotations[i] : 0;
            float[]? box = boxes is { Length: > 0 } && i < boxes.Length ? boxes[i] : null;
            lines.Add(new PredLine(row.Texts[i], rotation, box));
        }
        return lines;
    }

    private readonly record struct GtLine(string Text, int ClsDegrees, float X0, float Y0, float X1, float Y1);
    private readonly record struct PredLine(string Text, int Rotation, float[]? Box);

    private static Dictionary<string, List<GtLine>>? LoadGroundTruth(string? metadataPath)
    {
        if (metadataPath is null || !File.Exists(metadataPath)) return null;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(metadataPath));
        var result = new Dictionary<string, List<GtLine>>(StringComparer.OrdinalIgnoreCase);
        foreach (JsonElement image in document.RootElement.GetProperty("images").EnumerateArray())
        {
            string file = image.GetProperty("file").GetString()!;
            var lines = new List<GtLine>();
            foreach (JsonElement line in image.GetProperty("lines").EnumerateArray())
            {
                string text = line.GetProperty("text").GetString() ?? "";
                if (!line.TryGetProperty("cls_degrees", out JsonElement clsEl))
                    throw new InvalidDataException($"{file} is missing required cls_degrees.");
                int clsDegrees = (int)Math.Round(clsEl.GetDouble());
                if (clsDegrees is not (0 or 180))
                    throw new InvalidDataException($"{file} cls_degrees must be 0 or 180, got {clsDegrees}.");
                float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
                if (line.TryGetProperty("bbox", out JsonElement bbox) && bbox.GetArrayLength() >= 4)
                {
                    x0 = bbox[0].GetSingle();
                    y0 = bbox[1].GetSingle();
                    x1 = bbox[2].GetSingle();
                    y1 = bbox[3].GetSingle();
                }
                lines.Add(new GtLine(text, clsDegrees, x0, y0, x1, y1));
            }
            result[file] = lines;
        }
        return result;
    }

    private static List<PredLine> ReadPredictions(JsonObject row)
    {
        var texts = new List<string>();
        if (row["texts"] is JsonArray textNodes)
            foreach (JsonNode? text in textNodes)
                texts.Add(text?.GetValue<string>() ?? "");
        int[] rotations = ReadIntArray(row["rotations"], texts.Count);
        float[][]? boxes = ReadBoxes(row["boxes"], texts.Count);
        var lines = new List<PredLine>(texts.Count);
        for (int i = 0; i < texts.Count; i++)
            lines.Add(new PredLine(texts[i], rotations[i], boxes?[i]));
        return lines;
    }

    private static int[] ReadIntArray(JsonNode? node, int count)
    {
        int[] values = new int[count];
        if (node is not JsonArray array) return values;
        for (int i = 0; i < count && i < array.Count; i++)
            values[i] = array[i]?.GetValue<int>() ?? 0;
        return values;
    }

    private static float[][]? ReadBoxes(JsonNode? node, int count)
    {
        if (node is not JsonArray array || array.Count == 0) return null;
        var boxes = new float[count][];
        for (int i = 0; i < count; i++)
        {
            if (i < array.Count && array[i] is JsonArray box && box.Count >= 4)
            {
                boxes[i] =
                [
                    box[0]!.GetValue<float>(),
                    box[1]!.GetValue<float>(),
                    box[2]!.GetValue<float>(),
                    box[3]!.GetValue<float>(),
                ];
            }
        }
        return boxes;
    }

    /// <summary>
    /// Every GT line with required <c>cls_degrees</c> (0/180 after DET unwarp).
    /// Prefer IoU against predicted AABBs; fall back to exact-text pairing.
    /// </summary>
    private static (int Correct, int Total) ScoreCls(List<GtLine> gtLines, List<PredLine> predicted)
    {
        if (gtLines.Count == 0 || predicted.Count == 0) return (0, 0);

        bool useBoxes = predicted.Any(p => p.Box is { Length: >= 4 })
            && gtLines.Any(g => g.X1 > g.X0 && g.Y1 > g.Y0);
        if (useBoxes)
            return ScoreClsByBox(gtLines, predicted);
        return ScoreClsByText(gtLines, predicted);
    }

    private static (int Correct, int Total) ScoreClsByBox(List<GtLine> gtLines, List<PredLine> predicted)
    {
        const float minIou = 0.3f;
        var pairs = new List<(int Gi, int Pi, float Iou)>();
        for (int gi = 0; gi < gtLines.Count; gi++)
        {
            GtLine g = gtLines[gi];
            for (int pi = 0; pi < predicted.Count; pi++)
            {
                float[]? box = predicted[pi].Box;
                if (box is not { Length: >= 4 }) continue;
                float iou = AabbIou(g.X0, g.Y0, g.X1, g.Y1, box[0], box[1], box[2], box[3]);
                if (iou >= minIou) pairs.Add((gi, pi, iou));
            }
        }
        pairs.Sort((a, b) => b.Iou.CompareTo(a.Iou));
        var gtUsed = new bool[gtLines.Count];
        var predUsed = new bool[predicted.Count];
        int correct = 0, total = 0;
        foreach ((int gi, int pi, _) in pairs)
        {
            if (gtUsed[gi] || predUsed[pi]) continue;
            gtUsed[gi] = predUsed[pi] = true;
            total++;
            if (predicted[pi].Rotation == gtLines[gi].ClsDegrees) correct++;
        }
        return (correct, total);
    }

    private static (int Correct, int Total) ScoreClsByText(List<GtLine> gtLines, List<PredLine> predicted)
    {
        var remaining = predicted.ToList();
        int correct = 0, total = 0;
        foreach (GtLine gt in gtLines)
        {
            int index = remaining.FindIndex(p => p.Text == gt.Text);
            if (index < 0) continue;
            total++;
            if (remaining[index].Rotation == gt.ClsDegrees) correct++;
            remaining.RemoveAt(index);
        }
        return (correct, total);
    }

    private static float AabbIou(float ax0, float ay0, float ax1, float ay1,
        float bx0, float by0, float bx1, float by1)
    {
        float ix0 = Math.Max(ax0, bx0), iy0 = Math.Max(ay0, by0);
        float ix1 = Math.Min(ax1, bx1), iy1 = Math.Min(ay1, by1);
        float iw = ix1 - ix0, ih = iy1 - iy0;
        if (iw <= 0 || ih <= 0) return 0;
        float inter = iw * ih;
        float areaA = Math.Max(0, ax1 - ax0) * Math.Max(0, ay1 - ay0);
        float areaB = Math.Max(0, bx1 - bx0) * Math.Max(0, by1 - by0);
        float union = areaA + areaB - inter;
        return union <= 0 ? 0 : inter / union;
    }

    public static (double Mean, double Median, double P95) Stats(IReadOnlyList<double> values)
    {
        if (values.Count == 0) return (0, 0, 0);
        double[] s = values.OrderBy(x => x).ToArray();
        int n = s.Length;
        double median = n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
        double p95 = s[Math.Min(n - 1, (int)(0.95 * (n - 1) + 0.5))];
        return (s.Average(), median, p95);
    }

    public static void Print(RunResult run)
    {
        var (mean, median, p95) = Stats(run.Totals.Select(t => t.Value).ToArray());
        Console.WriteLine();
        Console.WriteLine($"=== summary: {run.Label} (n={run.Totals.Count}, timing excl warmup) ===");
        Console.WriteLine($"total_ms mean={mean:F1} median={median:F1} p95={p95:F1}");
        if (run.Accuracy is { } accuracy)
        {
            Console.WriteLine(
                $"accuracy" +
                (accuracy.ClsTotal is { } clsTotal
                    ? $" cls={accuracy.ClsCorrect}/{clsTotal}"
                    : "") +
                $" exact_lines={accuracy.ExactLines}/{accuracy.TotalLines} " +
                $"({Percent(accuracy.ExactLines, accuracy.TotalLines):F2}%) " +
                $"exact_img={accuracy.ExactImages}/{accuracy.Images} " +
                $"({Percent(accuracy.ExactImages, accuracy.Images):F2}%) " +
                $"CER={accuracy.Cer * 100:F2}% char_acc={accuracy.CharacterAccuracy * 100:F2}%");
        }
        if (run.StageValues.Count > 0)
        {
            Console.WriteLine("stage".PadRight(20) + " " + "mean".PadLeft(8));
            foreach (string k in OrderedKeys(run.StageValues))
                Console.WriteLine($"{k,-20} {run.StageValues[k].Average(),8:F2}");
        }
        if (run.OperatorValues.Count > 0)
        {
            Console.WriteLine("operator".PadRight(20) + " " + "mean".PadLeft(8));
            foreach (string k in run.OperatorValues.OrderByDescending(kv => kv.Value.Average()).Select(kv => kv.Key))
                Console.WriteLine($"{k,-20} {run.OperatorValues[k].Average(),8:F2}");
        }
        if (run.ConvValues.Count > 0)
        {
            Console.WriteLine("conv_class".PadRight(20) + " " + "mean".PadLeft(8));
            foreach (string k in run.ConvValues.OrderByDescending(kv => kv.Value.Average()).Select(kv => kv.Key))
                Console.WriteLine($"{k,-20} {run.ConvValues[k].Average(),8:F2}");
        }
    }

    public static int Compare(string[] args)
    {
        var paths = new List<string>();
        string? outMd = null, input = null;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--out-md": outMd = args[++i]; break;
                case "--input": input = args[++i]; break;
                default: paths.Add(args[i]); break;
            }
        }
        if (paths.Count == 0)
        {
            Console.Error.WriteLine("usage: --summarize <file...> [--input <dataset>] [--out-md <path>]");
            return 2;
        }

        string? metadataPath = input is null ? null : Path.Combine(input, "metadata.json");
        var runs = paths.Select(p => Parse(Resolve(p), null, metadataPath)).ToList();
        var sb = new System.Text.StringBuilder();
        void Emit(string s = "") { Console.WriteLine(s); sb.AppendLine(s); }

        Emit();
        Emit("=== e2e (ms/request, excl warmup) ===");
        Emit("run".PadRight(36) + " " + "n".PadLeft(4) + " " + "mean".PadLeft(8) + " " + "median".PadLeft(8) + " " + "p95".PadLeft(8) + " " + "img/s".PadLeft(8));
        foreach (RunResult r in runs)
        {
            var (m, md, p) = Stats(r.Totals.Select(t => t.Value).ToArray());
            double tps = m > 0 ? 1000.0 / m : 0;
            Emit($"{r.Label,-36} {r.Totals.Count,4} {m,8:F1} {md,8:F1} {p,8:F1} {tps,8:F2}");
        }

        if (runs.Any(r => r.Accuracy is not null))
        {
            Emit();
            Emit("=== accuracy (GT, all images) ===");
            Emit("run".PadRight(36) + " " + "cls".PadLeft(12) + " " + "exact_lines".PadLeft(16) + " " +
                "exact_img".PadLeft(12) + " " + "CER".PadLeft(8) + " " + "char_acc".PadLeft(9));
            foreach (RunResult r in runs)
            {
                if (r.Accuracy is not { } accuracy)
                {
                    Emit($"{r.Label,-36} {"-",12} {"-",16} {"-",12} {"-",8} {"-",9}");
                    continue;
                }
                string cls = accuracy.ClsTotal is { } clsTotal
                    ? $"{accuracy.ClsCorrect}/{clsTotal}" : "-";
                Emit($"{r.Label,-36} {cls,12} {accuracy.ExactLines + "/" + accuracy.TotalLines,16} " +
                    $"{accuracy.ExactImages + "/" + accuracy.Images,12} " +
                    $"{accuracy.Cer * 100,7:F2}% {accuracy.CharacterAccuracy * 100,8:F2}%");
            }
        }

        if (runs.Any(r => r.StageValues.Count > 0))
        {
            Emit();
            var header = new System.Text.StringBuilder($"{"stage",-20}");
            foreach (RunResult r in runs) header.Append($" {Short(r.Label!),10}");
            Emit(header.ToString());
            foreach (string stage in OrderedUnion(runs.Select(r => r.StageValues)))
            {
                var line = new System.Text.StringBuilder($"{stage,-20}");
                foreach (RunResult r in runs)
                {
                    double? v = MeanOf(r.StageValues, stage);
                    line.Append(v.HasValue ? $" {v.Value,10:F2}" : $" {"-",10}");
                }
                Emit(line.ToString());
            }
        }

        if (outMd != null)
        {
            File.WriteAllText(outMd, sb.ToString());
            Console.WriteLine($"markdown written: {outMd}");
        }
        return 0;
    }

    public static string? LabelFromMeta(JsonObject meta)
    {
        string rid = meta["rid"]?.GetValue<string>() ?? "";
        string engine = meta["mode"]?.GetValue<string>() ?? "sharp";
        string model = meta["model"]?.GetValue<string>() ?? "";
        int? workers = meta["workers"]?.GetValue<int>();
        string simd = SimdSuffix(meta);
        if (rid.Length == 0 && model.Length == 0) return null;
        return $"{rid} {engine} {model} {workers}w{simd}".Trim();
    }

    public static string SimdSuffix(JsonObject meta)
    {
        if (IsNetStandard20(meta["libraryTfm"])) return " ns2";
        if (IsOff(meta["hwintrinsic"])) return " scalar";
        if (IsOff(meta["avx"])) return " noavx";
        if (IsOff(meta["avx2"])) return " noavx2";
        if (IsOff(meta["avx512"])) return " noavx512";
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

    private static BenchmarkAccuracy? ReadAccuracy(JsonObject? meta)
    {
        if (meta?["accuracy"] is not JsonObject a) return null;
        int exactLines = a["exact_lines"]?.GetValue<int>() ?? 0;
        int totalLines = a["total_lines"]?.GetValue<int>() ?? 0;
        int exactImg = a["exact_img"]?.GetValue<int>() ?? 0;
        int images = a["images"]?.GetValue<int>() ?? 0;
        long errors = a["errors"]?.GetValue<long>() ?? 0;
        long totalChars = a["total_chars"]?.GetValue<long>() ?? 0;
        int? clsCorrect = a["cls_correct"]?.GetValue<int>();
        int? clsTotal = a["cls_total"]?.GetValue<int>();
        return new BenchmarkAccuracy(exactLines, totalLines, exactImg, images, errors, totalChars,
            clsCorrect, clsTotal);
    }

    private static string? GuessMetadata(string resultPath)
    {
        string? dir = Path.GetDirectoryName(Path.GetFullPath(resultPath));
        if (dir is null) return null;
        string beside = Path.Combine(dir, "metadata.json");
        return File.Exists(beside) ? beside : null;
    }

    private static string Resolve(string path)
    {
        if (File.Exists(path)) return path;
        throw new FileNotFoundException($"result file not found: {path}");
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

    private static double? MeanOf(Dictionary<string, List<double>> dict, string key)
        => dict.TryGetValue(key, out List<double>? v) && v.Count > 0 ? v.Average() : null;

    private static double Percent(int numerator, int denominator)
        => denominator > 0 ? numerator * 100.0 / denominator : 0;

    private static string Short(string label)
        => label.Length <= 10 ? label : label[..10];

    private static IEnumerable<string> OrderedKeys(Dictionary<string, List<double>> dict)
    {
        var rest = dict.Keys.ToHashSet();
        foreach (string k in StageOrder)
            if (rest.Remove(k)) yield return k;
        foreach (string k in rest.OrderBy(k => k)) yield return k;
    }

    private static IEnumerable<string> OrderedUnion(IEnumerable<Dictionary<string, List<double>>> dicts)
    {
        var rest = dicts.SelectMany(d => d.Keys).ToHashSet();
        foreach (string k in StageOrder)
            if (rest.Remove(k)) yield return k;
        foreach (string k in rest.OrderBy(k => k)) yield return k;
    }
}
