using System.Globalization;
using System.Text;

namespace Sdcb.SimdPaddleOCR.TestReport;

static class Markdown
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

    public static string Build(List<Run> runs)
    {
        var sb = new StringBuilder();
        string git = runs.Select(r => r.Git).FirstOrDefault(g => !string.IsNullOrEmpty(g)) ?? "n/a";
        string ts = DateTimeOffset.Now.ToString("yyyy-MM-dd HH:mm zzz", CultureInfo.InvariantCulture);
        sb.AppendLine("# Sdcb.SimdPaddleOCR CI report");
        sb.AppendLine();
        sb.AppendLine($"git `{git}` · generated {ts} · {runs.Count} runs · warmup excluded");
        sb.AppendLine();
        Section(sb, "All runs", runs);
        Section(sb, "tiny 4w across platforms", runs.Where(r => r.IsTiny4wDefault).ToList());
        Section(sb, "x64 SIMD Comparison",
            runs.Where(r => r.IsLinuxX64Tiny4w).OrderBy(r => r.SimdRank).ToList());
        Section(sb, "x64 Model Comparison",
            runs.Where(r => r.IsLinuxX64Model).OrderBy(r => r.ModelRank).ToList());
        SharpVsC(sb, runs.Where(r => r.IsWinX64TinyCSharpVsC).ToList());

        sb.AppendLine("## Details");
        sb.AppendLine();
        foreach (Run run in runs)
            Detail(sb, run);
        return sb.ToString();
    }

    private static void Section(StringBuilder sb, string title, List<Run> runs)
    {
        sb.AppendLine($"## {title}");
        sb.AppendLine();
        if (runs.Count == 0)
        {
            sb.AppendLine("No matching runs.");
            sb.AppendLine();
            return;
        }
        sb.AppendLine("| run | engine | RID | model | w | SIMD | n | mean | median | P95 | img/s | exact_lines | exact_img | CER | char_acc | WS loaded | WS last | WS peak | Δ WS |");
        sb.AppendLine("| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        foreach (Run r in runs)
        {
            double? delta = r.WsLast is { } last && r.WsLoaded is { } loaded ? last - loaded : null;
            sb.Append($"| {Cell(r.Label)} | {Cell(r.Engine)} | {Cell(r.Rid)} | {Cell(r.Model)} | {r.Workers} | {Cell(r.Simd.Length == 0 ? "default" : r.Simd)} | {r.N}");
            sb.Append($" | {r.Mean:F1} | {r.Median:F1} | {r.P95:F1} | {r.Throughput:F2}");
            sb.Append($" | {Frac(r.ExactLines, r.TotalLines)} | {Frac(r.ExactImages, r.Images)} | {Pct(r.Cer)} | {Pct(r.CharAcc)}");
            sb.Append($" | {Mb(r.WsLoaded)} | {Mb(r.WsLast)} | {Mb(r.WsPeak)} | {Mb(delta)}");
            sb.AppendLine(" |");
        }
        sb.AppendLine();
    }

    private static void Detail(StringBuilder sb, Run run)
    {
        sb.AppendLine($"### {run.Label}");
        sb.AppendLine();
        var bits = new List<string>();
        if (!string.IsNullOrEmpty(run.Os)) bits.Add(run.Os);
        if (!string.IsNullOrEmpty(run.CpuName)) bits.Add(run.CpuName);
        if (run.Cpu > 0) bits.Add($"{run.Cpu} logical processors");
        if (run.MemoryMb is { } mem) bits.Add($"{Ram(mem)} RAM");
        if (!string.IsNullOrEmpty(run.Timestamp)) bits.Add(run.Timestamp);
        if (bits.Count > 0)
        {
            sb.AppendLine(string.Join(" · ", bits));
            sb.AppendLine();
        }
        MetricTable(sb, "stage", StageOrder.Where(run.Stages.ContainsKey).Concat(run.Stages.Keys.Except(StageOrder).OrderBy(k => k)), run.Stages);
        MetricTable(sb, "operator", run.Operators.OrderByDescending(kv => kv.Value).Select(kv => kv.Key), run.Operators);
        MetricTable(sb, "conv_class", run.Conv.OrderByDescending(kv => kv.Value).Select(kv => kv.Key), run.Conv);
        if (run.Stages.Count == 0 && run.Operators.Count == 0 && run.Conv.Count == 0)
        {
            sb.AppendLine("No stage/operator breakdown (total latency and memory only).");
            sb.AppendLine();
        }
    }

    private static void SharpVsC(StringBuilder sb, List<Run> runs)
    {
        sb.AppendLine("## C# vs C (win-x64 tiny)");
        sb.AppendLine();
        var pairs = runs
            .GroupBy(r => r.Workers)
            .OrderBy(g => g.Key)
            .Select(g => (
                Workers: g.Key,
                Sharp: g.FirstOrDefault(r => r.IsSharp),
                C: g.FirstOrDefault(r => !r.IsSharp)))
            .Where(p => p.Sharp is not null || p.C is not null)
            .ToList();
        if (pairs.Count == 0)
        {
            sb.AppendLine("No matching runs.");
            sb.AppendLine();
            return;
        }

        sb.AppendLine("| w | engine | n | mean | median | P95 | img/s | C/C# | exact_lines | CER | WS loaded | WS last | WS peak | Δ WS |");
        sb.AppendLine("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        foreach (var (workers, sharp, c) in pairs)
        {
            AppendVsRow(sb, workers, sharp, c);
            AppendVsRow(sb, workers, c, sharp);
        }
        sb.AppendLine();
    }

    private static void AppendVsRow(StringBuilder sb, int? workers, Run? run, Run? other)
    {
        if (run is null) return;
        double? delta = run.WsLast is { } last && run.WsLoaded is { } loaded ? last - loaded : null;
        string ratio = "—";
        if (!run.IsSharp && other is not null && other.Mean > 0)
            ratio = (run.Mean / other.Mean).ToString("F2", CultureInfo.InvariantCulture);
        else if (run.IsSharp)
            ratio = "1.00";
        sb.Append($"| {workers} | {Cell(run.Engine)} | {run.N}");
        sb.Append($" | {run.Mean:F1} | {run.Median:F1} | {run.P95:F1} | {run.Throughput:F2}");
        sb.Append($" | {ratio} | {Frac(run.ExactLines, run.TotalLines)} | {Pct(run.Cer)}");
        sb.Append($" | {Mb(run.WsLoaded)} | {Mb(run.WsLast)} | {Mb(run.WsPeak)} | {Mb(delta)}");
        sb.AppendLine(" |");
    }

    private static void MetricTable(StringBuilder sb, string kind, IEnumerable<string> keys, Dictionary<string, double> values)
    {
        var list = keys.Where(values.ContainsKey).ToList();
        if (list.Count == 0) return;
        sb.AppendLine($"| {kind} | mean ms |");
        sb.AppendLine("| --- | ---: |");
        foreach (string key in list)
            sb.AppendLine($"| {Cell(key)} | {values[key]:F2} |");
        sb.AppendLine();
    }

    private static string Frac(int? num, int? den) =>
        num is null || den is null ? "—" : $"{num}/{den}";

    private static string Pct(double? v) =>
        v is null ? "—" : (v.Value * 100).ToString("F2", CultureInfo.InvariantCulture) + "%";

    private static string Mb(double? v) =>
        v is null ? "—" : v.Value.ToString("F1", CultureInfo.InvariantCulture);

    private static string Ram(double memoryMb) =>
        memoryMb >= 1024
            ? (memoryMb / 1024d).ToString("F1", CultureInfo.InvariantCulture) + " GB"
            : memoryMb.ToString("F0", CultureInfo.InvariantCulture) + " MB";

    private static string Cell(string? s) =>
        (s ?? "").Replace("|", "\\|", StringComparison.Ordinal);
}
