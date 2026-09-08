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
        SmokeSection(sb, "Platform smoke", runs.Where(r => r.IsSmoke).ToList());
        BenchSections(sb, "SIMD / ISA benchmarks", runs.Where(r => r.IsSimdBenchmark).ToList());
        BenchSections(sb, "Engine comparison benchmarks", runs.Where(r => r.IsEngineBenchmark).ToList());
        sb.AppendLine("## Benchmark details");
        sb.AppendLine();
        foreach (Run run in runs.Where(r => r.IsBenchmark).OrderBy(r => r.Rid).ThenBy(r => r.Machine).ThenBy(r => r.Replica).ThenBy(r => r.Label))
            Detail(sb, run);

        return sb.ToString();
    }

    private static void SmokeSection(StringBuilder sb, string title, List<Run> runs)
    {
        sb.AppendLine($"## {title}");
        sb.AppendLine();
        if (runs.Count == 0) { sb.AppendLine("No matching runs."); sb.AppendLine(); return; }
        sb.AppendLine("| run | RID | engine | model | w | samples | effective ISA | median ms | accuracy | CPU |");
        sb.AppendLine("| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |");
        foreach (Run r in runs.OrderBy(r => r.Rid).ThenBy(r => r.Model).ThenBy(r => r.Label))
            sb.AppendLine($"| {Cell(r.Label)} | {Cell(r.Rid)} | {Cell(r.Engine)} | {Cell(r.Model)} | {r.Workers} | {r.N} | {Cell(r.EffectiveIsa)} | {r.Median:F1} | {Frac(r.ExactLines, r.TotalLines)} | {Cell(r.CpuName)} |");
        sb.AppendLine();
    }

    private static void BenchSections(StringBuilder sb, string title, List<Run> runs)
    {
        sb.AppendLine($"## {title}");
        sb.AppendLine();
        List<IGrouping<string, Run>> groups = runs
            .GroupBy(r => r.Rid ?? "unknown", StringComparer.OrdinalIgnoreCase)
            .OrderBy(g => g.Key, StringComparer.OrdinalIgnoreCase).ToList();
        if (groups.Count == 0) { sb.AppendLine("No benchmark groups."); sb.AppendLine(); return; }
        foreach (IGrouping<string, Run> group in groups)
        {
            sb.AppendLine($"### {Cell(group.Key)}");
            sb.AppendLine();
            sb.AppendLine("All cases in a replica run on the same machine. Ratios are calculated within each replica; absolute values from different CPUs are not pooled.");
            sb.AppendLine();
            sb.AppendLine("| replica | case | engine | model | w | effective ISA | median ms | P95 ms | img/s | vs replica baseline | WS loaded | WS peak | Δ WS | CPU |");
            sb.AppendLine("| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
            foreach (Run r in group.OrderBy(r => r.Replica).ThenBy(r => r.CaseId).ThenBy(r => r.Label))
            {
                Run? baseline = Baseline(group, r.Replica);
                string ratio = baseline is not null && baseline.Mean > 0
                    ? (r.Mean / baseline.Mean).ToString("F2", CultureInfo.InvariantCulture) : "—";
                sb.AppendLine($"| {r.Replica} | {Cell(string.IsNullOrEmpty(r.CaseId) ? r.Label : r.CaseId)} | {Cell(r.Engine)} | {Cell(r.Model)} | {r.Workers} | {Cell(r.EffectiveIsa)} | {r.Median:F1} | {r.P95:F1} | {r.Throughput:F2} | {ratio} | {Mb(r.WsLoaded)} | {Mb(r.WsPeak)} | {Mb(WorkingSetDelta(r))} | {Cell(r.CpuName)} |");
            }
            sb.AppendLine();
            foreach (IGrouping<string, Run> caseGroup in group.GroupBy(r => r.CaseId.Length == 0 ? r.Label : r.CaseId))
            {
                double[] ratios = caseGroup.Select(r =>
                {
                    Run? baseline = Baseline(group, r.Replica);
                    return baseline is not null && baseline.Mean > 0 ? r.Mean / baseline.Mean : double.NaN;
                }).Where(double.IsFinite).OrderBy(x => x).ToArray();
                double[] memoryDeltas = caseGroup.Select(WorkingSetDelta).Where(v => v.HasValue).Select(v => v!.Value).OrderBy(v => v).ToArray();
                if (ratios.Length > 1)
                    sb.AppendLine($"- `{Cell(caseGroup.Key)}` normalized median `{Median(ratios):F2}x`, range `{ratios[0]:F2}–{ratios[^1]:F2}x` ({ratios.Length} replicas); Δ WS median `{(memoryDeltas.Length == 0 ? "—" : Median(memoryDeltas).ToString("F1", CultureInfo.InvariantCulture) + " MB")}`");
            }
            sb.AppendLine();
        }
    }

    private static double Median(double[] values) => values.Length == 0 ? double.NaN :
        values.Length % 2 == 1 ? values[values.Length / 2] : (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2;

    private static double? WorkingSetDelta(Run run) =>
        run.WsLast is { } last && run.WsLoaded is { } loaded ? last - loaded : null;

    private static Run? Baseline(IGrouping<string, Run> group, int replica) => group
        .Where(x => x.Replica == replica)
        .OrderBy(x => x.CaseId is "tiny-4w" ? 0 : 1)
        .ThenBy(x => x.CaseId)
        .FirstOrDefault();

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
            double? delta = WorkingSetDelta(r);
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
