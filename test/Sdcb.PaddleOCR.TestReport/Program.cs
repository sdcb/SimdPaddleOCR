using System.Text;
using Sdcb.PaddleOCR.TestReport;

string? inDir = null;
string? outPath = null;
for (int i = 0; i < args.Length; i++)
{
    string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"missing value for {args[i]}");
    switch (args[i])
    {
        case "--in": inDir = Next(); break;
        case "--out": outPath = Next(); break;
        case "-h" or "--help":
            Console.WriteLine("usage: --in <dir> --out <report.md>");
            return 0;
        default: throw new ArgumentException($"unknown argument: {args[i]}");
    }
}
if (inDir is null || outPath is null)
{
    Console.Error.WriteLine("usage: --in <dir> --out <report.md>");
    return 2;
}

inDir = Path.GetFullPath(inDir);
outPath = Path.GetFullPath(outPath);
string[] files = Directory.GetFiles(inDir, "*.json", SearchOption.AllDirectories)
    .OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();
if (files.Length == 0)
    throw new InvalidOperationException($"no JSON results under {inDir}");

var runs = files.Select(Run.TryLoad)
    .OfType<Run>()
    .OrderBy(r => r.SortKey, StringComparer.OrdinalIgnoreCase)
    .ToList();
if (runs.Count == 0)
    throw new InvalidOperationException($"no benchmark result JSON under {inDir}");
string? outDir = Path.GetDirectoryName(outPath);
if (!string.IsNullOrEmpty(outDir))
    Directory.CreateDirectory(outDir);
File.WriteAllText(outPath, Markdown.Build(runs), new UTF8Encoding(false));
Console.WriteLine($"report: {outPath} ({runs.Count} runs)");
return 0;
