using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.ModelProvider;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Medium;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Small;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Pdf;
using SkiaSharp;

Console.OutputEncoding = Encoding.UTF8;
if (args.Length < 2 || args.Contains("--help"))
{
    Console.WriteLine("Usage: SkiaSharp.SearchablePdf <image-folder-or-file> <output.pdf> " +
        "[--model medium|small|tiny] [--backend auto|cpu|vulkan] [--dpi 300] [--det-size 1600] [--max-pages N] " +
        "[--separate --repeat 3] (with --separate, output is a directory)");
    return args.Contains("--help") ? 0 : 1;
}

string? temporaryPdf = null;
try
{
    string input = Path.GetFullPath(args[0]);
    string output = Path.GetFullPath(args[1]);
    string model = "medium";
    OcrBackend backend = OcrBackend.Auto;
    float dpi = 300;
    int detSize = 1600, maxPages = int.MaxValue;
    bool separate = false;
    int repeat = 1;
    for (int i = 2; i < args.Length; i++)
    {
        string option = args[i];
        if (option == "--separate") { separate = true; continue; }
        if (++i >= args.Length) throw new ArgumentException($"Missing value for {option}.");
        string value = args[i];
        switch (option)
        {
            case "--model": model = value.ToLowerInvariant(); break;
            case "--backend": backend = value.ToLowerInvariant() switch
                {
                    "auto" => OcrBackend.Auto, "cpu" => OcrBackend.Cpu, "vulkan" => OcrBackend.Vulkan,
                    _ => throw new ArgumentException("Backend must be auto, cpu, or vulkan.")
                }; break;
            case "--dpi": dpi = float.Parse(value, CultureInfo.InvariantCulture); break;
            case "--det-size": detSize = int.Parse(value, CultureInfo.InvariantCulture); break;
            case "--max-pages": maxPages = int.Parse(value, CultureInfo.InvariantCulture); break;
            case "--repeat": repeat = int.Parse(value, CultureInfo.InvariantCulture); break;
            default: throw new ArgumentException($"Unknown option: {option}");
        }
    }
    if (!float.IsFinite(dpi) || dpi <= 0 || detSize < 32 || detSize > 4096 || maxPages <= 0)
        throw new ArgumentException("dpi and max-pages must be positive; det-size must be 32..4096.");
    if (repeat is < 1 or > 10) throw new ArgumentException("repeat must be 1..10.");
    if (!separate && !output.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
        throw new ArgumentException("Output must end in .pdf.");
    PaddleOcrModelBundle bundle = model switch
    {
        "medium" => ChineseV6MediumModels.Default,
        "small" => ChineseV6SmallModels.Default,
        "tiny" => ChineseV6TinyModels.Default,
        _ => throw new ArgumentException("Model must be medium, small, or tiny.")
    };
    HashSet<string> extensions = new(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff" };
    string[] files = (Directory.Exists(input)
            ? Directory.EnumerateFiles(input).Where(p => extensions.Contains(Path.GetExtension(p)))
            : File.Exists(input) && extensions.Contains(Path.GetExtension(input)) ? [input]
            : throw new FileNotFoundException("Image folder or supported image file not found.", input))
        .OrderBy(p => Path.GetFileName(p), NaturalFileNameComparer.Instance).Take(maxPages).ToArray();
    if (files.Length == 0) throw new InvalidDataException("No supported images in input folder.");
    PaddleOcrOptions options = new()
    {
        DocumentOrientationThreshold = 0.85f,
        DocumentOrientation = new() { Backend = backend },
        Detector = new() { Backend = backend, LimitSideLength = detSize },
        Classifier = new() { Backend = backend },
        Recognizer = new() { Backend = backend }
    };
    if (separate) return SinglePdfBenchmark.Run(files, output, bundle, options, model, backend, dpi, detSize, repeat);
    Directory.CreateDirectory(Path.GetDirectoryName(output)!);
    Console.WriteLine($"Loading {model} OCR ({backend}); {files.Length} pages, detector={detSize}, dpi={dpi}...");
    using PaddleOcrAll ocr = PaddleOcrAll.Load(bundle, DocumentOrientationModel.Provider, options);
    List<object> pages = [];
    StringBuilder transcript = new();
    Stopwatch total = Stopwatch.StartNew();
    temporaryPdf = output + "." + Guid.NewGuid().ToString("N") + ".tmp";
    using (FileStream stream = File.Create(temporaryPdf))
    using (PaddleOcrPdfWriter pdf = new(stream, dpi))
    {
        foreach (string file in files)
        {
            Stopwatch timer = Stopwatch.StartNew();
            byte[] encoded = File.ReadAllBytes(file);
            using SKBitmap bitmap = PdfImage.Decode(encoded);
            PaddleOcrResult rawResult = ocr.Run(bitmap.GetPixelSpan(), bitmap.Width, bitmap.Height,
                returnCtcAlignment: true, sourceStride: bitmap.RowBytes, format: ImagePixelFormat.Bgra32);
            PaddleOcrResult result = PdfTextAlignment.RemovePaddingEmissions(rawResult);
            int paddingTokensRemoved = rawResult.Lines.Sum(x => x.CtcSpans!.Length) - result.Lines.Sum(x => x.CtcSpans!.Length);
            pdf.AddPage(bitmap, result, encoded);
            pages.Add(new
            {
                Page = pdf.PageCount, File = Path.GetFileName(file), bitmap.Width, bitmap.Height,
                result.AppliedDocumentRotationDegrees,
                PaddingTokensRemoved = paddingTokensRemoved,
                OriginalOcrText = paddingTokensRemoved == 0 ? null : rawResult.Text,
                Lines = result.Lines.Select(line => new
                {
                    line.Text, line.RecognitionScore, line.AppliedRotationDegrees,
                    line.Box, Characters = PdfTextAlignment.EstimateCharacterBoxes(line)
                }).ToArray()
            });
            transcript.AppendLine($"--- Page {pdf.PageCount}: {Path.GetFileName(file)} ---").AppendLine(result.Text).AppendLine();
            Console.WriteLine($"[{pdf.PageCount}/{files.Length}] {Path.GetFileName(file)}: " +
                $"{result.Lines.Length} lines, {result.Text.Length} chars, {timer.Elapsed.TotalSeconds:F2}s" +
                (paddingTokensRemoved > 0 ? $" (excluded {paddingTokensRemoved} padding tokens)" : ""));
        }
        pdf.Complete();
    }
    File.Move(temporaryPdf, output, overwrite: true);
    temporaryPdf = null;
    File.WriteAllText(Path.ChangeExtension(output, ".ocr.json"), JsonSerializer.Serialize(new
    {
        Model = model, Backend = backend.ToString(), Dpi = dpi, DetectorSize = detSize, Pages = pages
    }, new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }), new UTF8Encoding(false));
    File.WriteAllText(Path.ChangeExtension(output, ".txt"), transcript.ToString(), new UTF8Encoding(false));
    Console.WriteLine($"Saved {output} ({new FileInfo(output).Length / 1048576.0:F2} MiB, {total.Elapsed.TotalSeconds:F1}s).");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex);
    return 1;
}
finally
{
    if (temporaryPdf is not null && File.Exists(temporaryPdf)) File.Delete(temporaryPdf);
}

internal sealed class NaturalFileNameComparer : IComparer<string>
{
    public static NaturalFileNameComparer Instance { get; } = new();
    public int Compare(string? x, string? y)
    {
        string[] a = Regex.Split(x ?? "", "([0-9]+)"), b = Regex.Split(y ?? "", "([0-9]+)");
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            string left = a[i], right = b[i];
            int compare;
            if (i % 2 == 1)
            {
                left = left.TrimStart('0'); right = right.TrimStart('0');
                compare = left.Length.CompareTo(right.Length);
                if (compare == 0) compare = string.CompareOrdinal(left, right);
            }
            else compare = StringComparer.OrdinalIgnoreCase.Compare(left, right);
            if (compare != 0) return compare;
        }
        int count = a.Length.CompareTo(b.Length);
        return count != 0 ? count : StringComparer.Ordinal.Compare(x, y);
    }
}
