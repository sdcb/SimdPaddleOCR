using Sdcb.SimdPaddleOCR;

namespace SimdCodecs.AspNetCore;

public sealed class OcrServerOptions
{
    public const string SectionName = "OCR";

    public string DefaultModel { get; set; } = OcrEngine.DefaultModelName;
    public string? PreloadModels { get; set; }
    public long MaxUploadBytes { get; set; } = 20 * 1024 * 1024;
    public long MaxImagePixels { get; set; } = 40_000_000;
    public string? ApiKey { get; set; }
    public OcrBackend Backend { get; set; } = OcrBackend.Auto;
    public int LineWorkerCount { get; set; }
    public int DetIntraOpThreads { get; set; }
    public bool UseDirectionClassification { get; set; } = true;

    public string GetNormalizedDefaultModel() =>
        OcrEngine.TryNormalizeModel(DefaultModel, OcrEngine.DefaultModelName, out string normalized)
            ? normalized
            : throw new InvalidOperationException(
                $"Invalid OCR:DefaultModel '{DefaultModel}'. Allowed values: {string.Join(", ", OcrEngine.Models)}.");

    public string[] ResolvePreloadModels()
    {
        if (string.IsNullOrWhiteSpace(PreloadModels))
            return [];

        string[] tokens = PreloadModels.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (tokens.Any(t => t.Equals("all", StringComparison.OrdinalIgnoreCase)))
            return [.. OcrEngine.Models];

        List<string> models = [];
        foreach (string token in tokens)
        {
            if (!OcrEngine.TryNormalizeModel(token, DefaultModel, out string normalized))
            {
                throw new InvalidOperationException(
                    $"Invalid model '{token}' in OCR:PreloadModels. Allowed values: {string.Join(", ", OcrEngine.Models)}, all.");
            }
            if (!models.Contains(normalized, StringComparer.Ordinal))
                models.Add(normalized);
        }
        return [.. models];
    }

    public PaddleOcrOptions ToPaddleOcrOptions() => new()
    {
        UseDirectionClassification = UseDirectionClassification,
        LineWorkerCount = LineWorkerCount,
        DetIntraOpThreads = DetIntraOpThreads,
        Detector = new PaddleOcrDetectorOptions
        {
            Backend = Backend,
            MaxImagePixels = MaxImagePixels,
        },
        Classifier = new PaddleOcrClassifierOptions
        {
            Backend = Backend,
            MaxImagePixels = MaxImagePixels,
        },
        Recognizer = new PaddleOcrRecognizerOptions
        {
            Backend = Backend,
            MaxImagePixels = MaxImagePixels,
        },
    };
}
