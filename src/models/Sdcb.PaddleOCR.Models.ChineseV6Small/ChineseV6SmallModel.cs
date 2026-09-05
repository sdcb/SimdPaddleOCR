using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Sdcb.PaddleOCR.ModelProvider;
using Sdcb.PaddleOCR.Models.TextLineOrientation;

namespace Sdcb.PaddleOCR.Models.ChineseV6Small;

public static class ChineseV6SmallModel
{
    public static IPaddleOcrModelProvider Detection { get; } = new ProviderImpl(
        "PP-OCRv6_small_det", 
        PaddleOcrModelKind.Detection, 
        "det.onnx", 
        "onnx");

    public static IPaddleOcrModelProvider Recognition { get; } = new ProviderImpl(
        "PP-OCRv6_small_rec", 
        PaddleOcrModelKind.Recognition, 
        "rec.onnx", 
        "onnx");

    public static IPaddleOcrModelProvider Dictionary { get; } = new ProviderImpl(
        "ppocrv6_dict.txt", 
        PaddleOcrModelKind.Dictionary, 
        "dict.txt", 
        "utf-8");

    public static IPaddleOcrModelProvider[] All { get; } = 
    [
        Detection, 
        Recognition, 
        Dictionary
    ];

    public static Stream OpenRead() => Detection.OpenRead();

    private sealed class ProviderImpl(string name, PaddleOcrModelKind kind, string resource, string format) : IPaddleOcrModelProvider
    {
        public string Name { get; } = name;
        public PaddleOcrModelKind Kind { get; } = kind;
        public string Format { get; } = format;
        public string? LanguageCode { get; } = "zh";
        public string? Version { get; } = "v6-small";
        public Stream OpenRead() => typeof(ChineseV6SmallModel).Assembly.GetManifestResourceStream("Sdcb.PaddleOCR.Models.ChineseV6Small." + resource)
            ?? throw new InvalidOperationException("Embedded model resource was not found.");
        public Task<Stream> OpenReadAsync(CancellationToken cancellationToken = default)
        { cancellationToken.ThrowIfCancellationRequested(); return Task.FromResult(OpenRead()); }
    }
}

public static class ChineseV6SmallModels
{
    public static PaddleOcrModelBundle Default { get; } = new(
        "PP-OCRv6_small", 
        "zh", 
        ChineseV6SmallModel.Detection,
        ChineseV6SmallModel.Recognition, 
        ChineseV6SmallModel.Dictionary,
        TextLineOrientationModel.Provider);
        
    public static PaddleOcrModelBundle[] All { get; } = 
    [
        Default
    ];
}
