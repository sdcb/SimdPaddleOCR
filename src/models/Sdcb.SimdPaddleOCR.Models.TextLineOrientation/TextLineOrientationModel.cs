using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR.ModelProvider;

namespace Sdcb.SimdPaddleOCR.Models.TextLineOrientation;

public static class TextLineOrientationModel
{
    public static IPaddleOcrModelProvider Provider { get; } = new ProviderImpl();

    public static IPaddleOcrModelProvider[] All { get; } = 
    [
        Provider
    ];

    public static Stream OpenRead() => Provider.OpenRead();

    private sealed class ProviderImpl : IPaddleOcrModelProvider
    {
        public string Name => "PP-LCNet_x0_25_textline_ori";
        public PaddleOcrModelKind Kind => PaddleOcrModelKind.Classification;
        public string Format => "onnx";
        public string? LanguageCode => null;
        public string? Version => null;

        public Stream OpenRead() => typeof(TextLineOrientationModel).Assembly.GetManifestResourceStream(
            "Sdcb.SimdPaddleOCR.Models.TextLineOrientation.cls.onnx")
            ?? throw new InvalidOperationException("Embedded model resource was not found.");

        public Task<Stream> OpenReadAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested(); 
            return Task.FromResult(OpenRead()); 
        }
    }
}
