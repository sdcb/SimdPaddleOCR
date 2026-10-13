using SkiaSharp;

namespace Sdcb.SimdPaddleOCR.Pdf;

/// <summary>Decodes the same opaque BGRA pixels used by OCR and the PDF image layer.</summary>
public static class PdfImage
{
    public static SKBitmap Decode(byte[] encoded, long maxPixels = 40_000_000)
    {
        ArgumentNullException.ThrowIfNull(encoded);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxPixels);
        using SKData data = SKData.CreateCopy(encoded);
        using SKCodec codec = SKCodec.Create(data)
            ?? throw new InvalidDataException("Unsupported or damaged image.");
        SKImageInfo info = codec.Info;
        if (info.Width <= 0 || info.Height <= 0 || (long)info.Width * info.Height > maxPixels)
            throw new InvalidDataException($"Image exceeds the {maxPixels:N0}-pixel limit.");
        using SKBitmap decoded = SKBitmap.Decode(codec)
            ?? throw new InvalidDataException("Unable to decode image.");
        SKBitmap opaque = new(info.Width, info.Height, SKColorType.Bgra8888, SKAlphaType.Opaque);
        try
        {
            using SKCanvas canvas = new(opaque);
            canvas.Clear(SKColors.White);
            canvas.DrawBitmap(decoded, 0, 0, new SKSamplingOptions(SKFilterMode.Nearest));
            return opaque;
        }
        catch
        {
            opaque.Dispose();
            throw;
        }
    }
}
