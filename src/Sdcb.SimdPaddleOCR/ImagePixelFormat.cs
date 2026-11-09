namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Interleaved source pixel layout for DET / CLS / REC / <see cref="PaddleOcrAll.Run"/>.
/// The default <see cref="Bgr24"/> is the historical contract. Other layouts are
/// read in-place during resize and crop; they are not converted into an extra
/// BGR image.
/// </summary>
public enum ImagePixelFormat
{
    /// <summary>3 bytes, B,G,R. OpenCV / GDI+ 24bpp memory order.</summary>
    Bgr24 = 0,
    /// <summary>
    /// 3 bytes, R,G,B. Horizontal gather stays scalar (only 32-bit layouts use SIMD).
    /// Still cheaper than converting to BGR first. Decoders that can emit a 32-bit
    /// layout should pass <see cref="Rgba32"/> or <see cref="Bgra32"/>.
    /// </summary>
    Rgb24 = 1,
    /// <summary>4 bytes, B,G,R,A. Alpha ignored. Skia / GDI+ 32bpp.</summary>
    Bgra32 = 2,
    /// <summary>4 bytes, R,G,B,A. Alpha ignored. Byte order of most RGBA8 decoders.</summary>
    Rgba32 = 3,
}

internal static class ImagePixels
{
    internal static int BytesPerPixel(ImagePixelFormat format) => format switch
    {
        ImagePixelFormat.Bgr24 or ImagePixelFormat.Rgb24 => 3,
        ImagePixelFormat.Bgra32 or ImagePixelFormat.Rgba32 => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(format)),
    };

    internal static bool SwapRedBlue(ImagePixelFormat format) =>
        format is ImagePixelFormat.Rgb24 or ImagePixelFormat.Rgba32;

    internal static int ResolveStride(int width, int stride, ImagePixelFormat format)
    {
        int bpp = BytesPerPixel(format);
        if (stride == 0) return checked(width * bpp);
        if (stride < checked(width * bpp)) throw new ArgumentException("Source stride is too small.");
        return stride;
    }
}
