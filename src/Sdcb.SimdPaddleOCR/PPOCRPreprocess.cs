using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Sdcb.SimdOnnx.Imaging;

namespace Sdcb.SimdPaddleOCR;

internal static class PPOCRPreprocess
{
    // PaddleOCR normalizes after cv2.resize on the 8-bit BGR image; the tables give the exact float
    // of every byte (one lookup per element instead of a divide / multiply in the hot loops).
    // DET: ImageNet (x/255 - mean) / std on B,G,R; REC / CLS: RecNorm (x/255 - 0.5) / 0.5.
    private static readonly PixelNormalization DetNormalization =
        PixelNormalization.MeanStd([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    private static readonly PixelNormalization RecNormalization =
        PixelNormalization.FromFunction((_, value) => value * (2.0f / 255.0f) - 1.0f);

    /// <summary>
    /// Runs <paramref name="body"/> for every line index on up to
    /// <paramref name="workers"/> threads (shared cursor, so wide lines do not
    /// leave a tail). Lines must write disjoint output.
    /// </summary>
    internal static void ForLines(int count, int workers, Action<int> body)
    {
        workers = Math.Min(workers, count);
        if (workers <= 1)
        {
            for (int k = 0; k < count; k++) body(k);
            return;
        }
        int cursor = -1;
        Parallel.For(0, workers, _ =>
        {
            int k;
            while ((k = Interlocked.Increment(ref cursor)) < count) body(k);
        });
    }

    public static (int Width, int Height, float WidthRatio, float HeightRatio) ComputeDetSize(
        int sourceWidth, int sourceHeight, int limitSideLength)
    {
        if (sourceWidth <= 0) throw new ArgumentOutOfRangeException(nameof(sourceWidth));
        if (sourceHeight <= 0) throw new ArgumentOutOfRangeException(nameof(sourceHeight));
        if (limitSideLength < 32 || limitSideLength > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(limitSideLength));
        // PaddleOCR pads very small images before DetResizeForTest. The
        // padding is part of the detector input (not the source image used
        // for mapping boxes back), and prevents zero-sized stride blocks.
        int effectiveWidth = sourceWidth;
        int effectiveHeight = sourceHeight;
        if (sourceWidth + (long)sourceHeight < 64)
        {
            effectiveWidth = Math.Max(32, sourceWidth);
            effectiveHeight = Math.Max(32, sourceHeight);
        }
        int maximumSide = Math.Max(effectiveWidth, effectiveHeight);
        double ratio = maximumSide > limitSideLength ? (double)limitSideLength / maximumSide : 1.0;
        // Match PaddleOCR's DetResizeForTest.resize_image_type0 exactly:
        // first truncate h*ratio/w*ratio, then round those integer sizes to
        // the nearest multiple of 32 (with a minimum of 32).
        int scaledWidth = checked((int)((double)effectiveWidth * ratio));
        int scaledHeight = checked((int)((double)effectiveHeight * ratio));
        long roundedWidth = (long)Math.Floor(scaledWidth / 32.0 + 0.5) * 32;
        long roundedHeight = (long)Math.Floor(scaledHeight / 32.0 + 0.5) * 32;
        roundedWidth = Math.Max(32, roundedWidth);
        roundedHeight = Math.Max(32, roundedHeight);
        if (roundedWidth > int.MaxValue || roundedHeight > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(limitSideLength));
        return ((int)roundedWidth, (int)roundedHeight,
            (float)((double)roundedWidth / effectiveWidth),
            (float)((double)roundedHeight / effectiveHeight));
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void Det(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int resizedWidth, int resizedHeight, Span<float> output,
        ImagePixelFormat format = ImagePixelFormat.Bgr24) =>
        Det(source, sourceWidth, sourceHeight, sourceStride, resizedWidth, resizedHeight,
            output, intraOpThreads: 1, format: format);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void Det(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int resizedWidth, int resizedHeight, Span<float> output,
        int intraOpThreads, bool nhwc = false, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride, format);
        if (output.Length != checked(resizedWidth * resizedHeight * 3)) throw new ArgumentException("Invalid DET output size.");
        // cv2.resize INTER_LINEAR's observable 8-bit result (11-bit coefficients, x64 SIMD vertical
        // rounding), so threshold-boundary detector pixels agree with the reference; rows split
        // across the intra-op threads bit-identically
        ImageTensor.ResizeLinear(source, sourceWidth, sourceHeight, sourceStride, (PixelFormat)format,
            resizedWidth, resizedHeight, output, nhwc ? TensorLayout.Nhwc : TensorLayout.Nchw, DetNormalization,
            threads: intraOpThreads);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int Cls(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, Span<float> output, ImagePixelFormat format = ImagePixelFormat.Bgr24) =>
        Cls(source, sourceWidth, sourceHeight, sourceStride, output, nhwc: false, format: format);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int Cls(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, Span<float> output, bool nhwc, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride, format);
        const int height = 80, width = 160;
        if (output.Length != 3 * height * width) throw new ArgumentException("Invalid CLS output size.");
        // Historical recipes we no longer run:
        // - PaddleX (this library 1.3): stretch the whole line to fill
        //   160×80 (no keep-aspect), gather as RGB, ImageNet
        //   (x/255-mean)/std with mean 0.485/0.456/0.406 and
        //   std 0.229/0.224/0.225 (same LUT as DET). Short lines are
        //   stretched; long lines are flattened.
        // - PaddleOCR ClsResizeImg (1.4): keep-aspect the whole line (height 80,
        //   width min(160, ceil(80*w/h))), pad unused columns with -1, BGR
        //   RecNorm (x/255-0.5)/0.5. Long lines are squeezed into 160 px and
        //   that costs 0/180 accuracy.
        // Current (1.4.2+): same keep-aspect + RecNorm write as ClsResizeImg
        // (so RGB/RGBA/padded stride stay bit-identical to BGR), but a DET
        // line wider than 4:1 is sampled from the left 4×height window only.
        // Squeezing a long crop into 160 px flattens glyphs until 0/180 is
        // noise; a left 2:1 window was too short on inverted Latin tails.
        // 4:1 is the accuracy vs simplicity tradeoff that held up. The
        // window is a smaller sourceWidth on the original buffer — stride
        // is unchanged, no crop allocation. INTER_LINEAR is output-sized;
        // a wider window only changes the 160 source taps, not the number
        // of output pixels.
        int windowWidth = sourceWidth;
        if ((long)height * sourceWidth > (long)(width * 2) * sourceHeight)
            windowWidth = (int)((long)(width * 2) * sourceHeight / height);
        int actualWidth = (int)Math.Min(width,
            ((long)height * windowWidth + sourceHeight - 1L) / sourceHeight);
        if (actualWidth < width) output.Fill(-1f);
        ImageTensor.ResizeLinear(source, windowWidth, sourceHeight, sourceStride, (PixelFormat)format,
            actualWidth, height, output, nhwc ? TensorLayout.Nhwc : TensorLayout.Nchw, RecNormalization, tensorWidth: width);
        return actualWidth;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int Rec(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int targetWidth, Span<float> output,
        ImagePixelFormat format = ImagePixelFormat.Bgr24) =>
        Rec(source, sourceWidth, sourceHeight, sourceStride, targetWidth, output, nhwc: false, format: format);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int Rec(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int targetWidth, Span<float> output, bool nhwc,
        ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride, format);
        if (targetWidth <= 0) throw new ArgumentOutOfRangeException(nameof(targetWidth));
        int height = 48;
        if (output.Length != checked(3 * height * targetWidth)) throw new ArgumentException("Invalid REC output size.");
        int actualWidth = (int)Math.Min(targetWidth,
            ((long)height * sourceWidth + sourceHeight - 1L) / sourceHeight);
        output.Clear();
        ImageTensor.ResizeLinear(source, sourceWidth, sourceHeight, sourceStride, (PixelFormat)format,
            actualWidth, height, output, nhwc ? TensorLayout.Nhwc : TensorLayout.Nchw, RecNormalization, tensorWidth: targetWidth);
        return actualWidth;
    }

    private static void ValidateSource(ReadOnlySpan<byte> source, int width, int height, int stride,
        ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        int bpp = ImagePixels.BytesPerPixel(format);
        if (stride < checked(width * bpp)) throw new ArgumentException("Source stride is too small.");
        long required = checked((long)(height - 1) * stride + width * (long)bpp);
        if (required > source.Length) throw new ArgumentException("Source buffer is too small.");
    }

}
