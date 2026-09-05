using System.Buffers;
using System.Runtime.CompilerServices;
using Sdcb.PaddleOCR.OnnxSharp;

namespace Sdcb.PaddleOCR;

internal static class PPOCRPreprocess
{
    private static readonly double[] DetMean = [0.485, 0.456, 0.406];
    private static readonly double[] DetInverseStd = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225];
    private static readonly double[] ClsMean = [0.485, 0.456, 0.406];
    private static readonly double[] ClsInverseStd = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225];
    // Normalization is applied to millions of pixels per request.  Looking up
    // the exact float result for each 8-bit value removes a floating-point
    // divide/multiply from the hot loops while retaining the same rounding.
    private static readonly float[] DetNormalized = BuildChannelLut(DetMean, DetInverseStd);
    private static readonly float[] ClsNormalized = BuildChannelLut(ClsMean, ClsInverseStd);
    private static readonly float[] RecNormalized = BuildRecLut();

    private static float[] BuildChannelLut(double[] mean, double[] inverseStd)
    {
        float[] values = new float[3 * 256];
        for (int channel = 0; channel < 3; channel++)
            for (int value = 0; value < 256; value++)
                values[channel * 256 + value] = (float)((value / 255.0 - mean[channel]) * inverseStd[channel]);
        return values;
    }

    private static float[] BuildRecLut()
    {
        float[] values = new float[256];
        for (int value = 0; value < 256; value++)
            values[value] = value * (2.0f / 255.0f) - 1.0f;
        return values;
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
    public static unsafe void DetBgrToNchw(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int resizedWidth, int resizedHeight, Span<float> output) =>
        DetBgrToNchw(source, sourceWidth, sourceHeight, sourceStride, resizedWidth, resizedHeight,
            output, null);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void DetBgrToNchw(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int resizedWidth, int resizedHeight, Span<float> output,
        ResizeWorkspace? workspace)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride);
        int plane = checked(resizedWidth * resizedHeight);
        if (output.Length != checked(plane * 3)) throw new ArgumentException("Invalid DET output size.");
        // PaddleOCR performs NormalizeImage after cv2.resize on an 8-bit BGR
        // image. OpenCV's regular INTER_LINEAR path uses 11-bit integer
        // coefficients and, on x64, its SIMD vertical pass shifts each
        // horizontal accumulator by four before the high multiply. Reproduce
        // that observable 8-bit result before normalizing so threshold-boundary
        // detector pixels agree with the reference implementation.
        bool pooled = workspace is null;
        workspace?.Ensure(resizedWidth);
        int[] xOffsets = workspace?.XOffsets ?? PooledArrays.Rent<int>(resizedWidth);
        short[] xCoefficients = workspace?.XCoefficients ?? PooledArrays.Rent<short>(checked(resizedWidth * 2));
        int[] row0 = workspace?.Row0 ?? PooledArrays.Rent<int>(checked(resizedWidth * 3));
        int[] row1 = workspace?.Row1 ?? PooledArrays.Rent<int>(checked(resizedWidth * 3));
        try
        {
            BuildLinearCoefficients(sourceWidth, resizedWidth, xOffsets, xCoefficients);
            fixed (byte* sourcePtr = source)
            fixed (float* outputPtr = output)
            fixed (float* normalizedPtr = DetNormalized)
            {
                for (int oy = 0; oy < resizedHeight; oy++)
                {
                    GetLinearCoordinate(oy, sourceHeight, resizedHeight,
                        out int sy, out short beta0, out short beta1);
                    int sy0 = MathCompat.Clamp(sy, 0, sourceHeight - 1);
                    int sy1 = MathCompat.Clamp(sy + 1, 0, sourceHeight - 1);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy0,
                        resizedWidth, xOffsets, xCoefficients, row0);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy1,
                        resizedWidth,
                        xOffsets, xCoefficients, row1);
                    int destination = oy * resizedWidth;
                    for (int ox = 0; ox < resizedWidth; ox++)
                    {
                        int rowOffset = ox * 3;
                        for (int channel = 0; channel < 3; channel++)
                        {
                            int h0 = row0[rowOffset + channel];
                            int h1 = row1[rowOffset + channel];
                            // VResizeLinearVec_32s8u from OpenCV's resize.cpp.
                            int value = (((h0 >> 4) * beta0 >> 16) +
                                ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                            if (value < 0) value = 0;
                            else if (value > 255) value = 255;
                                outputPtr[channel * plane + destination + ox] =
                                normalizedPtr[channel * 256 + value];
                        }
                    }
                }
            }
        }
        finally
        {
            if (pooled)
            {
                PooledArrays.Return(xOffsets);
                PooledArrays.Return(xCoefficients);
                PooledArrays.Return(row0);
                PooledArrays.Return(row1);
            }
        }
    }

    private static void BuildLinearCoefficients(int sourceSize, int destinationSize,
        int[] offsets, short[] coefficients)
    {
        const int scale = 1 << 11;
        for (int d = 0; d < destinationSize; d++)
        {
            // OpenCV computes the interpolation coordinate in softdouble
            // (effectively double precision).  Keeping the coordinate in
            // float can move coefficients by one at exact/tie boundaries for
            // narrow text crops (for example 28 -> 160), changing the final
            // 8-bit pixel after the vertical fixed-point pass.
            double coordinate = ((double)d + 0.5) * sourceSize / destinationSize - 0.5;
            int source = checked((int)Math.Floor(coordinate));
            double fraction = coordinate - source;
            if (source < 0) { source = 0; fraction = 0; }
            if (source >= sourceSize - 1) { source = sourceSize - 1; fraction = 0; }
            offsets[d] = source;
            coefficients[d * 2] = CvRoundToShort((1 - fraction) * scale);
            coefficients[d * 2 + 1] = CvRoundToShort(fraction * scale);
        }
    }

    private static void GetLinearCoordinate(int destination, int sourceSize, int destinationSize,
        out int source, out short coefficient0, out short coefficient1)
    {
        const int scale = 1 << 11;
        double coordinate = ((double)destination + 0.5) * sourceSize / destinationSize - 0.5;
        source = checked((int)Math.Floor(coordinate));
        double fraction = coordinate - source;
        // Keep the raw floor/fraction at the vertical edges. OpenCV's
        // separable resize clips the *source rows* to [0, sourceSize-1]
        // while retaining the fractional coefficients. Both rows therefore
        // point at the edge pixel, but the two fixed-point products are still
        // evaluated separately; replacing them with (2048, 0) changes the
        // result by one for some byte values.
        coefficient0 = CvRoundToShort((1 - fraction) * scale);
        coefficient1 = CvRoundToShort(fraction * scale);
    }

    private static short CvRoundToShort(double value) =>
        checked((short)Math.Round(value, MidpointRounding.ToEven));

    private static unsafe void BuildHorizontalRow(byte* source, int sourceStride, int sourceWidth,
        int sourceY, int destinationWidth, int[] offsets, short[] coefficients, int[] destination)
    {
        byte* row = source + sourceY * sourceStride;
        for (int x = 0; x < destinationWidth; x++)
        {
            int sx = offsets[x], sx1 = Math.Min(sx + 1, sourceWidth - 1);
            short coefficient0 = coefficients[x * 2], coefficient1 = coefficients[x * 2 + 1];
            int sourceOffset = sx * 3, sourceOffset1 = sx1 * 3, destinationOffset = x * 3;
            destination[destinationOffset] = row[sourceOffset] * coefficient0 + row[sourceOffset1] * coefficient1;
            destination[destinationOffset + 1] = row[sourceOffset + 1] * coefficient0 + row[sourceOffset1 + 1] * coefficient1;
            destination[destinationOffset + 2] = row[sourceOffset + 2] * coefficient0 + row[sourceOffset1 + 2] * coefficient1;
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe int ClsBgrToNchw(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, Span<float> output) =>
        ClsBgrToNchw(source, sourceWidth, sourceHeight, sourceStride, output, null);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe int ClsBgrToNchw(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, Span<float> output, ResizeWorkspace? workspace)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride);
        const int height = 80, width = 160;
        int plane = height * width;
        if (output.Length != 3 * plane) throw new ArgumentException("Invalid CLS output size.");
        // PP-LCNet_x0_25_textline_ori uses the PaddleX classification pipeline:
        // ReadImage(format="RGB") then ResizeImage([160, 80]) then ImageNet
        // NormalizeImage. Resize in the 8-bit BGR domain first (cv2 INTER_LINEAR
        // coefficients), then write RGB NCHW planes with per-channel LUTs.
        // Bilinear is per-channel, so swapping after resize matches BGR2RGB
        // before resize.
        ResizeBgrInterLinearToClsNchw(source, sourceWidth, sourceHeight, sourceStride,
            width, height, output, workspace);
        return width;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ResizeBgrInterLinearToClsNchw(ReadOnlySpan<byte> source,
        int sourceWidth, int sourceHeight, int sourceStride, int resizedWidth,
        int resizedHeight, Span<float> output, ResizeWorkspace? workspace)
    {
        bool pooled = workspace is null;
        workspace?.Ensure(resizedWidth);
        int[] xOffsets = workspace?.XOffsets ?? PooledArrays.Rent<int>(resizedWidth);
        short[] xCoefficients = workspace?.XCoefficients ?? PooledArrays.Rent<short>(checked(resizedWidth * 2));
        int[] row0 = workspace?.Row0 ?? PooledArrays.Rent<int>(checked(resizedWidth * 3));
        int[] row1 = workspace?.Row1 ?? PooledArrays.Rent<int>(checked(resizedWidth * 3));
        int plane = checked(resizedHeight * resizedWidth);
        try
        {
            BuildLinearCoefficients(sourceWidth, resizedWidth, xOffsets, xCoefficients);
            fixed (byte* sourcePtr = source)
            fixed (float* outputPtr = output)
            fixed (float* normalizedPtr = ClsNormalized)
            {
                for (int oy = 0; oy < resizedHeight; oy++)
                {
                    GetLinearCoordinate(oy, sourceHeight, resizedHeight,
                        out int sy, out short beta0, out short beta1);
                    int sy0 = MathCompat.Clamp(sy, 0, sourceHeight - 1);
                    int sy1 = MathCompat.Clamp(sy + 1, 0, sourceHeight - 1);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy0,
                        resizedWidth, xOffsets, xCoefficients, row0);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy1,
                        resizedWidth, xOffsets, xCoefficients, row1);
                    int destination = oy * resizedWidth;
                    for (int ox = 0; ox < resizedWidth; ox++)
                    {
                        int rowOffset = ox * 3;
                        // Packed source is BGR; PaddleX ReadImage converts to RGB
                        // before ImageNet, so NCHW planes are R, G, B.
                        int h0 = row0[rowOffset + 2], h1 = row1[rowOffset + 2];
                        int r = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        h0 = row0[rowOffset + 1]; h1 = row1[rowOffset + 1];
                        int g = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        h0 = row0[rowOffset]; h1 = row1[rowOffset];
                        int b = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        outputPtr[destination + ox] =
                            normalizedPtr[MathCompat.Clamp(r, 0, 255)];
                        outputPtr[plane + destination + ox] =
                            normalizedPtr[256 + MathCompat.Clamp(g, 0, 255)];
                        outputPtr[2 * plane + destination + ox] =
                            normalizedPtr[512 + MathCompat.Clamp(b, 0, 255)];
                    }
                }
            }
        }
        finally
        {
            if (pooled)
            {
                PooledArrays.Return(xOffsets);
                PooledArrays.Return(xCoefficients);
                PooledArrays.Return(row0);
                PooledArrays.Return(row1);
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int RecBgrToNchw(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int targetWidth, Span<float> output) =>
        RecBgrToNchw(source, sourceWidth, sourceHeight, sourceStride, targetWidth, output, null);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int RecBgrToNchw(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int targetWidth, Span<float> output, ResizeWorkspace? workspace)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride);
        if (targetWidth <= 0) throw new ArgumentOutOfRangeException(nameof(targetWidth));
        int height = 48;
        int plane = checked(height * targetWidth);
        if (output.Length != checked(3 * plane)) throw new ArgumentException("Invalid REC output size.");
        int actualWidth = (int)Math.Min(targetWidth,
            ((long)height * sourceWidth + sourceHeight - 1L) / sourceHeight);
        output.Clear();
        ResizeBgrInterLinearToNchw(source, sourceWidth, sourceHeight, sourceStride,
            actualWidth, height, targetWidth, output, workspace);
        return actualWidth;
    }

    // REC only needs the byte-domain interpolation result as an intermediate
    // for normalization. Keep the exact OpenCV fixed-point rounding, but write
    // the three normalized planes directly and avoid a temporary byte image
    // plus a second full-frame traversal.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ResizeBgrInterLinearToNchw(ReadOnlySpan<byte> source,
        int sourceWidth, int sourceHeight, int sourceStride, int resizedWidth,
        int resizedHeight, int outputWidth, Span<float> output, ResizeWorkspace? workspace)
    {
        bool pooled = workspace is null;
        workspace?.Ensure(resizedWidth);
        int[] xOffsets = workspace?.XOffsets ?? PooledArrays.Rent<int>(resizedWidth);
        short[] xCoefficients = workspace?.XCoefficients ?? PooledArrays.Rent<short>(checked(resizedWidth * 2));
        int[] row0 = workspace?.Row0 ?? PooledArrays.Rent<int>(checked(resizedWidth * 3));
        int[] row1 = workspace?.Row1 ?? PooledArrays.Rent<int>(checked(resizedWidth * 3));
        int plane = checked(resizedHeight * outputWidth);
        try
        {
            BuildLinearCoefficients(sourceWidth, resizedWidth, xOffsets, xCoefficients);
            fixed (byte* sourcePtr = source)
            fixed (float* outputPtr = output)
            fixed (float* normalizedPtr = RecNormalized)
            {
                for (int oy = 0; oy < resizedHeight; oy++)
                {
                    GetLinearCoordinate(oy, sourceHeight, resizedHeight,
                        out int sy, out short beta0, out short beta1);
                    int sy0 = MathCompat.Clamp(sy, 0, sourceHeight - 1);
                    int sy1 = MathCompat.Clamp(sy + 1, 0, sourceHeight - 1);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy0,
                        resizedWidth, xOffsets, xCoefficients, row0);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy1,
                        resizedWidth, xOffsets, xCoefficients, row1);
                    int destination = oy * outputWidth;
                    for (int ox = 0; ox < resizedWidth; ox++)
                    {
                        int rowOffset = ox * 3;
                        int h0 = row0[rowOffset], h1 = row1[rowOffset];
                        int value = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        outputPtr[destination + ox] = normalizedPtr[MathCompat.Clamp(value, 0, 255)];
                        h0 = row0[rowOffset + 1]; h1 = row1[rowOffset + 1];
                        value = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        outputPtr[plane + destination + ox] = normalizedPtr[MathCompat.Clamp(value, 0, 255)];
                        h0 = row0[rowOffset + 2]; h1 = row1[rowOffset + 2];
                        value = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        outputPtr[plane * 2 + destination + ox] = normalizedPtr[MathCompat.Clamp(value, 0, 255)];
                    }
                }
            }
        }
        finally
        {
            if (pooled)
            {
                PooledArrays.Return(xOffsets);
                PooledArrays.Return(xCoefficients);
                PooledArrays.Return(row0);
                PooledArrays.Return(row1);
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void NormalizeBgrResize(ReadOnlySpan<byte> resized, int resizedWidth,
        int resizedHeight, int outputWidth, Span<float> output)
    {
        int plane = checked(resizedHeight * outputWidth);
        fixed (byte* sourcePtr = resized)
        fixed (float* outputPtr = output)
        {
            for (int oy = 0; oy < resizedHeight; oy++)
            {
                for (int ox = 0; ox < resizedWidth; ox++)
                {
                    int sourceOffset = (oy * resizedWidth + ox) * 3;
                    int destination = oy * outputWidth + ox;
                    outputPtr[destination] = RecNormalized[sourcePtr[sourceOffset]];
                    outputPtr[plane + destination] = RecNormalized[sourcePtr[sourceOffset + 1]];
                    outputPtr[plane * 2 + destination] = RecNormalized[sourcePtr[sourceOffset + 2]];
                }
            }
        }
    }

    /// <summary>
    /// Resizes an interleaved BGR image using OpenCV's integer coefficient
    /// path for 8-bit INTER_LINEAR images. PaddleOCR normalizes only after
    /// this byte-domain rounding step.
    /// </summary>
    private static unsafe void ResizeBgrInterLinear(ReadOnlySpan<byte> source, int sourceWidth,
        int sourceHeight, int sourceStride, int destinationWidth, int destinationHeight,
        Span<byte> destination)
    {
        int required = checked(destinationWidth * destinationHeight * 3);
        if (destination.Length < required) throw new ArgumentException("Destination buffer is too small.");
        int[] xOffsets = PooledArrays.Rent<int>(destinationWidth);
        short[] xCoefficients = PooledArrays.Rent<short>(checked(destinationWidth * 2));
        int[] row0 = PooledArrays.Rent<int>(checked(destinationWidth * 3));
        int[] row1 = PooledArrays.Rent<int>(checked(destinationWidth * 3));
        try
        {
            BuildLinearCoefficients(sourceWidth, destinationWidth, xOffsets, xCoefficients);
            fixed (byte* sourcePtr = source)
            fixed (byte* destinationPtr = destination)
            {
                for (int oy = 0; oy < destinationHeight; oy++)
                {
                    GetLinearCoordinate(oy, sourceHeight, destinationHeight,
                        out int sy, out short beta0, out short beta1);
                    int sy0 = MathCompat.Clamp(sy, 0, sourceHeight - 1);
                    int sy1 = MathCompat.Clamp(sy + 1, 0, sourceHeight - 1);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy0,
                        destinationWidth, xOffsets, xCoefficients, row0);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy1,
                        destinationWidth,
                        xOffsets, xCoefficients, row1);
                    int destinationOffset = oy * destinationWidth * 3;
                    for (int ox = 0; ox < destinationWidth; ox++)
                    {
                        int rowOffset = ox * 3;
                        int pixelOffset = destinationOffset + rowOffset;
                        int h0 = row0[rowOffset], h1 = row1[rowOffset];
                        int value = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        destinationPtr[pixelOffset] = (byte)MathCompat.Clamp(value, 0, 255);
                        h0 = row0[rowOffset + 1]; h1 = row1[rowOffset + 1];
                        value = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        destinationPtr[pixelOffset + 1] = (byte)MathCompat.Clamp(value, 0, 255);
                        h0 = row0[rowOffset + 2]; h1 = row1[rowOffset + 2];
                        value = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        destinationPtr[pixelOffset + 2] = (byte)MathCompat.Clamp(value, 0, 255);
                    }
                }
            }
        }
        finally
        {
            PooledArrays.Return(xOffsets);
            PooledArrays.Return(xCoefficients);
            PooledArrays.Return(row0);
            PooledArrays.Return(row1);
        }
    }

    private static void ValidateSource(ReadOnlySpan<byte> source, int width, int height, int stride)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (stride < checked(width * 3)) throw new ArgumentException("Source stride is too small.");
        long required = checked((long)(height - 1) * stride + width * 3L);
        if (required > source.Length) throw new ArgumentException("Source buffer is too small.");
    }

}
