using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR.Kernels;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR;

internal static class PPOCRPreprocess
{
    private static readonly double[] DetMean = [0.485, 0.456, 0.406];
    private static readonly double[] DetInverseStd = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225];
    // Normalization is applied to millions of pixels per request.  Looking up
    // the exact float result for each 8-bit value removes a floating-point
    // divide/multiply from the hot loops while retaining the same rounding.
    private static readonly float[] DetNormalized = BuildChannelLut(DetMean, DetInverseStd);
    private static readonly float[] RecNormalized = BuildRecLut();

    [ThreadStatic] private static ResizeWorkspace? t_workspace;

    /// <summary>
    /// Runs <paramref name="body"/> for every line index on up to
    /// <paramref name="workers"/> threads (shared cursor, so wide lines do not
    /// leave a tail). Each thread gets its own resize scratch; the serial path
    /// uses <paramref name="serialWorkspace"/>. Lines must write disjoint output.
    /// </summary>
    internal static void ForLines(int count, int workers, ResizeWorkspace serialWorkspace,
        Action<int, ResizeWorkspace> body)
    {
        workers = Math.Min(workers, count);
        if (workers <= 1)
        {
            for (int k = 0; k < count; k++) body(k, serialWorkspace);
            return;
        }
        int cursor = -1;
        Parallel.For(0, workers, _ =>
        {
            ResizeWorkspace workspace = t_workspace ??= new ResizeWorkspace();
            int k;
            while ((k = Interlocked.Increment(ref cursor)) < count) body(k, workspace);
        });
    }

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
    public static unsafe void Det(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int resizedWidth, int resizedHeight, Span<float> output,
        ImagePixelFormat format = ImagePixelFormat.Bgr24) =>
        Det(source, sourceWidth, sourceHeight, sourceStride, resizedWidth, resizedHeight,
            output, null, format: format);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void Det(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int resizedWidth, int resizedHeight, Span<float> output,
        ResizeWorkspace? workspace, int intraOpThreads = 1, bool nhwc = false,
        ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride, format);
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
                int workers = ResolveRowWorkers(resizedHeight, resizedWidth, intraOpThreads, workspace);
                if (workers <= 1)
                {
                    for (int oy = 0; oy < resizedHeight; oy++)
                        DetRow(sourcePtr, sourceStride, sourceWidth, sourceHeight, resizedWidth,
                            resizedHeight, oy, xOffsets, xCoefficients, row0, row1, outputPtr,
                            normalizedPtr, plane, nhwc, format);
                }
                else
                {
                    // Destination rows are independent and write disjoint output
                    // ranges, so splitting them is pure distribution: the per-row
                    // arithmetic is untouched and results stay bit-identical. Only
                    // the shared row scratch moves to per-worker buffers.
                    nint sourceAddress = (nint)sourcePtr, outputAddress = (nint)outputPtr,
                        normalizedAddress = (nint)normalizedPtr;
                    ResizeWorkspace shared = workspace!;
                    Parallel.For(0, workers, worker =>
                    {
                        byte* rowSource = (byte*)sourceAddress;
                        float* rowOutput = (float*)outputAddress;
                        float* rowLut = (float*)normalizedAddress;
                        (int[] workerRow0, int[] workerRow1) = shared.RowsFor(worker);
                        for (int oy = worker; oy < resizedHeight; oy += workers)
                            DetRow(rowSource, sourceStride, sourceWidth, sourceHeight, resizedWidth,
                                resizedHeight, oy, xOffsets, xCoefficients, workerRow0, workerRow1,
                                rowOutput, rowLut, plane, nhwc, format);
                    });
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

    // Below this many rows per worker the per-worker setup (row scratch, task
    // dispatch) outweighs the split, so the DET resize stays serial.
    private const int MinRowsPerWorker = 24;

    private static int ResolveRowWorkers(int resizedHeight, int resizedWidth, int intraOpThreads,
        ResizeWorkspace? workspace)
    {
        if (workspace is null || intraOpThreads <= 1) return 1;
        int workers = Math.Min(intraOpThreads, Math.Max(1, resizedHeight / MinRowsPerWorker));
        if (workers < 2) return 1;
        // Serial pre-allocation: the parallel body must not grow shared
        // buffers, or concurrent growth loses a resize and leaves null slots.
        workspace.PrepareRows(workers, resizedWidth);
        return workers;
    }

    // One destination row: two horizontal passes into the caller-owned row
    // scratch, then the vertical fixed-point blend and the normalization LUT.
    // Split out so the row loop can run serially or sharded across workers
    // without duplicating any arithmetic.
    private static unsafe void DetRow(byte* sourcePtr, int sourceStride, int sourceWidth,
        int sourceHeight, int resizedWidth, int resizedHeight, int oy, int[] xOffsets,
        short[] xCoefficients, int[] row0, int[] row1, float* outputPtr, float* normalizedPtr,
        int plane, bool nhwc, ImagePixelFormat format)
    {
        GetLinearCoordinate(oy, sourceHeight, resizedHeight,
            out int sy, out short beta0, out short beta1);
        int sy0 = MathCompat.Clamp(sy, 0, sourceHeight - 1);
        int sy1 = MathCompat.Clamp(sy + 1, 0, sourceHeight - 1);
        BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy0,
            resizedWidth, xOffsets, xCoefficients, row0, format);
        BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy1,
            resizedWidth, xOffsets, xCoefficients, row1, format);
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
                if (nhwc)
                    outputPtr[(destination + ox) * 3 + channel] = normalizedPtr[channel * 256 + value];
                else
                    outputPtr[channel * plane + destination + ox] = normalizedPtr[channel * 256 + value];
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
        int sourceY, int destinationWidth, int[] offsets, short[] coefficients, int[] destination,
        ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        byte* row = source + sourceY * sourceStride;
        if (format == ImagePixelFormat.Bgr24)
        {
            for (int x = 0; x < destinationWidth; x++)
            {
                int sx = offsets[x], sx1 = Math.Min(sx + 1, sourceWidth - 1);
                short coefficient0 = coefficients[x * 2], coefficient1 = coefficients[x * 2 + 1];
                int sourceOffset = sx * 3, sourceOffset1 = sx1 * 3, destinationOffset = x * 3;
                destination[destinationOffset] = row[sourceOffset] * coefficient0 + row[sourceOffset1] * coefficient1;
                destination[destinationOffset + 1] = row[sourceOffset + 1] * coefficient0 + row[sourceOffset1 + 1] * coefficient1;
                destination[destinationOffset + 2] = row[sourceOffset + 2] * coefficient0 + row[sourceOffset1 + 2] * coefficient1;
            }
            return;
        }
        if (format == ImagePixelFormat.Rgb24)
        {
            PixelRow.GatherRgb24(row, sourceWidth, destinationWidth, offsets, coefficients, destination);
            return;
        }
        PixelRow.Gather32(row, sourceWidth, destinationWidth, offsets, coefficients, destination,
            format == ImagePixelFormat.Rgba32);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int Cls(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, Span<float> output, ImagePixelFormat format = ImagePixelFormat.Bgr24) =>
        Cls(source, sourceWidth, sourceHeight, sourceStride, output, null, format: format);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int Cls(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, Span<float> output, ResizeWorkspace? workspace, bool nhwc = false,
        ImagePixelFormat format = ImagePixelFormat.Bgr24)
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
        ResizeBgrInterLinearToNchw(source, windowWidth, sourceHeight, sourceStride,
            actualWidth, height, width, output, workspace, nhwc, format);
        return actualWidth;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int Rec(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int targetWidth, Span<float> output,
        ImagePixelFormat format = ImagePixelFormat.Bgr24) =>
        Rec(source, sourceWidth, sourceHeight, sourceStride, targetWidth, output, null,
            format: format);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static int Rec(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, int targetWidth, Span<float> output, ResizeWorkspace? workspace, bool nhwc = false,
        ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride, format);
        if (targetWidth <= 0) throw new ArgumentOutOfRangeException(nameof(targetWidth));
        int height = 48;
        int plane = checked(height * targetWidth);
        if (output.Length != checked(3 * plane)) throw new ArgumentException("Invalid REC output size.");
        int actualWidth = (int)Math.Min(targetWidth,
            ((long)height * sourceWidth + sourceHeight - 1L) / sourceHeight);
        output.Clear();
        ResizeBgrInterLinearToNchw(source, sourceWidth, sourceHeight, sourceStride,
            actualWidth, height, targetWidth, output, workspace, nhwc, format);
        return actualWidth;
    }

    // REC only needs the byte-domain interpolation result as an intermediate
    // for normalization. Keep the exact OpenCV fixed-point rounding, but write
    // the three normalized planes directly and avoid a temporary byte image
    // plus a second full-frame traversal.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ResizeBgrInterLinearToNchw(ReadOnlySpan<byte> source,
        int sourceWidth, int sourceHeight, int sourceStride, int resizedWidth,
        int resizedHeight, int outputWidth, Span<float> output, ResizeWorkspace? workspace, bool nhwc,
        ImagePixelFormat format)
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
                        resizedWidth, xOffsets, xCoefficients, row0, format);
                    BuildHorizontalRow(sourcePtr, sourceStride, sourceWidth, sy1,
                        resizedWidth, xOffsets, xCoefficients, row1, format);
                    int destination = oy * outputWidth;
                    for (int ox = 0; ox < resizedWidth; ox++)
                    {
                        int rowOffset = ox * 3;
                        int h0 = row0[rowOffset], h1 = row1[rowOffset];
                        int b = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        h0 = row0[rowOffset + 1]; h1 = row1[rowOffset + 1];
                        int g = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        h0 = row0[rowOffset + 2]; h1 = row1[rowOffset + 2];
                        int r = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
                        float bv = normalizedPtr[MathCompat.Clamp(b, 0, 255)];
                        float gv = normalizedPtr[MathCompat.Clamp(g, 0, 255)];
                        float rv = normalizedPtr[MathCompat.Clamp(r, 0, 255)];
                        if (nhwc)
                        {
                            int pixel = (destination + ox) * 3;
                            outputPtr[pixel] = bv;
                            outputPtr[pixel + 1] = gv;
                            outputPtr[pixel + 2] = rv;
                        }
                        else
                        {
                            outputPtr[destination + ox] = bv;
                            outputPtr[plane + destination + ox] = gv;
                            outputPtr[plane * 2 + destination + ox] = rv;
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
