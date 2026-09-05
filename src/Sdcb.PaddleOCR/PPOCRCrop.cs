using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

using Sdcb.PaddleOCR.OnnxSharp;
using Sdcb.PaddleOCR.Kernels;

namespace Sdcb.PaddleOCR;

internal static class PPOCRCrop
{
    private const double PerspectiveEpsilon = 1e-12;

    private readonly record struct Point(double X, double Y);

    private readonly record struct PerspectiveTransform(
        double A, double B, double C, double D, double E, double F, double G, double H,
        int UnrotatedWidth, int UnrotatedHeight, bool RotateVertical);

    public static (int Width, int Height, int ByteCount) GetSize(in PaddleOcrDetectionBox box)
    {
        if (!TryComputePerspective(box, out PerspectiveTransform transform))
            throw new InvalidDataException("Invalid detection quadrilateral.");
        int width = transform.RotateVertical ? transform.UnrotatedHeight : transform.UnrotatedWidth;
        int height = transform.RotateVertical ? transform.UnrotatedWidth : transform.UnrotatedHeight;
        long bytes = checked((long)width * height * 3);
        return (width, height, checked((int)bytes));
    }

    public static byte[] Extract(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, in PaddleOcrDetectionBox box, out int outputWidth, out int outputHeight)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride);
        (int Width, int Height, int ByteCount) size = GetSize(box);
        outputWidth = size.Width;
        outputHeight = size.Height;
        byte[] crop = new byte[size.ByteCount];
        ExtractCore(source, sourceWidth, sourceHeight, sourceStride, box, crop, outputWidth, outputHeight);
        return crop;
    }

    public static void ExtractInto(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, in PaddleOcrDetectionBox box, Span<byte> destination, out int outputWidth,
        out int outputHeight)
    {
        ValidateSource(source, sourceWidth, sourceHeight, sourceStride);
        (int Width, int Height, int ByteCount) size = GetSize(box);
        if (destination.Length < size.ByteCount)
            throw new ArgumentException("Destination buffer is too small.", nameof(destination));
        outputWidth = size.Width;
        outputHeight = size.Height;
        ExtractCore(source, sourceWidth, sourceHeight, sourceStride, box, destination,
            outputWidth, outputHeight);
    }

    private static unsafe void ExtractCore(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, in PaddleOcrDetectionBox box, Span<byte> crop, int outputWidth, int outputHeight)
    {
        if (!TryComputePerspective(box, out PerspectiveTransform transform))
            throw new InvalidDataException("Invalid perspective transform.");
        double a = transform.A, b = transform.B, c = transform.C, d = transform.D,
            e = transform.E, f = transform.F, g = transform.G, h = transform.H;
        int unrotatedWidth = transform.UnrotatedWidth, unrotatedHeight = transform.UnrotatedHeight;
        bool rotateVertical = transform.RotateVertical;

        fixed (byte* sourcePtr = source)
        fixed (byte* cropPtr = crop)
        {
            double* sxArr = stackalloc double[8];
            double* syArr = stackalloc double[8];
            for (int y = 0; y < unrotatedHeight; y++)
            {
                double v = (double)y / unrotatedHeight;
                int x = 0;
                if (Avx512F.IsSupported && unrotatedWidth >= 8)
                {
                    // Eight output pixels per iteration via Vector512<double>.
                    Vector512<double> vLane = Vector512.Create(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
                    Vector512<double> vWidth = Vector512.Create((double)unrotatedWidth);
                    Vector512<double> vOne = Vector512.Create(1.0);
                    Vector512<double> vEps = Vector512.Create(1e-12);
                    Vector512<double> vInf = Vector512.Create(double.PositiveInfinity);
                    Vector512<double> vAbsMask = Vector512.Create(long.MaxValue).AsDouble();
                    Vector512<double> vA = Vector512.Create(a), vD = Vector512.Create(d), vG = Vector512.Create(g);
                    Vector512<double> vV = Vector512.Create(v);
                    Vector512<double> bv = Avx512F.Multiply(Vector512.Create(b), vV);
                    Vector512<double> ev = Avx512F.Multiply(Vector512.Create(e), vV);
                    Vector512<double> hv = Avx512F.Multiply(Vector512.Create(h), vV);
                    Vector512<double> vC = Vector512.Create(c), vF = Vector512.Create(f);
                    for (; x <= unrotatedWidth - 8; x += 8)
                    {
                        Vector512<double> u = Avx512F.Divide(Avx512F.Add(Vector512.Create((double)x), vLane), vWidth);
                        Vector512<double> denominator = Avx512F.Add(Avx512F.Add(Avx512F.Multiply(vG, u), hv), vOne);
                        Vector512<double> absDenominator = Avx512F.And(denominator.AsInt64(), vAbsMask.AsInt64()).AsDouble();
                        Vector512<double> sx = Avx512F.Divide(Avx512F.Add(Avx512F.Add(Avx512F.Multiply(vA, u), bv), vC), denominator);
                        Vector512<double> sy = Avx512F.Divide(Avx512F.Add(Avx512F.Add(Avx512F.Multiply(vD, u), ev), vF), denominator);
                        Vector512<double> ok = Avx512F.And(
                            Avx512F.And(
                                Avx512F.Compare(absDenominator, vEps, FloatComparisonMode.OrderedGreaterThanNonSignaling).AsInt64(),
                                Avx512F.Compare(absDenominator, vInf, FloatComparisonMode.OrderedLessThanNonSignaling).AsInt64()),
                            Avx512F.And(
                                Avx512F.Compare(Avx512F.And(sx.AsInt64(), vAbsMask.AsInt64()).AsDouble(), vInf, FloatComparisonMode.OrderedLessThanNonSignaling).AsInt64(),
                                Avx512F.Compare(Avx512F.And(sy.AsInt64(), vAbsMask.AsInt64()).AsDouble(), vInf, FloatComparisonMode.OrderedLessThanNonSignaling).AsInt64())).AsDouble();
                        int okMask = Avx512DQ.IsSupported
                            ? Avx512DQ.MoveMask(ok)
                            : (int)Vector512.ExtractMostSignificantBits(ok);
                        if (okMask != 0xFF)
                        {
                            for (int lane = 0; lane < 8; lane++)
                                ProcessPixelScalar(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                                    cropPtr, outputWidth, unrotatedWidth, unrotatedHeight,
                                    rotateVertical, a, b, c, d, e, f, g, h, x + lane, y, v);
                            continue;
                        }
                        Avx512F.Store(sxArr, sx);
                        Avx512F.Store(syArr, sy);
                        for (int lane = 0; lane < 8; lane++)
                        {
                            double pixelX = sxArr[lane], pixelY = syArr[lane];
                            int xBase = (int)Math.Floor(pixelX), yBase = (int)Math.Floor(pixelY);
                            int destinationX = rotateVertical ? unrotatedHeight - 1 - y : x + lane;
                            int destinationY = rotateVertical ? x + lane : y;
                            int destination = checked((destinationY * outputWidth + destinationX) * 3);
                            if (xBase >= 1 && xBase < sourceWidth - 3 && yBase >= 1 && yBase < sourceHeight - 2)
                                Warp.SampleCubicAvx(sourcePtr, sourceStride, pixelX, pixelY,
                                    xBase, yBase, cropPtr, destination);
                            else
                                SamplePixelCubic(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                                    pixelX, pixelY, cropPtr, destination);
                        }
                    }
                }
                else if (Avx2.IsSupported && unrotatedWidth >= 4)
                {
                    // Four output pixels per iteration.  Every lane performs the
                    // exact scalar operation sequence (same associativity, no
                    // FMA contraction), so results are bit-identical.
                    Vector256<double> vLane = Vector256.Create(0.0, 1.0, 2.0, 3.0);
                    Vector256<double> vWidth = Vector256.Create((double)unrotatedWidth);
                    Vector256<double> vOne = Vector256.Create(1.0);
                    Vector256<double> vEps = Vector256.Create(1e-12);
                    Vector256<double> vInf = Vector256.Create(double.PositiveInfinity);
                    Vector256<double> vAbsMask = Vector256.Create(long.MaxValue).AsDouble();
                    Vector256<double> vA = Vector256.Create(a), vD = Vector256.Create(d), vG = Vector256.Create(g);
                    Vector256<double> vV = Vector256.Create(v);
                    Vector256<double> bv = Avx.Multiply(Vector256.Create(b), vV);
                    Vector256<double> ev = Avx.Multiply(Vector256.Create(e), vV);
                    Vector256<double> hv = Avx.Multiply(Vector256.Create(h), vV);
                    Vector256<double> vC = Vector256.Create(c), vF = Vector256.Create(f);
                    for (; x <= unrotatedWidth - 4; x += 4)
                    {
                        Vector256<double> u = Avx.Divide(Avx.Add(Vector256.Create((double)x), vLane), vWidth);
                        Vector256<double> denominator = Avx.Add(Avx.Add(Avx.Multiply(vG, u), hv), vOne);
                        Vector256<double> absDenominator = Avx.And(denominator, vAbsMask);
                        Vector256<double> sx = Avx.Divide(Avx.Add(Avx.Add(Avx.Multiply(vA, u), bv), vC), denominator);
                        Vector256<double> sy = Avx.Divide(Avx.Add(Avx.Add(Avx.Multiply(vD, u), ev), vF), denominator);
                        Vector256<double> ok = Avx.And(
                            Avx.And(
                                Avx.Compare(absDenominator, vEps, FloatComparisonMode.OrderedGreaterThanNonSignaling),
                                Avx.Compare(absDenominator, vInf, FloatComparisonMode.OrderedLessThanNonSignaling)),
                            Avx.And(
                                Avx.Compare(Avx.And(sx, vAbsMask), vInf, FloatComparisonMode.OrderedLessThanNonSignaling),
                                Avx.Compare(Avx.And(sy, vAbsMask), vInf, FloatComparisonMode.OrderedLessThanNonSignaling)));
                        if (Avx.MoveMask(ok) != 0xF)
                        {
                            for (int lane = 0; lane < 4; lane++)
                                ProcessPixelScalar(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                                    cropPtr, outputWidth, unrotatedWidth, unrotatedHeight,
                                    rotateVertical, a, b, c, d, e, f, g, h, x + lane, y, v);
                            continue;
                        }
                        Avx.Store(sxArr, sx);
                        Avx.Store(syArr, sy);
                        for (int lane = 0; lane < 4; lane++)
                        {
                            double pixelX = sxArr[lane], pixelY = syArr[lane];
                            int xBase = (int)Math.Floor(pixelX), yBase = (int)Math.Floor(pixelY);
                            int destinationX = rotateVertical ? unrotatedHeight - 1 - y : x + lane;
                            int destinationY = rotateVertical ? x + lane : y;
                            int destination = checked((destinationY * outputWidth + destinationX) * 3);
                            // The SIMD sampler loads 4 bytes per tap, so it needs
                            // one extra in-bounds column on the right.
                            if (xBase >= 1 && xBase < sourceWidth - 3 && yBase >= 1 && yBase < sourceHeight - 2)
                                Warp.SampleCubicAvx(sourcePtr, sourceStride, pixelX, pixelY,
                                    xBase, yBase, cropPtr, destination);
                            else
                                SamplePixelCubic(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                                    pixelX, pixelY, cropPtr, destination);
                        }
                    }
                }
                for (; x < unrotatedWidth; x++)
                    ProcessPixelScalar(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                        cropPtr, outputWidth, unrotatedWidth, unrotatedHeight,
                        rotateVertical, a, b, c, d, e, f, g, h, x, y, v);
            }
        }
    }

    private static unsafe void ProcessPixelScalar(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedWidth, int unrotatedHeight,
        bool rotateVertical, double a, double b, double c, double d, double e, double f,
        double g, double h, int x, int y, double v)
    {
        double u = (double)x / unrotatedWidth;
        double denominator = g * u + h * v + 1;
        if (!double.IsFinite(denominator) || Math.Abs(denominator) <= PerspectiveEpsilon)
            throw new InvalidDataException("Invalid perspective transform.");
        double sx = (a * u + b * v + c) / denominator;
        double sy = (d * u + e * v + f) / denominator;
        if (!double.IsFinite(sx) || !double.IsFinite(sy))
            throw new InvalidDataException("Invalid perspective transform.");
        int destinationX = rotateVertical ? unrotatedHeight - 1 - y : x;
        int destinationY = rotateVertical ? x : y;
        int destination = checked((destinationY * outputWidth + destinationX) * 3);
        SamplePixelCubic(sourcePtr, sourceWidth, sourceHeight, sourceStride,
            sx, sy, cropPtr, destination);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Rotate180(Span<byte> pixels, int width, int height)
    {
        int count = checked(width * height);
        fixed (byte* data = pixels)
        {
            for (int left = 0; left < count / 2; left++)
            {
                int right = count - 1 - left;
                byte* l = data + left * 3, r = data + right * 3;
                byte value = l[0]; l[0] = r[0]; r[0] = value;
                value = l[1]; l[1] = r[1]; r[1] = value;
                value = l[2]; l[2] = r[2]; r[2] = value;
            }
        }
    }

    private static bool TryComputePerspective(in PaddleOcrDetectionBox box,
        out PerspectiveTransform transform)
    {
        transform = default;
        Span<Point> points = stackalloc Point[4];
        LoadPoints(box, points);
        double widthValue = Math.Max(Distance(points[0], points[1]), Distance(points[2], points[3]));
        double heightValue = Math.Max(Distance(points[0], points[3]), Distance(points[1], points[2]));
        if (!double.IsFinite(widthValue) || !double.IsFinite(heightValue) || widthValue < 1 ||
            heightValue < 1 || widthValue > uint.MaxValue || heightValue > uint.MaxValue)
            return false;
        int unrotatedWidth = Math.Max(1, checked((int)Math.Floor(widthValue)));
        int unrotatedHeight = Math.Max(1, checked((int)Math.Floor(heightValue)));
        bool rotateVertical = unrotatedHeight >= unrotatedWidth * 1.5;

        double dx1 = points[1].X - points[2].X;
        double dx2 = points[3].X - points[2].X;
        double dx3 = points[0].X - points[1].X + points[2].X - points[3].X;
        double dy1 = points[1].Y - points[2].Y;
        double dy2 = points[3].Y - points[2].Y;
        double dy3 = points[0].Y - points[1].Y + points[2].Y - points[3].Y;
        double g = 0, h = 0;
        if (Math.Abs(dx3) > PerspectiveEpsilon || Math.Abs(dy3) > PerspectiveEpsilon)
        {
            double denominator = dx1 * dy2 - dx2 * dy1;
            if (!double.IsFinite(denominator) || Math.Abs(denominator) <= PerspectiveEpsilon)
                return false;
            g = (dx3 * dy2 - dx2 * dy3) / denominator;
            h = (dx1 * dy3 - dx3 * dy1) / denominator;
            if (!double.IsFinite(g) || !double.IsFinite(h))
                return false;
        }
        double a = points[1].X - points[0].X + g * points[1].X;
        double b = points[3].X - points[0].X + h * points[3].X;
        double c = points[0].X;
        double d = points[1].Y - points[0].Y + g * points[1].Y;
        double e = points[3].Y - points[0].Y + h * points[3].Y;
        double f = points[0].Y;
        transform = new PerspectiveTransform(a, b, c, d, e, f, g, h,
            unrotatedWidth, unrotatedHeight, rotateVertical);
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LoadPoints(in PaddleOcrDetectionBox box, Span<Point> points)
    {
        points[0] = new(box.X1, box.Y1);
        points[1] = new(box.X2, box.Y2);
        points[2] = new(box.X3, box.Y3);
        points[3] = new(box.X4, box.Y4);
    }

    private static double Distance(Point a, Point b) => Math.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SamplePixelCubic(byte* source, int width, int height, int stride,
        double x, double y, byte* destination, int destinationOffset)
    {
        int xBase = (int)Math.Floor(x), yBase = (int)Math.Floor(y);
        Span<double> xWeights =
        [
            CubicWeight(x - (xBase - 1)),
            CubicWeight(x - xBase),
            CubicWeight(x - (xBase + 1)),
            CubicWeight(x - (xBase + 2)),
        ];
        Span<double> yWeights =
        [
            CubicWeight(y - (yBase - 1)),
            CubicWeight(y - yBase),
            CubicWeight(y - (yBase + 1)),
            CubicWeight(y - (yBase + 2)),
        ];
        double value0 = 0, value1 = 0, value2 = 0;
        if (xBase >= 1 && xBase < width - 2 && yBase >= 1 && yBase < height - 2)
        {
            for (int ky = 0; ky < 4; ky++)
            {
                double wy = yWeights[ky];
                int sourceOffset = (yBase + ky - 1) * stride + (xBase - 1) * 3;
                for (int kx = 0; kx < 4; kx++, sourceOffset += 3)
                {
                    double wx = xWeights[kx];
                    value0 += source[sourceOffset] * wx * wy;
                    value1 += source[sourceOffset + 1] * wx * wy;
                    value2 += source[sourceOffset + 2] * wx * wy;
                }
            }
        }
        else
        {
            for (int ky = 0; ky < 4; ky++)
            {
                double wy = yWeights[ky];
                int sy = Clamp(yBase + ky - 1, height);
                for (int kx = 0; kx < 4; kx++)
                {
                    double wx = xWeights[kx];
                    int sx = Clamp(xBase + kx - 1, width);
                    int sourceOffset = sy * stride + sx * 3;
                    value0 += source[sourceOffset] * wx * wy;
                    value1 += source[sourceOffset + 1] * wx * wy;
                    value2 += source[sourceOffset + 2] * wx * wy;
                }
            }
        }
        destination[destinationOffset] = value0 <= 0 ? (byte)0 :
            value0 >= 255 ? (byte)255 : checked((byte)Math.Floor(value0 + 0.5));
        destination[destinationOffset + 1] = value1 <= 0 ? (byte)0 :
            value1 >= 255 ? (byte)255 : checked((byte)Math.Floor(value1 + 0.5));
        destination[destinationOffset + 2] = value2 <= 0 ? (byte)0 :
            value2 >= 255 ? (byte)255 : checked((byte)Math.Floor(value2 + 0.5));
    }

    // OpenCV INTER_CUBIC uses the Keys cubic kernel with a=-0.75.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double CubicWeight(double x)
    {
        x = Math.Abs(x);
        const double a = -0.75;
        return x <= 1 ? (a + 2) * x * x * x - (a + 3) * x * x + 1
            : x < 2 ? a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a : 0;
    }

    private static int Clamp(int value, int limit) => value < 0 ? 0 : value >= limit ? limit - 1 : value;

    private static void ValidateSource(ReadOnlySpan<byte> source, int width, int height, int stride)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (stride < checked(width * 3)) throw new ArgumentException("Source stride is too small.");
        long required = checked((long)(height - 1) * stride + width * 3L);
        if (required > source.Length) throw new ArgumentException("Source buffer is too small.");
    }
}
