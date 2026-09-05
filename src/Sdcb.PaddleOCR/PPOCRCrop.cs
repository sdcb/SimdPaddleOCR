using System.Runtime.CompilerServices;

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
            for (int y = 0; y < unrotatedHeight; y++)
            {
                double v = (double)y / unrotatedHeight;
                int x = 0;
                Warp.MapRow(sourcePtr, sourceWidth, sourceHeight, sourceStride, cropPtr, outputWidth,
                    unrotatedWidth, unrotatedHeight, rotateVertical, a, b, c, d, e, f, g, h, y, v, ref x);
                for (; x < unrotatedWidth; x++)
                    Warp.ProcessPixel(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                        cropPtr, outputWidth, unrotatedWidth, unrotatedHeight,
                        rotateVertical, a, b, c, d, e, f, g, h, x, y, v);
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
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
        if (!MathCompat.IsFinite(widthValue) || !MathCompat.IsFinite(heightValue) || widthValue < 1 ||
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
            if (!MathCompat.IsFinite(denominator) || Math.Abs(denominator) <= PerspectiveEpsilon)
                return false;
            g = (dx3 * dy2 - dx2 * dy3) / denominator;
            h = (dx1 * dy3 - dx3 * dy1) / denominator;
            if (!MathCompat.IsFinite(g) || !MathCompat.IsFinite(h))
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

    private static void ValidateSource(ReadOnlySpan<byte> source, int width, int height, int stride)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (stride < checked(width * 3)) throw new ArgumentException("Source stride is too small.");
        long required = checked((long)(height - 1) * stride + width * 3L);
        if (required > source.Length) throw new ArgumentException("Source buffer is too small.");
    }
}
