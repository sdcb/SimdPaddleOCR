using System.Runtime.CompilerServices;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Warp
{
    internal const double PerspectiveEpsilon = 1e-12;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void StoreClampedRgb(byte* destination, int destinationOffset,
        double value0, double value1, double value2)
    {
        destination[destinationOffset] = value0 <= 0 ? (byte)0 :
            value0 >= 255 ? (byte)255 : checked((byte)Math.Floor(value0 + 0.5));
        destination[destinationOffset + 1] = value1 <= 0 ? (byte)0 :
            value1 >= 255 ? (byte)255 : checked((byte)Math.Floor(value1 + 0.5));
        destination[destinationOffset + 2] = value2 <= 0 ? (byte)0 :
            value2 >= 255 ? (byte)255 : checked((byte)Math.Floor(value2 + 0.5));
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void ProcessPixel(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedWidth, int unrotatedHeight,
        bool rotateVertical, double a, double b, double c, double d, double e, double f,
        double g, double h, int x, int y, double v)
    {
        double u = (double)x / unrotatedWidth;
        double denominator = g * u + h * v + 1;
        if (!MathCompat.IsFinite(denominator) || Math.Abs(denominator) <= PerspectiveEpsilon)
            throw new InvalidDataException("Invalid perspective transform.");
        double sx = (a * u + b * v + c) / denominator;
        double sy = (d * u + e * v + f) / denominator;
        if (!MathCompat.IsFinite(sx) || !MathCompat.IsFinite(sy))
            throw new InvalidDataException("Invalid perspective transform.");
        int destinationX = rotateVertical ? unrotatedHeight - 1 - y : x;
        int destinationY = rotateVertical ? x : y;
        int destination = checked((destinationY * outputWidth + destinationX) * 3);
        SampleCubicScalar(sourcePtr, sourceWidth, sourceHeight, sourceStride,
            sx, sy, cropPtr, destination);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void SampleCubicScalar(byte* source, int width, int height, int stride,
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
        StoreClampedRgb(destination, destinationOffset, value0, value1, value2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double CubicWeight(double x)
    {
        x = Math.Abs(x);
        const double a = -0.75;
        return x <= 1 ? (a + 2) * x * x * x - (a + 3) * x * x + 1
            : x < 2 ? a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a : 0;
    }

    private static int Clamp(int value, int limit) => value < 0 ? 0 : value >= limit ? limit - 1 : value;
}
