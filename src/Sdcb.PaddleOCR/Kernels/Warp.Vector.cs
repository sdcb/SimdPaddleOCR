using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Warp
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MapRowVector(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedWidth, int unrotatedHeight,
        bool rotateVertical, double a, double b, double c, double d, double e, double f,
        double g, double h, int y, double v, ref int x)
    {
        int width = Vector<double>.Count;
        Vector<double> vLane = Vector<double>.Zero;
        for (int lane = 0; lane < width; lane++)
            vLane = vLane.WithElement(lane, lane);
        Vector<double> vWidth = new((double)unrotatedWidth);
        Vector<double> vOne = new(1.0);
        Vector<double> vA = new(a), vD = new(d), vG = new(g);
        Vector<double> vV = new(v);
        Vector<double> bv = new Vector<double>(b) * vV;
        Vector<double> ev = new Vector<double>(e) * vV;
        Vector<double> hv = new Vector<double>(h) * vV;
        Vector<double> vC = new(c), vF = new(f);
        for (; x <= unrotatedWidth - width; x += width)
        {
            Vector<double> u = (new Vector<double>(x) + vLane) / vWidth;
            Vector<double> denominator = vG * u + hv + vOne;
            Vector<double> sx = (vA * u + bv + vC) / denominator;
            Vector<double> sy = (vD * u + ev + vF) / denominator;
            bool allOk = true;
            for (int lane = 0; lane < width; lane++)
            {
                double denom = denominator.GetElement(lane);
                if (!MathCompat.IsFinite(denom) || Math.Abs(denom) <= PerspectiveEpsilon ||
                    !MathCompat.IsFinite(sx.GetElement(lane)) || !MathCompat.IsFinite(sy.GetElement(lane)))
                {
                    allOk = false;
                    break;
                }
            }
            if (!allOk)
            {
                for (int lane = 0; lane < width; lane++)
                    ProcessPixel(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                        cropPtr, outputWidth, unrotatedWidth, unrotatedHeight,
                        rotateVertical, a, b, c, d, e, f, g, h, x + lane, y, v);
                continue;
            }
            for (int lane = 0; lane < width; lane++)
                SampleMappedPixel(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                    cropPtr, outputWidth, unrotatedHeight, rotateVertical,
                    sx.GetElement(lane), sy.GetElement(lane), x + lane, y);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void SampleCubicVector(byte* source, int stride,
        double x, double y, int xBase, int yBase, byte* destination, int destinationOffset)
    {
        Vector<double> tap = Vector<double>.Zero;
        tap = tap.WithElement(0, xBase - 1.0).WithElement(1, xBase)
            .WithElement(2, xBase + 1.0).WithElement(3, xBase + 2.0);
        Vector<double> wx = CubicWeightVector(new Vector<double>(x) - tap);
        tap = Vector<double>.Zero;
        tap = tap.WithElement(0, yBase - 1.0).WithElement(1, yBase)
            .WithElement(2, yBase + 1.0).WithElement(3, yBase + 2.0);
        Vector<double> wy = CubicWeightVector(new Vector<double>(y) - tap);
        Vector<double> wx0 = new(wx.GetElement(0)), wx1 = new(wx.GetElement(1));
        Vector<double> wx2 = new(wx.GetElement(2)), wx3 = new(wx.GetElement(3));
        byte* row = source + (yBase - 1) * stride + (xBase - 1) * 3;
        Vector<double> acc = AccumulateRowVector(row, wx0, wx1, wx2, wx3,
            new Vector<double>(wy.GetElement(0)), Vector<double>.Zero);
        acc = AccumulateRowVector(row + stride, wx0, wx1, wx2, wx3, new Vector<double>(wy.GetElement(1)), acc);
        acc = AccumulateRowVector(row + 2 * stride, wx0, wx1, wx2, wx3, new Vector<double>(wy.GetElement(2)), acc);
        acc = AccumulateRowVector(row + 3 * stride, wx0, wx1, wx2, wx3, new Vector<double>(wy.GetElement(3)), acc);
        StoreClampedRgb(destination, destinationOffset,
            acc.GetElement(0), acc.GetElement(1), acc.GetElement(2));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector<double> AccumulateRowVector(byte* row,
        Vector<double> wx0, Vector<double> wx1, Vector<double> wx2, Vector<double> wx3,
        Vector<double> wy, Vector<double> acc)
    {
        acc += LoadPixelVector(row) * wx0 * wy;
        acc += LoadPixelVector(row + 3) * wx1 * wy;
        acc += LoadPixelVector(row + 6) * wx2 * wy;
        acc += LoadPixelVector(row + 9) * wx3 * wy;
        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector<double> LoadPixelVector(byte* pixel)
    {
        Vector<double> value = Vector<double>.Zero;
        return value.WithElement(0, pixel[0]).WithElement(1, pixel[1]).WithElement(2, pixel[2]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> CubicWeightVector(Vector<double> value)
    {
        Vector<double> t = Vector.Abs(value);
        Vector<double> one = new(1.0);
        Vector<double> inner = new Vector<double>(1.25) * t * t * t - new Vector<double>(2.25) * t * t + one;
        Vector<double> outer = new Vector<double>(-0.75) * t * t * t - new Vector<double>(-3.75) * t * t
            + new Vector<double>(-6.0) * t - new Vector<double>(-3.0);
        Vector<double> result = Vector.ConditionalSelect(Vector.LessThan(t, new Vector<double>(2.0)),
            outer, Vector<double>.Zero);
        return Vector.ConditionalSelect(Vector.LessThanOrEqual(t, one), inner, result);
    }
}
