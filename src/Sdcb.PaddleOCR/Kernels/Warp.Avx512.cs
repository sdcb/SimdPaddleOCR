using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Warp
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MapRowAvx512(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedWidth, int unrotatedHeight,
        bool rotateVertical, double a, double b, double c, double d, double e, double f,
        double g, double h, int y, double v, ref int x)
    {
        double* sxArr = stackalloc double[8];
        double* syArr = stackalloc double[8];
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
                    ProcessPixel(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                        cropPtr, outputWidth, unrotatedWidth, unrotatedHeight,
                        rotateVertical, a, b, c, d, e, f, g, h, x + lane, y, v);
                continue;
            }
            Avx512F.Store(sxArr, sx);
            Avx512F.Store(syArr, sy);
            for (int lane = 0; lane < 8; lane++)
                SampleMappedPixel(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                    cropPtr, outputWidth, unrotatedHeight, rotateVertical,
                    sxArr[lane], syArr[lane], x + lane, y);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void SampleCubicAvx512(byte* source, int stride,
        double x, double y, int xBase, int yBase, byte* destination, int destinationOffset)
    {
        Vector512<double> tx = Avx512F.Subtract(Vector512.Create(x),
            Vector512.Create((double)(xBase - 1), xBase, xBase + 1, xBase + 2, 0, 0, 0, 0));
        Vector512<double> ty = Avx512F.Subtract(Vector512.Create(y),
            Vector512.Create((double)(yBase - 1), yBase, yBase + 1, yBase + 2, 0, 0, 0, 0));
        Vector512<double> wx = CubicWeightVectorAvx512(tx);
        Vector512<double> wy = CubicWeightVectorAvx512(ty);
        Vector512<double> wx0 = Vector512.Create(wx.GetElement(0));
        Vector512<double> wx1 = Vector512.Create(wx.GetElement(1));
        Vector512<double> wx2 = Vector512.Create(wx.GetElement(2));
        Vector512<double> wx3 = Vector512.Create(wx.GetElement(3));
        byte* row = source + (yBase - 1) * stride + (xBase - 1) * 3;
        Vector512<double> acc = AccumulateRowAvx512(row, wx0, wx1, wx2, wx3,
            Vector512.Create(wy.GetElement(0)), Vector512<double>.Zero);
        acc = AccumulateRowAvx512(row + stride, wx0, wx1, wx2, wx3, Vector512.Create(wy.GetElement(1)), acc);
        acc = AccumulateRowAvx512(row + 2 * stride, wx0, wx1, wx2, wx3, Vector512.Create(wy.GetElement(2)), acc);
        acc = AccumulateRowAvx512(row + 3 * stride, wx0, wx1, wx2, wx3, Vector512.Create(wy.GetElement(3)), acc);
        StoreClampedRgb(destination, destinationOffset,
            acc.GetElement(0), acc.GetElement(1), acc.GetElement(2));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector512<double> AccumulateRowAvx512(byte* row,
        Vector512<double> wx0, Vector512<double> wx1, Vector512<double> wx2, Vector512<double> wx3,
        Vector512<double> wy, Vector512<double> acc)
    {
        acc = Avx512F.Add(acc, Avx512F.Multiply(Avx512F.Multiply(LoadPixelAvx512(row), wx0), wy));
        acc = Avx512F.Add(acc, Avx512F.Multiply(Avx512F.Multiply(LoadPixelAvx512(row + 3), wx1), wy));
        acc = Avx512F.Add(acc, Avx512F.Multiply(Avx512F.Multiply(LoadPixelAvx512(row + 6), wx2), wy));
        acc = Avx512F.Add(acc, Avx512F.Multiply(Avx512F.Multiply(LoadPixelAvx512(row + 9), wx3), wy));
        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector512<double> LoadPixelAvx512(byte* pixel) =>
        Vector512.Create(Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(pixel)),
            Vector256<double>.Zero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<double> CubicWeightVectorAvx512(Vector512<double> value)
    {
        Vector512<double> t = Avx512F.And(value.AsInt64(), Vector512.Create(long.MaxValue)).AsDouble();
        Vector512<double> one = Vector512.Create(1.0);
        Vector512<double> inner = Avx512F.Add(Avx512F.Subtract(
            Avx512F.Multiply(Avx512F.Multiply(Avx512F.Multiply(Vector512.Create(1.25), t), t), t),
            Avx512F.Multiply(Avx512F.Multiply(Vector512.Create(2.25), t), t)), one);
        Vector512<double> outer = Avx512F.Subtract(Avx512F.Add(Avx512F.Subtract(
            Avx512F.Multiply(Avx512F.Multiply(Avx512F.Multiply(Vector512.Create(-0.75), t), t), t),
            Avx512F.Multiply(Avx512F.Multiply(Vector512.Create(-3.75), t), t)),
            Avx512F.Multiply(Vector512.Create(-6.0), t)), Vector512.Create(-3.0));
        Vector512<double> result = Avx512F.BlendVariable(Vector512<double>.Zero, outer,
            Avx512F.Compare(t, Vector512.Create(2.0), FloatComparisonMode.OrderedLessThanNonSignaling));
        return Avx512F.BlendVariable(result, inner,
            Avx512F.Compare(t, one, FloatComparisonMode.OrderedLessThanOrEqualNonSignaling));
    }
}
