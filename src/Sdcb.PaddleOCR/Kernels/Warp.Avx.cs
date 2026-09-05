using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Warp
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MapRowAvx(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedWidth, int unrotatedHeight,
        bool rotateVertical, double a, double b, double c, double d, double e, double f,
        double g, double h, int y, double v, ref int x)
    {
        double* sxArr = stackalloc double[4];
        double* syArr = stackalloc double[4];
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
                    ProcessPixel(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                        cropPtr, outputWidth, unrotatedWidth, unrotatedHeight,
                        rotateVertical, a, b, c, d, e, f, g, h, x + lane, y, v);
                continue;
            }
            Avx.Store(sxArr, sx);
            Avx.Store(syArr, sy);
            for (int lane = 0; lane < 4; lane++)
                SampleMappedPixel(sourcePtr, sourceWidth, sourceHeight, sourceStride,
                    cropPtr, outputWidth, unrotatedHeight, rotateVertical,
                    sxArr[lane], syArr[lane], x + lane, y);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void SampleCubicAvx(byte* source, int stride,
        double x, double y, int xBase, int yBase, byte* destination, int destinationOffset)
    {
        Vector256<double> tx = Avx.Subtract(Vector256.Create(x),
            Vector256.Create((double)(xBase - 1), xBase, xBase + 1, xBase + 2));
        Vector256<double> ty = Avx.Subtract(Vector256.Create(y),
            Vector256.Create((double)(yBase - 1), yBase, yBase + 1, yBase + 2));
        Vector256<double> wx = CubicWeightVectorAvx(tx);
        Vector256<double> wy = CubicWeightVectorAvx(ty);
        Vector256<double> wx0 = Avx2.Permute4x64(wx, 0x00), wx1 = Avx2.Permute4x64(wx, 0x55);
        Vector256<double> wx2 = Avx2.Permute4x64(wx, 0xAA), wx3 = Avx2.Permute4x64(wx, 0xFF);
        byte* row = source + (yBase - 1) * stride + (xBase - 1) * 3;
        Vector256<double> acc = AccumulateRowAvx(row, wx0, wx1, wx2, wx3,
            Avx2.Permute4x64(wy, 0x00), Vector256<double>.Zero);
        acc = AccumulateRowAvx(row + stride, wx0, wx1, wx2, wx3, Avx2.Permute4x64(wy, 0x55), acc);
        acc = AccumulateRowAvx(row + 2 * stride, wx0, wx1, wx2, wx3, Avx2.Permute4x64(wy, 0xAA), acc);
        acc = AccumulateRowAvx(row + 3 * stride, wx0, wx1, wx2, wx3, Avx2.Permute4x64(wy, 0xFF), acc);
        StoreClampedRgb(destination, destinationOffset,
            acc.GetElement(0), acc.GetElement(1), acc.GetElement(2));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<double> AccumulateRowAvx(byte* row,
        Vector256<double> wx0, Vector256<double> wx1, Vector256<double> wx2, Vector256<double> wx3,
        Vector256<double> wy, Vector256<double> acc)
    {
        acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(LoadPixelAvx(row), wx0), wy));
        acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(LoadPixelAvx(row + 3), wx1), wy));
        acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(LoadPixelAvx(row + 6), wx2), wy));
        acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(LoadPixelAvx(row + 9), wx3), wy));
        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<double> LoadPixelAvx(byte* pixel) =>
        Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(pixel));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> CubicWeightVectorAvx(Vector256<double> value)
    {
        Vector256<double> t = Avx.And(value, Vector256.Create(long.MaxValue).AsDouble());
        Vector256<double> one = Vector256.Create(1.0);
        Vector256<double> inner = Avx.Add(Avx.Subtract(
            Avx.Multiply(Avx.Multiply(Avx.Multiply(Vector256.Create(1.25), t), t), t),
            Avx.Multiply(Avx.Multiply(Vector256.Create(2.25), t), t)), one);
        Vector256<double> outer = Avx.Subtract(Avx.Add(Avx.Subtract(
            Avx.Multiply(Avx.Multiply(Avx.Multiply(Vector256.Create(-0.75), t), t), t),
            Avx.Multiply(Avx.Multiply(Vector256.Create(-3.75), t), t)),
            Avx.Multiply(Vector256.Create(-6.0), t)), Vector256.Create(-3.0));
        Vector256<double> result = Avx.BlendVariable(Vector256<double>.Zero, outer,
            Avx.Compare(t, Vector256.Create(2.0), FloatComparisonMode.OrderedLessThanNonSignaling));
        return Avx.BlendVariable(result, inner,
            Avx.Compare(t, one, FloatComparisonMode.OrderedLessThanOrEqualNonSignaling));
    }
}
