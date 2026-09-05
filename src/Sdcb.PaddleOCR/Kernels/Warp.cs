using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.PaddleOCR.Kernels;

internal static class Warp
{
    // Interior-only SIMD sampler: channels live in vector lanes (lane 3 is a
    // don't-care), so each channel keeps the scalar accumulation order and
    // the result is bit-identical to PPOCRCrop.SamplePixelCubic.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void SampleCubicAvx(byte* source, int stride,
        double x, double y, int xBase, int yBase, byte* destination, int destinationOffset)
    {
        Vector256<double> tx = Avx.Subtract(Vector256.Create(x),
            Vector256.Create((double)(xBase - 1), xBase, xBase + 1, xBase + 2));
        Vector256<double> ty = Avx.Subtract(Vector256.Create(y),
            Vector256.Create((double)(yBase - 1), yBase, yBase + 1, yBase + 2));
        Vector256<double> wx = CubicWeightVector(tx);
        Vector256<double> wy = CubicWeightVector(ty);
        Vector256<double> wx0 = Avx2.Permute4x64(wx, 0x00), wx1 = Avx2.Permute4x64(wx, 0x55);
        Vector256<double> wx2 = Avx2.Permute4x64(wx, 0xAA), wx3 = Avx2.Permute4x64(wx, 0xFF);
        byte* row = source + (yBase - 1) * stride + (xBase - 1) * 3;
        Vector256<double> acc = AccumulateRow(row, wx0, wx1, wx2, wx3,
            Avx2.Permute4x64(wy, 0x00), Vector256<double>.Zero);
        acc = AccumulateRow(row + stride, wx0, wx1, wx2, wx3, Avx2.Permute4x64(wy, 0x55), acc);
        acc = AccumulateRow(row + 2 * stride, wx0, wx1, wx2, wx3, Avx2.Permute4x64(wy, 0xAA), acc);
        acc = AccumulateRow(row + 3 * stride, wx0, wx1, wx2, wx3, Avx2.Permute4x64(wy, 0xFF), acc);
        double value0 = acc.GetElement(0), value1 = acc.GetElement(1), value2 = acc.GetElement(2);
        destination[destinationOffset] = value0 <= 0 ? (byte)0 :
            value0 >= 255 ? (byte)255 : checked((byte)Math.Floor(value0 + 0.5));
        destination[destinationOffset + 1] = value1 <= 0 ? (byte)0 :
            value1 >= 255 ? (byte)255 : checked((byte)Math.Floor(value1 + 0.5));
        destination[destinationOffset + 2] = value2 <= 0 ? (byte)0 :
            value2 >= 255 ? (byte)255 : checked((byte)Math.Floor(value2 + 0.5));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<double> AccumulateRow(byte* row,
        Vector256<double> wx0, Vector256<double> wx1, Vector256<double> wx2, Vector256<double> wx3,
        Vector256<double> wy, Vector256<double> acc)
    {
        acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(LoadPixel(row), wx0), wy));
        acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(LoadPixel(row + 3), wx1), wy));
        acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(LoadPixel(row + 6), wx2), wy));
        acc = Avx.Add(acc, Avx.Multiply(Avx.Multiply(LoadPixel(row + 9), wx3), wy));
        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<double> LoadPixel(byte* pixel) =>
        Avx.ConvertToVector256Double(Sse41.ConvertToVector128Int32(pixel));

    // Same piecewise Keys polynomials as PPOCRCrop.CubicWeight, evaluated with
    // the exact scalar operation order per lane.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> CubicWeightVector(Vector256<double> value)
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
