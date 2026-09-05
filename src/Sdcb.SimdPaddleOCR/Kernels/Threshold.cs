using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static class Threshold
{
    // 0x0101.. expansion table: bit i of the movemask becomes byte i (0 or 1).
    private static readonly ulong[] MaskExpand = BuildMaskExpand();

    private static ulong[] BuildMaskExpand()
    {
        ulong[] table = new ulong[256];
        for (int mask = 0; mask < 256; mask++)
        {
            ulong value = 0;
            for (int bit = 0; bit < 8; bit++)
                if ((mask & (1 << bit)) != 0) value |= 1UL << (bit * 8);
            table[mask] = value;
        }
        return table;
    }

    internal static unsafe void Binarize(ReadOnlySpan<float> prediction, byte[] bitmap,
        int pixels, float threshold)
    {
        int i = 0;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported && pixels >= 16)
        {
            fixed (float* predictionPtr = prediction)
            fixed (byte* bitmapPtr = bitmap)
            {
                Vector512<float> vThreshold = Vector512.Create(threshold);
                Vector512<float> vAbsMask = Vector512.Create(int.MaxValue).AsSingle();
                Vector512<float> vInfinity = Vector512.Create(float.PositiveInfinity);
                Vector512<float> finiteAccumulator = Vector512<float>.Zero;
                fixed (ulong* expandPtr = MaskExpand)
                {
                    for (; i <= pixels - 16; i += 16)
                    {
                        Vector512<float> value = Avx512F.LoadVector512(predictionPtr + i);
                        finiteAccumulator = Avx512F.Or(
                            finiteAccumulator.AsInt32(),
                            Avx512F.Compare(
                                Avx512F.And(value.AsInt32(), vAbsMask.AsInt32()).AsSingle(),
                                vInfinity,
                                FloatComparisonMode.UnorderedNotLessThanNonSignaling).AsInt32()).AsSingle();
                        int mask = Avx512F.MoveMask(Avx512F.Compare(value, vThreshold,
                            FloatComparisonMode.OrderedGreaterThanNonSignaling));
                        *(ulong*)(bitmapPtr + i) = expandPtr[mask & 0xFF];
                        *(ulong*)(bitmapPtr + i + 8) = expandPtr[(mask >> 8) & 0xFF];
                    }
                }
                if (Avx512F.MoveMask(finiteAccumulator) != 0)
                    throw new InvalidDataException("Detector output contains a non-finite value.");
            }
        }
        else if (Avx.IsSupported && pixels >= 8)
        {
            fixed (float* predictionPtr = prediction)
            fixed (byte* bitmapPtr = bitmap)
            {
                Vector256<float> vThreshold = Vector256.Create(threshold);
                Vector256<float> vAbsMask = Vector256.Create(int.MaxValue).AsSingle();
                Vector256<float> vInfinity = Vector256.Create(float.PositiveInfinity);
                Vector256<float> finiteAccumulator = Vector256<float>.Zero;
                fixed (ulong* expandPtr = MaskExpand)
                {
                    for (; i <= pixels - 8; i += 8)
                    {
                        Vector256<float> value = Avx.LoadVector256(predictionPtr + i);
                        finiteAccumulator = Avx.Or(finiteAccumulator, Avx.Compare(
                            Avx.And(value, vAbsMask), vInfinity,
                            FloatComparisonMode.UnorderedNotLessThanNonSignaling));
                        int mask = Avx.MoveMask(Avx.Compare(value, vThreshold,
                            FloatComparisonMode.OrderedGreaterThanNonSignaling));
                        *(ulong*)(bitmapPtr + i) = expandPtr[mask];
                    }
                }
                if (Avx.MoveMask(finiteAccumulator) != 0)
                    throw new InvalidDataException("Detector output contains a non-finite value.");
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated && pixels >= Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            Vector<float> vThreshold = new(threshold);
            Vector<int> invalid = Vector<int>.Zero;
            fixed (float* predictionPtr = prediction)
            {
                for (; i <= pixels - width; i += width)
                {
                    Vector<float> value = SimdOps.VectorLoad(predictionPtr + i);
                    invalid |= SimdOps.VectorNonFiniteMask(value);
                    StoreBitmapBits(bitmap, i, Vector.AsVectorInt32(Vector.GreaterThan(value, vThreshold)), width);
                }
            }
            if (SimdOps.VectorAnyNonZero(invalid))
                throw new InvalidDataException("Detector output contains a non-finite value.");
        }
        for (; i < pixels; i++)
        {
            float value = prediction[i];
            if (!MathCompat.IsFinite(value))
                throw new InvalidDataException("Detector output contains a non-finite value.");
            bitmap[i] = value > threshold ? (byte)1 : (byte)0;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreBitmapBits(byte[] bitmap, int offset, Vector<int> bits, int width)
    {
        if (width == 8)
        {
            bitmap[offset] = (byte)(bits.GetElement(0) & 1);
            bitmap[offset + 1] = (byte)(bits.GetElement(1) & 1);
            bitmap[offset + 2] = (byte)(bits.GetElement(2) & 1);
            bitmap[offset + 3] = (byte)(bits.GetElement(3) & 1);
            bitmap[offset + 4] = (byte)(bits.GetElement(4) & 1);
            bitmap[offset + 5] = (byte)(bits.GetElement(5) & 1);
            bitmap[offset + 6] = (byte)(bits.GetElement(6) & 1);
            bitmap[offset + 7] = (byte)(bits.GetElement(7) & 1);
            return;
        }
        if (width == 4)
        {
            bitmap[offset] = (byte)(bits.GetElement(0) & 1);
            bitmap[offset + 1] = (byte)(bits.GetElement(1) & 1);
            bitmap[offset + 2] = (byte)(bits.GetElement(2) & 1);
            bitmap[offset + 3] = (byte)(bits.GetElement(3) & 1);
            return;
        }
        for (int lane = 0; lane < width; lane++)
            bitmap[offset + lane] = (byte)(bits.GetElement(lane) & 1);
    }
}
