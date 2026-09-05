using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.PaddleOCR.Kernels;

internal static class ArgMax
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe int Find(ReadOnlySpan<float> row, out float bestValue)
    {
        bestValue = row[0];
        if (!MathCompat.IsFinite(bestValue))
            throw new InvalidDataException("Recognizer output is invalid.");
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported && row.Length >= 16)
        {
            fixed (float* rowPtr = row)
            {
                Vector512<float> maxVector = Vector512.Create(bestValue);
                Vector512<float> absMask = Vector512.Create(unchecked((int)0x7fffffff)).AsSingle();
                Vector512<float> finiteMax = Vector512.Create(float.MaxValue);
                int i = 1;
                for (; i <= row.Length - 16; i += 16)
                {
                    Vector512<float> value = Avx512F.LoadVector512(rowPtr + i);
                    Vector512<float> invalid = Avx512F.Or(
                        Avx512F.Compare(value, value, FloatComparisonMode.UnorderedNonSignaling).AsInt32(),
                        Avx512F.Compare(
                            Avx512F.And(value.AsInt32(), absMask.AsInt32()).AsSingle(),
                            finiteMax,
                            FloatComparisonMode.OrderedGreaterThanNonSignaling).AsInt32()).AsSingle();
                    if (Avx512F.MoveMask(invalid) != 0)
                        throw new InvalidDataException("Recognizer output is invalid.");
                    maxVector = Avx512F.Max(maxVector, value);
                }
                float maximum = maxVector.GetElement(0);
                for (int lane = 1; lane < 16; lane++) maximum = MathF.Max(maximum, maxVector.GetElement(lane));
                for (; i < row.Length; i++)
                {
                    float value = row[i];
                    if (!MathCompat.IsFinite(value)) throw new InvalidDataException("Recognizer output is invalid.");
                    if (value > maximum) maximum = value;
                }
                if (!(maximum > bestValue))
                    return 0;
                Vector512<float> maximumVector = Vector512.Create(maximum);
                i = 1;
                for (; i <= row.Length - 16; i += 16)
                {
                    int mask = Avx512F.MoveMask(Avx512F.Compare(Avx512F.LoadVector512(rowPtr + i), maximumVector,
                        FloatComparisonMode.OrderedEqualNonSignaling));
                    if (mask != 0)
                    {
                        int index = i + BitOperations.TrailingZeroCount((uint)mask);
                        bestValue = row[index];
                        return index;
                    }
                }
                for (; i < row.Length; i++)
                    if (row[i] == maximum) { bestValue = row[i]; return i; }
            }
            throw new InvalidDataException("Recognizer output is invalid.");
        }
        else if (Avx2.IsSupported && row.Length >= 8)
        {
            fixed (float* rowPtr = row)
            {
                Vector256<float> maxVector = Vector256.Create(bestValue);
                Vector256<float> absMask = Vector256.Create(unchecked((int)0x7fffffff)).AsSingle();
                Vector256<float> finiteMax = Vector256.Create(float.MaxValue);
                int i = 1;
                for (; i <= row.Length - 8; i += 8)
                {
                    Vector256<float> value = Avx.LoadVector256(rowPtr + i);
                    Vector256<float> invalid = Avx.Or(
                        Avx.Compare(value, value, FloatComparisonMode.UnorderedNonSignaling),
                        Avx.Compare(Avx.And(value, absMask), finiteMax,
                            FloatComparisonMode.OrderedGreaterThanNonSignaling));
                    if (Avx.MoveMask(invalid) != 0)
                        throw new InvalidDataException("Recognizer output is invalid.");
                    maxVector = Avx.Max(maxVector, value);
                }
                float maximum = maxVector.GetElement(0);
                for (int lane = 1; lane < 8; lane++) maximum = MathF.Max(maximum, maxVector.GetElement(lane));
                for (; i < row.Length; i++)
                {
                    float value = row[i];
                    if (!MathCompat.IsFinite(value)) throw new InvalidDataException("Recognizer output is invalid.");
                    if (value > maximum) maximum = value;
                }
                if (!(maximum > bestValue))
                    return 0;
                Vector256<float> maximumVector = Vector256.Create(maximum);
                i = 1;
                for (; i <= row.Length - 8; i += 8)
                {
                    int mask = Avx.MoveMask(Avx.Compare(Avx.LoadVector256(rowPtr + i), maximumVector,
                        FloatComparisonMode.OrderedEqualNonSignaling));
                    if (mask != 0)
                    {
                        int index = i + BitOperations.TrailingZeroCount((uint)mask);
                        bestValue = row[index];
                        return index;
                    }
                }
                for (; i < row.Length; i++)
                    if (row[i] == maximum) { bestValue = row[i]; return i; }
            }
            throw new InvalidDataException("Recognizer output is invalid.");
        }
        else
#endif
        if (Vector.IsHardwareAccelerated && row.Length >= Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            fixed (float* rowPtr = row)
            {
                Vector<float> maxVector = new(bestValue);
                int i = 1;
                for (; i <= row.Length - width; i += width)
                {
                    Vector<float> value = SimdOps.VectorLoad(rowPtr + i);
                    for (int lane = 0; lane < width; lane++)
                    {
                        float laneValue = value.GetElement(lane);
                        if (!MathCompat.IsFinite(laneValue))
                            throw new InvalidDataException("Recognizer output is invalid.");
                    }
                    maxVector = Vector.Max(maxVector, value);
                }
                float maximum = maxVector[0];
                for (int lane = 1; lane < width; lane++) maximum = MathF.Max(maximum, maxVector.GetElement(lane));
                for (; i < row.Length; i++)
                {
                    float value = row[i];
                    if (!MathCompat.IsFinite(value)) throw new InvalidDataException("Recognizer output is invalid.");
                    if (value > maximum) maximum = value;
                }
                if (!(maximum > bestValue))
                    return 0;
                i = 1;
                for (; i < row.Length; i++)
                    if (row[i] == maximum) { bestValue = row[i]; return i; }
            }
            throw new InvalidDataException("Recognizer output is invalid.");
        }
        int scalarBest = 0;
        for (int c = 1; c < row.Length; c++)
        {
            float value = row[c];
            if (!MathCompat.IsFinite(value)) throw new InvalidDataException("Recognizer output is invalid.");
            if (value > bestValue) { bestValue = value; scalarBest = c; }
        }
        return scalarBest;
    }
}
