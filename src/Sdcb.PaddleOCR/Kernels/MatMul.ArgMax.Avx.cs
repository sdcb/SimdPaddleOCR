using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class MatMul
{
    private static readonly Vector256<float> ArgMaxLanes256 =
        Vector256.Create(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxPackedAvx(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores,
        int batch, int rows, int inner, int columns)
    {
        Span<Vector256<float>> vectorMaxima = stackalloc Vector256<float>[4];
        Span<Vector256<float>> vectorIndices = stackalloc Vector256<float>[4];
        Span<float> maxima = stackalloc float[4];
        Span<int> best = stackalloc int[4];
        fixed (float* inputPtr = input, weightsPtr = weights,
            packedPtr = packedWeights, biasPtr = bias)
        {
            for (int b = 0; b < batch; b++)
                for (int row = 0; row < rows; row += 4)
                {
                    vectorMaxima.Fill(Vector256.Create(float.NegativeInfinity));
                    int inputBase = (b * rows + row) * inner;
                    int col = 0;
                    for (; col <= columns - 16; col += 16)
                    {
                        Vector256<float> a0l = Vector256<float>.Zero, a0h = a0l;
                        Vector256<float> a1l = a0l, a1h = a0l;
                        Vector256<float> a2l = a0l, a2h = a0l;
                        Vector256<float> a3l = a0l, a3h = a0l;
                        float* tile = packedPtr + (col / 16) * inner * 16;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector256<float> wl = Avx.LoadVector256(tile + k * 16);
                            Vector256<float> wh = Avx.LoadVector256(tile + k * 16 + 8);
                            Vector256<float> v0 = Vector256.Create(inputPtr[inputBase + k]);
                            Vector256<float> v1 = Vector256.Create(inputPtr[inputBase + inner + k]);
                            Vector256<float> v2 = Vector256.Create(inputPtr[inputBase + inner * 2 + k]);
                            Vector256<float> v3 = Vector256.Create(inputPtr[inputBase + inner * 3 + k]);
                            a0l = AddMul(a0l, v0, wl); a0h = AddMul(a0h, v0, wh);
                            a1l = AddMul(a1l, v1, wl); a1h = AddMul(a1h, v1, wh);
                            a2l = AddMul(a2l, v2, wl); a2h = AddMul(a2h, v2, wh);
                            a3l = AddMul(a3l, v3, wl); a3h = AddMul(a3h, v3, wh);
                        }
                        if (!bias.IsEmpty)
                        {
                            Vector256<float> bl = Avx.LoadVector256(biasPtr + col);
                            Vector256<float> bh = Avx.LoadVector256(biasPtr + col + 8);
                            a0l = Avx.Add(a0l, bl); a0h = Avx.Add(a0h, bh);
                            a1l = Avx.Add(a1l, bl); a1h = Avx.Add(a1h, bh);
                            a2l = Avx.Add(a2l, bl); a2h = Avx.Add(a2h, bh);
                            a3l = Avx.Add(a3l, bl); a3h = Avx.Add(a3h, bh);
                        }
                        Update256(a0l, col, 0, vectorMaxima, vectorIndices);
                        Update256(a0h, col + 8, 0, vectorMaxima, vectorIndices);
                        Update256(a1l, col, 1, vectorMaxima, vectorIndices);
                        Update256(a1h, col + 8, 1, vectorMaxima, vectorIndices);
                        Update256(a2l, col, 2, vectorMaxima, vectorIndices);
                        Update256(a2h, col + 8, 2, vectorMaxima, vectorIndices);
                        Update256(a3l, col, 3, vectorMaxima, vectorIndices);
                        Update256(a3h, col + 8, 3, vectorMaxima, vectorIndices);
                    }
                    Reduce256(vectorMaxima, vectorIndices, maxima, best);
                    FinishScalarTail(inputPtr, weightsPtr, biasPtr, !bias.IsEmpty,
                        indices, scores, b, row, 4, rows, inner, columns, col,
                        maxima, best);
                }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Update256(Vector256<float> values, int column, int row,
        Span<Vector256<float>> maxima, Span<Vector256<float>> indices)
    {
        Vector256<float> previous = maxima[row];
        Vector256<float> replace = Avx.Compare(values, previous,
            FloatComparisonMode.OrderedGreaterThanNonSignaling);
        maxima[row] = Avx.Max(previous, values);
        Vector256<float> candidate = Avx.Add(Vector256.Create((float)column), ArgMaxLanes256);
        indices[row] = Avx.BlendVariable(indices[row], candidate, replace);
    }

    private static void Reduce256(ReadOnlySpan<Vector256<float>> vectorMaxima,
        ReadOnlySpan<Vector256<float>> vectorIndices, Span<float> maxima, Span<int> best)
    {
        for (int row = 0; row < vectorMaxima.Length; row++)
        {
            Vector256<float> values = vectorMaxima[row];
            Vector256<float> positions = vectorIndices[row];
            float maximum = values.GetElement(0);
            int index = (int)positions.GetElement(0);
            for (int lane = 1; lane < 8; lane++)
            {
                float value = values.GetElement(lane);
                int candidate = (int)positions.GetElement(lane);
                if (value > maximum || value == maximum && candidate < index)
                {
                    maximum = value;
                    index = candidate;
                }
            }
            if (!MathCompat.IsFinite(maximum))
                throw new InvalidDataException("Recognizer output is invalid.");
            maxima[row] = maximum;
            best[row] = index;
        }
    }
}
