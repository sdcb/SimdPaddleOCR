using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class MatMul
{
    internal static bool CanFuseArgMax(int rows, int inner, int columns, float[]? packedWeights)
    {
        if (packedWeights is null || inner < 64 || columns < 1024 || rows < 1)
            return false;
#if !NETSTANDARD2_0
        if (Avx512F.IsSupported && rows >= 8 && (rows & 7) == 0)
            return true;
        else if (Avx2.IsSupported && rows >= 4 && (rows & 3) == 0)
            return true;
        else
#endif
        return Vector.IsHardwareAccelerated && Vector<float>.Count > 0 &&
            (16 % Vector<float>.Count) == 0;
    }

    internal static bool TryArgMax(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias,
        Span<int> indices, Span<float> scores, int batch, int rows,
        int inner, int columns, float[]? packedWeights)
    {
        if (!CanFuseArgMax(rows, inner, columns, packedWeights) ||
            indices.Length != batch * rows || scores.Length != batch * rows)
            return false;
        if (!bias.IsEmpty && bias.Length != columns) return false;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported && rows >= 8 && (rows & 7) == 0)
        {
            MatMulArgMaxPackedAvx512(input, weights, packedWeights, bias,
                indices, scores, batch, rows, inner, columns);
            return true;
        }
        else if (Avx2.IsSupported && rows >= 4 && (rows & 3) == 0)
        {
            MatMulArgMaxPackedAvx(input, weights, packedWeights, bias,
                indices, scores, batch, rows, inner, columns);
            return true;
        }
        else
        #endif
        if (Vector.IsHardwareAccelerated && (16 % Vector<float>.Count) == 0)
        {
            MatMulArgMaxPackedVector(input, weights, packedWeights!, bias,
                indices, scores, batch, rows, inner, columns);
            return true;
        }

        return false;
    }

    private static unsafe void FinishScalarTail(float* input, float* weights,
        float* bias, bool hasBias, Span<int> indices, Span<float> scores,
        int batch, int row, int blockRows, int rows, int inner, int columns,
        int firstColumn, Span<float> maxima, Span<int> best)
    {
        int inputBase = (batch * rows + row) * inner;
        for (int r = 0; r < blockRows; r++)
        {
            for (int column = firstColumn; column < columns; column++)
            {
                float value = hasBias ? bias[column] : 0;
                for (int k = 0; k < inner; k++)
                    value += input[inputBase + r * inner + k] * weights[k * columns + column];
                if (!MathCompat.IsFinite(value))
                    throw new InvalidDataException("Recognizer output is invalid.");
                float oldMax = maxima[r];
                if (value > oldMax)
                {
                    maxima[r] = value;
                    best[r] = column;
                }
            }
            int outputRow = batch * rows + row + r;
            indices[outputRow] = best[r];
            scores[outputRow] = maxima[r];
        }
    }
}
