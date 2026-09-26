using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class MatMul
{
    internal static bool CanFuseArgMax(int rows, int inner, int columns, float[]? packedWeights)
    {
        _ = packedWeights;
        return inner >= 64 && columns >= 1024 && rows >= 1;
    }

    internal static bool TryArgMax(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias,
        Span<int> indices, Span<float> scores, int batch, int rows,
        int inner, int columns, float[]? packedWeights, int threads = 1)
    {
        if (!CanFuseArgMax(rows, inner, columns, packedWeights) ||
            indices.Length != batch * rows || scores.Length != batch * rows)
            return false;
        if (!bias.IsEmpty && bias.Length != columns) return false;
        #if !NETSTANDARD2_0
        if (packedWeights is not null && Avx512F.IsSupported && rows >= 8 && (rows & 7) == 0)
        {
            MatMulArgMaxPackedAvx512(input, weights, packedWeights, bias,
                indices, scores, batch, rows, inner, columns, threads);
            return true;
        }
        else if (packedWeights is not null && Avx2.IsSupported && rows >= 4 && (rows & 3) == 0)
        {
            MatMulArgMaxPackedAvx(input, weights, packedWeights, bias,
                indices, scores, batch, rows, inner, columns, threads);
            return true;
        }
        else if (packedWeights is not null && AdvSimd.Arm64.IsSupported)
        {
            MatMulArgMaxPackedAdvSimd(input, weights, packedWeights, bias,
                indices, scores, batch, rows, inner, columns, threads);
            return true;
        }
        else
        #endif
        if (packedWeights is not null && Vector.IsHardwareAccelerated && (16 % Vector<float>.Count) == 0)
        {
            MatMulArgMaxPackedVector(input, weights, packedWeights!, bias,
                indices, scores, batch, rows, inner, columns, threads);
            return true;
        }

        MatMulArgMaxScalar(input, weights, bias, indices, scores, batch, rows, inner, columns, threads);
        return true;
    }

    // Row-block sharding shared by every ArgMax impl: jobs are (batch, row
    // block) pairs writing disjoint output rows, so workers stride the job
    // space. Work is total MACs; 2M is well past the point where Parallel.For
    // dispatch is noise (the conv kernels use a higher 8M cutoff, but CTC
    // projections sit just under it while still being worth sharding).
    private static int ArgMaxWorkers(int batch, int rows, int inner, int columns, int threads)
    {
        long work = (long)batch * rows * inner * columns;
        return threads > 1 && work >= 2_000_000 ? Math.Min(threads, Math.Max(1, batch * rows)) : 1;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxScalar(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias,
        Span<int> indices, Span<float> scores, int batch, int rows,
        int inner, int columns, int threads)
    {
        bool hasBias = !bias.IsEmpty;
        int full4 = rows / 4, tail = rows - full4 * 4;
        int rowJobs = full4 + tail;
        int jobs = checked(batch * rowJobs);
        int workers = Math.Min(ArgMaxWorkers(batch, rows, inner, columns, threads), jobs);
        fixed (float* inputPin = input, weightsPin = weights, biasPin = bias)
        fixed (int* indicesPin = indices)
        fixed (float* scoresPin = scores)
        {
            // Captured by the worker lambda, so they cannot be the fixed
            // locals themselves (CS1764); copies stay valid while pinned.
            float* inputPtr = inputPin, weightsPtr = weightsPin, biasPtr = biasPin;
            int* indicesPtr = indicesPin;
            float* scoresPtr = scoresPin;
            void Block4(int b, int row)
            {
                float m0 = float.NegativeInfinity, m1 = float.NegativeInfinity;
                float m2 = float.NegativeInfinity, m3 = float.NegativeInfinity;
                int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
                int col = 0;
                for (; col <= columns - 4; col += 4)
                {
                    float a00 = hasBias ? biasPtr[col] : 0, a01 = hasBias ? biasPtr[col + 1] : 0;
                    float a02 = hasBias ? biasPtr[col + 2] : 0, a03 = hasBias ? biasPtr[col + 3] : 0;
                    float a10 = a00, a11 = a01, a12 = a02, a13 = a03;
                    float a20 = a00, a21 = a01, a22 = a02, a23 = a03;
                    float a30 = a00, a31 = a01, a32 = a02, a33 = a03;
                    int inputBase = (b * rows + row) * inner;
                    float* weightCursor = weightsPtr + col;
                    for (int k = 0; k < inner; k++)
                    {
                        float v0 = inputPtr[inputBase + k];
                        float v1 = inputPtr[inputBase + inner + k];
                        float v2 = inputPtr[inputBase + inner * 2 + k];
                        float v3 = inputPtr[inputBase + inner * 3 + k];
                        float w0 = weightCursor[0], w1 = weightCursor[1], w2 = weightCursor[2], w3 = weightCursor[3];
                        a00 += v0 * w0; a01 += v0 * w1; a02 += v0 * w2; a03 += v0 * w3;
                        a10 += v1 * w0; a11 += v1 * w1; a12 += v1 * w2; a13 += v1 * w3;
                        a20 += v2 * w0; a21 += v2 * w1; a22 += v2 * w2; a23 += v2 * w3;
                        a30 += v3 * w0; a31 += v3 * w1; a32 += v3 * w2; a33 += v3 * w3;
                        weightCursor += columns;
                    }
                    UpdateArgMax(a00, col, ref m0, ref i0);
                    UpdateArgMax(a01, col + 1, ref m0, ref i0);
                    UpdateArgMax(a02, col + 2, ref m0, ref i0);
                    UpdateArgMax(a03, col + 3, ref m0, ref i0);
                    UpdateArgMax(a10, col, ref m1, ref i1);
                    UpdateArgMax(a11, col + 1, ref m1, ref i1);
                    UpdateArgMax(a12, col + 2, ref m1, ref i1);
                    UpdateArgMax(a13, col + 3, ref m1, ref i1);
                    UpdateArgMax(a20, col, ref m2, ref i2);
                    UpdateArgMax(a21, col + 1, ref m2, ref i2);
                    UpdateArgMax(a22, col + 2, ref m2, ref i2);
                    UpdateArgMax(a23, col + 3, ref m2, ref i2);
                    UpdateArgMax(a30, col, ref m3, ref i3);
                    UpdateArgMax(a31, col + 1, ref m3, ref i3);
                    UpdateArgMax(a32, col + 2, ref m3, ref i3);
                    UpdateArgMax(a33, col + 3, ref m3, ref i3);
                }
                for (; col < columns; col++)
                {
                    float s0 = hasBias ? biasPtr[col] : 0, s1 = s0, s2 = s0, s3 = s0;
                    int inputBase = (b * rows + row) * inner;
                    for (int k = 0; k < inner; k++)
                    {
                        float w = weightsPtr[k * columns + col];
                        s0 += inputPtr[inputBase + k] * w;
                        s1 += inputPtr[inputBase + inner + k] * w;
                        s2 += inputPtr[inputBase + inner * 2 + k] * w;
                        s3 += inputPtr[inputBase + inner * 3 + k] * w;
                    }
                    UpdateArgMax(s0, col, ref m0, ref i0);
                    UpdateArgMax(s1, col, ref m1, ref i1);
                    UpdateArgMax(s2, col, ref m2, ref i2);
                    UpdateArgMax(s3, col, ref m3, ref i3);
                }
                int outputRow = b * rows + row;
                WriteArgMax(indicesPtr, scoresPtr, outputRow, i0, m0);
                WriteArgMax(indicesPtr, scoresPtr, outputRow + 1, i1, m1);
                WriteArgMax(indicesPtr, scoresPtr, outputRow + 2, i2, m2);
                WriteArgMax(indicesPtr, scoresPtr, outputRow + 3, i3, m3);
            }

            void Row1(int b, int row)
            {
                float max = float.NegativeInfinity;
                int best = 0;
                int inputBase = (b * rows + row) * inner;
                for (int col = 0; col < columns; col++)
                {
                    float sum = hasBias ? biasPtr[col] : 0;
                    for (int k = 0; k < inner; k++)
                        sum += inputPtr[inputBase + k] * weightsPtr[k * columns + col];
                    UpdateArgMax(sum, col, ref max, ref best);
                }
                WriteArgMax(indicesPtr, scoresPtr, b * rows + row, best, max);
            }

            void Worker(int w)
            {
                for (int job = w; job < jobs; job += workers)
                {
                    int b = job / rowJobs, j = job - b * rowJobs;
                    if (j < full4) Block4(b, j * 4);
                    else Row1(b, full4 * 4 + j - full4);
                }
            }

            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UpdateArgMax(float value, int column, ref float max, ref int best)
    {
        if (!MathCompat.IsFinite(value))
            throw new InvalidDataException("Recognizer output is invalid.");
        if (value > max)
        {
            max = value;
            best = column;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void WriteArgMax(int* indices, float* scores, int row, int index, float score)
    {
        indices[row] = index;
        scores[row] = score;
    }

    private static unsafe void FinishScalarTail(float* input, float* weights,
        float* bias, bool hasBias, int* indices, float* scores,
        int batch, int row, int blockRows, int rows, int inner, int columns,
        int firstColumn, float* maxima, int* best)
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
