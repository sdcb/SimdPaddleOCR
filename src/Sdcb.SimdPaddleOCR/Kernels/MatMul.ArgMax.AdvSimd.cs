using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
#endif
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class MatMul
{
#if !NETSTANDARD2_0
    // NEON fused projection+argmax. The MAC loop mirrors
    // MatMulArgMaxPackedVector but runs on Vector128 with lane-indexed FMA
    // (no per-k broadcast); the argmax epilogue converts each accumulator
    // back to Vector<float> and reuses UpdateVector/ReduceVector verbatim.

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxPackedAdvSimd(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores,
        int batch, int rows, int inner, int columns, int threads)
    {
        int full4 = rows / 4, tail = rows - full4 * 4;
        int rowJobs = full4 + tail;
        int jobs = checked(batch * rowJobs);
        int workers = Math.Min(ArgMaxWorkers(batch, rows, inner, columns, threads), jobs);
        bool hasBias = !bias.IsEmpty;
        fixed (float* inputPin = input, weightsPin = weights,
            packedPin = packedWeights, biasPin = bias)
        fixed (int* indicesPin = indices)
        fixed (float* scoresPin = scores)
        {
            float* inputPtr = inputPin, weightsPtr = weightsPin,
                packedPtr = packedPin, biasPtr = biasPin;
            int* indicesPtr = indicesPin;
            float* scoresPtr = scoresPin;
            void Worker(int w)
            {
                for (int job = w; job < jobs; job += workers)
                {
                    int b = job / rowJobs, j = job - b * rowJobs;
                    if (j < full4)
                        Rows4(inputPtr, weightsPtr, packedPtr,
                            biasPtr, hasBias, indicesPtr, scoresPtr, b, j * 4,
                            rows, inner, columns);
                    else
                        Rows1(inputPtr, weightsPtr, packedPtr,
                            biasPtr, hasBias, indicesPtr, scoresPtr, b,
                            full4 * 4 + j - full4, rows, inner, columns);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Rows4(float* input, float* weights,
        float* packed, float* bias, bool hasBias,
        int* indices, float* scores, int batch, int row,
        int rows, int inner, int columns)
    {
        Span<Vector<float>> vectorMaxima = stackalloc Vector<float>[4];
        Span<Vector<float>> vectorIndices = stackalloc Vector<float>[4];
        float* maxima = stackalloc float[4];
        int* best = stackalloc int[4];
        vectorMaxima[0] = vectorMaxima[1] = vectorMaxima[2] = vectorMaxima[3] =
            new Vector<float>(float.NegativeInfinity);
        int inputBase = (batch * rows + row) * inner;
        int col = 0;
        for (; col <= columns - 16; col += 16)
        {
            Vector128<float> a00 = Vector128<float>.Zero, a01 = a00, a02 = a00, a03 = a00;
            Vector128<float> a10 = Vector128<float>.Zero, a11 = a10, a12 = a10, a13 = a10;
            Vector128<float> a20 = Vector128<float>.Zero, a21 = a20, a22 = a20, a23 = a20;
            Vector128<float> a30 = Vector128<float>.Zero, a31 = a30, a32 = a30, a33 = a30;
            float* tile = packed + (col / 16) * inner * 16;
            int k = 0;
            for (; k <= inner - 4; k += 4)
            {
                Vector128<float> x0 = AdvSimd.LoadVector128(input + inputBase + k);
                Vector128<float> x1 = AdvSimd.LoadVector128(input + inputBase + inner + k);
                Vector128<float> x2 = AdvSimd.LoadVector128(input + inputBase + inner * 2 + k);
                Vector128<float> x3 = AdvSimd.LoadVector128(input + inputBase + inner * 3 + k);
                float* wrow = tile + k * 16;
                // Lane index must be an instruction constant, so j unrolls.
                Vector128<float> w0 = AdvSimd.LoadVector128(wrow);
                Vector128<float> w1 = AdvSimd.LoadVector128(wrow + 4);
                Vector128<float> w2 = AdvSimd.LoadVector128(wrow + 8);
                Vector128<float> w3 = AdvSimd.LoadVector128(wrow + 12);
                a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, x0, 0);
                a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, x0, 0);
                a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, x0, 0);
                a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, x0, 0);
                a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, x1, 0);
                a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, x1, 0);
                a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, x1, 0);
                a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, x1, 0);
                a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, x2, 0);
                a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, x2, 0);
                a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, x2, 0);
                a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, x2, 0);
                a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, x3, 0);
                a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, x3, 0);
                a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, x3, 0);
                a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, x3, 0);
                wrow += 16;
                w0 = AdvSimd.LoadVector128(wrow);
                w1 = AdvSimd.LoadVector128(wrow + 4);
                w2 = AdvSimd.LoadVector128(wrow + 8);
                w3 = AdvSimd.LoadVector128(wrow + 12);
                a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, x0, 1);
                a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, x0, 1);
                a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, x0, 1);
                a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, x0, 1);
                a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, x1, 1);
                a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, x1, 1);
                a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, x1, 1);
                a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, x1, 1);
                a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, x2, 1);
                a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, x2, 1);
                a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, x2, 1);
                a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, x2, 1);
                a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, x3, 1);
                a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, x3, 1);
                a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, x3, 1);
                a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, x3, 1);
                wrow += 16;
                w0 = AdvSimd.LoadVector128(wrow);
                w1 = AdvSimd.LoadVector128(wrow + 4);
                w2 = AdvSimd.LoadVector128(wrow + 8);
                w3 = AdvSimd.LoadVector128(wrow + 12);
                a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, x0, 2);
                a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, x0, 2);
                a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, x0, 2);
                a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, x0, 2);
                a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, x1, 2);
                a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, x1, 2);
                a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, x1, 2);
                a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, x1, 2);
                a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, x2, 2);
                a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, x2, 2);
                a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, x2, 2);
                a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, x2, 2);
                a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, x3, 2);
                a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, x3, 2);
                a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, x3, 2);
                a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, x3, 2);
                wrow += 16;
                w0 = AdvSimd.LoadVector128(wrow);
                w1 = AdvSimd.LoadVector128(wrow + 4);
                w2 = AdvSimd.LoadVector128(wrow + 8);
                w3 = AdvSimd.LoadVector128(wrow + 12);
                a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, x0, 3);
                a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, x0, 3);
                a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, x0, 3);
                a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, x0, 3);
                a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, x1, 3);
                a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, x1, 3);
                a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, x1, 3);
                a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, x1, 3);
                a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, x2, 3);
                a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, x2, 3);
                a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, x2, 3);
                a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, x2, 3);
                a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, x3, 3);
                a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, x3, 3);
                a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, x3, 3);
                a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, x3, 3);
            }
            for (; k < inner; k++)
            {
                float* wrow = tile + k * 16;
                Vector128<float> w0 = AdvSimd.LoadVector128(wrow);
                Vector128<float> w1 = AdvSimd.LoadVector128(wrow + 4);
                Vector128<float> w2 = AdvSimd.LoadVector128(wrow + 8);
                Vector128<float> w3 = AdvSimd.LoadVector128(wrow + 12);
                Vector64<float> x0 = Vector64.CreateScalarUnsafe(input[inputBase + k]);
                Vector64<float> x1 = Vector64.CreateScalarUnsafe(input[inputBase + inner + k]);
                Vector64<float> x2 = Vector64.CreateScalarUnsafe(input[inputBase + inner * 2 + k]);
                Vector64<float> x3 = Vector64.CreateScalarUnsafe(input[inputBase + inner * 3 + k]);
                a00 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a00, w0, x0);
                a01 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a01, w1, x0);
                a02 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a02, w2, x0);
                a03 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a03, w3, x0);
                a10 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a10, w0, x1);
                a11 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a11, w1, x1);
                a12 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a12, w2, x1);
                a13 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a13, w3, x1);
                a20 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a20, w0, x2);
                a21 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a21, w1, x2);
                a22 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a22, w2, x2);
                a23 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a23, w3, x2);
                a30 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a30, w0, x3);
                a31 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a31, w1, x3);
                a32 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a32, w2, x3);
                a33 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a33, w3, x3);
            }
            if (hasBias)
            {
                for (int p = 0; p < 4; p++)
                {
                    Vector128<float> bv = AdvSimd.LoadVector128(bias + col + p * 4);
                    if (p == 0) { a00 += bv; a10 += bv; a20 += bv; a30 += bv; }
                    else if (p == 1) { a01 += bv; a11 += bv; a21 += bv; a31 += bv; }
                    else if (p == 2) { a02 += bv; a12 += bv; a22 += bv; a32 += bv; }
                    else { a03 += bv; a13 += bv; a23 += bv; a33 += bv; }
                }
            }
            UpdateVector(a00.AsVector(), col, 0, vectorMaxima, vectorIndices);
            UpdateVector(a01.AsVector(), col + 4, 0, vectorMaxima, vectorIndices);
            UpdateVector(a02.AsVector(), col + 8, 0, vectorMaxima, vectorIndices);
            UpdateVector(a03.AsVector(), col + 12, 0, vectorMaxima, vectorIndices);
            UpdateVector(a10.AsVector(), col, 1, vectorMaxima, vectorIndices);
            UpdateVector(a11.AsVector(), col + 4, 1, vectorMaxima, vectorIndices);
            UpdateVector(a12.AsVector(), col + 8, 1, vectorMaxima, vectorIndices);
            UpdateVector(a13.AsVector(), col + 12, 1, vectorMaxima, vectorIndices);
            UpdateVector(a20.AsVector(), col, 2, vectorMaxima, vectorIndices);
            UpdateVector(a21.AsVector(), col + 4, 2, vectorMaxima, vectorIndices);
            UpdateVector(a22.AsVector(), col + 8, 2, vectorMaxima, vectorIndices);
            UpdateVector(a23.AsVector(), col + 12, 2, vectorMaxima, vectorIndices);
            UpdateVector(a30.AsVector(), col, 3, vectorMaxima, vectorIndices);
            UpdateVector(a31.AsVector(), col + 4, 3, vectorMaxima, vectorIndices);
            UpdateVector(a32.AsVector(), col + 8, 3, vectorMaxima, vectorIndices);
            UpdateVector(a33.AsVector(), col + 12, 3, vectorMaxima, vectorIndices);
        }
        ReduceVector(vectorMaxima, vectorIndices, maxima, best);
        FinishScalarTail(input, weights, bias, hasBias, indices, scores,
            batch, row, 4, rows, inner, columns, col, maxima, best);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Rows1(float* input, float* weights,
        float* packed, float* bias, bool hasBias,
        int* indices, float* scores, int batch, int row,
        int rows, int inner, int columns)
    {
        Span<Vector<float>> vectorMaxima = stackalloc Vector<float>[1];
        Span<Vector<float>> vectorIndices = stackalloc Vector<float>[1];
        float* maxima = stackalloc float[1];
        int* best = stackalloc int[1];
        vectorMaxima[0] = new Vector<float>(float.NegativeInfinity);
        int inputBase = (batch * rows + row) * inner;
        int col = 0;
        for (; col <= columns - 16; col += 16)
        {
            Vector128<float> a0 = Vector128<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
            float* tile = packed + (col / 16) * inner * 16;
            int k = 0;
            for (; k <= inner - 4; k += 4)
            {
                Vector128<float> x = AdvSimd.LoadVector128(input + inputBase + k);
                float* wrow = tile + k * 16;
                a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0,
                    AdvSimd.LoadVector128(wrow), x, 0);
                a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1,
                    AdvSimd.LoadVector128(wrow + 4), x, 0);
                a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2,
                    AdvSimd.LoadVector128(wrow + 8), x, 0);
                a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3,
                    AdvSimd.LoadVector128(wrow + 12), x, 0);
                wrow += 16;
                a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0,
                    AdvSimd.LoadVector128(wrow), x, 1);
                a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1,
                    AdvSimd.LoadVector128(wrow + 4), x, 1);
                a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2,
                    AdvSimd.LoadVector128(wrow + 8), x, 1);
                a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3,
                    AdvSimd.LoadVector128(wrow + 12), x, 1);
                wrow += 16;
                a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0,
                    AdvSimd.LoadVector128(wrow), x, 2);
                a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1,
                    AdvSimd.LoadVector128(wrow + 4), x, 2);
                a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2,
                    AdvSimd.LoadVector128(wrow + 8), x, 2);
                a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3,
                    AdvSimd.LoadVector128(wrow + 12), x, 2);
                wrow += 16;
                a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0,
                    AdvSimd.LoadVector128(wrow), x, 3);
                a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1,
                    AdvSimd.LoadVector128(wrow + 4), x, 3);
                a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2,
                    AdvSimd.LoadVector128(wrow + 8), x, 3);
                a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3,
                    AdvSimd.LoadVector128(wrow + 12), x, 3);
            }
            for (; k < inner; k++)
            {
                float* wrow = tile + k * 16;
                Vector64<float> x = Vector64.CreateScalarUnsafe(input[inputBase + k]);
                a0 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a0, AdvSimd.LoadVector128(wrow), x);
                a1 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a1, AdvSimd.LoadVector128(wrow + 4), x);
                a2 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a2, AdvSimd.LoadVector128(wrow + 8), x);
                a3 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a3, AdvSimd.LoadVector128(wrow + 12), x);
            }
            if (hasBias)
            {
                a0 += AdvSimd.LoadVector128(bias + col);
                a1 += AdvSimd.LoadVector128(bias + col + 4);
                a2 += AdvSimd.LoadVector128(bias + col + 8);
                a3 += AdvSimd.LoadVector128(bias + col + 12);
            }
            UpdateVector(a0.AsVector(), col, 0, vectorMaxima, vectorIndices);
            UpdateVector(a1.AsVector(), col + 4, 0, vectorMaxima, vectorIndices);
            UpdateVector(a2.AsVector(), col + 8, 0, vectorMaxima, vectorIndices);
            UpdateVector(a3.AsVector(), col + 12, 0, vectorMaxima, vectorIndices);
        }
        ReduceVector(vectorMaxima, vectorIndices, maxima, best);
        FinishScalarTail(input, weights, bias, hasBias, indices, scores,
            batch, row, 1, rows, inner, columns, col, maxima, best);
    }
#endif
}
