using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class MatMul
{
    private static readonly Vector512<float> ArgMaxLanes512 =
        Vector512.Create(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f,
            8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxPackedAvx512(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores,
        int batch, int rows, int inner, int columns, int threads)
    {
        // rows % 8 == 0 (dispatch gate): per batch there are rows/16
        // sixteen-row blocks plus at most one trailing eight-row block.
        int full16 = rows / 16, tail8 = rows - full16 * 16 != 0 ? 1 : 0;
        int rowJobs = full16 + tail8;
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
                    if (j < full16)
                        MatMulArgMaxBlock16Avx512(inputPtr, weightsPtr, packedPtr,
                            biasPtr, hasBias, indicesPtr, scoresPtr, b, j * 16,
                            rows, inner, columns);
                    else
                        MatMulArgMaxBlock8Avx512(inputPtr, weightsPtr, packedPtr,
                            biasPtr, hasBias, indicesPtr, scoresPtr, b, full16 * 16,
                            rows, inner, columns);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    /// <summary>
    /// One pass over every row of a streamed REC batch (units back to back,
    /// each with rows % 4 == 0, i.e. rows the AVX-512 or AVX2 packed kernel
    /// would take) with the column tiles, not the rows, split across workers:
    /// the vocab matrix is streamed once per call instead of once per row
    /// block per worker per unit. Every column value is the same FMA chain
    /// as in those kernels (both are fused), and the per-lane maxima of the
    /// workers' column ranges are merged in column order with the Update512
    /// rule, which reproduces the sequential lane state for finite values;
    /// the lane reduction and scalar column tail are the shared ones.
    /// </summary>
    private static unsafe bool TryArgMaxUnitsColumnsAvx512(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias,
        Span<int> indices, Span<float> scores, ReadOnlySpan<int> batches, ReadOnlySpan<int> rows,
        int inner, int columns, float[]? packedWeights, int threads)
    {
        if (packedWeights is null || !Avx512F.IsSupported || threads <= 1) return false;
        int totalRows = 0;
        for (int u = 0; u < batches.Length; u++)
        {
            if (rows[u] < 4 || (rows[u] & 3) != 0) return false;
            totalRows = checked(totalRows + batches[u] * rows[u]);
        }
        if (totalRows == 0 || !CanFuseArgMax(totalRows, inner, columns, packedWeights)
            || indices.Length != totalRows || scores.Length != totalRows
            || input.Length != (long)totalRows * inner
            || (!bias.IsEmpty && bias.Length != columns))
            return false;
        int tiles = columns / 16;
        int workers = Math.Min(Math.Min(threads, tiles),
            ArgMaxWorkers(1, totalRows, inner, columns, threads));
        if (workers <= 1) return false;
        bool hasBias = !bias.IsEmpty;
        int blockTiles = Math.Max(1, 256 * 1024 / (inner * 16 * sizeof(float)));
        int stateLen = workers * totalRows;
        Vector512<float>[] allMaxima = System.Buffers.ArrayPool<Vector512<float>>.Shared.Rent(stateLen);
        Vector512<float>[] allIndices = System.Buffers.ArrayPool<Vector512<float>>.Shared.Rent(stateLen);
        try
        {
            fixed (float* inputPin = input, weightsPin = weights,
                packedPin = packedWeights, biasPin = bias)
            fixed (int* indicesPin = indices)
            fixed (float* scoresPin = scores)
            {
                float* inputPtr = inputPin, packedPtr = packedPin, biasPtr = biasPin;
                void Worker(int w)
                {
                    int tBeg = (int)((long)tiles * w / workers), tEnd = (int)((long)tiles * (w + 1) / workers);
                    Span<Vector512<float>> vm = allMaxima.AsSpan(w * totalRows, totalRows);
                    Span<Vector512<float>> vi = allIndices.AsSpan(w * totalRows, totalRows);
                    vm.Fill(Vector512.Create(float.NegativeInfinity));
                    vi.Clear();
                    for (int t0 = tBeg; t0 < tEnd; t0 += blockTiles)
                    {
                        int t1 = Math.Min(tEnd, t0 + blockTiles);
                        int row = 0;
                        for (; row + 16 <= totalRows; row += 16)
                            ArgMaxTiles16Avx512(inputPtr, packedPtr, biasPtr, hasBias,
                                row * inner, inner, t0, t1, vm.Slice(row, 16), vi.Slice(row, 16));
                        if (row + 8 <= totalRows)
                        {
                            ArgMaxTiles8Avx512(inputPtr, packedPtr, biasPtr, hasBias,
                                row * inner, inner, t0, t1, vm.Slice(row, 8), vi.Slice(row, 8));
                            row += 8;
                        }
                        if (row < totalRows)
                            ArgMaxTiles4Avx512(inputPtr, packedPtr, biasPtr, hasBias,
                                row * inner, inner, t0, t1, vm.Slice(row, 4), vi.Slice(row, 4));
                    }
                }
                Parallel.For(0, workers, Worker);

                float* maxima = stackalloc float[1];
                int* best = stackalloc int[1];
                Span<Vector512<float>> m1 = stackalloc Vector512<float>[1];
                Span<Vector512<float>> i1 = stackalloc Vector512<float>[1];
                for (int row = 0; row < totalRows; row++)
                {
                    Vector512<float> m = allMaxima[row], ix = allIndices[row];
                    for (int w = 1; w < workers; w++)
                    {
                        Vector512<float> v = allMaxima[w * totalRows + row];
                        Vector512<float> replace = Avx512F.Compare(v, m,
                            FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        m = Avx512F.Max(m, v);
                        ix = Avx512F.BlendVariable(ix, allIndices[w * totalRows + row], replace);
                    }
                    m1[0] = m; i1[0] = ix;
                    Reduce512(m1, i1, maxima, best);
                    FinishScalarTail(inputPin, weightsPin, biasPin, hasBias, indicesPin, scoresPin,
                        0, row, 1, totalRows, inner, columns, tiles * 16, maxima, best);
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<Vector512<float>>.Shared.Return(allMaxima);
            System.Buffers.ArrayPool<Vector512<float>>.Shared.Return(allIndices);
        }
        return true;
    }

    // column tiles [t0, t1) of a sixteen-row block, updating its lane maxima
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ArgMaxTiles16Avx512(float* input, float* packed, float* bias,
        bool hasBias, int inputBase, int inner, int t0, int t1,
        Span<Vector512<float>> vectorMaxima, Span<Vector512<float>> vectorIndices)
    {
        for (int t = t0; t < t1; t++)
        {
            int col = t * 16;
            Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
            Vector512<float> a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            Vector512<float> a8 = a0, a9 = a0, a10 = a0, a11 = a0;
            Vector512<float> a12 = a0, a13 = a0, a14 = a0, a15 = a0;
            float* tile = packed + t * inner * 16;
            for (int k = 0; k < inner; k++)
            {
                Vector512<float> w = Avx512F.LoadVector512(tile + k * 16);
                a0 = AddMul512(a0, Vector512.Create(input[inputBase + k]), w);
                a1 = AddMul512(a1, Vector512.Create(input[inputBase + inner + k]), w);
                a2 = AddMul512(a2, Vector512.Create(input[inputBase + inner * 2 + k]), w);
                a3 = AddMul512(a3, Vector512.Create(input[inputBase + inner * 3 + k]), w);
                a4 = AddMul512(a4, Vector512.Create(input[inputBase + inner * 4 + k]), w);
                a5 = AddMul512(a5, Vector512.Create(input[inputBase + inner * 5 + k]), w);
                a6 = AddMul512(a6, Vector512.Create(input[inputBase + inner * 6 + k]), w);
                a7 = AddMul512(a7, Vector512.Create(input[inputBase + inner * 7 + k]), w);
                a8 = AddMul512(a8, Vector512.Create(input[inputBase + inner * 8 + k]), w);
                a9 = AddMul512(a9, Vector512.Create(input[inputBase + inner * 9 + k]), w);
                a10 = AddMul512(a10, Vector512.Create(input[inputBase + inner * 10 + k]), w);
                a11 = AddMul512(a11, Vector512.Create(input[inputBase + inner * 11 + k]), w);
                a12 = AddMul512(a12, Vector512.Create(input[inputBase + inner * 12 + k]), w);
                a13 = AddMul512(a13, Vector512.Create(input[inputBase + inner * 13 + k]), w);
                a14 = AddMul512(a14, Vector512.Create(input[inputBase + inner * 14 + k]), w);
                a15 = AddMul512(a15, Vector512.Create(input[inputBase + inner * 15 + k]), w);
            }
            if (hasBias)
            {
                Vector512<float> bv = Avx512F.LoadVector512(bias + col);
                a0 = Avx512F.Add(a0, bv); a1 = Avx512F.Add(a1, bv);
                a2 = Avx512F.Add(a2, bv); a3 = Avx512F.Add(a3, bv);
                a4 = Avx512F.Add(a4, bv); a5 = Avx512F.Add(a5, bv);
                a6 = Avx512F.Add(a6, bv); a7 = Avx512F.Add(a7, bv);
                a8 = Avx512F.Add(a8, bv); a9 = Avx512F.Add(a9, bv);
                a10 = Avx512F.Add(a10, bv); a11 = Avx512F.Add(a11, bv);
                a12 = Avx512F.Add(a12, bv); a13 = Avx512F.Add(a13, bv);
                a14 = Avx512F.Add(a14, bv); a15 = Avx512F.Add(a15, bv);
            }
            Update512(a0, col, 0, vectorMaxima, vectorIndices);
            Update512(a1, col, 1, vectorMaxima, vectorIndices);
            Update512(a2, col, 2, vectorMaxima, vectorIndices);
            Update512(a3, col, 3, vectorMaxima, vectorIndices);
            Update512(a4, col, 4, vectorMaxima, vectorIndices);
            Update512(a5, col, 5, vectorMaxima, vectorIndices);
            Update512(a6, col, 6, vectorMaxima, vectorIndices);
            Update512(a7, col, 7, vectorMaxima, vectorIndices);
            Update512(a8, col, 8, vectorMaxima, vectorIndices);
            Update512(a9, col, 9, vectorMaxima, vectorIndices);
            Update512(a10, col, 10, vectorMaxima, vectorIndices);
            Update512(a11, col, 11, vectorMaxima, vectorIndices);
            Update512(a12, col, 12, vectorMaxima, vectorIndices);
            Update512(a13, col, 13, vectorMaxima, vectorIndices);
            Update512(a14, col, 14, vectorMaxima, vectorIndices);
            Update512(a15, col, 15, vectorMaxima, vectorIndices);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ArgMaxTiles8Avx512(float* input, float* packed, float* bias,
        bool hasBias, int inputBase, int inner, int t0, int t1,
        Span<Vector512<float>> vectorMaxima, Span<Vector512<float>> vectorIndices)
    {
        for (int t = t0; t < t1; t++)
        {
            int col = t * 16;
            Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
            Vector512<float> a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            float* tile = packed + t * inner * 16;
            for (int k = 0; k < inner; k++)
            {
                Vector512<float> w = Avx512F.LoadVector512(tile + k * 16);
                a0 = AddMul512(a0, Vector512.Create(input[inputBase + k]), w);
                a1 = AddMul512(a1, Vector512.Create(input[inputBase + inner + k]), w);
                a2 = AddMul512(a2, Vector512.Create(input[inputBase + inner * 2 + k]), w);
                a3 = AddMul512(a3, Vector512.Create(input[inputBase + inner * 3 + k]), w);
                a4 = AddMul512(a4, Vector512.Create(input[inputBase + inner * 4 + k]), w);
                a5 = AddMul512(a5, Vector512.Create(input[inputBase + inner * 5 + k]), w);
                a6 = AddMul512(a6, Vector512.Create(input[inputBase + inner * 6 + k]), w);
                a7 = AddMul512(a7, Vector512.Create(input[inputBase + inner * 7 + k]), w);
            }
            if (hasBias)
            {
                Vector512<float> bv = Avx512F.LoadVector512(bias + col);
                a0 = Avx512F.Add(a0, bv); a1 = Avx512F.Add(a1, bv);
                a2 = Avx512F.Add(a2, bv); a3 = Avx512F.Add(a3, bv);
                a4 = Avx512F.Add(a4, bv); a5 = Avx512F.Add(a5, bv);
                a6 = Avx512F.Add(a6, bv); a7 = Avx512F.Add(a7, bv);
            }
            Update512(a0, col, 0, vectorMaxima, vectorIndices);
            Update512(a1, col, 1, vectorMaxima, vectorIndices);
            Update512(a2, col, 2, vectorMaxima, vectorIndices);
            Update512(a3, col, 3, vectorMaxima, vectorIndices);
            Update512(a4, col, 4, vectorMaxima, vectorIndices);
            Update512(a5, col, 5, vectorMaxima, vectorIndices);
            Update512(a6, col, 6, vectorMaxima, vectorIndices);
            Update512(a7, col, 7, vectorMaxima, vectorIndices);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ArgMaxTiles4Avx512(float* input, float* packed, float* bias,
        bool hasBias, int inputBase, int inner, int t0, int t1,
        Span<Vector512<float>> vectorMaxima, Span<Vector512<float>> vectorIndices)
    {
        for (int t = t0; t < t1; t++)
        {
            int col = t * 16;
            Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
            float* tile = packed + t * inner * 16;
            for (int k = 0; k < inner; k++)
            {
                Vector512<float> w = Avx512F.LoadVector512(tile + k * 16);
                a0 = AddMul512(a0, Vector512.Create(input[inputBase + k]), w);
                a1 = AddMul512(a1, Vector512.Create(input[inputBase + inner + k]), w);
                a2 = AddMul512(a2, Vector512.Create(input[inputBase + inner * 2 + k]), w);
                a3 = AddMul512(a3, Vector512.Create(input[inputBase + inner * 3 + k]), w);
            }
            if (hasBias)
            {
                Vector512<float> bv = Avx512F.LoadVector512(bias + col);
                a0 = Avx512F.Add(a0, bv); a1 = Avx512F.Add(a1, bv);
                a2 = Avx512F.Add(a2, bv); a3 = Avx512F.Add(a3, bv);
            }
            Update512(a0, col, 0, vectorMaxima, vectorIndices);
            Update512(a1, col, 1, vectorMaxima, vectorIndices);
            Update512(a2, col, 2, vectorMaxima, vectorIndices);
            Update512(a3, col, 3, vectorMaxima, vectorIndices);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxBlock16Avx512(float* input,
        float* weights, float* packed, float* bias, bool hasBias,
        int* indices, float* scores, int batch, int row,
        int rows, int inner, int columns)
    {
        Span<Vector512<float>> vectorMaxima = stackalloc Vector512<float>[16];
        Span<Vector512<float>> vectorIndices = stackalloc Vector512<float>[16];
        vectorMaxima.Fill(Vector512.Create(float.NegativeInfinity));
        int inputBase = (batch * rows + row) * inner;
        int col = 0;
        for (; col <= columns - 16; col += 16)
        {
            Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
            Vector512<float> a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            Vector512<float> a8 = a0, a9 = a0, a10 = a0, a11 = a0;
            Vector512<float> a12 = a0, a13 = a0, a14 = a0, a15 = a0;
            float* tile = packed + (col / 16) * inner * 16;
            for (int k = 0; k < inner; k++)
            {
                Vector512<float> w = Avx512F.LoadVector512(tile + k * 16);
                a0 = AddMul512(a0, Vector512.Create(input[inputBase + k]), w);
                a1 = AddMul512(a1, Vector512.Create(input[inputBase + inner + k]), w);
                a2 = AddMul512(a2, Vector512.Create(input[inputBase + inner * 2 + k]), w);
                a3 = AddMul512(a3, Vector512.Create(input[inputBase + inner * 3 + k]), w);
                a4 = AddMul512(a4, Vector512.Create(input[inputBase + inner * 4 + k]), w);
                a5 = AddMul512(a5, Vector512.Create(input[inputBase + inner * 5 + k]), w);
                a6 = AddMul512(a6, Vector512.Create(input[inputBase + inner * 6 + k]), w);
                a7 = AddMul512(a7, Vector512.Create(input[inputBase + inner * 7 + k]), w);
                a8 = AddMul512(a8, Vector512.Create(input[inputBase + inner * 8 + k]), w);
                a9 = AddMul512(a9, Vector512.Create(input[inputBase + inner * 9 + k]), w);
                a10 = AddMul512(a10, Vector512.Create(input[inputBase + inner * 10 + k]), w);
                a11 = AddMul512(a11, Vector512.Create(input[inputBase + inner * 11 + k]), w);
                a12 = AddMul512(a12, Vector512.Create(input[inputBase + inner * 12 + k]), w);
                a13 = AddMul512(a13, Vector512.Create(input[inputBase + inner * 13 + k]), w);
                a14 = AddMul512(a14, Vector512.Create(input[inputBase + inner * 14 + k]), w);
                a15 = AddMul512(a15, Vector512.Create(input[inputBase + inner * 15 + k]), w);
            }
            if (hasBias)
            {
                Vector512<float> bv = Avx512F.LoadVector512(bias + col);
                a0 = Avx512F.Add(a0, bv); a1 = Avx512F.Add(a1, bv);
                a2 = Avx512F.Add(a2, bv); a3 = Avx512F.Add(a3, bv);
                a4 = Avx512F.Add(a4, bv); a5 = Avx512F.Add(a5, bv);
                a6 = Avx512F.Add(a6, bv); a7 = Avx512F.Add(a7, bv);
                a8 = Avx512F.Add(a8, bv); a9 = Avx512F.Add(a9, bv);
                a10 = Avx512F.Add(a10, bv); a11 = Avx512F.Add(a11, bv);
                a12 = Avx512F.Add(a12, bv); a13 = Avx512F.Add(a13, bv);
                a14 = Avx512F.Add(a14, bv); a15 = Avx512F.Add(a15, bv);
            }
            Update512(a0, col, 0, vectorMaxima, vectorIndices);
            Update512(a1, col, 1, vectorMaxima, vectorIndices);
            Update512(a2, col, 2, vectorMaxima, vectorIndices);
            Update512(a3, col, 3, vectorMaxima, vectorIndices);
            Update512(a4, col, 4, vectorMaxima, vectorIndices);
            Update512(a5, col, 5, vectorMaxima, vectorIndices);
            Update512(a6, col, 6, vectorMaxima, vectorIndices);
            Update512(a7, col, 7, vectorMaxima, vectorIndices);
            Update512(a8, col, 8, vectorMaxima, vectorIndices);
            Update512(a9, col, 9, vectorMaxima, vectorIndices);
            Update512(a10, col, 10, vectorMaxima, vectorIndices);
            Update512(a11, col, 11, vectorMaxima, vectorIndices);
            Update512(a12, col, 12, vectorMaxima, vectorIndices);
            Update512(a13, col, 13, vectorMaxima, vectorIndices);
            Update512(a14, col, 14, vectorMaxima, vectorIndices);
            Update512(a15, col, 15, vectorMaxima, vectorIndices);
        }
        float* maxima = stackalloc float[16];
        int* best = stackalloc int[16];
        Reduce512(vectorMaxima, vectorIndices, maxima, best);
        FinishScalarTail(input, weights, bias, hasBias, indices, scores,
            batch, row, 16, rows, inner, columns, col, maxima, best);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxBlock8Avx512(float* input,
        float* weights, float* packed, float* bias, bool hasBias,
        int* indices, float* scores, int batch, int row,
        int rows, int inner, int columns)
    {
        Span<Vector512<float>> vectorMaxima = stackalloc Vector512<float>[8];
        Span<Vector512<float>> vectorIndices = stackalloc Vector512<float>[8];
        vectorMaxima.Fill(Vector512.Create(float.NegativeInfinity));
        int inputBase = (batch * rows + row) * inner;
        int col = 0;
        for (; col <= columns - 16; col += 16)
        {
            Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
            Vector512<float> a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            float* tile = packed + (col / 16) * inner * 16;
            for (int k = 0; k < inner; k++)
            {
                Vector512<float> w = Avx512F.LoadVector512(tile + k * 16);
                a0 = AddMul512(a0, Vector512.Create(input[inputBase + k]), w);
                a1 = AddMul512(a1, Vector512.Create(input[inputBase + inner + k]), w);
                a2 = AddMul512(a2, Vector512.Create(input[inputBase + inner * 2 + k]), w);
                a3 = AddMul512(a3, Vector512.Create(input[inputBase + inner * 3 + k]), w);
                a4 = AddMul512(a4, Vector512.Create(input[inputBase + inner * 4 + k]), w);
                a5 = AddMul512(a5, Vector512.Create(input[inputBase + inner * 5 + k]), w);
                a6 = AddMul512(a6, Vector512.Create(input[inputBase + inner * 6 + k]), w);
                a7 = AddMul512(a7, Vector512.Create(input[inputBase + inner * 7 + k]), w);
            }
            if (hasBias)
            {
                Vector512<float> bv = Avx512F.LoadVector512(bias + col);
                a0 = Avx512F.Add(a0, bv); a1 = Avx512F.Add(a1, bv);
                a2 = Avx512F.Add(a2, bv); a3 = Avx512F.Add(a3, bv);
                a4 = Avx512F.Add(a4, bv); a5 = Avx512F.Add(a5, bv);
                a6 = Avx512F.Add(a6, bv); a7 = Avx512F.Add(a7, bv);
            }
            Update512(a0, col, 0, vectorMaxima, vectorIndices);
            Update512(a1, col, 1, vectorMaxima, vectorIndices);
            Update512(a2, col, 2, vectorMaxima, vectorIndices);
            Update512(a3, col, 3, vectorMaxima, vectorIndices);
            Update512(a4, col, 4, vectorMaxima, vectorIndices);
            Update512(a5, col, 5, vectorMaxima, vectorIndices);
            Update512(a6, col, 6, vectorMaxima, vectorIndices);
            Update512(a7, col, 7, vectorMaxima, vectorIndices);
        }
        float* maxima = stackalloc float[8];
        int* best = stackalloc int[8];
        Reduce512(vectorMaxima, vectorIndices, maxima, best);
        FinishScalarTail(input, weights, bias, hasBias, indices, scores,
            batch, row, 8, rows, inner, columns, col, maxima, best);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Update512(Vector512<float> values, int column, int row,
        Span<Vector512<float>> maxima, Span<Vector512<float>> indices)
    {
        Vector512<float> previous = maxima[row];
        Vector512<float> replace = Avx512F.Compare(values, previous,
            FloatComparisonMode.OrderedGreaterThanNonSignaling);
        maxima[row] = Avx512F.Max(previous, values);
        Vector512<float> candidate = Avx512F.Add(Vector512.Create((float)column), ArgMaxLanes512);
        indices[row] = Avx512F.BlendVariable(indices[row], candidate, replace);
    }

    private static unsafe void Reduce512(ReadOnlySpan<Vector512<float>> vectorMaxima,
        ReadOnlySpan<Vector512<float>> vectorIndices, float* maxima, int* best)
    {
        for (int row = 0; row < vectorMaxima.Length; row++)
        {
            Vector512<float> values = vectorMaxima[row];
            Vector512<float> positions = vectorIndices[row];
            float maximum = values.GetElement(0);
            int index = (int)positions.GetElement(0);
            for (int lane = 1; lane < 16; lane++)
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
