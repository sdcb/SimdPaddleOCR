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
    private static readonly Vector256<float> ArgMaxLanes256 =
        Vector256.Create(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxPackedAvx(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores,
        int batch, int rows, int inner, int columns, int threads)
    {
        int rowJobs = rows / 4;
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
            // Column-block-major walk: a block of packed weight tiles
            // (~256 KB, L2-resident) is reused by every row quad of this
            // worker before moving on, instead of each quad streaming the
            // whole vocab matrix from L3. Each row still visits columns in
            // ascending order with the same per-element FMA chain, so the
            // maxima, indices and tie-breaks are unchanged.
            int tiles = columns >= 16 ? columns / 16 : 0;
            int blockTiles = Math.Max(1, 256 * 1024 / (inner * 16 * sizeof(float)));
            void Worker(int w)
            {
                int myJobs = (jobs - w + workers - 1) / workers;
                var allMaxima = new Vector256<float>[myJobs * 4];
                var allIndices = new Vector256<float>[myJobs * 4];
                allMaxima.AsSpan().Fill(Vector256.Create(float.NegativeInfinity));
                float* maxima = stackalloc float[4];
                int* best = stackalloc int[4];
                for (int t0 = 0; t0 < tiles; t0 += blockTiles)
                {
                    int t1 = Math.Min(tiles, t0 + blockTiles);
                    for (int j = 0; j < myJobs; j++)
                    {
                        int job = w + j * workers;
                        int b = job / rowJobs, row = (job - b * rowJobs) * 4;
                        Span<Vector256<float>> vectorMaxima = allMaxima.AsSpan(j * 4, 4);
                        Span<Vector256<float>> vectorIndices = allIndices.AsSpan(j * 4, 4);
                        int inputBase = (b * rows + row) * inner;
                        for (int t = t0; t < t1; t++)
                        {
                            int col = t * 16;
                            Vector256<float> a0l = Vector256<float>.Zero, a0h = a0l;
                            Vector256<float> a1l = a0l, a1h = a0l;
                            Vector256<float> a2l = a0l, a2h = a0l;
                            Vector256<float> a3l = a0l, a3h = a0l;
                            float* tile = packedPtr + t * inner * 16;
                            for (int k = 0; k < inner; k++)
                            {
                                Vector256<float> wl = Avx.LoadVector256(tile + k * 16);
                                Vector256<float> wh = Avx.LoadVector256(tile + k * 16 + 8);
                                Vector256<float> v0 = Avx.BroadcastScalarToVector256(inputPtr + inputBase + k);
                                Vector256<float> v1 = Avx.BroadcastScalarToVector256(inputPtr + inputBase + inner + k);
                                Vector256<float> v2 = Avx.BroadcastScalarToVector256(inputPtr + inputBase + inner * 2 + k);
                                Vector256<float> v3 = Avx.BroadcastScalarToVector256(inputPtr + inputBase + inner * 3 + k);
                                a0l = AddMul(a0l, v0, wl); a0h = AddMul(a0h, v0, wh);
                                a1l = AddMul(a1l, v1, wl); a1h = AddMul(a1h, v1, wh);
                                a2l = AddMul(a2l, v2, wl); a2h = AddMul(a2h, v2, wh);
                                a3l = AddMul(a3l, v3, wl); a3h = AddMul(a3h, v3, wh);
                            }
                            if (hasBias)
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
                    }
                }
                for (int j = 0; j < myJobs; j++)
                {
                    int job = w + j * workers;
                    int b = job / rowJobs, row = (job - b * rowJobs) * 4;
                    Reduce256(allMaxima.AsSpan(j * 4, 4), allIndices.AsSpan(j * 4, 4), maxima, best);
                    FinishScalarTail(inputPtr, weightsPtr, biasPtr, hasBias,
                        indicesPtr, scoresPtr, b, row, 4, rows, inner, columns, tiles * 16,
                        maxima, best);
                }
            }

            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
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

    private static unsafe void Reduce256(ReadOnlySpan<Vector256<float>> vectorMaxima,
        ReadOnlySpan<Vector256<float>> vectorIndices, float* maxima, int* best)
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
