using System.Buffers;
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class MatMul
{
    // A maximum logit identifies a label but cannot give its confidence without
    // the other classes. Keep a small block of projected rows per worker, not [N,T,C].
    internal static unsafe bool TryCtcProjection(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores, int rows, int inner,
        int columns, float[]? packedWeights, int threads)
    {
        if (!CanFuseArgMax(rows, inner, columns, packedWeights) ||
            indices.Length != rows || scores.Length != rows || (!bias.IsEmpty && bias.Length != columns))
            return false;

        if ((long)rows * inner > input.Length || (long)inner * columns > weights.Length ||
            (packedWeights is not null && (long)(columns / 16) * inner * 16 > packedWeights.Length))
            throw new InvalidDataException("CTC projection operands do not match their shape.");

        const int block = 8;
        int jobs = (rows - 1) / block + 1;
        int workers = Math.Min(jobs, ArgMaxWorkers(1, rows, inner, columns, threads));
        fixed (float* inputPin = input, weightsPin = weights, biasPin = bias, scoresPin = scores)
        fixed (int* indicesPin = indices)
        {
            float* activations = inputPin, matrix = weightsPin, biases = biasPin, probabilities = scoresPin;
            int* labels = indicesPin;
            void Worker(int worker)
            {
                // Space is bounded by the model's vocabulary, independent of batch/image dimensions.
                float[] scratch = ArrayPool<float>.Shared.Rent(checked(Math.Min(rows, block) * columns));
                try
                {
                    for (int job = worker; job < jobs; job += workers)
                    {
                        int row = job * block, count = Math.Min(block, rows - row);
                        Try(new ReadOnlySpan<float>(activations + row * inner, count * inner),
                            new ReadOnlySpan<float>(matrix, inner * columns), scratch.AsSpan(0, count * columns),
                            1, count, inner, columns, packedWeights);
                        for (int i = 0; i < count; i++)
                        {
                            Span<float> values = scratch.AsSpan(i * columns, columns);
                            if (biases != null)
                                SimdKernels.Add(values, new ReadOnlySpan<float>(biases, columns), values);
                            int label = ArgMax.Find(values, out float maximum);
                            labels[row + i] = label;
                            probabilities[row + i] = label == 0 ? 0 : SimdKernels.SoftmaxMaximumProbability(values, maximum);
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(scratch);
                }
            }
            if (workers == 1) Worker(0);
            else Parallel.For(0, workers, Worker);
        }
        return true;
    }
}
