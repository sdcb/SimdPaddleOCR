using Sdcb.SimdPaddleOCR.Kernels;
using static Sdcb.SimdPaddleOCR.UnitTests.KernelCorrectnessTests;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class ArgMaxWorkspaceTests
{
    [Theory]
    [InlineData(4, 1)]
    [InlineData(256, 4)]
    [InlineData(512, 4)]
    [InlineData(516, 4)]
    [InlineData(1024, 4)]
    [InlineData(2048, 1)]
    public void ReusedWorkspaceResetsMaximaAndTieIndices(int rows, int threads)
    {
        const int batch = 2, inner = 64, columns = 1040;
        float[] input = new float[batch * rows * inner];
        float[] weights = new float[inner * columns];
        float[] packed = PackMatMul(weights, inner, columns);
        float[] bias = new float[columns];
        int[] indices = new int[batch * rows];
        float[] scores = new float[indices.Length];
        for (int iteration = 0; iteration < 8; iteration++)
        {
            bool tied = (iteration & 1) != 0;
            Array.Fill(bias, -20f);
            int expected = tied ? 0 : columns - 1 - iteration;
            if (!tied) bias[expected] = 5f;
            Assert.True(MatMul.TryArgMax(input, weights, bias, indices, scores,
                batch, rows, inner, columns, packed, threads));
            Assert.All(indices, value => Assert.Equal(expected, value));
            Assert.All(scores, value => Assert.Equal(tied ? -20f : 5f, value));
        }
    }
}
