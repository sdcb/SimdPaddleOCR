namespace Sdcb.SimdPaddleOCR.UnitTests;

public class ParallelismTests
{
    [Theory]
    [InlineData(4, 2, 2)]
    [InlineData(0, 2, 2)]
    [InlineData(0, 16, 4)]
    [InlineData(0, 1, 1)]
    [InlineData(0, 3, 3)]
    [InlineData(1, 2, 1)]
    [InlineData(8, 4, 4)]
    [InlineData(16, 32, 16)]
    public void ResolveLineWorkers(int requested, int processorCount, int expected) =>
        Assert.Equal(expected, Parallelism.ResolveLineWorkers(requested, processorCount));

    [Theory]
    [InlineData(2, 2, 1)]
    [InlineData(1, 2, 2)]
    [InlineData(4, 16, 4)]
    [InlineData(4, 8, 2)]
    [InlineData(1, 8, 4)]
    public void ResolveRecognizerIntraOp(int lineWorkers, int processorCount, int expected) =>
        Assert.Equal(expected, Parallelism.ResolveRecognizerIntraOp(lineWorkers, processorCount));
}
