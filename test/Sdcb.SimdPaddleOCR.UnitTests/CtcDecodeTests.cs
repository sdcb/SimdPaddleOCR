using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class CtcDecodeTests
{
    // Class ids: 0 blank, 1 你, 2 好, 3 space.
    private static readonly string[] Labels = ["你", "好"];

    [Fact]
    public void RepeatsStayInsideAndBlanksStayOutside()
    {
        int[] indices = [0, 1, 1, 1, 0];
        float[] scores = [0.5f, 0.2f, 0.9f, 0.8f, 0.4f];
        PaddleOcrRecognitionResult result = DecodeCompact(indices, scores, capture: true);
        Assert.Equal("你", result.Text);
        Assert.Equal(0.2f, result.Score);
        Assert.Equal(1, result.EmittedCount);
        PaddleOcrCtcSpan span = Assert.Single(result.CtcSpans!);
        Assert.Equal(new PaddleOcrCtcSpan("你", 0.2f, 1, 4), span);
    }

    [Fact]
    public void OpenRunClosesAtTheLastStep()
    {
        int[] indices = [0, 1, 1];
        float[] scores = [0.3f, 0.6f, 0.1f];
        PaddleOcrRecognitionResult result = DecodeCompact(indices, scores, capture: true);
        PaddleOcrCtcSpan span = Assert.Single(result.CtcSpans!);
        Assert.Equal(new PaddleOcrCtcSpan("你", 0.6f, 1, indices.Length), span);
    }

    [Fact]
    public void BlankSeparatesTwoEmissionsOfTheSameClass()
    {
        int[] indices = [1, 0, 1];
        float[] scores = [0.4f, 0.9f, 0.8f];
        PaddleOcrRecognitionResult result = DecodeCompact(indices, scores, capture: true);
        Assert.Equal("你你", result.Text);
        Assert.Equal(0.6f, result.Score);
        Assert.Equal(2, result.EmittedCount);
        Assert.NotNull(result.CtcSpans);
        Assert.Equal<PaddleOcrCtcSpan>(
            [new PaddleOcrCtcSpan("你", 0.4f, 0, 1), new PaddleOcrCtcSpan("你", 0.8f, 2, 3)],
            result.CtcSpans);
    }

    [Fact]
    public void SpaceTokenIsItsOwnRun()
    {
        int[] indices = [3, 3, 2];
        float[] scores = [0.5f, 0.2f, 0.7f];
        PaddleOcrRecognitionResult result = DecodeCompact(indices, scores, capture: true);
        Assert.Equal(" 好", result.Text);
        Assert.NotNull(result.CtcSpans);
        Assert.Equal<PaddleOcrCtcSpan>(
            [new PaddleOcrCtcSpan(" ", 0.5f, 0, 2), new PaddleOcrCtcSpan("好", 0.7f, 2, 3)],
            result.CtcSpans);
    }

    [Fact]
    public void AllBlanksCaptureAnEmptyAlignment()
    {
        int[] indices = [0, 0, 0];
        float[] scores = [0.9f, 0.9f, 0.9f];
        PaddleOcrRecognitionResult result = DecodeCompact(indices, scores, capture: true);
        Assert.Equal("", result.Text);
        Assert.Equal(0, result.EmittedCount);
        Assert.NotNull(result.CtcSpans);
        Assert.Empty(result.CtcSpans);
    }

    [Fact]
    public void CaptureOffKeepsTheTextAndDropsSpans()
    {
        int[] indices = [1, 0, 1];
        float[] scores = [0.4f, 0.9f, 0.8f];
        PaddleOcrRecognitionResult result = DecodeCompact(indices, scores, capture: false);
        Assert.Equal("你你", result.Text);
        Assert.Equal(0.6f, result.Score);
        Assert.Null(result.CtcSpans);
    }

    [Fact]
    public void DenseLogitsUseArgMaxAndTheEmittedSoftmax()
    {
        // Same alignment as BlankSeparatesTwoEmissionsOfTheSameClass: 你, blank, 你.
        // Classes: blank, 你, 好, space.
        float[] logits =
        [
            0, 4, 0, 0,
            4, 0, 0, 0,
            0, 4, 1, 0
        ];
        PaddleOcrRecognitionResult result = PaddleOcrRecognizer.DecodeCtc(
            Labels, maxLabelChars: 1, capture: true,
            logits, [], [], timeSteps: 3, resizedWidth: 3, tensorWidth: 4, denseIsLogits: true);
        Assert.Equal("你你", result.Text);
        Assert.Equal(4, result.TensorWidth);
        Assert.NotNull(result.CtcSpans);
        Assert.Equal(2, result.CtcSpans.Length);
        Assert.Equal(0, result.CtcSpans[0].StartColumn);
        Assert.Equal(1, result.CtcSpans[0].EndColumn);
        Assert.Equal(2, result.CtcSpans[1].StartColumn);
        Assert.Equal(3, result.CtcSpans[1].EndColumn);
        float first = Softmax(logits.AsSpan(0, 4), 4f);
        float second = Softmax(logits.AsSpan(8, 4), 4f);
        Assert.Equal(first, result.CtcSpans[0].Score);
        Assert.Equal(second, result.CtcSpans[1].Score);
        Assert.Equal((float)(((double)first + second) / 2), result.Score);
    }

    private static PaddleOcrRecognitionResult DecodeCompact(int[] indices, float[] scores, bool capture)
        => PaddleOcrRecognizer.DecodeCtc(
            Labels, maxLabelChars: 1, capture,
            [], indices, scores, indices.Length, resizedWidth: indices.Length, tensorWidth: indices.Length,
            denseIsLogits: false);

    private static float Softmax(ReadOnlySpan<float> row, float maximum)
    {
        // reference: softmax at the row maximum, 1 / sum exp(x - max)
        float sum = 0;
        foreach (float v in row) sum += MathF.Exp(v - maximum);
        return 1f / sum;
    }
}
