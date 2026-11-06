using System.IO.Compression;
using System.Text;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class CtcProjectionScoreTests
{
    private const int Width = 4, Height = 48, Steps = 8, Inner = 72, Classes = 1024;
    private const int TailRows = 9, TailClasses = 1025;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProjectionMatchesSoftmaxAcrossRowAndColumnTails(bool withBias)
    {
        // Only the blank and the final, unpacked class have nonzero weights.
        float[] weights = new float[Inner * TailClasses];
        weights[(Inner - 1) * TailClasses] = -1;
        weights[^1] = 1;
        float[] packed = KernelCorrectnessTests.PackMatMul(weights, Inner, TailClasses);
        float[] bias = withBias ? new float[TailClasses] : [];
        if (withBias) bias[^1] = 3.5f;
        float[] input = new float[TailRows * Inner];
        for (int r = 0; r < TailRows; r++) input[(r + 1) * Inner - 1] = r - 4;
        int[] indices = new int[TailRows];
        float[] scores = new float[TailRows];
        Assert.True(MatMul.TryCtcProjection(input, weights, bias, indices, scores,
            TailRows, Inner, TailClasses, packed, threads: 1));

        for (int r = 0; r < TailRows; r++)
        {
            double blank = 4 - r, last = r - 4 + (withBias ? 3.5 : 0);
            int winner = last > blank ? TailClasses - 1 : 0;
            Assert.Equal(winner, indices[r]);
            if (winner == 0) continue; // Blank scores are never emitted.
            double probability = 1 / (1 + Math.Exp(blank - last) + (TailClasses - 2) * Math.Exp(-last));
            Assert.InRange(Math.Abs(scores[r] - probability), 0, 1e-6);
        }
    }

    [Theory]
    [InlineData("input")]
    [InlineData("weights")]
    [InlineData("packed")]
    public void ProjectionRejectsShortOperand(string operand)
    {
        float[] input = new float[TailRows * Inner - (operand == "input" ? 1 : 0)];
        float[] weights = new float[Inner * TailClasses - (operand == "weights" ? 1 : 0)];
        float[] packed = new float[(TailClasses / 16) * Inner * 16 - (operand == "packed" ? 1 : 0)];
        Assert.Throws<InvalidDataException>(() => MatMul.TryCtcProjection(input, weights, [],
            new int[TailRows], new float[TailRows], TailRows, Inner, TailClasses, packed, threads: 1));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.NegativeInfinity)]
    [InlineData(float.PositiveInfinity)]
    public void ProjectionRejectsNonFiniteLogit(float invalid)
    {
        float[] bias = new float[TailClasses];
        bias[^1] = invalid;
        Assert.Throws<InvalidDataException>(() => MatMul.TryCtcProjection(new float[Inner],
            new float[Inner * TailClasses], bias, new int[1], new float[1], 1, Inner, TailClasses,
            packedWeights: null, threads: 1));
    }

    [Theory]
    [InlineData(-1024)]
    [InlineData(1024)]
    public void FusedSingleAndBatchedScoresMatchSoftmax(int commonShift)
    {
        // The synthetic ONNX head reshapes NCHW into [N,8,72], then applies
        // MatMul + bias + Softmax. Its 1024 classes exercise fused ArgMax.
        // With s = commonShift, the logits are [s+4-4*x, s+3+4*x+y, s, ..., s].
        using Stream fixture = typeof(CtcProjectionScoreTests).Assembly.GetManifestResourceStream(
            "Sdcb.SimdPaddleOCR.UnitTests.Fixtures.FusedCtcScore.onnx.gz")!;
        using var onnx = new GZipStream(fixture, CompressionMode.Decompress);
        string dictionary = string.Join("\n", Enumerable.Repeat("A", Classes - 2));
        using var recognizer = new PaddleOcrRecognizer(onnx, Encoding.UTF8.GetBytes(dictionary),
            new PaddleOcrRecognizerOptions { TargetWidth = Width, AdaptiveWidth = false, Backend = OcrBackend.Cpu });

        // The last A in the first line and first A in the second line must both be emitted.
        byte[][] crops =
        [
            Crop([1, 1, 0, 1, 1, 0, 1, 1], commonShift),
            Crop([1, 0, 1, 1, 1, 0, 0, 1], commonShift),
            Crop([0, 0, 0, 0, 0, 0, 0, 0], commonShift)
        ];
        (int Start, int End)[][] expected =
        [
            [(0, 2), (3, 5), (6, 8)],
            [(0, 1), (2, 5), (7, 8)],
            []
        ];
        for (int i = 0; i < crops.Length; i++)
        {
            var line = recognizer.Recognize(crops[i], Width, Height, returnCtcAlignment: true);
            CheckLine(line, expected[i], commonShift);
        }

        int size = crops[0].Length;
        byte[] buffer = crops.SelectMany(crop => crop).ToArray();
        var batch = new PaddleOcrRecognitionResult[crops.Length];
        recognizer.RecognizeBatch(buffer, [0, size, 2 * size], [size, size, size],
            [Width, Width, Width], [Height, Height, Height], [0, 1, 2], Width,
            batch, intraOpThreads: 2, returnCtcAlignment: true);
        for (int i = 0; i < batch.Length; i++)
            CheckLine(batch[i], expected[i], commonShift);
    }

    private static byte[] Crop(int[] classes, int commonShift)
    {
        // Input bytes 0 and 255 normalize to -1 and +1. Only three features
        // of each row have nonzero weights: x, the shared offset, and y.
        byte[] pixels = new byte[Width * Height * 3];
        for (int step = 0; step < Steps; step++)
        {
            Put(step * Inner, classes[step] == 1);
            Put(step * Inner + 1, commonShift > 0);
            Put(step * Inner + 2, step % 2 == 0);
        }
        return pixels;

        void Put(int nchwIndex, bool positive)
        {
            int pixel = nchwIndex % (Width * Height);
            int channel = nchwIndex / (Width * Height);
            pixels[pixel * 3 + channel] = positive ? (byte)255 : (byte)0;
        }
    }

    private static void CheckLine(PaddleOcrRecognitionResult result,
        (int Start, int End)[] expected, int shift)
    {
        Assert.Equal(new string('A', expected.Length), result.Text);
        Assert.Equal(expected.Length, result.EmittedCount);
        Assert.Equal(Steps, result.TimeSteps);
        Assert.NotNull(result.CtcSpans);
        Assert.Equal(expected.Length, result.CtcSpans.Length);
        double sum = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            var span = result.CtcSpans[i];
            Assert.Equal("A", span.Text);
            Assert.Equal(expected[i], (span.StartColumn, span.EndColumn));
            // Independent float64 softmax. On emitted rows x=1, making the
            // blank logit equal to shift, like each of the unused classes.
            double winning = shift + 7 + (span.StartColumn % 2 == 0 ? 1 : -1);
            double probability = 1 / (1 + (Classes - 1) * Math.Exp(shift - winning));
            Assert.InRange(span.Score, 0f, 1f);
            Assert.InRange(Math.Abs(span.Score - probability), 0, 1e-5);
            sum += probability;
        }
        double lineScore = expected.Length == 0 ? 0 : sum / expected.Length;
        Assert.InRange(result.Score, 0f, 1f);
        Assert.InRange(Math.Abs(result.Score - lineScore), 0, 1e-5);
    }
}
