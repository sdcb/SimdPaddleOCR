using System.IO.Compression;
using System.Text.Json;

namespace Sdcb.SimdPaddleOCR.UnitTests;

/// <summary>
/// Document orientation stage: PaddleX preprocessing against OpenCV golden values, page rotation and
/// its inverse mapping, and the classifier on tiny stand-in models (build/create-orientation-test-fixtures.py).
/// Ported from sdcb/SimdPaddleOCR#32 by hnlyf168.
/// </summary>
public class DocumentOrientationTests
{
    private static Stream Resource(string name) => typeof(DocumentOrientationTests).Assembly
        .GetManifestResourceStream(name) ?? throw new FileNotFoundException(name);

    private static PaddleOcrDocumentOrientationClassifier Tiny(int classes, PaddleOcrDocumentOrientationOptions? options = null)
    {
        using Stream gz = Resource($"Fixtures.DocOrientationTiny{classes}.onnx.gz");
        using var onnx = new GZipStream(gz, CompressionMode.Decompress);
        return new PaddleOcrDocumentOrientationClassifier(onnx, options ?? new() { Backend = OcrBackend.Cpu });
    }

    private static byte[] Solid(int width, int height, byte b, byte g, byte r)
    {
        byte[] image = new byte[width * height * 3];
        for (int i = 0; i < image.Length; i += 3) { image[i] = b; image[i + 1] = g; image[i + 2] = r; }
        return image;
    }

    [Theory]
    [InlineData(ImagePixelFormat.Bgr24, false)]
    [InlineData(ImagePixelFormat.Rgb24, false)]
    [InlineData(ImagePixelFormat.Bgra32, true)]
    [InlineData(ImagePixelFormat.Rgba32, true)]
    [InlineData(ImagePixelFormat.Bgr24, true)]
    [InlineData(ImagePixelFormat.Rgba32, false)]
    public void CenterCropRgbNormalization_MatchesOpenCv(ImagePixelFormat format, bool nhwc)
    {
        const int width = 17, height = 19, size = 224;
        int bpp = ImagePixels.BytesPerPixel(format), stride = width * bpp + 7;
        byte[] source = new byte[stride * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int c = 0; c < 3; c++)
                {
                    // the golden pattern is defined on BGR bytes
                    int bgrChannel = ImagePixels.SwapRedBlue(format) ? 2 - c : c;
                    source[y * stride + x * bpp + c] = (byte)((((y * width + x) * 3 + bgrChannel) * 37 + 11) % 256);
                }
        float[] output = new float[3 * size * size];
        PPOCRPreprocess.DocumentOrientation(source, width, height, stride, output, nhwc, format);
        using Stream json = Resource("Assets.Orientation.preprocess.json");
        using JsonDocument samples = JsonDocument.Parse(json);
        foreach (JsonElement sample in samples.RootElement.EnumerateArray())
        {
            int c = sample.GetProperty("channel").GetInt32(), x = sample.GetProperty("x").GetInt32(), y = sample.GetProperty("y").GetInt32();
            float expected = sample.GetProperty("value").GetSingle();
            float actual = output[nhwc ? (y * size + x) * 3 + c : c * size * size + y * size + x];
            Assert.True(Math.Abs(expected - actual) <= 3e-7f, $"{format} ({x},{y},{c}): {expected} != {actual}");
        }
    }

    [Theory]
    [InlineData(0, new int[] { 0, 1, 2, 3, 4, 5 })]
    [InlineData(90, new int[] { 2, 5, 1, 4, 0, 3 })]
    [InlineData(180, new int[] { 5, 4, 3, 2, 1, 0 })]
    [InlineData(270, new int[] { 3, 0, 4, 1, 5, 2 })]
    public void RectangularRotation_AndSourceCoordinatesAgree(int degrees, int[] order)
    {
        // 3 x 2 BGR image with a padded 13-byte stride; pixel id in the blue byte
        byte[] source = new byte[2 * 13];
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 3; x++)
                source[y * 13 + x * 3] = (byte)(y * 3 + x);
        byte[] rotated = new byte[18];
        PaddleOcrDocumentRotation.Rotate(source, 3, 2, 13, ImagePixelFormat.Bgr24, degrees, rotated);
        Assert.Equal(order, Enumerable.Range(0, 6).Select(i => (int)rotated[i * 3]).ToArray());
        (int w, _) = PaddleOcrDocumentRotation.RotatedSize(3, 2, degrees);
        for (int i = 0; i < 6; i++)
        {
            float x = i % w, y = i / w;
            PaddleOcrDetectionBox p = PaddleOcrDocumentRotation.ToSource(new PaddleOcrDetectionBox(x, y, x, y, x, y, x, y, 1), 3, 2, degrees);
            Assert.Equal(order[i] % 3, p.X1);
            Assert.Equal(order[i] / 3, p.Y1);
        }
    }

    [Fact]
    public void Rotation_RejectsInvalidArguments()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => PaddleOcrDocumentRotation.RotatedSize(3, 2, 45));
        Assert.Throws<ArgumentException>(() => PaddleOcrDocumentRotation.Rotate(new byte[17], 3, 2, 0, ImagePixelFormat.Bgr24, 90, new byte[18]));
        Assert.Throws<ArgumentException>(() => PaddleOcrDocumentRotation.Rotate(new byte[18], 3, 2, 0, ImagePixelFormat.Bgr24, 90, new byte[17]));
    }

    [Fact]
    public void LineToSource_KeepsEverythingButTheBox()
    {
        var line = new PaddleOcrLine
        {
            Box = new PaddleOcrDetectionBox(1, 2, 5, 2, 5, 4, 1, 4, 0.9f), Text = "abc",
            RecognitionScore = 0.8f, ClassificationScore = 0.7f, ClassificationLabel = 1, AppliedRotationDegrees = 180,
            EmittedCount = 3, CtcSpans = [], RecognitionContentWidth = 11, RecognitionTensorWidth = 12, RecognitionTimeSteps = 13,
        };
        PaddleOcrLine back = PaddleOcrDocumentRotation.ToSource(line, 20, 10, 90);
        Assert.Equal(line.Box with { X1 = 17, Y1 = 1, X2 = 17, Y2 = 5, X3 = 15, Y3 = 5, X4 = 15, Y4 = 1 }, back.Box);
        Assert.True(new LineComparer().Equals(line, back, compareBox: false));
    }

    private sealed class LineComparer
    {
        public bool Equals(PaddleOcrLine a, PaddleOcrLine b, bool compareBox) =>
            (!compareBox || a.Box == b.Box) && a.Text == b.Text && a.RecognitionScore == b.RecognitionScore &&
            a.ClassificationScore == b.ClassificationScore && a.ClassificationLabel == b.ClassificationLabel &&
            a.AppliedRotationDegrees == b.AppliedRotationDegrees && a.EmittedCount == b.EmittedCount &&
            ReferenceEquals(a.CtcSpans, b.CtcSpans) && a.RecognitionContentWidth == b.RecognitionContentWidth &&
            a.RecognitionTensorWidth == b.RecognitionTensorWidth && a.RecognitionTimeSteps == b.RecognitionTimeSteps;
    }

    [Theory]
    [InlineData(0, 0, 255, 0u)]     // red
    [InlineData(0, 255, 0, 1u)]     // green
    [InlineData(255, 0, 0, 2u)]     // blue
    [InlineData(0, 0, 0, 3u)]       // black
    public void Classifier_ReturnsArgmaxAndDegrees(byte b, byte g, byte r, uint label)
    {
        using var classifier = Tiny(4);
        PaddleOcrDocumentOrientationResult result = classifier.Classify(Solid(40, 30, b, g, r), 40, 30);
        Assert.Equal(label, result.Label);
        Assert.Equal((int)label * 90, result.OrientationDegrees);
        Assert.InRange(result.Score, 0.5f, 1f);
    }

    [Fact]
    public void Classifier_RejectsTwoClassModelAndInvalidInput()
    {
        Assert.Throws<ArgumentException>(() => Tiny(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => Tiny(4, new() { Backend = OcrBackend.Cpu, IntraOpThreads = 17 }));
        using var classifier = Tiny(4);
        Assert.Throws<ArgumentException>(() => classifier.Classify(new byte[3], 2, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => classifier.Classify([], 0, 1));
        using var small = Tiny(4, new() { Backend = OcrBackend.Cpu, MaxImagePixels = 100 });
        Assert.Throws<InvalidOperationException>(() => small.Classify(new byte[11 * 10 * 3], 11, 10));
    }

    [Fact]
    public void Classifier_ConcurrentCallsUseSeparateBuffers()
    {
        using var classifier = Tiny(4, new() { Backend = OcrBackend.Cpu, IntraOpThreads = 1, MaxPooledSessions = 2 });
        byte[][] images = [Solid(32, 48, 0, 0, 255), Solid(48, 32, 255, 0, 0), Solid(32, 32, 0, 255, 0), Solid(300, 20, 0, 0, 0)];
        (int W, int H)[] sizes = [(32, 48), (48, 32), (32, 32), (300, 20)];
        PaddleOcrDocumentOrientationResult[] expected = images.Select((im, i) => classifier.Classify(im, sizes[i].W, sizes[i].H)).ToArray();
        Assert.Equal([0u, 2u, 1u, 3u], expected.Select(e => e.Label).ToArray());
        Parallel.For(0, 64, i => Assert.Equal(expected[i % 4], classifier.Classify(images[i % 4], sizes[i % 4].W, sizes[i % 4].H)));
    }
}
