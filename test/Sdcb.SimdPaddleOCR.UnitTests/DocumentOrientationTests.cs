using System.Reflection;
using System.Text.Json;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.UnitTests;

/// <summary>验证纸张方向的算子适配、预处理、旋转、并发及流水线开关。</summary>
public class DocumentOrientationTests
{
    /// <summary>打开由独立 Python 工具生成并内嵌到测试程序集的参考资源。</summary>
    /// <param name="name">不带 orientation. 前缀的资源文件名。</param>
    /// <returns>新的参考资源流，由测试方法释放。</returns>
    private static Stream Asset(string name) => typeof(DocumentOrientationTests).Assembly
        .GetManifestResourceStream("orientation." + name) ?? throw new Exception(name);

    /// <summary>验证 Gemm 的转置、缩放、偏置广播，以及 Constant 内部名称与输出边不同的情形。</summary>
    [Fact]
    public void Gemm_TransposeScalingBroadcastBias_AndConstantNames()
    {
        using Stream json = Asset("gemm.json");
        using JsonDocument cases = JsonDocument.Parse(json);
        foreach (JsonElement c in cases.RootElement.EnumerateArray())
        {
            using Stream stream = Asset(c.GetProperty("name").GetString() + ".onnx");
            using Model model = Model.Load(stream);
            int[] shape = c.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            var compiled = new CompiledModel(model, shape, 1);
            try
            {
                using InferenceSession session = compiled.CreateRequest();
                float[] input = c.GetProperty("input").EnumerateArray().Select(v => v.GetSingle()).ToArray();
                float[] expected = c.GetProperty("expected").EnumerateArray().Select(v => v.GetSingle()).ToArray();
                float[] actual = session.Run(input);
                KernelCorrectnessTests.AssertClose(expected, actual);
            }
            finally { compiled.Dispose(); }
        }
    }

    /// <summary>验证不同像素格式和张量布局的中心裁剪归一化与 OpenCV 黄金值一致。</summary>
    /// <param name="format">输入像素格式，包含三通道和四通道格式。</param>
    /// <param name="nhwc">是否使用交错通道的 NHWC 输出布局。</param>
    [Theory]
    [InlineData(ImagePixelFormat.Bgr24, false)]
    [InlineData(ImagePixelFormat.Rgb24, false)]
    [InlineData(ImagePixelFormat.Bgra32, true)]
    [InlineData(ImagePixelFormat.Rgba32, true)]
    public void CenterCropRgbNormalization_MatchesOpenCv(ImagePixelFormat format, bool nhwc)
    {
        const int width = 17, height = 19;
        int bpp = ImagePixels.BytesPerPixel(format), stride = width * bpp + 7;
        byte[] source = new byte[stride * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int c = 0; c < 3; c++)
                {
                    int sourceChannel = ImagePixels.SwapRedBlue(format) ? 2 - c : c;
                    source[y * stride + x * bpp + c] = (byte)((((y * width + x) * 3 + sourceChannel) * 37 + 11) % 256);
                }
        float[] output = new float[3 * 224 * 224];
        PPOCRPreprocess.DocumentOrientation(source, width, height, stride, output, new ResizeWorkspace(), nhwc, format);
        using Stream json = Asset("preprocess.json");
        using JsonDocument samples = JsonDocument.Parse(json);
        foreach (JsonElement sample in samples.RootElement.EnumerateArray())
        {
            int c = sample.GetProperty("channel").GetInt32(), x = sample.GetProperty("x").GetInt32(), y = sample.GetProperty("y").GetInt32();
            float expected = sample.GetProperty("value").GetSingle();
            float actual = output[nhwc ? (y * 224 + x) * 3 + c : c * 224 * 224 + y * 224 + x];
            Assert.True(Math.Abs(expected - actual) <= 3e-7f, $"{format} ({x},{y},{c}): {expected} != {actual}");
        }
    }

    /// <summary>验证非正方形、带行尾填充图像的直角旋转及原图坐标逆变换。</summary>
    /// <param name="angle">待测试的逆时针旋转角度。</param>
    /// <param name="order">旋转后按行展开时，每个像素对应的原图像素编号。</param>
    [Theory]
    [InlineData(90, new int[] { 2, 5, 1, 4, 0, 3 })]
    [InlineData(180, new int[] { 5, 4, 3, 2, 1, 0 })]
    [InlineData(270, new int[] { 3, 0, 4, 1, 5, 2 })]
    public void RectangularRotation_AndSourceCoordinatesAgree(int angle, int[] order)
    {
        byte[] source = new byte[2 * 13];
        for (int y = 0; y < 2; y++) for (int x = 0; x < 3; x++)
            source[y * 13 + x * 3] = (byte)(y * 3 + x);
        byte[] rotated = new byte[18];
        DocumentRotation.Rotate(source, 3, 2, 13, ImagePixelFormat.Bgr24, angle, rotated);
        Assert.Equal(order, Enumerable.Range(0, 6).Select(i => (int)rotated[i * 3]).ToArray());
        var size = DocumentRotation.Size(3, 2, angle);
        for (int i = 0; i < 6; i++)
        {
            float x = i % size.Width, y = i / size.Width;
            var point = DocumentRotation.ToSource(new PaddleOcrDetectionBox(x, y, x, y, x, y, x, y, 1), 3, 2, angle);
            Assert.Equal(order[i] % 3, point.X1); Assert.Equal(order[i] / 3, point.Y1);
        }
    }

    /// <summary>验证模型四分类输出、角度映射及非法图像输入检查。</summary>
    [Fact]
    public void DocumentModel_HasFourClasses_AndRejectsInvalidInputs()
    {
        using Stream stream = DocumentOrientationModel.OpenRead();
        using var classifier = new PaddleOcrDocumentOrientationClassifier(stream, new() { Backend = OcrBackend.Cpu });
        Assert.Throws<ArgumentException>(() => classifier.Classify(new byte[3], 2, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => classifier.Classify([], 0, 1));
        var result = classifier.Classify(new byte[16 * 24 * 3], 16, 24);
        Assert.InRange(result.Label, 0u, 3u);
        Assert.InRange(result.Score, 0f, 1f);
        Assert.Equal((int)result.Label * 90, result.OrientationDegrees);
    }

    /// <summary>交替并发分类黑白图像，检查各请求不会覆盖共享分类器中的输入缓冲区。</summary>
    [Fact]
    public void ConcurrentClassification_UsesSeparateRequestBuffers()
    {
        using Stream stream = DocumentOrientationModel.OpenRead();
        using var classifier = new PaddleOcrDocumentOrientationClassifier(stream, new() { Backend = OcrBackend.Cpu, IntraOpThreads = 1 });
        byte[][] images = [new byte[32 * 48 * 3], Enumerable.Repeat((byte)255, 32 * 48 * 3).ToArray()];
        var expected = images.Select(image => classifier.Classify(image, 32, 48)).ToArray();
        Parallel.For(0, 8, i => Assert.Equal(expected[i % 2], classifier.Classify(images[i % 2], 32, 48)));
    }

    /// <summary>验证独立模型加载遵守取消令牌，并拒绝文本行分类模型提供方。</summary>
    /// <returns>完成两种错误路径验证的异步任务。</returns>
    [Fact]
    public async Task DocumentProviderLoad_ObservesCancellationAndKind()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => PaddleOcrAll.LoadAsync(
            Models.ChineseV6Tiny.ChineseV6TinyModels.Default, DocumentOrientationModel.Provider, null, cancellation.Token));
        Assert.Throws<ArgumentException>(() => PaddleOcrAll.Load(Models.ChineseV6Tiny.ChineseV6TinyModels.Default,
            Models.TextLineOrientation.TextLineOrientationModel.Provider, null));
    }

    /// <summary>验证关闭纸张阶段时不运行方向图，并返回空方向结果及零校正角度。</summary>
    [Fact]
    public void OrientationStage_CanBeDisabledWithoutLoadingItIntoTheGraph()
    {
        using PaddleOcrAll ocr = PaddleOcrAll.Load(Models.ChineseV6Tiny.ChineseV6TinyModels.Default,
            DocumentOrientationModel.Provider, new() { UseDocumentOrientationClassification = false,
                Detector = new() { Backend = OcrBackend.Cpu }, Classifier = new() { Backend = OcrBackend.Cpu },
                Recognizer = new() { Backend = OcrBackend.Cpu } });
        var result = ocr.Run(new byte[32 * 48 * 3], 32, 48);
        Assert.Null(result.DocumentOrientation);
        Assert.Equal(0, result.AppliedDocumentRotationDegrees);
    }
}
