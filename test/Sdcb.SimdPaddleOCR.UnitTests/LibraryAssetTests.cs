using System.Reflection;
using System.Runtime.Versioning;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class LibraryAssetTests
{
    [Fact]
    public void ConsumesExpectedLibraryTfm()
    {
        string? tfm = typeof(PaddleOcrAll).Assembly
            .GetCustomAttribute<TargetFrameworkAttribute>()?.FrameworkName;
        Assert.NotNull(tfm);
#if USE_NS20_LIBRARY
        Assert.Contains("NETStandard", tfm, StringComparison.OrdinalIgnoreCase);
#else
        Assert.Contains("NETCoreApp", tfm, StringComparison.OrdinalIgnoreCase);
#endif
    }

    [Fact]
    public void LibraryAssemblyReferencesLoad()
    {
        foreach (AssemblyName name in typeof(PaddleOcrAll).Assembly.GetReferencedAssemblies())
            Assembly.Load(name);
    }

    [Fact]
    public async Task PaddleOcrAll_ExposesSharedStageComponents_AndGuardsDisposed()
    {
        using PaddleOcrAll withCls = await PaddleOcrAll.LoadAsync(
            Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default,
            new PaddleOcrOptions { UseDirectionClassification = true },
            TestContext.Current.CancellationToken);
        Assert.NotNull(withCls.Detector);
        Assert.NotNull(withCls.Classifier);
        Assert.NotNull(withCls.Recognizer);

        byte[] tinyBlank = new byte[32 * 32 * 3];
        PaddleOcrDetectionResult det = withCls.Detector.Detect(tinyBlank, 32, 32);
        Assert.NotNull(det.Boxes);
        PaddleOcrClassificationResult cls = withCls.Classifier!.Classify(tinyBlank, 32, 32);
        Assert.True(cls.OrientationDegrees is 0 or 180);
        PaddleOcrRecognitionResult rec = withCls.Recognizer.Recognize(tinyBlank, 32, 32);
        Assert.NotNull(rec.Text);

        withCls.Dispose();
        Assert.Throws<ObjectDisposedException>(() => withCls.Detector);
        Assert.Throws<ObjectDisposedException>(() => withCls.Classifier);
        Assert.Throws<ObjectDisposedException>(() => withCls.Recognizer);

        using PaddleOcrAll withoutCls = await PaddleOcrAll.LoadAsync(
            Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default,
            new PaddleOcrOptions { UseDirectionClassification = false },
            TestContext.Current.CancellationToken);
        Assert.NotNull(withoutCls.Detector);
        Assert.Null(withoutCls.Classifier);
        Assert.NotNull(withoutCls.Recognizer);
    }
}
