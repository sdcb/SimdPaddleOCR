using System.Reflection;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class PreprocessThreadBudgetTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(16)]
    public async Task ResolvesPreprocessBudgetIndependentlyOfLineWorkers(int requested)
    {
        using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default,
            new PaddleOcrOptions
            {
                Detector = new() { Backend = OcrBackend.Cpu },
                Classifier = new() { Backend = OcrBackend.Cpu },
                Recognizer = new() { Backend = OcrBackend.Cpu },
                LineWorkerCount = 4, PreprocessWorkerCount = requested
            }, cancellationToken: TestContext.Current.CancellationToken);

        int expected = requested == 0 ? Math.Min(Environment.ProcessorCount, 16)
            : Math.Min(requested, Environment.ProcessorCount);
        Assert.Equal(expected, ReadBudget(ocr, "_cropWorkers"));
        Assert.Equal(Math.Min(Environment.ProcessorCount, 4), ocr.EffectiveLineWorkerCount);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(17)]
    [InlineData(int.MaxValue)]
    public void RejectsInvalidPreprocessBudget(int requested)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PaddleOcrAll(null!, null, null!, Array.Empty<byte>(),
                new PaddleOcrOptions
                {
                    Detector = new() { Backend = OcrBackend.Cpu },
                    Classifier = new() { Backend = OcrBackend.Cpu },
                    Recognizer = new() { Backend = OcrBackend.Cpu },
                    PreprocessWorkerCount = requested
                }));
        Assert.Equal("options", error.ParamName);
        Assert.Contains(nameof(PaddleOcrOptions.PreprocessWorkerCount), error.Message);
    }

    [Fact]
    public async Task SerialPreprocessingPreservesGraphBudgets()
    {
        using PaddleOcrAll automatic = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default,
            new PaddleOcrOptions
            {
                Detector = new() { Backend = OcrBackend.Cpu },
                Classifier = new() { Backend = OcrBackend.Cpu },
                Recognizer = new() { Backend = OcrBackend.Cpu },
                LineWorkerCount = 4, DetIntraOpThreads = 1
            }, cancellationToken: TestContext.Current.CancellationToken);
        using PaddleOcrAll limited = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default,
            new PaddleOcrOptions
            {
                Detector = new() { Backend = OcrBackend.Cpu },
                Classifier = new() { Backend = OcrBackend.Cpu },
                Recognizer = new() { Backend = OcrBackend.Cpu },
                LineWorkerCount = 4, DetIntraOpThreads = 1, PreprocessWorkerCount = 1
            }, cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(1, ReadBudget(limited, "_cropWorkers"));
        foreach (string budget in new[] { "_recIntraOpBase", "_recIntraOpMax", "_lineWorkers", "_gpuCtcIntraOpThreads" })
            Assert.Equal(ReadBudget(automatic, budget), ReadBudget(limited, budget));

    }

    private static int ReadBudget(PaddleOcrAll ocr, string field) =>
        (int)typeof(PaddleOcrAll).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(ocr)!;
}
