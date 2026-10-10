using System.Reflection;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class GpuCtcThreadBudgetTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(16)]
    public async Task ResolvesAutomaticAndExplicitGpuCtcBudgets(int requested)
    {
        using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default,
            new PaddleOcrOptions
            {
                Detector = new() { Backend = OcrBackend.Cpu },
                Classifier = new() { Backend = OcrBackend.Cpu },
                Recognizer = new() { Backend = OcrBackend.Cpu },
                GpuCtcIntraOpThreads = requested
            }, cancellationToken: TestContext.Current.CancellationToken);

        int expected = requested == 0 ? Math.Min(Environment.ProcessorCount, 16)
            : Math.Min(requested, Environment.ProcessorCount);
        Assert.Equal(expected, ReadBudget(ocr, "_gpuCtcIntraOpThreads"));
        if (requested == 0)
            Assert.Equal(ReadBudget(ocr, "_recIntraOpMax"), ReadBudget(ocr, "_gpuCtcIntraOpThreads"));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(17)]
    [InlineData(int.MaxValue)]
    public void RejectsInvalidGpuCtcBudget(int requested)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PaddleOcrAll(null!, null, null!, Array.Empty<byte>(),
                new PaddleOcrOptions
                {
                    Detector = new() { Backend = OcrBackend.Cpu },
                    Classifier = new() { Backend = OcrBackend.Cpu },
                    Recognizer = new() { Backend = OcrBackend.Cpu },
                    GpuCtcIntraOpThreads = requested
                }));
        Assert.Equal("options", error.ParamName);
        Assert.Contains(nameof(PaddleOcrOptions.GpuCtcIntraOpThreads), error.Message);
    }

    [Fact]
    public async Task GpuCtcLimitLeavesCpuBudgetsUnchanged()
    {
        using PaddleOcrAll automatic = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default,
            new PaddleOcrOptions
            {
                Detector = new() { Backend = OcrBackend.Cpu },
                Classifier = new() { Backend = OcrBackend.Cpu },
                Recognizer = new() { Backend = OcrBackend.Cpu },
                LineWorkerCount = 1, DetIntraOpThreads = 1
            }, cancellationToken: TestContext.Current.CancellationToken);
        using PaddleOcrAll limited = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default,
            new PaddleOcrOptions
            {
                Detector = new() { Backend = OcrBackend.Cpu },
                Classifier = new() { Backend = OcrBackend.Cpu },
                Recognizer = new() { Backend = OcrBackend.Cpu },
                LineWorkerCount = 1, DetIntraOpThreads = 1, GpuCtcIntraOpThreads = 1
            }, cancellationToken: TestContext.Current.CancellationToken);

        foreach (string budget in new[] { "_recIntraOpBase", "_recIntraOpMax", "_lineWorkers", "_cropWorkers" })
            Assert.Equal(ReadBudget(automatic, budget), ReadBudget(limited, budget));
        MethodInfo lineBudget = typeof(PaddleOcrAll).GetMethod("LineIntraOpBudget",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        foreach (int count in new[] { 1, 2, 25 })
            Assert.Equal(lineBudget.Invoke(automatic, new object[] { count }),
                lineBudget.Invoke(limited, new object[] { count }));

    }

    private static int ReadBudget(PaddleOcrAll ocr, string field) =>
        (int)typeof(PaddleOcrAll).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(ocr)!;
}
