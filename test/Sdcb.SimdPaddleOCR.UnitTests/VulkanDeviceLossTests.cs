#if !USE_NS20_LIBRARY
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;

namespace Sdcb.SimdPaddleOCR.UnitTests;

/// <summary>
/// Marks the process-wide Vulkan device lost, which disturbs every other Vulkan test running in the
/// process: opt-in (SIMDONNX_TEST_DEVICELOSS=1), run it on its own.
/// </summary>
public class VulkanDeviceLossTests
{
    [Fact]
    public async Task Ocr_RecoversFromDeviceLoss()
    {
        if (Environment.GetEnvironmentVariable("SIMDONNX_TEST_DEVICELOSS") != "1") Assert.Skip("opt-in: SIMDONNX_TEST_DEVICELOSS=1");
        VkBackend before;
        try { before = VkBackend.Default; }
        catch (Exception ex) { Assert.Skip($"no Vulkan device: {ex.Message}"); return; }

        using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default,
            new PaddleOcrOptions
            {
                Detector = new() { Backend = OcrBackend.Vulkan },
                Classifier = new() { Backend = OcrBackend.Vulkan },
                Recognizer = new() { Backend = OcrBackend.Vulkan },
            }, cancellationToken: TestContext.Current.CancellationToken);
        (byte[] bgr, int w, int h) = Stripes();

        PaddleOcrResult first = ocr.Run(bgr, w, h, w * 3);
        before.Dev.Lost = true;
        PaddleOcrResult second = ocr.Run(bgr, w, h, w * 3);   // the lost call may finish on the CPU
        PaddleOcrResult third = ocr.Run(bgr, w, h, w * 3);    // back on a fresh device

        Assert.NotSame(before, VkBackend.Default);
        Assert.NotEqual(0, first.DetectedCount);
        Assert.Equal(first.Text, second.Text);
        Assert.Equal(first.Text, third.Text);
    }

    /// <summary>Rows of dark glyph-like blocks on white: enough for the detector to find text lines.</summary>
    private static (byte[] bgr, int w, int h) Stripes()
    {
        const int w = 640, h = 240;
        byte[] bgr = new byte[w * h * 3];
        Array.Fill(bgr, (byte)255);
        var rng = new Random(7);
        for (int row = 0; row < 4; row++)
        {
            int y0 = 30 + row * 50, x = 30;
            while (x < w - 60)
            {
                int gw = rng.Next(8, 22), gh = rng.Next(14, 26), stroke = rng.Next(2, 5);
                for (int y = y0; y < y0 + gh; y++)
                    for (int xx = x; xx < x + gw; xx++)
                    {
                        bool edge = y < y0 + stroke || y >= y0 + gh - stroke || xx < x + stroke || xx >= x + gw - stroke
                            || (xx - x) == gw / 2;
                        if (!edge) continue;
                        int o = (y * w + xx) * 3;
                        bgr[o] = bgr[o + 1] = bgr[o + 2] = 20;
                    }
                x += gw + rng.Next(3, 10);
            }
        }
        return (bgr, w, h);
    }
}
#endif
