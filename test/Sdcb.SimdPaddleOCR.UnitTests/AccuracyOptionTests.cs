namespace Sdcb.SimdPaddleOCR.UnitTests;

public class AccuracyOptionTests
{
    [Fact]
    public void DetSizeDefaultOnlyDownscales()
    {
        var size = PPOCRPreprocess.ComputeDetSize(400, 200, 960);
        Assert.Equal((416, 192), (size.Width, size.Height));
    }

    [Fact]
    public void DetSizeMinSideUpscalesTowardItUnderTheLongSideCap()
    {
        // Short side 200 → 736 wants 3.68x, but the long side caps the ratio at 960/400 = 2.4.
        var capped = PPOCRPreprocess.ComputeDetSize(400, 200, 960, minSideLength: 736);
        Assert.Equal((960, 480), (capped.Width, capped.Height));
        // Square 300x300 → 736x736 (rounded to the 32 grid: 736).
        var square = PPOCRPreprocess.ComputeDetSize(300, 300, 960, minSideLength: 736);
        Assert.Equal((736, 736), (square.Width, square.Height));
        // Already large enough: unchanged from the default rule.
        var big = PPOCRPreprocess.ComputeDetSize(1600, 900, 960, minSideLength: 736);
        Assert.Equal(PPOCRPreprocess.ComputeDetSize(1600, 900, 960), big);
    }

    // A 12-px-tall white crop with glyph blocks at the given x ranges.
    private static byte[] Crop(int width, int height, params (int X0, int X1)[] glyphs)
    {
        byte[] bgr = new byte[width * height * 3];
        bgr.AsSpan().Fill(255);
        foreach ((int x0, int x1) in glyphs)
            for (int y = 2; y < height - 2; y++)
                for (int x = x0; x < x1; x++)
                    bgr.AsSpan((y * width + x) * 3, 3).Clear();
        return bgr;
    }

    // One time step per crop column: tensorWidth == contentWidth == timeSteps == width.
    private static string Apply(string text, PaddleOcrCtcSpan[] spans, byte[] crop, int width, int height, float ratio)
        => PPOCRSpaceGap.Apply(text, spans, width, width, width, crop, width, height, ratio);

    [Fact]
    public void WideEmptyRunInsertsASpace()
    {
        byte[] crop = Crop(40, 12, (0, 8), (20, 28));
        PaddleOcrCtcSpan[] spans = [new("中", 1, 0, 1), new("A", 1, 20, 21)];
        Assert.Equal("中 A", Apply("中A", spans, crop, 40, 12, 0.5f));
    }

    [Fact]
    public void NarrowGapIsLeftAlone()
    {
        byte[] crop = Crop(40, 12, (0, 8), (11, 19));
        PaddleOcrCtcSpan[] spans = [new("中", 1, 0, 1), new("文", 1, 11, 12)];
        Assert.Equal("中文", Apply("中文", spans, crop, 40, 12, 0.5f));
    }

    [Fact]
    public void NoSpaceNextToExistingSpaceOrCjkPunctuation()
    {
        byte[] crop = Crop(60, 12, (0, 8), (30, 38));
        PaddleOcrCtcSpan[] withSpace = [new("A", 1, 0, 1), new(" ", 1, 15, 16), new("B", 1, 30, 31)];
        Assert.Equal("A B", Apply("A B", withSpace, crop, 60, 12, 0.5f));
        PaddleOcrCtcSpan[] comma = [new("，", 1, 0, 1), new("中", 1, 30, 31)];
        Assert.Equal("，中", Apply("，中", comma, crop, 60, 12, 0.5f));
    }

    [Fact]
    public void LightTextOnDarkBackgroundUsesTheMedianBackground()
    {
        byte[] crop = Crop(40, 12, (0, 8), (20, 28));
        for (int i = 0; i < crop.Length; i++) crop[i] = (byte)(255 - crop[i]);
        PaddleOcrCtcSpan[] spans = [new("中", 1, 0, 1), new("A", 1, 20, 21)];
        Assert.Equal("中 A", Apply("中A", spans, crop, 40, 12, 0.5f));
    }

    [Fact]
    public void UnclipRatioDefaultsAreModelAware()
    {
        using var tiny = new PaddleOcrDetector(Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModel.Detection.OpenRead());
        using var small = new PaddleOcrDetector(Sdcb.SimdPaddleOCR.Models.ChineseV6Small.ChineseV6SmallModel.Detection.OpenRead());
        Assert.Equal((1.4f, 0.4f), (tiny.ResolvedOptions.UnclipRatio, tiny.ResolvedOptions.BoxThreshold));
        Assert.Equal((1.6f, 0.45f), (small.ResolvedOptions.UnclipRatio, small.ResolvedOptions.BoxThreshold));
        // An explicit value always wins, and setting one does not disturb the other default.
        using var smallExplicit = new PaddleOcrDetector(Sdcb.SimdPaddleOCR.Models.ChineseV6Small.ChineseV6SmallModel.Detection.OpenRead(),
            new PaddleOcrDetectorOptions { UnclipRatio = 1.4f });
        Assert.Equal((1.4f, 0.45f), (smallExplicit.ResolvedOptions.UnclipRatio, smallExplicit.ResolvedOptions.BoxThreshold));
        using var tinyExplicit = new PaddleOcrDetector(Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModel.Detection.OpenRead(),
            new PaddleOcrDetectorOptions { UnclipRatio = 1.6f });
        Assert.Equal((1.6f, 0.4f), (tinyExplicit.ResolvedOptions.UnclipRatio, tinyExplicit.ResolvedOptions.BoxThreshold));
    }
}
