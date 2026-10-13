using System.Text;
using Sdcb.SimdPaddleOCR.Pdf;
using SkiaSharp;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Core;

namespace Sdcb.SimdPaddleOCR.Pdf.Tests;

public sealed class SearchablePdfTests
{
    [Fact]
    public void IndependentReaderExtractsChineseAsciiAndSupplementaryUnicode()
    {
        string text = "中文复制对应 A0-9，。㎡²🙂𠀀";
        using SKBitmap image = BlankImage();
        byte[] bytes = Write(image, Result(Line(text)));
        using PdfDocument pdf = PdfDocument.Open(bytes);
        Assert.Equal(text, pdf.GetPage(1).Text);
        Assert.All(pdf.GetPage(1).Letters, x => Assert.Equal(TextRenderingMode.Neither, x.RenderingMode));
        Assert.Equal(1000, pdf.GetPage(1).Width);
        Assert.Equal(200, pdf.GetPage(1).Height);
    }

    [Fact]
    public void MoreThan255DistinctCharactersAndMultiplePagesKeepExactText()
    {
        string text = string.Concat(Enumerable.Range(0x4E00, 600).Select(char.ConvertFromUtf32));
        using SKBitmap image = BlankImage();
        byte[] bytes = Write(image, Result(Line(text)), Result(Line("第二页123")));
        using PdfDocument pdf = PdfDocument.Open(bytes);
        Assert.Equal(2, pdf.NumberOfPages);
        Assert.Equal(text, pdf.GetPage(1).Text);
        Assert.Equal("第二页123", pdf.GetPage(2).Text);
    }

    [Fact]
    public void CtcCharacterPositionsAndDpiSurvivePdfCoordinateConversion()
    {
        using SKBitmap image = BlankImage();
        PaddleOcrLine line = Line("甲乙", new(50, 40, 250, 40, 250, 100, 50, 100, 1));
        using MemoryStream stream = new();
        using (PaddleOcrPdfWriter writer = new(stream, 144)) writer.AddPage(image, Result(line));
        using PdfDocument pdf = PdfDocument.Open(stream.ToArray());
        var letters = pdf.GetPage(1).Letters;
        Assert.Equal("甲乙", pdf.GetPage(1).Text);
        Assert.Equal(25, letters[0].Location.X, 4);
        Assert.Equal(50, letters[0].Location.Y, 4);
        Assert.Equal(75, letters[1].Location.X, 4);
        Assert.Equal(50, letters[0].Width, 4);
    }

    [Fact]
    public void UpsideDownLineKeepsTextOrderAndReversedBaseline()
    {
        using SKBitmap image = BlankImage();
        byte[] bytes = Write(image, Result(Line("甲乙", new(50, 40, 250, 40, 250, 100, 50, 100, 1), 180)));
        using PdfDocument pdf = PdfDocument.Open(bytes);
        var letters = pdf.GetPage(1).Letters;
        Assert.Equal("甲乙", pdf.GetPage(1).Text);
        Assert.Equal(250, letters[0].Location.X, 4);
        Assert.Equal(160, letters[0].Location.Y, 4);
        Assert.True(letters[0].EndBaseLine.X < letters[0].Location.X);
    }

    [Fact]
    public void TallLineKeepsVerticalSelectionDirection()
    {
        using SKBitmap image = BlankImage();
        using PdfDocument pdf = PdfDocument.Open(Write(image,
            Result(Line("甲乙", new(50, 40, 70, 40, 70, 160, 50, 160, 1)))));
        var letter = pdf.GetPage(1).Letters[0];
        Assert.Equal("甲乙", pdf.GetPage(1).Text);
        Assert.True(letter.EndBaseLine.Y > letter.Location.Y);
    }

    [Fact]
    public void OriginalJpegIsEmbeddedWithoutRecompression()
    {
        using SKBitmap image = BlankImage();
        using SKImage skImage = SKImage.FromBitmap(image);
        using SKData jpeg = skImage.Encode(SKEncodedImageFormat.Jpeg, 90);
        byte[] encoded = jpeg.ToArray();
        using MemoryStream stream = new();
        using (PaddleOcrPdfWriter writer = new(stream, 72)) writer.AddPage(image, Result(Line("测试")), encoded);
        Assert.True(stream.ToArray().AsSpan().IndexOf(encoded) >= 0);
        using PdfDocument pdf = PdfDocument.Open(stream.ToArray());
        Assert.Single(pdf.GetPage(1).GetImages());
    }

    [Fact]
    public void TransparentImageIsFlattenedOnWhite()
    {
        using SKBitmap image = new(5, 5, SKColorType.Bgra8888, SKAlphaType.Premul);
        image.Erase(SKColors.Transparent);
        using SKImage skImage = SKImage.FromBitmap(image);
        using SKData png = skImage.Encode(SKEncodedImageFormat.Png, 100);
        using SKBitmap decoded = PdfImage.Decode(png.ToArray());
        Assert.Equal(SKAlphaType.Opaque, decoded.AlphaType);
        Assert.Equal(SKColors.White, decoded.GetPixel(0, 0));
        Assert.Throws<InvalidDataException>(() => PdfImage.Decode(png.ToArray(), maxPixels: 10));
    }

    [Fact]
    public void OuterSelectionEdgesCoverTheFirstAndLastVisibleCharacter()
    {
        // CTC fires near character centers. The PDF must also cover the visible
        // leading/trailing half, while retaining the interior midpoint at x=140.
        PaddleOcrLine line = new()
        {
            Text = "甲乙", Box = new(50, 40, 250, 40, 250, 100, 50, 100, 1),
            EmittedCount = 2, RecognitionScore = 1, ClassificationScore = 1,
            ClassificationLabel = 0, AppliedRotationDegrees = 0,
            RecognitionContentWidth = 100, RecognitionTensorWidth = 100, RecognitionTimeSteps = 100,
            CtcSpans = [new("甲", 1, 20, 30), new("乙", 1, 60, 70)]
        };
        using SKBitmap image = BlankImage();
        using PdfDocument pdf = PdfDocument.Open(Write(image, Result(line)));
        var letters = pdf.GetPage(1).Letters;
        Assert.Equal(50, letters[0].Location.X, 4);
        Assert.Equal(140, letters[1].Location.X, 4);
        Assert.Equal(250, letters[1].EndBaseLine.X, 4);
        Assert.Equal(20, line.CtcSpans[0].StartColumn); // Original alignment is retained.
        Assert.Equal(70, line.CtcSpans[1].EndColumn);
    }

    [Fact]
    public void PaddingHallucinationHasNoSelectableSourcePositionAndIsExcluded()
    {
        // Reproduces the handwritten line on scan 40.jpg: the third token fires
        // at steps 7..8, beyond the 48 / 64 * 8 = 6 content steps.
        PaddleOcrLine line = new()
        {
            Text = "医务级", Box = new(50, 40, 150, 40, 150, 140, 50, 140, 1),
            EmittedCount = 3, RecognitionScore = 1, ClassificationScore = 1,
            ClassificationLabel = 0, AppliedRotationDegrees = 0,
            RecognitionContentWidth = 48, RecognitionTensorWidth = 64, RecognitionTimeSteps = 8,
            CtcSpans = [new("医", 1, 1, 3), new("务", 1, 4, 5), new("级", 1, 7, 8)]
        };
        using SKBitmap image = BlankImage();
        using PdfDocument pdf = PdfDocument.Open(Write(image, Result(line)));
        Assert.Equal("医务", pdf.GetPage(1).Text);
        Assert.Equal("医务级", line.Text); // Caller-owned OCR data is retained.
        var filtered = PdfTextAlignment.RemovePaddingEmissions(Result(line));
        Assert.Equal(2, filtered.Lines[0].CtcSpans!.Length);
        Assert.All(filtered.Lines[0].EstimateCharacterBoxes(), x => Assert.True(x.X2 > x.X1));
    }

    [Fact]
    public void MissingAlignmentFailsInsteadOfProducingMisalignedCopyText()
    {
        using SKBitmap image = BlankImage();
        PaddleOcrLine line = Line("中文");
        line = new PaddleOcrLine
        {
            Text = line.Text, Box = line.Box, EmittedCount = line.EmittedCount,
            RecognitionScore = 1, ClassificationScore = 1, ClassificationLabel = 0,
            AppliedRotationDegrees = 0
        };
        using MemoryStream stream = new();
        using PaddleOcrPdfWriter writer = new(stream);
        Assert.Throws<InvalidOperationException>(() => writer.AddPage(image, Result(line)));
        Assert.Equal(0, writer.PageCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void InvalidDpiIsRejected(float dpi)
    {
        using MemoryStream stream = new();
        Assert.Throws<ArgumentOutOfRangeException>(() => new PaddleOcrPdfWriter(stream, dpi));
    }

    private static SKBitmap BlankImage()
    {
        SKBitmap bitmap = new(1000, 200, SKColorType.Bgra8888, SKAlphaType.Opaque);
        bitmap.Erase(SKColors.White);
        return bitmap;
    }

    private static byte[] Write(SKBitmap image, params PaddleOcrResult[] results)
    {
        using MemoryStream stream = new();
        using (PaddleOcrPdfWriter writer = new(stream, 72))
            foreach (PaddleOcrResult result in results) writer.AddPage(image, result);
        return stream.ToArray();
    }

    private static PaddleOcrResult Result(params PaddleOcrLine[] lines) => new()
    {
        Lines = lines, DetectedCount = lines.Length,
        DetectorResizedWidth = 1000, DetectorResizedHeight = 200
    };

    private static PaddleOcrLine Line(string text, PaddleOcrDetectionBox? box = null, int rotation = 0)
    {
        Rune[] runes = text.EnumerateRunes().ToArray();
        int width = runes.Length * 10;
        return new()
        {
            Text = text, Box = box ?? new(50, 40, 950, 40, 950, 100, 50, 100, 1),
            RecognitionScore = 1, ClassificationScore = 1, ClassificationLabel = rotation == 180 ? 1u : 0u,
            AppliedRotationDegrees = rotation, EmittedCount = (uint)runes.Length,
            RecognitionContentWidth = width, RecognitionTensorWidth = width,
            RecognitionTimeSteps = width,
            CtcSpans = runes.Select((r, i) => new PaddleOcrCtcSpan(r.ToString(), 1, i * 10, (i + 1) * 10)).ToArray()
        };
    }
}
