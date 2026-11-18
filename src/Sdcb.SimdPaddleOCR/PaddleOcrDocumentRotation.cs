namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Right-angle page rotation for <see cref="PaddleOcrDocumentOrientationClassifier"/>, and the inverse
/// mapping of OCR results from the rotated (upright) image back to the source image.
/// All angles are counterclockwise rotations of the source; pass
/// <see cref="PaddleOcrDocumentOrientationResult.OrientationDegrees"/> to make a page upright.
/// </summary>
public static class PaddleOcrDocumentRotation
{
    /// <summary>Size after rotating by <paramref name="degrees"/>: width and height swap for 90 and 270.</summary>
    public static (int Width, int Height) RotatedSize(int width, int height, int degrees)
    {
        CheckDegrees(degrees);
        return degrees is 90 or 270 ? (height, width) : (width, height);
    }

    /// <summary>
    /// Rotates the image counterclockwise by <paramref name="degrees"/> (0, 90, 180, 270), copying whole
    /// pixels (no interpolation). <paramref name="destination"/> is tightly packed, the same
    /// <paramref name="format"/>, at least width x height pixels, and must not overlap the source.
    /// <paramref name="stride"/> 0 means tightly packed source rows.
    /// </summary>
    public static void Rotate(ReadOnlySpan<byte> source, int width, int height, int stride,
        ImagePixelFormat format, int degrees, Span<byte> destination)
    {
        CheckDegrees(degrees);
        if (width < 1) throw new ArgumentOutOfRangeException(nameof(width));
        if (height < 1) throw new ArgumentOutOfRangeException(nameof(height));
        int bpp = ImagePixels.BytesPerPixel(format);
        stride = ImagePixels.ResolveStride(width, stride, format);
        if ((long)(height - 1) * stride + (long)width * bpp > source.Length)
            throw new ArgumentException("Source buffer is too small.", nameof(source));
        int rowBytes = checked(width * bpp);
        if (destination.Length < checked(rowBytes * height))
            throw new ArgumentException("Destination buffer is too small.", nameof(destination));
        if (degrees == 0)
        {
            for (int y = 0; y < height; y++)
                source.Slice(y * stride, rowBytes).CopyTo(destination.Slice(y * rowBytes, rowBytes));
            return;
        }
        int rotatedWidth = degrees == 180 ? width : height;
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> row = source.Slice(y * stride, rowBytes);
            for (int x = 0; x < width; x++)
            {
                (int dx, int dy) = degrees switch
                {
                    90 => (y, width - 1 - x),
                    180 => (width - 1 - x, height - 1 - y),
                    _ => (height - 1 - y, x), // 270
                };
                row.Slice(x * bpp, bpp).CopyTo(destination.Slice((dy * rotatedWidth + dx) * bpp, bpp));
            }
        }
    }

    /// <summary>
    /// Maps a box found in the image rotated by <paramref name="degrees"/> back to the source image
    /// (<paramref name="sourceWidth"/> x <paramref name="sourceHeight"/>, before rotation). Vertex order
    /// is kept, so X1/Y1 is still the reading-order top-left of the text.
    /// </summary>
    public static PaddleOcrDetectionBox ToSource(PaddleOcrDetectionBox box, int sourceWidth, int sourceHeight, int degrees)
    {
        CheckDegrees(degrees);
        (float X, float Y) P(float x, float y) => degrees switch
        {
            90 => (sourceWidth - 1 - y, x),
            180 => (sourceWidth - 1 - x, sourceHeight - 1 - y),
            270 => (y, sourceHeight - 1 - x),
            _ => (x, y),
        };
        var p1 = P(box.X1, box.Y1); var p2 = P(box.X2, box.Y2);
        var p3 = P(box.X3, box.Y3); var p4 = P(box.X4, box.Y4);
        return new(p1.X, p1.Y, p2.X, p2.Y, p3.X, p3.Y, p4.X, p4.Y, box.Score);
    }

    /// <summary>
    /// Copy of <paramref name="line"/> with its box mapped back to the source image (see
    /// <see cref="ToSource(PaddleOcrDetectionBox, int, int, int)"/>). Text, scores and CTC alignment are
    /// kept, so <see cref="PaddleOcrLine.EstimateCharacterBoxes"/> yields source-image boxes too.
    /// </summary>
    public static PaddleOcrLine ToSource(PaddleOcrLine line, int sourceWidth, int sourceHeight, int degrees)
    {
        if (line is null) throw new ArgumentNullException(nameof(line));
        return new PaddleOcrLine
        {
            Box = ToSource(line.Box, sourceWidth, sourceHeight, degrees),
            Text = line.Text,
            RecognitionScore = line.RecognitionScore,
            ClassificationScore = line.ClassificationScore,
            ClassificationLabel = line.ClassificationLabel,
            AppliedRotationDegrees = line.AppliedRotationDegrees,
            EmittedCount = line.EmittedCount,
            CtcSpans = line.CtcSpans,
            RecognitionContentWidth = line.RecognitionContentWidth,
            RecognitionTensorWidth = line.RecognitionTensorWidth,
            RecognitionTimeSteps = line.RecognitionTimeSteps,
        };
    }

    private static void CheckDegrees(int degrees)
    {
        if (degrees is not (0 or 90 or 180 or 270)) throw new ArgumentOutOfRangeException(nameof(degrees));
    }
}
