namespace Sdcb.SimdPaddleOCR;

/// <summary>整页逆时针直角旋转，以及将校正图中的检测框还原到原图的坐标工具。</summary>
internal static class DocumentRotation
{
    /// <summary>计算旋转后的图像尺寸；旋转 90° 或 270° 时交换宽高。</summary>
    /// <param name="width">旋转前的宽度，单位为像素。</param>
    /// <param name="height">旋转前的高度，单位为像素。</param>
    /// <param name="degrees">逆时针旋转角度，调用方应传入 0、90、180 或 270。</param>
    /// <returns>旋转后的宽度和高度。</returns>
    internal static (int Width, int Height) Size(int width, int height, int degrees) =>
        degrees is 90 or 270 ? (height, width) : (width, height);

    /// <summary>按整数像素坐标逆时针旋转整页，保留各像素的原始通道值，不执行插值。</summary>
    /// <param name="source">旋转前的像素缓冲区；不得与目标缓冲区重叠。</param>
    /// <param name="width">原图宽度，单位为像素，必须为正数。</param>
    /// <param name="height">原图高度，单位为像素，必须为正数。</param>
    /// <param name="stride">原图每行的字节跨度，允许行尾填充，至少为一行像素所需字节数。</param>
    /// <param name="format">原图和目标图共同使用的像素格式。</param>
    /// <param name="degrees">逆时针旋转角度，只允许 0、90、180、270。</param>
    /// <param name="destination">目标像素缓冲区，按行紧密排列，容量至少为 width×height×每像素字节数。</param>
    internal static void Rotate(ReadOnlySpan<byte> source, int width, int height, int stride,
        ImagePixelFormat format, int degrees, Span<byte> destination)
    {
        int bpp = ImagePixels.BytesPerPixel(format);
        if (width < 1 || height < 1 || stride < checked(width * bpp) ||
            (long)(height - 1) * stride + (long)width * bpp > source.Length)
            throw new ArgumentException("Invalid source image.");
        if (degrees is not (0 or 90 or 180 or 270)) throw new ArgumentOutOfRangeException(nameof(degrees));
        if (destination.Length < checked(width * height * bpp)) throw new ArgumentException("Destination is too small.");
        (int rotatedWidth, _) = Size(width, height, degrees);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                // 将原图像素映射到目标位置；使用 width/height - 1 对齐像素中心的零起始坐标。
                (int dx, int dy) = degrees switch
                {
                    90 => (y, width - 1 - x),
                    180 => (width - 1 - x, height - 1 - y),
                    270 => (height - 1 - y, x),
                    _ => (x, y)
                };
                source.Slice(y * stride + x * bpp, bpp).CopyTo(destination.Slice((dy * rotatedWidth + dx) * bpp, bpp));
            }
    }

    /// <summary>把旋转校正图中的四边形逐点逆变换到原图，保留顶点顺序和检测分数。</summary>
    /// <param name="box">在校正图坐标系中的检测框。</param>
    /// <param name="width">旋转前的原图宽度，单位为像素。</param>
    /// <param name="height">旋转前的原图高度，单位为像素。</param>
    /// <param name="degrees">生成校正图时执行的逆时针旋转角度。</param>
    /// <returns>原图坐标系中的检测框；保留校正图中的顶点顺序，不重新按原图左上角排序。</returns>
    internal static PaddleOcrDetectionBox ToSource(PaddleOcrDetectionBox box, int width, int height, int degrees)
    {
        // Point 的 x/y 来自校正图；这里采用 Rotate 中映射公式的逆变换。
        (float X, float Y) Point(float x, float y) => degrees switch
        {
            90 => (width - 1 - y, x),
            180 => (width - 1 - x, height - 1 - y),
            270 => (y, height - 1 - x),
            _ => (x, y)
        };
        var p1 = Point(box.X1, box.Y1); var p2 = Point(box.X2, box.Y2);
        var p3 = Point(box.X3, box.Y3); var p4 = Point(box.X4, box.Y4);
        return new(p1.X, p1.Y, p2.X, p2.Y, p3.X, p3.Y, p4.X, p4.Y, box.Score);
    }

    /// <summary>复制识别行并还原检测框坐标，保留文本、行方向和 CTC 对齐信息。</summary>
    /// <param name="line">在校正图中生成的识别行。</param>
    /// <param name="width">旋转前的原图宽度，单位为像素。</param>
    /// <param name="height">旋转前的原图高度，单位为像素。</param>
    /// <param name="degrees">生成校正图时执行的逆时针旋转角度。</param>
    /// <returns>坐标已还原的识别行；字符框可继续根据保留阅读顺序的顶点和 CTC 信息估算。</returns>
    internal static PaddleOcrLine ToSource(PaddleOcrLine line, int width, int height, int degrees) => new()
    {
        Box = ToSource(line.Box, width, height, degrees), Text = line.Text,
        RecognitionScore = line.RecognitionScore, ClassificationScore = line.ClassificationScore,
        ClassificationLabel = line.ClassificationLabel, AppliedRotationDegrees = line.AppliedRotationDegrees,
        EmittedCount = line.EmittedCount, CtcSpans = line.CtcSpans,
        RecognitionContentWidth = line.RecognitionContentWidth, RecognitionTensorWidth = line.RecognitionTensorWidth,
        RecognitionTimeSteps = line.RecognitionTimeSteps
    };
}
