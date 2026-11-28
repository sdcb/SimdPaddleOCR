using System.Diagnostics;
using System.Text.Json;
using Sdcb.SimdCodecs;
using Sdcb.SimdCodecs.Abstractions;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;

namespace Sdcb.SimdPaddleOCR.Tests;

/// <summary>
/// --doc-ori &lt;upright page image&gt; [--engine sharp|metal|vulkan|auto] [--out &lt;json&gt;]: the real document
/// orientation model on the page turned 0 / 90 / 180 / 270 degrees clockwise. Exit 1 when any turn is
/// classified wrong (correctness only; the timings are informational).
/// </summary>
internal static class DocOrientationSmoke
{
    public static int Run(string[] args)
    {
        string? image = null, outPath = null;
        string engine = "sharp";
        for (int i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"missing value for {args[i]}");
            switch (args[i])
            {
                case "--engine": engine = Next().ToLowerInvariant(); break;
                case "--out": outPath = Next(); break;
                default:
                    if (image is null && !args[i].StartsWith("--")) image = args[i];
                    else throw new ArgumentException($"unknown argument: {args[i]}");
                    break;
            }
        }
        if (image is null) throw new ArgumentException("usage: --doc-ori <image> [--engine sharp|metal|vulkan|auto] [--out <json>]");
        OcrBackend backend = engine switch
        {
            "sharp" or "cpu" => OcrBackend.Cpu,
            "metal" => OcrBackend.Metal,
            "vulkan" => OcrBackend.Vulkan,
            "auto" => OcrBackend.Auto,
            _ => throw new ArgumentException($"unknown engine: {engine}"),
        };

        using IImageRowReader reader = ImageCodecs.OpenRowReader(File.OpenRead(image), PixelFormat.Bgr24);
        int width = reader.Width, height = reader.Height, stride = checked(width * 3);
        byte[] pixels = new byte[checked(stride * height)];
        for (int y = 0; y < height;) y += reader.ReadRows(pixels.AsSpan(y * stride), height - y, stride);

        long t0 = Stopwatch.GetTimestamp();
        using var classifier = new PaddleOcrDocumentOrientationClassifier(DocumentOrientationModel.OpenRead(),
            new PaddleOcrDocumentOrientationOptions { Backend = backend });
        double loadMs = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;

        var rows = new List<object>();
        int wrong = 0;
        foreach (int degrees in (int[])[0, 90, 180, 270])
        {
            // a page turned `degrees` clockwise is the upright page rotated counterclockwise by 360 - degrees
            int ccw = (360 - degrees) % 360;
            (int w, int h) = PaddleOcrDocumentRotation.RotatedSize(width, height, ccw);
            byte[] turned = new byte[checked(w * h * 3)];
            PaddleOcrDocumentRotation.Rotate(pixels, width, height, stride, ImagePixelFormat.Bgr24, ccw, turned);
            long t1 = Stopwatch.GetTimestamp();
            PaddleOcrDocumentOrientationResult r = classifier.Classify(turned, w, h);
            double ms = Stopwatch.GetElapsedTime(t1).TotalMilliseconds;
            bool ok = r.OrientationDegrees == degrees;
            if (!ok) wrong++;
            Console.WriteLine($"[doc-ori {engine}] turned {degrees,3} -> predicted {r.OrientationDegrees,3} score {r.Score:F4} {ms,7:F1} ms {(ok ? "OK" : "WRONG")}");
            rows.Add(new { expected = degrees, predicted = r.OrientationDegrees, score = r.Score, ms });
        }
        Console.WriteLine($"[doc-ori {engine}] {Path.GetFileName(image)} {width}x{height}: {4 - wrong}/4 correct (model load {loadMs:F0} ms)");
        if (outPath is not null)
            File.WriteAllText(outPath, JsonSerializer.Serialize(new { image = Path.GetFileName(image), engine, width, height, loadMs, rows },
                new JsonSerializerOptions { WriteIndented = true }));
        return wrong == 0 ? 0 : 1;
    }
}
