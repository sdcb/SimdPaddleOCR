using System.Collections.Concurrent;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using SkiaSharp;

string outDir = ".";
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--out" && i + 1 < args.Length)
        outDir = args[++i];
    else if (args[i] is "-h" or "--help")
    {
        Console.WriteLine("usage: --out <dir>");
        return 0;
    }
}

outDir = Path.GetFullPath(outDir);
Directory.CreateDirectory(outDir);

const int Seed = 20260830;

string[] fontCandidates =
[
    @"C:\Windows\Fonts\arial.ttf",
    @"C:\Windows\Fonts\calibri.ttf",
    @"C:\Windows\Fonts\times.ttf",
    @"C:\Windows\Fonts\segoeui.ttf",
    @"C:\Windows\Fonts\consola.ttf",
    @"C:\Windows\Fonts\simhei.ttf",
    @"C:\Windows\Fonts\msyh.ttc",
    @"C:\Windows\Fonts\simsun.ttc",
];
string[] cjkCandidates =
[
    @"C:\Windows\Fonts\msyh.ttc",
    @"C:\Windows\Fonts\simhei.ttf",
    @"C:\Windows\Fonts\simsun.ttc",
];

string[] extraFonts = DiscoverExtraFonts();
string[] fonts = fontCandidates.Where(File.Exists).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
string[] cjkFonts = cjkCandidates.Where(File.Exists).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
if (cjkFonts.Length == 0)
{
    cjkFonts = extraFonts;
    if (cjkFonts.Length == 0)
        throw new InvalidOperationException(
            "No CJK font found. Install Windows CJK fonts, or set PPOCR_FONTS_DIR to a noto-cjk checkout.");
    Console.WriteLine("no Windows CJK fonts; using extra fonts");
}
if (fonts.Length == 0)
    fonts = cjkFonts;

Console.WriteLine($"seed={Seed} out={outDir} workers={Environment.ProcessorCount}");
Console.WriteLine($"fonts={fonts.Length} cjkFonts={cjkFonts.Length}");
foreach (string font in fonts)
    Console.WriteLine($"  font {font}");
foreach (string font in cjkFonts)
    Console.WriteLine($"  CJK {font}");
foreach (string missing in fontCandidates.Concat(cjkCandidates).Distinct(StringComparer.OrdinalIgnoreCase).Where(p => !File.Exists(p)))
    Console.WriteLine($"  missing {missing}");
Console.Out.Flush();

string[] texts =
[
    "PPOCRSharp benchmark", "OpenVINO dynamic width", "纯托管 C# 推理",
    "性能测试 2026", "文本识别结果", "Dynamic shape session",
    "The quick brown fox", "边界框与旋转", "天气晴朗 温度 28C",
    "Batch size eight", "small model validation", "图像处理流水线",
    "No native dependency", "读取发票金额 128.50", "方向分类 180 度",
    "Mixed English 中文", "可重复性测试", "PaddleOCR reference",
    "宽度排序与填充", "accuracy metadata",
    "PPOCRSharp dynamic recognition benchmark with variable width input",
    "OpenVINO and pure managed CSharp performance comparison 2026",
    "这是一段明显较长的中文文本用于测试动态识别宽度和压缩行为",
    "检测后处理与文字方向分类应该保持和 PaddleOCR 官方结果一致",
    "The quick brown fox jumps over the lazy dog while OCR reads every word",
    "纯托管推理引擎不依赖 native library 并且支持多个动态输入尺寸",
    "Invoice number 20260830 amount 128.50 date 2026-08-30 validation text",
    "宽文本行必须触发超过 320 像素的 REC 自然宽度并记录到 metadata",
    "Alternating backend benchmark measures cold shape creation and steady state",
    "这是用于性能基准的更长文本行包含中文 English 以及数字 1234567890",
];

(byte r, byte g, byte b)[] paletteBgs =
[
    (245,245,240), (232,242,250), (250,237,220), (236,248,232),
    (245,232,246), (235,235,235),
    (18, 18, 22), (12, 28, 48), (28, 12, 24), (10, 38, 34),
];
(byte r, byte g, byte b)[] paletteInks =
[
    (20,30,40), (10,55,100), (80,35,15), (25,75,35),
    (75,25,80), (30,30,30),
    (245, 245, 245), (245, 215, 72), (255, 150, 205), (95, 235, 205),
];

ConcurrentDictionary<string, SKTypeface> typefaces = new();

SKTypeface TypefaceFor(string path) =>
    typefaces.GetOrAdd(path, p => SKTypeface.FromFile(p) ?? throw new InvalidOperationException($"failed to load font: {p}"));

string FontPathForText(string text, int i)
{
    if (text.Any(ch => ch > 0x3000))
        return cjkFonts[i % cjkFonts.Length];
    return fonts[i % fonts.Length];
}

SKFont MakeFont(string path, float size) =>
    new(TypefaceFor(path), size) { Edging = SKFontEdging.Antialias, Subpixel = true };

foreach (string old in Directory.EnumerateFiles(outDir, "img-*.jpg"))
    File.Delete(old);
string oldMeta = Path.Combine(outDir, "metadata.json");
if (File.Exists(oldMeta))
    File.Delete(oldMeta);

JsonObject?[] imageEntries = new JsonObject?[100];
SKSamplingOptions cubic = new(SKCubicResampler.Mitchell);
Stopwatch sw = Stopwatch.StartNew();
int done = 0;
Console.WriteLine("generating 100 images...");
Console.Out.Flush();

Parallel.For(0, 100, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, i =>
{
    Random rng = new(Seed + i);
    int RandInt(int minInclusive, int maxInclusive) => rng.Next(minInclusive, maxInclusive + 1);
    double Uniform(double a, double b) => a + rng.NextDouble() * (b - a);

    int width = 640 + 16 * rng.Next(0, (1800 - 640) / 16 + 1);
    int height = 480 + 16 * rng.Next(0, (1400 - 480) / 16 + 1);

    int paletteIdx = rng.Next(paletteBgs.Length);
    var bg = paletteBgs[paletteIdx];
    var ink = paletteInks[paletteIdx];

    using SKBitmap bmp = new(new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Opaque));
    using SKCanvas canvas = new(bmp);

    int tw = Math.Max(8, width / 32), th = Math.Max(8, height / 32);
    using (SKBitmap noise = new(tw, th, SKColorType.Rgb888x, SKAlphaType.Opaque))
    {
        for (int yy = 0; yy < th; yy++)
        {
            int shade = (int)(8 * Math.Sin(yy / 3.0 + i));
            for (int xx = 0; xx < tw; xx++)
            {
                int n = rng.Next(-3, 4);
                noise.SetPixel(xx, yy, new SKColor(
                    Clamp8(bg.r + shade + n), Clamp8(bg.g + shade + n), Clamp8(bg.b + shade + n)));
            }
        }
        using SKImage noiseImg = SKImage.FromBitmap(noise);
        canvas.DrawImage(noiseImg, new SKRect(0, 0, tw, th), new SKRect(0, 0, width, height), cubic);
    }

    const int pad = 2;
    List<(int x0, int y0, int x1, int y1)> placedRects = new();
    bool RectFree(int x0, int y0, int rw, int rh)
    {
        int rx0 = x0 - pad, ry0 = y0 - pad, rx1 = x0 + rw + pad, ry1 = y0 + rh + pad;
        if (rx0 < 0) rx0 = 0;
        if (ry0 < 0) ry0 = 0;
        if (rx1 > width) rx1 = width;
        if (ry1 > height) ry1 = height;
        if (rx1 <= rx0 || ry1 <= ry0) return false;
        foreach (var (ax0, ay0, ax1, ay1) in placedRects)
        {
            if (rx0 < ax1 && ax0 < rx1 && ry0 < ay1 && ay0 < ry1) return false;
        }
        return true;
    }
    void MarkRect(int x0, int y0, int rw, int rh)
    {
        int rx0 = Math.Max(0, x0 - pad), ry0 = Math.Max(0, y0 - pad);
        int rx1 = Math.Min(width, x0 + rw + pad), ry1 = Math.Min(height, y0 + rh + pad);
        placedRects.Add((rx0, ry0, rx1, ry1));
    }

    int lineCount = RandInt(6, 16);
    int marginX = RandInt(24, 80);
    int y = RandInt(24, 70);
    JsonArray lines = new();
    int rightX = width - RandInt(24, 70);
    int vY = RandInt(24, 70);

    for (int lineNo = 0; lineNo < lineCount; lineNo++)
    {
        int minSide = Math.Min(width, height);
        bool vertical = rng.NextDouble() < 0.20;
        string text = rng.NextDouble() < 0.55 ? texts[20 + rng.Next(texts.Length - 20)] : texts[rng.Next(texts.Length)];

        string fontPath;
        SKRect bounds;
        int size;
        bool shortFallback = false;
        while (true)
        {
            size = RandInt(Math.Max(18, minSide / 40), Math.Max(26, minSide / 16));
            fontPath = FontPathForText(text, i + lineNo);
            while (true)
            {
                using var f = MakeFont(fontPath, size);
                f.MeasureText(text, out bounds);
                if (bounds.Width <= (vertical ? height : width) - 2 * marginX) break;
                if (size <= 14)
                {
                    if (shortFallback) break;
                    text = texts[rng.Next(20)];
                    fontPath = FontPathForText(text, i + lineNo);
                    shortFallback = true;
                    continue;
                }
                size -= 2;
            }
            if (bounds.Width <= (vertical ? height : width) - 2 * marginX || shortFallback) break;
        }
        int textW = Math.Max(1, CeilToInt(bounds.Width));
        int textH = Math.Max(1, CeilToInt(bounds.Height));

        double angle = vertical ? Uniform(-8.0, 8.0) : Uniform(-20.0, 20.0);
        double lineRotation = vertical
            ? (rng.Next(2) == 0 ? 90.0 : -90.0)
            : (rng.NextDouble() < 0.25 ? 180.0 : 0.0);
        SKColor color = new(
            Clamp8(ink.r + RandInt(-12, 12)),
            Clamp8(ink.g + RandInt(-12, 12)),
            Clamp8(ink.b + RandInt(-12, 12)));

        double layerW = textW + 20, layerH = textH + 20;
        double totalDeg = angle + lineRotation;
        double rad = totalDeg * Math.PI / 180.0;
        double cos = Math.Abs(Math.Cos(rad)), sin = Math.Abs(Math.Sin(rad));
        int rw = CeilToInt(layerW * cos + layerH * sin);
        int rh = CeilToInt(layerW * sin + layerH * cos);

        while (!vertical && rh > height / 7 && Math.Abs(angle) > 1.0)
        {
            angle *= 0.7;
            totalDeg = angle + lineRotation;
            rad = totalDeg * Math.PI / 180.0;
            cos = Math.Abs(Math.Cos(rad)); sin = Math.Abs(Math.Sin(rad));
            rw = CeilToInt(layerW * cos + layerH * sin);
            rh = CeilToInt(layerW * sin + layerH * cos);
        }

        int x = 0, drawY = 0;
        bool placed = false;
        if (vertical)
        {
            for (int col = 0; col < 8 && !placed && rightX - rw >= 20; col++)
            {
                foreach (int startY in new[] { vY, y })
                {
                    if (startY + rh >= height - 20) continue;
                    for (int attempt = 0; attempt < 10 && !placed; attempt++)
                    {
                        int cand = RandInt(Math.Max(20, rightX - rw - 24), rightX - rw);
                        if (RectFree(cand, startY, rw, rh))
                        {
                            x = cand; drawY = startY; placed = true;
                        }
                    }
                    if (placed) break;
                }
                if (!placed)
                {
                    rightX -= rw + RandInt(10, 30);
                    vY = RandInt(24, 70);
                }
            }
            if (!placed) break;
            vY = drawY + rh + RandInt(12, 36);
        }
        else
        {
            for (int rowTry = 0; rowTry < 4 && !placed; rowTry++)
            {
                if (y + rh >= height - 20) break;
                int xLo = Math.Min(marginX, Math.Max(marginX, width - marginX - rw));
                int xHi = Math.Max(marginX, width - marginX - rw);
                for (int attempt = 0; attempt < 12 && !placed; attempt++)
                {
                    int cand = RandInt(xLo, xHi);
                    if (RectFree(cand, y, rw, rh))
                    {
                        x = cand; drawY = y; placed = true;
                    }
                }
                if (!placed) y += Math.Max(10, textH / 2);
            }
            if (!placed) break;
        }

        using (SKPaint paint = new()
        { Color = color, IsAntialias = true })
        using (var f = MakeFont(fontPath, size))
        {
            canvas.Save();
            canvas.Translate(x + rw / 2f, drawY + rh / 2f);
            canvas.RotateDegrees((float)totalDeg);
            canvas.DrawText(text, (float)(-layerW / 2 + 10 - bounds.Left),
                (float)(-layerH / 2 + 10 - bounds.Top), SKTextAlign.Left, f, paint);
            canvas.Restore();
        }
        MarkRect(x, drawY, rw, rh);

        int naturalWidth48 = CeilToInt(textW * 48.0 / Math.Max(1, textH));
        lines.Add(LineJson(text, x, drawY, rw, rh, naturalWidth48, angle, lineRotation, fontPath, size, color));
        if (!vertical) y = drawY + rh + RandInt(6, 20);
    }

    int failedFills = 0;
    while (lines.Count < lineCount && failedFills < 60)
    {
        int minSide = Math.Min(width, height);
        bool vertical = rng.NextDouble() < 0.20;
        string text = texts[rng.Next(texts.Length)];
        string fontPath = FontPathForText(text, 999 + lines.Count);
        int size = RandInt(Math.Max(18, minSide / 40), Math.Max(26, minSide / 16));
        SKRect bounds;
        while (true)
        {
            using var f = MakeFont(fontPath, size);
            f.MeasureText(text, out bounds);
            if (bounds.Width <= (vertical ? height : width) - 60 || size <= 14) break;
            size -= 2;
        }
        int textW = Math.Max(1, CeilToInt(bounds.Width));
        int textH = Math.Max(1, CeilToInt(bounds.Height));
        double angle = vertical ? Uniform(-8.0, 8.0) : Uniform(-20.0, 20.0);
        double lineRotation = vertical
            ? (rng.Next(2) == 0 ? 90.0 : -90.0)
            : (rng.NextDouble() < 0.25 ? 180.0 : 0.0);
        SKColor color = new(
            Clamp8(ink.r + RandInt(-12, 12)),
            Clamp8(ink.g + RandInt(-12, 12)),
            Clamp8(ink.b + RandInt(-12, 12)));
        double layerW = textW + 20, layerH = textH + 20;
        double totalDeg = angle + lineRotation;
        double rad = totalDeg * Math.PI / 180.0;
        double cos = Math.Abs(Math.Cos(rad)), sin = Math.Abs(Math.Sin(rad));
        int rw = CeilToInt(layerW * cos + layerH * sin);
        int rh = CeilToInt(layerW * sin + layerH * cos);
        while (!vertical && rh > height / 7 && Math.Abs(angle) > 1.0)
        {
            angle *= 0.7;
            totalDeg = angle + lineRotation;
            rad = totalDeg * Math.PI / 180.0;
            cos = Math.Abs(Math.Cos(rad)); sin = Math.Abs(Math.Sin(rad));
            rw = CeilToInt(layerW * cos + layerH * sin);
            rh = CeilToInt(layerW * sin + layerH * cos);
        }

        bool placed = false;
        int x = 0, drawY = 0;
        for (int t = 0; t < 40 && !placed; t++)
        {
            int candX = RandInt(20, Math.Max(20, width - 20 - rw));
            int candY = RandInt(20, Math.Max(20, height - 20 - rh));
            if (RectFree(candX, candY, rw, rh))
            {
                x = candX; drawY = candY; placed = true;
            }
        }
        if (!placed) { failedFills++; continue; }
        failedFills = 0;

        using (SKPaint paint = new()
        { Color = color, IsAntialias = true })
        using (var f = MakeFont(fontPath, size))
        {
            canvas.Save();
            canvas.Translate(x + rw / 2f, drawY + rh / 2f);
            canvas.RotateDegrees((float)totalDeg);
            canvas.DrawText(text, (float)(-layerW / 2 + 10 - bounds.Left),
                (float)(-layerH / 2 + 10 - bounds.Top), SKTextAlign.Left, f, paint);
            canvas.Restore();
        }
        MarkRect(x, drawY, rw, rh);
        int naturalWidth48 = CeilToInt(textW * 48.0 / Math.Max(1, textH));
        lines.Add(LineJson(text, x, drawY, rw, rh, naturalWidth48, angle, lineRotation, fontPath, size, color));
    }

    while (lines.Count < 5)
    {
        string text = texts[rng.Next(20)];
        string fontPath = FontPathForText(text, 555 + lines.Count);
        const int size = 16;
        using var f = MakeFont(fontPath, size);
        f.MeasureText(text, out var bounds);
        int textW = Math.Max(1, CeilToInt(bounds.Width));
        int textH = Math.Max(1, CeilToInt(bounds.Height));
        double layerW = textW + 20, layerH = textH + 20;
        int rw = CeilToInt(layerW), rh = CeilToInt(layerH);

        bool placed = false;
        int x = 0, drawY = 0;
        for (int t = 0; t < 200 && !placed; t++)
        {
            int candX = RandInt(10, Math.Max(10, width - 10 - rw));
            int candY = RandInt(10, Math.Max(10, height - 10 - rh));
            if (RectFree(candX, candY, rw, rh))
            {
                x = candX; drawY = candY; placed = true;
            }
        }
        if (!placed) break;

        SKColor color = new(ink.r, ink.g, ink.b);
        using (SKPaint paint = new()
        { Color = color, IsAntialias = true })
        {
            canvas.DrawText(text, x + 10 - bounds.Left, drawY + 10 - bounds.Top,
                SKTextAlign.Left, f, paint);
        }
        MarkRect(x, drawY, rw, rh);
        int naturalWidth48 = CeilToInt(textW * 48.0 / Math.Max(1, textH));
        lines.Add(LineJson(text, x, drawY, rw, rh, naturalWidth48, 0, 0, fontPath, size, color));
    }

    if (lines.Count == 0)
    {
        string text = texts[i % texts.Length];
        string fontPath = FontPathForText(text, i);
        using var f = MakeFont(fontPath, 28);
        f.MeasureText(text, out var b);
        using SKPaint paint = new() { Color = new SKColor(ink.r, ink.g, ink.b), IsAntialias = true };
        canvas.DrawText(text, 24 - b.Left, 24 - b.Top, SKTextAlign.Left, f, paint);
        int bw = Math.Max(1, CeilToInt(b.Width)), bh = Math.Max(1, CeilToInt(b.Height));
        lines.Add(new JsonObject
        {
            ["text"] = text,
            ["bbox"] = new JsonArray(24 + (int)b.Left, 24 + (int)b.Top, 24 + (int)b.Right, 24 + (int)b.Bottom),
            ["natural_width_at_height_48"] = CeilToInt(bw * 48.0 / bh),
            ["angle_degrees"] = 0.0,
            ["orientation_degrees"] = 0,
            ["font"] = fontPath,
            ["font_size"] = 28,
            ["color_rgb"] = new JsonArray((int)ink.r, (int)ink.g, (int)ink.b),
        });
    }

    string name = $"img-{i + 1:000}.jpg";
    string path = Path.Combine(outDir, name);
    byte[] bytes;
    using (SKImage image = SKImage.FromBitmap(bmp))
    using (var data = image.Encode(SKEncodedImageFormat.Jpeg, 95))
        bytes = data.ToArray();
    File.WriteAllBytes(path, bytes);
    string sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    imageEntries[i] = new JsonObject
    {
        ["file"] = name,
        ["width"] = width,
        ["height"] = height,
        ["background_rgb"] = new JsonArray((int)bg.r, (int)bg.g, (int)bg.b),
        ["sha256"] = sha256,
        ["lines"] = lines,
    };

    int completed = Interlocked.Increment(ref done);
    Console.WriteLine($"  [{completed,3}/100] {name} {width}x{height} lines={lines.Count} {sw.Elapsed.TotalSeconds:0.0}s");
    Console.Out.Flush();
});

JsonObject metadata = new()
{
    ["version"] = 1,
    ["seed"] = Seed,
    ["images"] = new JsonArray(),
};
JsonArray images = (JsonArray)metadata["images"]!;
foreach (var entry in imageEntries) images.Add(entry);

File.WriteAllText(Path.Combine(outDir, "metadata.json"),
    metadata.ToJsonString(new JsonSerializerOptions
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    }), new UTF8Encoding(false));
Console.WriteLine($"generated {images.Count} images in {outDir} ({sw.Elapsed.TotalSeconds:0.0}s)");
return 0;

static string[] DiscoverExtraFonts()
{
    List<string> paths = new();
    string? single = Environment.GetEnvironmentVariable("PPOCR_CJK_FONT");
    if (!string.IsNullOrWhiteSpace(single) && File.Exists(single))
        paths.Add(single);

    List<string> dirs = new();
    string? dirEnv = Environment.GetEnvironmentVariable("PPOCR_FONTS_DIR");
    if (!string.IsNullOrWhiteSpace(dirEnv))
    {
        dirs.AddRange(dirEnv.Split(Path.PathSeparator,
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
    }
    dirs.Add(Path.Combine(Environment.CurrentDirectory, "fonts"));

    foreach (string dir in dirs)
    {
        if (!Directory.Exists(dir))
            continue;
        foreach (string file in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories))
        {
            string ext = Path.GetExtension(file);
            if (ext.Equals(".otf", StringComparison.OrdinalIgnoreCase) ||
                ext.Equals(".ttf", StringComparison.OrdinalIgnoreCase) ||
                ext.Equals(".ttc", StringComparison.OrdinalIgnoreCase))
                paths.Add(file);
        }
    }

    return paths.Distinct(StringComparer.OrdinalIgnoreCase)
        .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
        .ToArray();
}

static byte Clamp8(int v) => (byte)Math.Clamp(v, 0, 255);
static int CeilToInt(double v) => (int)Math.Ceiling(v);

static JsonObject LineJson(string text, int x, int y, int rw, int rh, int naturalWidth48,
    double angle, double lineRotation, string fontPath, int size, SKColor color) => new()
{
    ["text"] = text,
    ["bbox"] = new JsonArray(x, y, x + rw, y + rh),
    ["natural_width_at_height_48"] = naturalWidth48,
    ["angle_degrees"] = Math.Round(angle, 3),
    ["orientation_degrees"] = Math.Round(lineRotation, 3),
    ["font"] = fontPath,
    ["font_size"] = size,
    ["color_rgb"] = new JsonArray((int)color.Red, (int)color.Green, (int)color.Blue),
};
