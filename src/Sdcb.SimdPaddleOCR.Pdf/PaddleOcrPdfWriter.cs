using System.Globalization;
using System.IO.Compression;
using System.Numerics;
using System.Text;
using SkiaSharp;

namespace Sdcb.SimdPaddleOCR.Pdf;

/// <summary>
/// Streams scanned images and an invisible, Unicode searchable text layer to PDF.
/// Character positions come from CTC alignment, not equal division of a text line.
/// The embedded Type 3 font has an empty glyph and a ToUnicode entry per scalar;
/// it needs no system font and cannot substitute Chinese or supplementary characters.
/// </summary>
public sealed class PaddleOcrPdfWriter : IDisposable
{
    private readonly Stream _output;
    private readonly bool _leaveOpen;
    private readonly float _scale;
    private readonly List<long> _offsets = [0];
    private readonly List<int> _pages = [];
    private readonly List<FontSubset> _fonts = [];
    private readonly Dictionary<Rune, (int Font, int Code)> _characters = [];
    private readonly int _catalog;
    private readonly int _pageTree;
    private readonly int _emptyGlyph;
    private bool _completed;
    private bool _disposed;

    public int PageCount => _pages.Count;

    /// <param name="output">A writable, seekable, empty stream. Caller owns it by default.</param>
    /// <param name="dpi">Image pixels per inch. Pages preserve the source aspect ratio.</param>
    /// <param name="leaveOpen">Whether Dispose leaves output open.</param>
    public PaddleOcrPdfWriter(Stream output, float dpi = 300, bool leaveOpen = true)
    {
        ArgumentNullException.ThrowIfNull(output);
        if (!output.CanWrite || !output.CanSeek || output.Position != 0 || output.Length != 0)
            throw new ArgumentException("Output must be a writable, seekable, empty stream.", nameof(output));
        if (!float.IsFinite(dpi) || dpi <= 0)
            throw new ArgumentOutOfRangeException(nameof(dpi));
        _output = output;
        _leaveOpen = leaveOpen;
        _scale = 72 / dpi;
        _catalog = Reserve();
        _pageTree = Reserve();
        _emptyGlyph = Reserve();
        Write("%PDF-1.7\n%\xE2\xE3\xCF\xD3\n", Encoding.Latin1);
        WriteStream(_emptyGlyph, "", Encoding.ASCII.GetBytes("1000 0 0 0 1000 1000 d1\n"));
    }

    /// <summary>
    /// Adds one page. Run OCR on these exact pixels with returnCtcAlignment: true.
    /// Pass the original encoded image to preserve compatible RGB/grayscale JPEG bytes.
    /// Other images are embedded losslessly after flattening on white in PdfImage.Decode.
    /// The bitmap must remain alive until this method returns.
    /// </summary>
    public void AddPage(SKBitmap image, PaddleOcrResult result, ReadOnlySpan<byte> originalImage = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_completed) throw new InvalidOperationException("PDF has already been completed.");
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(result);
        result = PdfTextAlignment.RemovePaddingEmissions(result);
        if (image.Width <= 0 || image.Height <= 0 || image.ColorType != SKColorType.Bgra8888 ||
            image.AlphaType != SKAlphaType.Opaque)
            throw new ArgumentException("Use PdfImage.Decode to supply an opaque BGRA bitmap.", nameof(image));
        float width = image.Width * _scale, height = image.Height * _scale;
        if (!float.IsFinite(width) || !float.IsFinite(height) || Math.Max(width, height) > 14400)
            throw new ArgumentException("Page exceeds PDF's 14400-point limit. Increase dpi.", nameof(image));

        // Validate alignment before writing anything for this page.
        List<(PaddleOcrLine Line, PaddleOcrCharacterBox[] Boxes)> lines = [];
        foreach (PaddleOcrLine line in result.Lines)
        {
            if (line.Text.Length == 0) continue;
            PaddleOcrCharacterBox[] boxes = PdfTextAlignment.EstimateCharacterBoxes(line);
            if (string.Concat(boxes.Select(x => x.Text)) != line.Text)
                throw new InvalidDataException("CTC tokens must exactly reproduce the recognized line text.");
            foreach (PaddleOcrCharacterBox box in boxes) ValidateBox(box);
            lines.Add((line, boxes));
        }

        int imageId = Reserve(), contentId = Reserve(), pageId = Reserve();
        if (TryGetJpegComponents(originalImage, image.Width, image.Height, out int components))
        {
            WriteStream(imageId, $"/Type /XObject /Subtype /Image /Width {image.Width} /Height {image.Height} " +
                $"/ColorSpace /Device{(components == 1 ? "Gray" : "RGB")} /BitsPerComponent 8 /Filter /DCTDecode",
                originalImage);
        }
        else
        {
            byte[] rgb = CompressRgb(image);
            WriteStream(imageId, $"/Type /XObject /Subtype /Image /Width {image.Width} /Height {image.Height} " +
                "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode", rgb);
        }

        StringBuilder content = new($"q\n{N(width)} 0 0 {N(height)} 0 0 cm\n/Im0 Do\nQ\n");
        HashSet<int> usedFonts = [];
        foreach ((PaddleOcrLine line, PaddleOcrCharacterBox[] boxes) in lines)
        {
            content.Append("BT\n3 Tr\n"); // PDF rendering mode 3: neither fill nor stroke.
            int currentFont = -1;
            foreach (PaddleOcrCharacterBox box in boxes)
            {
                Rune[] runes = box.Text.EnumerateRunes().ToArray();
                if (runes.Length == 0) continue;
                Vector2 tl = new(box.X1, box.Y1), tr = new(box.X2, box.Y2);
                Vector2 br = new(box.X3, box.Y3), bl = new(box.X4, box.Y4);
                // EstimateCharacterBoxes reverses token positions for CLS=180;
                // reverse glyph axes as well so PDF selection follows the reading direction.
                if (line.AppliedRotationDegrees == 180)
                    (tl, tr, br, bl) = (br, bl, tl, tr);
                for (int i = 0; i < runes.Length; i++)
                {
                    (int font, int code) = GetCharacter(runes[i]);
                    usedFonts.Add(font);
                    if (font != currentFont)
                    {
                        content.Append($"/F{font} 1 Tf\n");
                        currentFont = font;
                    }
                    // Affine approximation of each tiny perspective quad. Unit glyph
                    // advance and height map to its actual width and height in pixels.
                    Vector2 leftTop = Vector2.Lerp(tl, tr, (float)i / runes.Length);
                    Vector2 leftBottom = Vector2.Lerp(bl, br, (float)i / runes.Length);
                    Vector2 across = ((tr - tl) + (br - bl)) / (2 * runes.Length) * _scale;
                    Vector2 up = (leftTop - leftBottom) * _scale;
                    Vector2 origin = leftBottom * _scale;
                    content.Append($"{N(across.X)} {N(-across.Y)} {N(up.X)} {N(-up.Y)} " +
                        $"{N(origin.X)} {N(height - origin.Y)} Tm\n<{code:X2}> Tj\n");
                }
            }
            content.Append("ET\n");
        }
        WriteStream(contentId, "/Filter /FlateDecode", Compress(Encoding.ASCII.GetBytes(content.ToString())));
        string fontResources = string.Join(" ", usedFonts.Order().Select(i => $"/F{i} {_fonts[i].ObjectId} 0 R"));
        WriteObject(pageId, $"<< /Type /Page /Parent {_pageTree} 0 R /MediaBox [0 0 {N(width)} {N(height)}] " +
            $"/Resources << /XObject << /Im0 {imageId} 0 R >> /Font << {fontResources} >> >> " +
            $"/Contents {contentId} 0 R >>");
        _pages.Add(pageId);
    }

    public void Complete()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_completed) return;
        if (_pages.Count == 0) throw new InvalidOperationException("A PDF must have at least one page.");
        for (int i = 0; i < _fonts.Count; i++) WriteFont(i);
        WriteObject(_pageTree, $"<< /Type /Pages /Count {_pages.Count} /Kids [{string.Join(" ", _pages.Select(x => $"{x} 0 R"))}] >>");
        WriteObject(_catalog, $"<< /Type /Catalog /Pages {_pageTree} 0 R >>");
        long xref = _output.Position;
        Write($"xref\n0 {_offsets.Count}\n0000000000 65535 f \n");
        foreach (long offset in _offsets.Skip(1))
        {
            if (offset <= 0 || offset > 9_999_999_999L)
                throw new InvalidOperationException("Invalid PDF object offset.");
            Write($"{offset.ToString("D10", CultureInfo.InvariantCulture)} 00000 n \n");
        }
        Write($"trailer\n<< /Size {_offsets.Count} /Root {_catalog} 0 R >>\nstartxref\n{xref.ToString(CultureInfo.InvariantCulture)}\n%%EOF\n");
        _output.Flush();
        _completed = true;
    }

    public void Dispose()
    {
        if (_disposed) return;
        try { if (!_completed && _pages.Count > 0) Complete(); }
        finally { _disposed = true; if (!_leaveOpen) _output.Dispose(); }
    }

    private (int Font, int Code) GetCharacter(Rune rune)
    {
        if (_characters.TryGetValue(rune, out var value)) return value;
        if (_fonts.Count == 0 || _fonts[^1].Runes.Count == 255)
            _fonts.Add(new FontSubset(Reserve()));
        FontSubset subset = _fonts[^1];
        subset.Runes.Add(rune);
        value = (_fonts.Count - 1, subset.Runes.Count);
        _characters.Add(rune, value);
        return value;
    }

    private void WriteFont(int index)
    {
        FontSubset font = _fonts[index];
        int cmapId = Reserve();
        StringBuilder cmap = new("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n" +
            $"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n/CMapName /OcrUnicode{index} def\n" +
            "/CMapType 2 def\n1 begincodespacerange\n<00> <FF>\nendcodespacerange\n");
        for (int start = 0; start < font.Runes.Count; start += 100)
        {
            int count = Math.Min(100, font.Runes.Count - start);
            cmap.Append($"{count} beginbfchar\n");
            for (int i = start; i < start + count; i++)
                cmap.Append($"<{i + 1:X2}> <{Convert.ToHexString(Encoding.BigEndianUnicode.GetBytes(font.Runes[i].ToString()))}>\n");
            cmap.Append("endbfchar\n");
        }
        cmap.Append("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n");
        WriteStream(cmapId, "/Filter /FlateDecode", Compress(Encoding.ASCII.GetBytes(cmap.ToString())));
        string names = string.Join(" ", Enumerable.Range(1, font.Runes.Count).Select(x => $"/g{x}"));
        string procs = string.Join(" ", Enumerable.Range(1, font.Runes.Count).Select(x => $"/g{x} {_emptyGlyph} 0 R"));
        string widths = string.Join(" ", Enumerable.Repeat("1000", font.Runes.Count));
        WriteObject(font.ObjectId, $"<< /Type /Font /Subtype /Type3 /Name /F{index} /FontBBox [0 0 1000 1000] " +
            "/FontMatrix [0.001 0 0 0.001 0 0] " +
            $"/CharProcs << {procs} >> /Encoding << /Type /Encoding /Differences [1 {names}] >> " +
            $"/FirstChar 1 /LastChar {font.Runes.Count} /Widths [{widths}] /Resources << >> /ToUnicode {cmapId} 0 R >>");
    }

    private int Reserve() { _offsets.Add(0); return _offsets.Count - 1; }
    private void WriteObject(int id, string value)
    {
        _offsets[id] = _output.Position;
        Write($"{id} 0 obj\n{value}\nendobj\n");
    }
    private void WriteStream(int id, string entries, ReadOnlySpan<byte> bytes)
    {
        _offsets[id] = _output.Position;
        Write($"{id} 0 obj\n<< {entries} /Length {bytes.Length} >>\nstream\n");
        _output.Write(bytes);
        Write("\nendstream\nendobj\n");
    }
    private void Write(string value, Encoding? encoding = null) => _output.Write((encoding ?? Encoding.ASCII).GetBytes(value));
    private static string N(float value) => value.ToString("0.######", CultureInfo.InvariantCulture);
    private static byte[] Compress(byte[] bytes)
    {
        using MemoryStream buffer = new();
        using (ZLibStream zlib = new(buffer, CompressionLevel.Optimal, leaveOpen: true)) zlib.Write(bytes);
        return buffer.ToArray();
    }
    private static byte[] CompressRgb(SKBitmap image)
    {
        using MemoryStream buffer = new();
        using (ZLibStream zlib = new(buffer, CompressionLevel.Optimal, leaveOpen: true))
        {
            ReadOnlySpan<byte> pixels = image.GetPixelSpan();
            byte[] row = new byte[checked(image.Width * 3)];
            for (int y = 0; y < image.Height; y++)
            {
                ReadOnlySpan<byte> bgra = pixels.Slice(y * image.RowBytes, image.Width * 4);
                for (int x = 0; x < image.Width; x++)
                {
                    row[x * 3] = bgra[x * 4 + 2];
                    row[x * 3 + 1] = bgra[x * 4 + 1];
                    row[x * 3 + 2] = bgra[x * 4];
                }
                zlib.Write(row);
            }
        }
        return buffer.ToArray();
    }
    private static void ValidateBox(PaddleOcrCharacterBox box)
    {
        float[] values = [box.X1, box.Y1, box.X2, box.Y2, box.X3, box.Y3, box.X4, box.Y4];
        if (values.Any(x => !float.IsFinite(x)) ||
            Vector2.Distance(new(box.X1, box.Y1), new(box.X2, box.Y2)) <= 0 ||
            Vector2.Distance(new(box.X1, box.Y1), new(box.X4, box.Y4)) <= 0)
            throw new InvalidDataException("Character quad is non-finite or degenerate.");
    }
    private static bool TryGetJpegComponents(ReadOnlySpan<byte> data, int width, int height, out int components)
    {
        components = 0;
        if (data.Length < 4 || data[0] != 0xFF || data[1] != 0xD8) return false;
        int pos = 2;
        while (pos + 4 <= data.Length)
        {
            if (data[pos++] != 0xFF) return false;
            while (pos < data.Length && data[pos] == 0xFF) pos++;
            if (pos >= data.Length) return false;
            int marker = data[pos++];
            if (marker is 0xDA or 0xD9) return false;
            if (marker == 0x01 || marker is >= 0xD0 and <= 0xD7) continue;
            if (pos + 2 > data.Length) return false;
            int length = (data[pos] << 8) | data[pos + 1];
            if (length < 2 || pos + length > data.Length) return false;
            if (marker is >= 0xC0 and <= 0xCF && marker is not (0xC4 or 0xC8 or 0xCC))
            {
                if (length < 8 || data[pos + 2] != 8) return false;
                components = data[pos + 7];
                return ((data[pos + 3] << 8) | data[pos + 4]) == height &&
                    ((data[pos + 5] << 8) | data[pos + 6]) == width && components is 1 or 3;
            }
            pos += length;
        }
        return false;
    }
    private sealed class FontSubset(int objectId)
    {
        public int ObjectId { get; } = objectId;
        public List<Rune> Runes { get; } = [];
    }
}
