using System.Text;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Pixel word-gap spaces (opt-in, <see cref="PaddleOcrOptions.SpaceGapRatio"/>).
/// The recognizer often drops the space between a CJK run and Latin / digits
/// ("方向分类 180 度" → "方向分类180度"). Between two decoded tokens this looks
/// at the crop columns from the first token's CTC start to the second's: when
/// the longest run of empty columns is at least <c>ratio × crop height</c> a
/// space is inserted. CTC blank frames alone cannot separate a word gap from
/// the normal pitch between CJK glyphs (one time step is ~1/6 of the line
/// height), so the decision is made on the pixels.
/// </summary>
internal static class PPOCRSpaceGap
{
    // A column is empty when no pixel in it is farther than this from the
    // background gray. Anti-aliasing leaves a little ink next to glyphs, so it
    // cannot be zero; 30/255 sits below the faintest real stroke.
    private const int EmptyInk = 30;

    public static string Apply(string text, PaddleOcrCtcSpan[] spans, int timeSteps, int contentWidth,
        int tensorWidth, ReadOnlySpan<byte> bgrCrop, int cropWidth, int cropHeight, float ratio)
    {
        if (spans.Length < 2 || timeSteps <= 0 || contentWidth <= 0 || tensorWidth <= 0 ||
            cropWidth <= 0 || cropHeight <= 0 || bgrCrop.Length < cropWidth * cropHeight * 3)
            return text;

        bool[] empty = PooledArrays.Rent<bool>(cropWidth);
        try
        {
            EmptyColumns(bgrCrop, cropWidth, cropHeight, empty);
            // Token k starts at the crop column of its first CTC frame.
            double scale = (double)tensorWidth / timeSteps / contentWidth;
            int Col(int k) => (int)(Math.Min(1.0, Math.Max(0.0, spans[k].StartColumn * scale)) * cropWidth);
            float minGap = ratio * cropHeight;
            StringBuilder? sb = null;
            int consumed = 0; // chars of `text` already copied into sb
            int textPos = spans[0].Text.Length;
            for (int k = 1; k < spans.Length; textPos += spans[k].Text.Length, k++)
            {
                string prev = spans[k - 1].Text, cur = spans[k].Text;
                if (IsSpace(prev) || IsSpace(cur) || IsCjkPunct(prev) || IsCjkPunct(cur)) continue;
                int b = Col(k - 1), a = Col(k);
                if (a <= b) continue;
                int run = 0, best = 0;
                for (int x = b; x < a; x++)
                {
                    if (empty[x]) { if (++run > best) best = run; }
                    else run = 0;
                }
                if (best < minGap) continue;
                sb ??= new StringBuilder(text.Length + 8);
                sb.Append(text, consumed, textPos - consumed).Append(' ');
                consumed = textPos;
            }
            // Spans must tile the decoded text; otherwise keep it unchanged.
            if (sb is null || textPos != text.Length) return text;
            return sb.Append(text, consumed, text.Length - consumed).ToString();
        }
        finally
        {
            PooledArrays.Return(empty);
        }
    }

    /// <summary>
    /// Polarity-free: the background is the median gray of the crop (dark text
    /// on light and light text on dark both occur); a column is empty when its
    /// most extreme pixel stays within <see cref="EmptyInk"/> of it. Gray is
    /// kept as r+g+b (0..765) so one pass builds the histogram and the
    /// per-column min / max without a division.
    /// </summary>
    private static void EmptyColumns(ReadOnlySpan<byte> bgr, int width, int height, bool[] empty)
    {
        Span<int> hist = stackalloc int[766];
        hist.Clear();
        int[] lo = PooledArrays.Rent<int>(width), hi = PooledArrays.Rent<int>(width);
        try
        {
            lo.AsSpan(0, width).Fill(int.MaxValue);
            hi.AsSpan(0, width).Clear();
            int stride = width * 3;
            for (int y = 0; y < height; y++)
            {
                ReadOnlySpan<byte> row = bgr.Slice(y * stride, stride);
                for (int x = 0, p = 0; x < width; x++, p += 3)
                {
                    int g = row[p] + row[p + 1] + row[p + 2];
                    hist[g]++;
                    if (g < lo[x]) lo[x] = g;
                    if (g > hi[x]) hi[x] = g;
                }
            }
            long half = ((long)width * height + 1) / 2, acc = 0;
            int bg = 384;
            for (int v = 0; v < hist.Length; v++)
            {
                acc += hist[v];
                if (acc >= half) { bg = v; break; }
            }
            const int limit = EmptyInk * 3;
            for (int x = 0; x < width; x++)
                empty[x] = hi[x] - bg <= limit && bg - lo[x] <= limit;
        }
        finally
        {
            PooledArrays.Return(lo);
            PooledArrays.Return(hi);
        }
    }

    private static bool IsSpace(string s) => s.Length > 0 && s.Trim().Length == 0;

    // Full-width CJK punctuation draws only part of its 1-em box (a comma inks
    // its lower-left corner), so its blank half reads like a word gap. The
    // tokens are known at this point, so they are excluded instead of widening
    // the threshold.
    private static bool IsCjkPunct(string s)
    {
        if (s.Length == 0) return false;
        int cp = s[0];
        return cp is >= 0x3000 and <= 0x303F
            or >= 0xFF01 and <= 0xFF0F
            or >= 0xFF1A and <= 0xFF20
            or >= 0xFF3B and <= 0xFF40
            or >= 0xFF5B and <= 0xFF65
            or 0x2018 or 0x2019 or 0x201C or 0x201D or 0x2014 or 0x2026 or 0x00B7;
    }
}
