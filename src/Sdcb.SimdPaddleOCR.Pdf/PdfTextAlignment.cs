namespace Sdcb.SimdPaddleOCR.Pdf;

/// <summary>Excludes recognizer emissions originating entirely in right padding.</summary>
public static class PdfTextAlignment
{
    /// <summary>
    /// Uses CTC midpoints for interior boundaries and the detection quad for the
    /// two outer boundaries. First/last tokens otherwise cover only their firing
    /// frames, often leaving half of a visible title character outside selection.
    /// Call after RemovePaddingEmissions when working with a full OCR result.
    /// </summary>
    public static PaddleOcrCharacterBox[] EstimateCharacterBoxes(PaddleOcrLine line)
    {
        ArgumentNullException.ThrowIfNull(line);
        _ = line.EstimateCharacterBoxes(); // Validate alignment without changing caller data.
        PaddleOcrCtcSpan[] spans = (PaddleOcrCtcSpan[])line.CtcSpans!.Clone();
        if (spans.Length == 0) return [];
        int contentEnd = (int)Math.Min(line.RecognitionTimeSteps,
            ((long)line.RecognitionContentWidth * line.RecognitionTimeSteps +
             line.RecognitionTensorWidth - 1) / line.RecognitionTensorWidth);
        spans[0] = spans[0] with { StartColumn = 0 };
        spans[^1] = spans[^1] with { EndColumn = contentEnd };
        return PaddleOcrCharacterBoxes.Estimate(line.Box, spans, line.RecognitionTimeSteps,
            line.RecognitionContentWidth, line.RecognitionTensorWidth, line.AppliedRotationDegrees);
    }

    /// <summary>
    /// Padding has no corresponding position on the source image. Do not attach
    /// its hallucinated tokens to the last real character or invent a selection box.
    /// Returns the original result when no tokens need removal.
    /// </summary>
    public static PaddleOcrResult RemovePaddingEmissions(PaddleOcrResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        List<PaddleOcrLine> lines = [];
        bool changed = false;
        foreach (PaddleOcrLine line in result.Lines)
        {
            if (line.Text.Length == 0) { lines.Add(line); continue; }
            if (line.CtcSpans is null)
                throw new InvalidOperationException("CTC alignment was not captured. Pass returnCtcAlignment: true to Run.");
            if (line.RecognitionTimeSteps <= 0 || line.RecognitionContentWidth <= 0 ||
                line.RecognitionTensorWidth < line.RecognitionContentWidth)
                throw new InvalidDataException("Invalid recognition alignment dimensions.");
            if (string.Concat(line.CtcSpans.Select(s => s.Text)) != line.Text)
                throw new InvalidDataException("CTC tokens must exactly reproduce the recognized line text.");
            // Validate original span ranges before filtering (do not conceal malformed input).
            _ = line.EstimateCharacterBoxes();
            PaddleOcrCtcSpan[] spans = line.CtcSpans.Where(s =>
                (long)s.StartColumn * line.RecognitionTensorWidth <
                (long)line.RecognitionTimeSteps * line.RecognitionContentWidth).ToArray();
            if (spans.Length == line.CtcSpans.Length) { lines.Add(line); continue; }
            changed = true;
            if (spans.Length == 0) continue;
            lines.Add(new PaddleOcrLine
            {
                Box = line.Box, Text = string.Concat(spans.Select(s => s.Text)),
                RecognitionScore = line.RecognitionScore, ClassificationScore = line.ClassificationScore,
                ClassificationLabel = line.ClassificationLabel, AppliedRotationDegrees = line.AppliedRotationDegrees,
                EmittedCount = (uint)spans.Length, CtcSpans = spans,
                RecognitionContentWidth = line.RecognitionContentWidth,
                RecognitionTensorWidth = line.RecognitionTensorWidth,
                RecognitionTimeSteps = line.RecognitionTimeSteps
            });
        }
        return changed ? new PaddleOcrResult
        {
            Lines = lines.ToArray(), DocumentOrientation = result.DocumentOrientation,
            AppliedDocumentRotationDegrees = result.AppliedDocumentRotationDegrees,
            DetectedCount = result.DetectedCount, DetectorResizedWidth = result.DetectorResizedWidth,
            DetectorResizedHeight = result.DetectorResizedHeight
        } : result;
    }
}
