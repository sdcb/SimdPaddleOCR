using System.Numerics;

using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR;

/// <summary>Options for the DB text detector.</summary>
public sealed class PaddleOcrDetectorOptions
{
    // Ignored: the detector keeps one shape-agnostic CompiledModel and pools
    // dynamically reshaped sessions, so there are no per-shape buckets to cap.
    public int MaxSessionCacheEntries { get; init; } = 10;
    /// <summary>
    /// Upper bound on pooled reusable sessions. Sessions returned while the pool
    /// is full are disposed instead of pooled, capping memory at the cost of
    /// re-creating sessions when concurrency exceeds the cap. The default is
    /// <see cref="Environment.ProcessorCount">, enough for one session per core;
    /// set to <see cref="int.MaxValue"> for an unbounded pool.
    /// </summary>
    public int MaxPooledSessions { get; init; } = Environment.ProcessorCount;
    public int LimitSideLength { get; init; } = 960;
    public int MaxCandidates { get; init; } = 3000;
    public bool UseDilation { get; init; }
    // PP-OCRv6 official DBPostProcess configuration uses bitmap threshold
    // 0.2 and unclip ratio 1.4. Tiny uses box_thresh=0.4 while small/medium
    // use 0.45; Detector applies that model-aware default when this property
    // is not explicitly set.
    public float BitmapThreshold { get; init; } = 0.2f;
    private float _boxThreshold = 0.45f;
    private bool _boxThresholdSet;
    public float BoxThreshold { get => _boxThreshold; init { _boxThreshold = value; _boxThresholdSet = true; } }
    internal bool HasExplicitBoxThreshold => _boxThresholdSet;
    public float UnclipRatio { get; init; } = 1.4f;
    public long MaxImagePixels { get; init; } = 40_000_000;
    /// <summary>Compute backend for the detection graph. Default <see cref="OcrBackend.Auto"/>.</summary>
    public OcrBackend Backend { get; init; } = OcrBackend.Auto;
}

/// <summary>Options for the direction classifier.</summary>
public sealed class PaddleOcrClassifierOptions
{
    public long MaxImagePixels { get; init; } = 40_000_000;
    /// <summary>
    /// Upper bound on pooled reusable sessions. Sessions returned while the pool
    /// is full are disposed instead of pooled. The default is
    /// <see cref="Environment.ProcessorCount">; set to <see cref="int.MaxValue">
    /// for an unbounded pool.
    /// </summary>
    public int MaxPooledSessions { get; init; } = Environment.ProcessorCount;
}

/// <summary>Options for the CTC recognizer.</summary>
public sealed class PaddleOcrRecognizerOptions
{
    /// <summary>
    /// Upper bound on pooled reusable sessions. Sessions returned while the pool
    /// is full are disposed instead of pooled, capping memory at the cost of
    /// re-creating sessions when concurrency exceeds the cap. The default is
    /// <see cref="Environment.ProcessorCount">, enough for one session per core;
    /// set to <see cref="int.MaxValue"> for an unbounded pool.
    /// </summary>
    public int MaxPooledSessions { get; init; } = Environment.ProcessorCount;
    public int TargetWidth { get; init; } = 320;
    /// <summary>Uses OpenVINO-style 32-pixel width buckets without a 320-pixel cap.</summary>
    public bool AdaptiveWidth { get; init; } = true;
    public long MaxImagePixels { get; init; } = 40_000_000;
    /// <summary>Compute backend for the recognition graph. Default <see cref="OcrBackend.Auto"/>.</summary>
    public OcrBackend Backend { get; init; } = OcrBackend.Auto;
}

/// <summary>
/// Options for a complete DET → CLS → REC request.
/// <para>
/// Two independent knobs: <see cref="DetIntraOpThreads"/> is threads inside
/// the single DET session; <see cref="LineWorkerCount"/> is how many CLS/REC
/// sessions run in parallel (the two stages share that count, clamped to
/// ProcessorCount). REC may still shard convolutions inside each session
/// with leftover cores; that budget is not customer-facing.
/// </para>
/// </summary>
public sealed class PaddleOcrOptions
{
    public bool UseDirectionClassification { get; init; } = true;
    /// <summary>
    /// Minimum class-1 probability required for a 180-degree rotation.
    /// The official PaddleX text-line orientation pipeline uses the top-1
    /// class directly, so the default is zero (no additional confidence
    /// gate). Set a positive value to retain the legacy threshold behavior.
    /// </summary>
    public float ClassifierThreshold { get; init; } = 0f;
    /// <summary>
    /// How many crop / CLS / REC line workers run at once. Each worker may
    /// hold its own CLS and REC session, so memory scales with this value.
    /// A positive value is a maximum, clamped to
    /// <see cref="Environment.ProcessorCount"/> (2-core machine requesting 4
    /// → 2). <c>0</c> (default) is <c>min(ProcessorCount, 4)</c>. Does not
    /// change <see cref="DetIntraOpThreads"/>. Leftover cores go to hidden
    /// REC intra-op (cap 4).
    /// </summary>
    public int LineWorkerCount { get; init; }
    /// <summary>
    /// Intra-op threads inside the detector's convolutions. DET runs in an
    /// exclusive window before line workers start, so this does not multiply
    /// with <see cref="LineWorkerCount"/>. <c>0</c> (default) uses up to 8.
    /// CLS and REC share <see cref="LineWorkerCount"/>; they are not given
    /// separate intra-op knobs.
    /// </summary>
    public int DetIntraOpThreads { get; init; }
    /// <summary>
    /// Maximum lines per batched recognizer graph call (same-target-width
    /// lines only, so results stay bit-identical to per-line execution).
    /// The default of 1 keeps per-line execution: on the interpreter each
    /// operator processes the whole batch before the next one runs, which
    /// inflates the activation working set and measured ~20% slower than
    /// per-line REC on an 8-core Zen 3. Raise only after profiling.
    /// When the recognizer runs on a GPU backend and this is left unset, a
    /// batch of 16 is used — line batching is the main GPU win.
    /// </summary>
    public int RecBatchLines { get => _recBatchLines; init { _recBatchLines = value; _recBatchLinesSet = true; } }
    private int _recBatchLines = 1;
    private bool _recBatchLinesSet;
    internal bool HasExplicitRecBatchLines => _recBatchLinesSet;
    public long MaxCropPixels { get; init; } = 16_000_000;
    public PaddleOcrDetectorOptions Detector { get; init; } = new();
    public PaddleOcrClassifierOptions Classifier { get; init; } = new();
    public PaddleOcrRecognizerOptions Recognizer { get; init; } = new();
}

public readonly record struct PaddleOcrDetectionBox(
    float X1, float Y1, float X2, float Y2,
    float X3, float Y3, float X4, float Y4, float Score)
{
    public Vector2 this[int index] => index switch
    {
        0 => new(X1, Y1),
        1 => new(X2, Y2),
        2 => new(X3, Y3),
        3 => new(X4, Y4),
        _ => throw new ArgumentOutOfRangeException(nameof(index))
    };

    public float Width => Vector2.Distance(new(X1, Y1), new(X2, Y2));
    public float Height => Vector2.Distance(new(X1, Y1), new(X4, Y4));
}

public sealed class PaddleOcrDetectionResult
{
    public required PaddleOcrDetectionBox[] Boxes { get; init; }
    public required int ResizedWidth { get; init; }
    public required int ResizedHeight { get; init; }
    public required float WidthRatio { get; init; }
    public required float HeightRatio { get; init; }
}

public readonly record struct PaddleOcrClassificationResult(uint Label, float Score, int ResizedWidth)
{
    public int OrientationDegrees => Label == 0 ? 0 : 180;
}

/// <summary>CTC decode of one crop. <see cref="CtcSpans"/> is null unless alignment was requested.</summary>
/// <param name="ResizedWidth">
/// Recognizer content width after the height-48 resize, before right padding.
/// </param>
public readonly record struct PaddleOcrRecognitionResult(string Text, float Score, int EmittedCount,
    int ResizedWidth, int TimeSteps)
{
    /// <summary>
    /// Recognizer tensor width, including right padding out to the width bucket.
    /// CTC time steps span this width, so content occupies the leading
    /// <c>ResizedWidth / TensorWidth</c> fraction. Zero when unset.
    /// </summary>
    public int TensorWidth { get; init; }

    /// <summary>
    /// One <see cref="PaddleOcrCtcSpan"/> per emitted token, in reading order.
    /// Null when this call did not pass <c>returnCtcAlignment: true</c>.
    /// An empty array means alignment was captured and the line emitted nothing.
    /// </summary>
    public PaddleOcrCtcSpan[]? CtcSpans { get; init; }
}

/// <summary>
/// Half-open time-step run of one emitted token: repeated frames of that class
/// are included, and blank frames are excluded.
/// </summary>
public readonly record struct PaddleOcrCtcSpan(string Text, float Score, int StartColumn, int EndColumn);

/// <summary>
/// Estimated quad of one CTC token on the source image. Neighboring boxes split
/// the blank gap between <see cref="PaddleOcrCtcSpan"/> runs at its midpoint and
/// share that edge; a side with no neighbor keeps the trigger edge. Point order
/// matches the character rectangle in the crop the recognizer read (left-top,
/// right-top, right-bottom, left-bottom) after undoing a 180° line rotation and,
/// for a tall quad, the same 90° rotation the crop uses.
/// </summary>
public readonly record struct PaddleOcrCharacterBox(
    string Text, float Score,
    float X1, float Y1, float X2, float Y2,
    float X3, float Y3, float X4, float Y4);

public sealed class PaddleOcrLine
{
    public required PaddleOcrDetectionBox Box { get; init; }
    public required string Text { get; init; }
    public required float RecognitionScore { get; init; }
    public required float ClassificationScore { get; init; }
    public required uint ClassificationLabel { get; init; }
    public required int AppliedRotationDegrees { get; init; }
    public required uint EmittedCount { get; init; }
    /// <summary>
    /// CTC emissions for this line. Null unless this call passed
    /// <c>returnCtcAlignment: true</c>.
    /// </summary>
    public PaddleOcrCtcSpan[]? CtcSpans { get; init; }
    /// <summary>Recognizer content width. See <see cref="PaddleOcrRecognitionResult.ResizedWidth"/>.</summary>
    public int RecognitionContentWidth { get; init; }
    /// <summary>Recognizer tensor width. See <see cref="PaddleOcrRecognitionResult.TensorWidth"/>.</summary>
    public int RecognitionTensorWidth { get; init; }
    /// <summary>CTC time steps spanning <see cref="RecognitionTensorWidth"/>.</summary>
    public int RecognitionTimeSteps { get; init; }

    /// <summary>
    /// Maps <see cref="CtcSpans"/> onto <see cref="Box"/>. Blank gaps between
    /// spans are split at their midpoint. Requires <c>returnCtcAlignment: true</c>
    /// on the <see cref="PaddleOcrAll.Run"/> or <see cref="PaddleOcrRecognizer.Recognize"/>
    /// call that produced this line.
    /// </summary>
    public PaddleOcrCharacterBox[] EstimateCharacterBoxes() => PaddleOcrCharacterBoxes.Estimate(this);
}

public sealed class PaddleOcrResult
{
    public required PaddleOcrLine[] Lines { get; init; }
    public required int DetectedCount { get; init; }
    public required int DetectorResizedWidth { get; init; }
    public required int DetectorResizedHeight { get; init; }

    public string Text => string.Join("\n", Lines.Select(x => x.Text));

    /// <summary>Matches the C benchmark's packed UTF-8 text (NUL after every line).</summary>
    public string PackedText => string.Concat(Lines.Select(x => x.Text + "\0"));

    public ulong PackedTextHash
    {
        get
        {
            ulong hash = 14695981039346656037UL;
            foreach (byte value in System.Text.Encoding.UTF8.GetBytes(PackedText))
            {
                hash ^= value;
                hash *= 1099511628211UL;
            }
            return hash;
        }
    }
}
