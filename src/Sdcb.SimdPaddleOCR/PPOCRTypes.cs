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
    /// <see cref="Environment.ProcessorCount"/>, enough for one session per core;
    /// set to <see cref="int.MaxValue"/> for an unbounded pool.
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
    /// <summary>Compute backend for the classification graph. Default <see cref="OcrBackend.Auto"/>.</summary>
    public OcrBackend Backend { get; init; } = OcrBackend.Auto;
    public long MaxImagePixels { get; init; } = 40_000_000;
    /// <summary>
    /// Upper bound on pooled reusable sessions. Sessions returned while the pool
    /// is full are disposed instead of pooled. The default is
    /// <see cref="Environment.ProcessorCount"/>; set to <see cref="int.MaxValue"/>
    /// for an unbounded pool.
    /// </summary>
    public int MaxPooledSessions { get; init; } = Environment.ProcessorCount;
}

/// <summary>四方向纸张识别阶段的计算后端、线程预算和资源上限。</summary>
public sealed class PaddleOcrDocumentOrientationOptions
{
    /// <summary>推理后端；默认 Auto，由运行平台和可用设备选择 CPU 或 GPU。</summary>
    public OcrBackend Backend { get; init; } = OcrBackend.Auto;
    /// <summary>CPU 算子线程预算；0 表示最多使用 4 个逻辑处理器，显式设置允许 1～16。</summary>
    public int IntraOpThreads { get; init; }
    /// <summary>输入图像允许的最大像素数，默认四千万；用于在预处理前限制资源消耗。</summary>
    public long MaxImagePixels { get; init; } = 40_000_000;
    /// <summary>保留的空闲推理会话上限，默认逻辑处理器数；0 表示每次使用后立即释放。</summary>
    public int MaxPooledSessions { get; init; } = Environment.ProcessorCount;
}

/// <summary>纸张在原图中的顺时针方向与对应置信度；校正时应按相同角度逆时针旋转。</summary>
/// <param name="Label">最高分的类别索引；0、1、2、3 分别对应 0°、90°、180°、270°。</param>
/// <param name="Score">该类别的模型输出分数；官方模型输出为 softmax 概率。</param>
public readonly record struct PaddleOcrDocumentOrientationResult(uint Label, float Score)
{
    /// <summary>原图的顺时针纸张方向，取值为 0、90、180、270 度。</summary>
    public int OrientationDegrees => checked((int)Label * 90);
}

/// <summary>Options for the CTC recognizer.</summary>
public sealed class PaddleOcrRecognizerOptions
{
    /// <summary>
    /// Upper bound on pooled reusable sessions. Sessions returned while the pool
    /// is full are disposed instead of pooled, capping memory at the cost of
    /// re-creating sessions when concurrency exceeds the cap. The default is
    /// <see cref="Environment.ProcessorCount"/>, enough for one session per core;
    /// set to <see cref="int.MaxValue"/> for an unbounded pool.
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
/// Separate budgets: <see cref="DetIntraOpThreads"/> is threads inside
/// the single DET session; <see cref="LineWorkerCount"/> is how many CLS/REC
/// sessions run in parallel (the two stages share that count, clamped to
/// ProcessorCount). REC may still shard convolutions inside each session
/// with leftover cores; that CPU graph budget is not customer-facing.
/// <see cref="PreprocessWorkerCount"/> controls crop/GPU preprocessing workers;
/// <see cref="GpuCtcIntraOpThreads"/> controls CPU projection after GPU recognition.
/// </para>
/// </summary>
public sealed class PaddleOcrOptions
{
    /// <summary>是否执行纸张方向识别；默认开启，但只有显式传入独立纸张方向模型时才运行第一阶段。</summary>
    public bool UseDocumentOrientationClassification { get; init; } = true;
    /// <summary>整页旋转的最低置信度，范围 0～1；默认 0，按模型最高分类执行校正。低于阈值时仍返回方向结果。</summary>
    public float DocumentOrientationThreshold { get; init; }
    /// <summary>纸张方向识别阶段的独立后端、线程和资源配置。</summary>
    public PaddleOcrDocumentOrientationOptions DocumentOrientation { get; init; } = new();
    /// <summary>是否对裁剪后的文本行执行 0°/180° 分类；与整页纸张方向识别分别控制。</summary>
    public bool UseDirectionClassification { get; init; } = true;
    /// <summary>
    /// Minimum class-1 probability required for a 180-degree rotation.
    /// The official PaddleX text-line orientation pipeline uses the top-1
    /// class directly, so the default is zero (no additional confidence
    /// gate). Set a positive value to retain the legacy threshold behavior.
    /// </summary>
    public float ClassifierThreshold { get; init; } = 0f;
    /// <summary>
    /// How many CLS / REC line workers run at once. Each worker may
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

    /// <summary>Crop and GPU batch resize/normalize workers. 0 preserves the automatic budget; 1..16 is capped by the logical processor count. Does not change detector preprocessing or graph operator threads.</summary>
    public int PreprocessWorkerCount { get; init; }

    /// <summary>CPU vocabulary projection/ArgMax threads after GPU batch recognition. 0 preserves the automatic budget; 1..16 is capped by the logical processor count. Does not reduce CPU inference or GPU-fallback operator budgets.</summary>
    public int GpuCtcIntraOpThreads { get; init; }
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

/// <summary>单个文本裁剪图的 CTC 识别结果；未请求对齐时 CtcSpans 为 null。</summary>
/// <param name="Text">CTC 解码后的文本。</param>
/// <param name="Score">该段文本的识别分数。</param>
/// <param name="EmittedCount">解码输出的词项数量。</param>
/// <param name="ResizedWidth">缩放至高 48 后、右侧补零之前的有效内容宽度。</param>
/// <param name="TimeSteps">识别模型输出的 CTC 时间步数。</param>
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
    /// 根据 <see cref="CtcSpans"/> 和 <see cref="Box"/> 估算原图坐标中的字符四边形，空白间隔从中点分割。
    /// 生成本行时须在 <see cref="PaddleOcrAll.Run(ReadOnlySpan{byte}, int, int, bool, int, ImagePixelFormat)"/>
    /// 或 <see cref="PaddleOcrRecognizer.Recognize(ReadOnlySpan{byte}, int, int, bool, int, ImagePixelFormat)"/>
    /// 中传入 <c>returnCtcAlignment: true</c>。
    /// </summary>
    /// <returns>已考虑整页及文本行旋转的字符框；没有输出字符时返回空数组。</returns>
    /// <exception cref="InvalidOperationException">生成本行时未保留 CTC 对齐信息。</exception>
    public PaddleOcrCharacterBox[] EstimateCharacterBoxes() => PaddleOcrCharacterBoxes.Estimate(this);
}

public sealed class PaddleOcrResult
{
    /// <summary>整页方向预测与置信度；未传入纸张模型或关闭该阶段时为 null。</summary>
    public PaddleOcrDocumentOrientationResult? DocumentOrientation { get; init; }
    /// <summary>文本检测前实际执行的逆时针整页旋转角度；为 0、90、180、270，低置信度跳过旋转时为 0。</summary>
    public int AppliedDocumentRotationDegrees { get; init; }
    /// <summary>按校正图阅读顺序排列的识别行；行框及可估算的字符框均已还原到原始输入图坐标。</summary>
    public required PaddleOcrLine[] Lines { get; init; }
    public required int DetectedCount { get; init; }
    /// <summary>检测器实际使用的缩放图宽度；纸张校正开启时基于旋转后的图像计算。</summary>
    public required int DetectorResizedWidth { get; init; }
    /// <summary>检测器实际使用的缩放图高度；纸张校正开启时基于旋转后的图像计算。</summary>
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
