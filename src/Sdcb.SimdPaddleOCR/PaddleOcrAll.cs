using System.Buffers;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.ModelProvider;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// 完整 OCR 流水线，接收原始像素，依次执行可选的纸张方向校正、文本检测、文本行分类及文字识别。
/// <para>
/// 各阶段共享只读编译模型，并分别池化推理会话；每次请求独占裁剪和整页旋转缓冲区，支持并发 Run。
/// 返回的行框和字符框使用原始输入图坐标；图像解码由调用方完成。
/// </para>
/// </summary>
public sealed class PaddleOcrAll : IDisposable
{
    private static bool s_profileEnabled;
    private static readonly long[] s_profileTicks = new long[6];
    private readonly PaddleOcrDetector _detector;
    private readonly PaddleOcrClassifier? _classifier;
    private readonly PaddleOcrRecognizer _recognizer;
    private readonly PaddleOcrDocumentOrientationClassifier? _documentOrientation; // 未提供纸张模型或禁用阶段时为 null。
    private Model? _ownedDocumentOrientationModel; // 通过独立 provider 加载的模型由引擎统一释放。
    private readonly PaddleOcrOptions _options;
    private readonly int _lineWorkers;
    private readonly int _cropWorkers;
    private readonly int _recIntraOpBase;
    private readonly int _recIntraOpMax;
    private readonly int _gpuCtcIntraOpThreads;
    private readonly bool _recGpu;
    private readonly int _recBatchEffective;
    private readonly List<byte[]> _cropBuffers = [];
    private readonly object _cropLock = new();
    private bool _disposed;

    /// <summary>
    /// CLS/REC sessions that actually run in parallel after clamping
    /// <see cref="PaddleOcrOptions.LineWorkerCount"/> to ProcessorCount.
    /// </summary>
    public int EffectiveLineWorkerCount => _lineWorkers;

    public PaddleOcrAll(PaddleOcrModelSet models, PaddleOcrOptions? options = null)
        : this(models?.DetectionModel ?? throw new ArgumentNullException(nameof(models)), models.ClassificationModel,
            models.RecognitionModel, models.DictionaryUtf8.Span, options) => _ownedModels = models;

    private PaddleOcrModelSet? _ownedModels;

    public static PaddleOcrAll Load(string detectionPath, string? classificationPath, string recognitionPath, string dictionaryPath, PaddleOcrOptions? options = null)
        => new(PaddleOcrModelSet.Load(detectionPath, classificationPath, recognitionPath, dictionaryPath), options);

    public static async Task<PaddleOcrAll> LoadAsync(string detectionPath, string? classificationPath,
        string recognitionPath, string dictionaryPath, PaddleOcrOptions? options = null,
        CancellationToken cancellationToken = default)
        => new(await PaddleOcrModelSet.LoadAsync(detectionPath, classificationPath, recognitionPath,
            dictionaryPath, cancellationToken).ConfigureAwait(false), options);

    public static PaddleOcrAll Load(PaddleOcrModelBundle bundle, PaddleOcrOptions? options = null)
        => new(PaddleOcrModelSet.Load(bundle), options);

    public static async Task<PaddleOcrAll> LoadAsync(PaddleOcrModelBundle bundle,
        PaddleOcrOptions? options = null, CancellationToken cancellationToken = default)
        => new(await PaddleOcrModelSet.LoadAsync(bundle, cancellationToken).ConfigureAwait(false), options);

    /// <summary>从 OCR 模型集合和独立纸张方向模型提供方加载带第一阶段的完整流水线。</summary>
    /// <param name="bundle">文本检测、文本行分类、识别和字典组成的 OCR 模型集合。</param>
    /// <param name="documentOrientation">独立四方向模型提供方，Kind 必须为 DocumentOrientation。</param>
    /// <param name="options">流水线和各阶段配置；传入 null 使用默认配置。</param>
    /// <returns>拥有本次加载模型的 OCR 引擎；使用完毕后由调用方释放。</returns>
    public static PaddleOcrAll Load(PaddleOcrModelBundle bundle, IPaddleOcrModelProvider documentOrientation,
        PaddleOcrOptions? options)
    {
        ValidateDocumentProvider(documentOrientation);
        using Stream stream = documentOrientation.OpenRead();
        Model orientation = Model.Load(stream);
        PaddleOcrModelSet? models = null;
        try
        {
            models = PaddleOcrModelSet.Load(bundle);
            return new PaddleOcrAll(models.DetectionModel, models.ClassificationModel, models.RecognitionModel,
                models.DictionaryUtf8.Span, options, orientation)
                { _ownedModels = models, _ownedDocumentOrientationModel = orientation };
        }
        catch { models?.Dispose(); orientation.Dispose(); throw; }
    }

    /// <summary>异步加载 OCR 模型集合及独立纸张方向模型，构造带整页校正的流水线。</summary>
    /// <param name="bundle">文本检测、文本行分类、识别和字典组成的 OCR 模型集合。</param>
    /// <param name="documentOrientation">独立四方向模型提供方，Kind 必须为 DocumentOrientation。</param>
    /// <param name="options">流水线和各阶段配置；传入 null 使用默认配置。</param>
    /// <param name="cancellationToken">取消模型流读取、解析及模型集合加载的令牌。</param>
    /// <returns>拥有本次加载模型的 OCR 引擎。</returns>
    public static async Task<PaddleOcrAll> LoadAsync(PaddleOcrModelBundle bundle,
        IPaddleOcrModelProvider documentOrientation, PaddleOcrOptions? options,
        CancellationToken cancellationToken = default)
    {
        ValidateDocumentProvider(documentOrientation);
        using Stream stream = await documentOrientation.OpenReadAsync(cancellationToken).ConfigureAwait(false);
        Model orientation = await Model.LoadAsync(stream, cancellationToken).ConfigureAwait(false);
        PaddleOcrModelSet? models = null;
        try
        {
            models = await PaddleOcrModelSet.LoadAsync(bundle, cancellationToken).ConfigureAwait(false);
            return new PaddleOcrAll(models.DetectionModel, models.ClassificationModel, models.RecognitionModel,
                models.DictionaryUtf8.Span, options, orientation)
                { _ownedModels = models, _ownedDocumentOrientationModel = orientation };
        }
        catch { models?.Dispose(); orientation.Dispose(); throw; }
    }

    /// <summary>验证提供方属于四方向纸张模型，避免误将文本行两分类模型接入第一阶段。</summary>
    /// <param name="provider">待验证的非空模型提供方。</param>
    private static void ValidateDocumentProvider(IPaddleOcrModelProvider provider)
    {
        if (provider is null) throw new ArgumentNullException(nameof(provider));
        if (provider.Kind != PaddleOcrModelKind.DocumentOrientation)
            throw new ArgumentException("Expected a four-way document orientation provider.", nameof(provider));
    }

    /// <summary>Loads a complete OCR pipeline directly from model streams.</summary>
    public static PaddleOcrAll Load(
        Stream detectionOnnx,
        Stream? classificationOnnx,
        Stream recognitionOnnx,
        Stream dictionaryUtf8,
        PaddleOcrOptions? options = null)
        => new(PaddleOcrModelSet.Load(detectionOnnx, classificationOnnx, recognitionOnnx, dictionaryUtf8), options);

    public static async Task<PaddleOcrAll> LoadAsync(
        Stream detectionOnnx,
        Stream? classificationOnnx,
        Stream recognitionOnnx,
        Stream dictionaryUtf8,
        PaddleOcrOptions? options = null,
        CancellationToken cancellationToken = default)
        => new(await PaddleOcrModelSet.LoadAsync(detectionOnnx, classificationOnnx, recognitionOnnx,
            dictionaryUtf8, cancellationToken).ConfigureAwait(false), options);

    public PaddleOcrAll(
        Stream detectorModel,
        Stream? classifierModel,
        Stream recognizerModel,
        Stream dictionaryUtf8,
        PaddleOcrOptions? options = null)
        : this(PaddleOcrModelSet.Load(detectorModel, classifierModel, recognizerModel, dictionaryUtf8), options)
    {
    }

    public PaddleOcrAll(
        Model detectorModel,
        Model? classifierModel,
        Model recognizerModel,
        Stream dictionaryUtf8,
        PaddleOcrOptions? options = null)
        : this(detectorModel, classifierModel, recognizerModel,
            ReadDictionary(dictionaryUtf8), options)
    {
    }

    /// <summary>使用已加载的 OCR 模型创建不带纸张方向阶段的流水线。</summary>
    /// <param name="detectorModel">文本检测模型，由调用方管理其生命周期。</param>
    /// <param name="classifierModel">文本行 0°/180° 分类模型；关闭文本行分类时可为 null。</param>
    /// <param name="recognizerModel">文字识别模型，由调用方管理其生命周期。</param>
    /// <param name="dictionaryUtf8">UTF-8 字典内容，定义识别类别对应的字符。</param>
    /// <param name="options">流水线配置；为 null 时使用默认配置。</param>
    public PaddleOcrAll(Model detectorModel, Model? classifierModel, Model recognizerModel,
        ReadOnlySpan<byte> dictionaryUtf8, PaddleOcrOptions? options = null)
        : this(detectorModel, classifierModel, recognizerModel, dictionaryUtf8, options, null) { }

    /// <summary>使用已加载模型创建流水线，可显式接入独立的四方向纸张模型。</summary>
    /// <param name="detectorModel">文本检测模型，由调用方管理其生命周期。</param>
    /// <param name="classifierModel">文本行 0°/180° 分类模型；关闭文本行分类时可为 null。</param>
    /// <param name="recognizerModel">文字识别模型，由调用方管理其生命周期。</param>
    /// <param name="dictionaryUtf8">UTF-8 字典内容，定义识别类别对应的字符。</param>
    /// <param name="options">流水线配置；为 null 时使用默认配置。</param>
    /// <param name="documentOrientationModel">四方向纸张模型；为 null 时跳过第一阶段，模型本身由调用方释放。</param>
    public PaddleOcrAll(Model detectorModel, Model? classifierModel, Model recognizerModel,
        ReadOnlySpan<byte> dictionaryUtf8, PaddleOcrOptions? options, Model? documentOrientationModel)
    {
        _options = options ?? new PaddleOcrOptions();
        if (_options.LineWorkerCount is < 0 or > Parallelism.MaxLineWorkers)
            throw new ArgumentOutOfRangeException(nameof(options));
        if (_options.GpuCtcIntraOpThreads is < 0 or > 16)
            throw new ArgumentOutOfRangeException(nameof(options), "GpuCtcIntraOpThreads must be between 0 and 16.");
        if (_options.PreprocessWorkerCount is < 0 or > 16)
            throw new ArgumentOutOfRangeException(nameof(options), "PreprocessWorkerCount must be between 0 and 16.");
        if (_options.RecBatchLines < 1) throw new ArgumentOutOfRangeException(nameof(options));
        if (_options.ClassifierThreshold is < 0 or > 1 || !MathCompat.IsFinite(_options.ClassifierThreshold))
            throw new ArgumentOutOfRangeException(nameof(options));
        if (_options.DocumentOrientationThreshold is < 0 or > 1 || !MathCompat.IsFinite(_options.DocumentOrientationThreshold))
            throw new ArgumentOutOfRangeException(nameof(options));
        if (!_options.UseDirectionClassification) classifierModel = null;
        if (_options.UseDirectionClassification && classifierModel is null)
            throw new ArgumentNullException(nameof(classifierModel));
        _lineWorkers = Parallelism.ResolveLineWorkers(_options.LineWorkerCount);
        _cropWorkers = _options.PreprocessWorkerCount == 0
            ? Parallelism.ResolveCropWorkers(_options.LineWorkerCount)
            : Math.Min(_options.PreprocessWorkerCount, Environment.ProcessorCount);
        _detector = new PaddleOcrDetector(detectorModel ?? throw new ArgumentNullException(nameof(detectorModel)), _options.Detector,
            ResolveDetectorIntraThreads(_options));
        try
        {
            _classifier = classifierModel is null ? null : new PaddleOcrClassifier(classifierModel, _options.Classifier);
            _recIntraOpBase = Parallelism.ResolveRecognizerIntraOp(_lineWorkers);
            // 剩余行减少时可使用空闲线程提升识别速度，最高不超过编译模型支持的 16 线程。
            // NHWC 算子受 FMA 吞吐限制，已有测量表明 SMT 线程也可提供额外计算吞吐。
            _recIntraOpMax = Math.Min(Environment.ProcessorCount, 16);
            _gpuCtcIntraOpThreads = _options.GpuCtcIntraOpThreads == 0
                ? _recIntraOpMax
                : Math.Min(_options.GpuCtcIntraOpThreads, Environment.ProcessorCount);
            _recognizer = new PaddleOcrRecognizer(recognizerModel ?? throw new ArgumentNullException(nameof(recognizerModel)),
                dictionaryUtf8, _options.Recognizer, ownsModel: false,
                _recIntraOpBase);
            _recGpu = OnnxSharp.OcrSessionFactory.IsGpuBackend(_options.Recognizer.Backend);
            _recBatchEffective = _options.HasExplicitRecBatchLines ? _options.RecBatchLines
                : _recGpu ? 16 : _options.RecBatchLines;
            // 模型独立且显式传入；仅开启配置不会隐式加载或下载纸张模型。
            _documentOrientation = documentOrientationModel is not null && _options.UseDocumentOrientationClassification
                ? new PaddleOcrDocumentOrientationClassifier(documentOrientationModel, _options.DocumentOrientation) : null;
        }
        catch
        {
            // 构造后续阶段失败时，释放此前已创建的会话和编译结果；调用方拥有的 Model 不在这里释放。
            _documentOrientation?.Dispose();
            _recognizer?.Dispose();
            _classifier?.Dispose();
            _detector.Dispose();
            throw;
        }
    }

    private static byte[] ReadDictionary(Stream source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        using MemoryStream buffer = new();
        source.CopyTo(buffer);
        if (buffer.Length == 0) throw new InvalidDataException("Dictionary stream is empty.");
        return buffer.ToArray();
    }

    /// <summary>执行可选整页校正、文本检测、透视裁剪、文本行分类和文字识别。</summary>
    /// <param name="source">原始输入像素，不会被修改。</param>
    /// <param name="sourceWidth">输入图宽度，单位为像素。</param>
    /// <param name="sourceHeight">输入图高度，单位为像素。</param>
    /// <param name="sourceStride">输入行字节跨度；0 表示紧密排列。</param>
    /// <param name="format">输入像素格式，默认 BGR24。</param>
    /// <returns>识别文本、原图坐标框及可选纸张方向结果。</returns>
    public PaddleOcrResult Run(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
        => Run(source, sourceWidth, sourceHeight, returnCtcAlignment: false, sourceStride, format);

    /// <summary>
    /// 执行完整 OCR，可在解码时保留 CTC 字符对齐，供原图坐标中的字符框估算使用。
    /// </summary>
    /// <param name="source">原始输入像素，可含行尾填充，不会被修改。</param>
    /// <param name="sourceWidth">原始输入图宽度，单位为像素。</param>
    /// <param name="sourceHeight">原始输入图高度，单位为像素。</param>
    /// <param name="returnCtcAlignment">是否保留各输出字符的 CTC 时间区间；false 时 CtcSpans 为 null。</param>
    /// <param name="sourceStride">输入行字节跨度；0 表示按宽度和格式计算紧密排列的跨度。</param>
    /// <param name="format">输入像素格式，默认 BGR24。</param>
    /// <returns>包含方向预测、实际校正角度及原图坐标识别行的结果。</returns>
    public PaddleOcrResult Run(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        bool returnCtcAlignment, int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrAll));
        if (_documentOrientation is null)
            return RunCore(source, sourceWidth, sourceHeight, returnCtcAlignment, sourceStride, format);
        sourceStride = ImagePixels.ResolveStride(sourceWidth, sourceStride, format);
        if ((long)sourceWidth * sourceHeight > _options.Detector.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
        PaddleOcrDocumentOrientationResult orientation = _documentOrientation.Classify(source,
            sourceWidth, sourceHeight, sourceStride, format);
        // 方向预测表示顺时针倾转；按相同角度逆时针校正。低置信度只跳过旋转，不丢弃预测结果。
        int rotation = orientation.Score >= _options.DocumentOrientationThreshold ? orientation.OrientationDegrees : 0;
        byte[]? rotated = null;
        try
        {
            PaddleOcrResult result;
            if (rotation == 0) result = RunCore(source, sourceWidth, sourceHeight, returnCtcAlignment, sourceStride, format);
            else
            {
                long started = PipelineProfiler.Enabled ? PipelineProfiler.Now() : 0;
                // 每次请求独占旋转缓冲区；目标像素紧密排列，因此 RunCore 的 stride 传 0。
                rotated = PooledArrays.Rent<byte>(checked(sourceWidth * sourceHeight * ImagePixels.BytesPerPixel(format)));
                DocumentRotation.Rotate(source, sourceWidth, sourceHeight, sourceStride, format, rotation, rotated);
                if (PipelineProfiler.Enabled) PipelineProfiler.Add(PipelineProfiler.DocumentRotate, started);
                var size = DocumentRotation.Size(sourceWidth, sourceHeight, rotation);
                result = RunCore(rotated, size.Width, size.Height, returnCtcAlignment, 0, format);
                // 逐点还原行框并保留阅读顺序，CTC 字符框后续也能正确映射回原图。
                for (int i = 0; i < result.Lines.Length; i++)
                    result.Lines[i] = DocumentRotation.ToSource(result.Lines[i], sourceWidth, sourceHeight, rotation);
            }
            return new PaddleOcrResult
            {
                Lines = result.Lines, DetectedCount = result.DetectedCount,
                DetectorResizedWidth = result.DetectorResizedWidth, DetectorResizedHeight = result.DetectorResizedHeight,
                DocumentOrientation = orientation, AppliedDocumentRotationDegrees = rotation
            };
        }
        finally { if (rotated is not null) PooledArrays.Return(rotated); }
    }

    /// <summary>对当前输入图像执行 DET、文本行 CLS 和 REC；本方法不执行整页方向预测或校正。</summary>
    /// <param name="source">当前图像的像素；可能是原图，也可能是整页旋转后的临时图。</param>
    /// <param name="sourceWidth">当前图像宽度，单位为像素。</param>
    /// <param name="sourceHeight">当前图像高度，单位为像素。</param>
    /// <param name="returnCtcAlignment">是否在识别解码时保留字符对齐信息。</param>
    /// <param name="sourceStride">当前图像行字节跨度；0 表示紧密排列。</param>
    /// <param name="format">当前图像的像素格式。</param>
    /// <returns>当前图像坐标系中的 OCR 结果；原图坐标还原由外层 Run 负责。</returns>
    private PaddleOcrResult RunCore(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        bool returnCtcAlignment, int sourceStride, ImagePixelFormat format)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrAll));
        sourceStride = ImagePixels.ResolveStride(sourceWidth, sourceStride, format);
        long stageStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
        PaddleOcrDetectionResult detection = _detector.Detect(source, sourceWidth, sourceHeight, sourceStride, format);
        if (s_profileEnabled) AddProfile(0, stageStart);
        int count = detection.Boxes.Length;
        if (count == 0)
            return new PaddleOcrResult
            {
                Lines = [],
                DetectedCount = 0,
                DetectorResizedWidth = detection.ResizedWidth,
                DetectorResizedHeight = detection.ResizedHeight
            };
        if ((long)sourceWidth * sourceHeight > _options.Detector.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");

        // Per-call crop workspace from the engine's grow-only pool so unique
        // cropTotal sizes are not discarded into LOH after every image.
        int[] cropOffsets = PooledArrays.Rent<int>(count);
        int[] cropBytes = PooledArrays.Rent<int>(count);
        int[] cropWidths = PooledArrays.Rent<int>(count);
        int[] cropHeights = PooledArrays.Rent<int>(count);
        byte[] cropBuffer = [];
        bool pipelineProfile = PipelineProfiler.Enabled;
        try
        {
            stageStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
            long pipelineStarted = pipelineProfile ? PipelineProfiler.Now() : 0;
            long cropTotal = 0;
            for (int i = 0; i < count; i++)
            {
                (int Width, int Height, int ByteCount) size = PPOCRCrop.GetSize(detection.Boxes[i]);
                if ((long)size.Width * size.Height > _options.MaxCropPixels)
                    throw new InvalidOperationException("OCR crop exceeds MaxCropPixels.");
                cropTotal = checked(cropTotal + size.ByteCount);
                if (cropTotal > int.MaxValue)
                    throw new InvalidOperationException("OCR crop workspace exceeds the managed array limit.");
                cropOffsets[i] = checked((int)(cropTotal - size.ByteCount));
                cropBytes[i] = size.ByteCount;
            }
            cropBuffer = RentCropBuffer(checked((int)cropTotal));
            if (pipelineProfile) PipelineProfiler.Add(PipelineProfiler.CropSetup, pipelineStarted);
            pipelineStarted = pipelineProfile ? PipelineProfiler.Now() : 0;
            // Crops write disjoint buffer ranges, so they parallelize freely.
            // This is an exclusive window before any line worker starts, so it
            // gets its own budget the same way DET does; reusing
            // LineWorkerCount here capped the crop at 4 threads on a 16-core
            // machine and left the phase latency-bound. The automatic budget
            // never falls below the line workers (Parallelism.ResolveCropWorkers)
            // and an explicit PreprocessWorkerCount may deliberately do so, so
            // the resolved budget is used as-is without a line-count lift.
            int lineWorkerCount = Math.Min(_lineWorkers, count);
            int cropWorkers = Math.Min(_cropWorkers, count);
            if (cropWorkers <= 1)
            {
                for (int i = 0; i < count; i++)
                {
                    PPOCRCrop.ExtractInto(source, sourceWidth, sourceHeight, sourceStride,
                        detection.Boxes[i], cropBuffer.AsSpan(cropOffsets[i], cropBytes[i]),
                        out cropWidths[i], out cropHeights[i], format);
                }
            }
            else
            {
                // Flatten (line, row band) into one work list. cropWorkers is
                // still min'd with `count` above, so a 12-line page on 16 cores
                // uses 12 workers, not 16. The 2D split still helps: bands of
                // one tall crop go to different workers and cut the slowest-line
                // tail. Rows within a crop write disjoint destination ranges, so
                // bands are independent and the per-row arithmetic is unchanged.
                List<(int Line, int Begin, int End)> bands = [];
                for (int i = 0; i < count; i++)
                {
                    int rows = PPOCRCrop.UnrotatedHeight(detection.Boxes[i]);
                    for (int y = 0; y < rows; y += CropRowsPerBand)
                        bands.Add((i, y, Math.Min(rows, y + CropRowsPerBand)));
                }
                int cropJobs = Math.Min(cropWorkers, bands.Count);
                if (cropJobs <= 1)
                {
                    for (int i = 0; i < count; i++)
                    {
                        PPOCRCrop.ExtractInto(source, sourceWidth, sourceHeight, sourceStride,
                            detection.Boxes[i], cropBuffer.AsSpan(cropOffsets[i], cropBytes[i]),
                            out cropWidths[i], out cropHeights[i], format);
                    }
                }
                else
                {
                    unsafe
                    {
                        fixed (byte* sourcePtr = source)
                        {
                            nint sourceAddress = (nint)sourcePtr;
                            int sourceLength = source.Length;
                            byte[] cropTarget = cropBuffer;
                            int[] offsets = cropOffsets, bytes = cropBytes, widths = cropWidths,
                                heights = cropHeights;
                            int workers = cropJobs;
                            PaddleOcrDetectionBox[] boxes = detection.Boxes;
                            List<(int Line, int Begin, int End)> jobs = bands;
                            Parallel.For(0, workers, worker =>
                                CropBands(worker, workers, sourceAddress, sourceLength,
                                    sourceWidth, sourceHeight, sourceStride, format, boxes, cropTarget,
                                    offsets, bytes, widths, heights, jobs));
                        }
                    }
                }
            }
            if (pipelineProfile) PipelineProfiler.Add(PipelineProfiler.Crop, pipelineStarted);
            if (s_profileEnabled) AddProfile(1, stageStart);

            PaddleOcrLine[] lines = new PaddleOcrLine[count];
            stageStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
            pipelineStarted = pipelineProfile ? PipelineProfiler.Now() : 0;
            int workerCount = lineWorkerCount;
            int recIntraOp = LineIntraOpBudget(count);
            // Same-width batched REC is opt-in: it is numerically exact, but
            // per-op whole-batch execution inflates the activation working set
            // and measured slower than per-line REC on desktop CPUs. On a GPU
            // backend batching is the main win, so the default is raised there
            // unless the caller set RecBatchLines explicitly.
            int maxBatch = Math.Max(1, _recBatchEffective);
            if (maxBatch > 1)
            {
                ProcessLinesBatched(count, workerCount, maxBatch, cropBuffer, cropOffsets,
                    cropBytes, cropWidths, cropHeights, detection.Boxes, lines, returnCtcAlignment);
            }
            else if (workerCount <= 1)
            {
                ProcessRange(0, count, 1, 0, recIntraOp, cropBuffer, cropOffsets, cropBytes,
                    cropWidths, cropHeights, detection.Boxes, lines, returnCtcAlignment);
            }
            else
            {
                // Longest-processing-time first with a shared cursor: REC cost
                // grows with the resized line width, so handing out the widest
                // crops first and letting idle workers pull the next line keeps
                // the tail (where fewer than workerCount lines remain) short.
                int[] order = PooledArrays.Rent<int>(count);
                try
                {
                    for (int i = 0; i < count; i++) order[i] = i;
                    int[] cost = cropWidths, denominator = cropHeights;
                    Array.Sort(order, 0, count, Comparer<int>.Create((a, b) =>
                    {
                        long costA = (long)cost[a] * Math.Max(1, denominator[b]);
                        long costB = (long)cost[b] * Math.Max(1, denominator[a]);
                        int byCost = costB.CompareTo(costA);
                        return byCost != 0 ? byCost : a.CompareTo(b);
                    }));
                    int cursor = -1;
                    Parallel.For(0, workerCount, _ =>
                    {
                        PaddleOcrClassifier? classifier = _classifier;
                        PaddleOcrRecognizer recognizer = _recognizer;
                        int next;
                        while ((next = Interlocked.Increment(ref cursor)) < count)
                            ProcessOne(order[next], classifier, recognizer, recIntraOp,
                                cropBuffer, cropOffsets, cropBytes,
                                cropWidths, cropHeights, detection.Boxes, lines, returnCtcAlignment);
                    });
                }
                finally
                {
                    PooledArrays.Return(order);
                }
            }
            if (pipelineProfile) PipelineProfiler.Add(PipelineProfiler.LinesWall, pipelineStarted);
            if (s_profileEnabled) AddProfile(2, stageStart);
            return new PaddleOcrResult
            {
                Lines = lines,
                DetectedCount = count,
                DetectorResizedWidth = detection.ResizedWidth,
                DetectorResizedHeight = detection.ResizedHeight
            };
        }
        finally
        {
            PooledArrays.Return(cropOffsets);
            PooledArrays.Return(cropBytes);
            PooledArrays.Return(cropWidths);
            PooledArrays.Return(cropHeights);
            if (cropBuffer.Length != 0) ReturnCropBuffer(cropBuffer);
        }
    }

    internal static void EnableProfiling(bool enabled)
    {
        s_profileEnabled = enabled;
        if (enabled) Array.Clear(s_profileTicks, 0, s_profileTicks.Length);
    }

    internal static double[] ProfileSnapshot()
    {
        double scale = 1000.0 / Stopwatch.Frequency;
        double[] result = new double[s_profileTicks.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = Interlocked.Read(ref s_profileTicks[i]) * scale;
        return result;
    }

    private static void AddProfile(int stage, long started)
    {
        if ((uint)stage < (uint)s_profileTicks.Length)
            Interlocked.Add(ref s_profileTicks[stage], Stopwatch.GetTimestamp() - started);
    }

    // Width-bucketed batched REC: CLS + rotation first (parallel per line),
    // then same-target-width lines grouped into batched recognizer runs.
    // Grouping is deterministic (ascending line index) and each sample's math
    // is per-batch-element independent, so texts match the per-line path.
    private void ProcessLinesBatched(int count, int workerCount, int maxBatch, byte[] cropBuffer,
        int[] offsets, int[] bytes, int[] widths, int[] heights, PaddleOcrDetectionBox[] boxes,
        PaddleOcrLine[] lines, bool returnCtcAlignment)
    {
        uint[] labels = PooledArrays.Rent<uint>(count);
        float[] clsScores = PooledArrays.Rent<float>(count);
        int[] rotations = PooledArrays.Rent<int>(count);
        int[] recWidths = PooledArrays.Rent<int>(count);
        PaddleOcrRecognitionResult[] recResults = new PaddleOcrRecognitionResult[count];
        try
        {
            if (_classifier is { GpuCapable: true })
            {
                // GPU: one batched classify for all lines (fixed [n,3,80,160]
                // input — single submit, head resolved on CPU), then the cheap
                // rotate/width-select tail fans out across workers.
                _classifier.ClassifyBatch(cropBuffer, offsets, bytes, widths, heights,
                    count, labels, clsScores, _cropWorkers);
                if (workerCount <= 1)
                {
                    PostClassifyRange(0, count, 1, cropBuffer, offsets, bytes, widths,
                        heights, labels, clsScores, rotations, recWidths);
                }
                else
                {
                    Parallel.For(0, workerCount, worker =>
                        PostClassifyRange(worker, count, workerCount, cropBuffer, offsets,
                            bytes, widths, heights, labels, clsScores, rotations, recWidths));
                }
            }
            else if (workerCount <= 1)
            {
                ClassifyRange(0, count, 1, cropBuffer, offsets, bytes, widths, heights,
                    labels, clsScores, rotations, recWidths);
            }
            else
            {
                Parallel.For(0, workerCount, worker =>
                    ClassifyRange(worker, count, workerCount, cropBuffer, offsets, bytes,
                        widths, heights, labels, clsScores, rotations, recWidths));
            }

            // Exact-width groups: each unit runs at its members' own target
            // width, so no line sees extra padding. That matters for SVTR
            // recognizers (global attention mixes every column, so padding a
            // line past its bucket changes its logits) and costs nothing on
            // GPU, where all units of the image go out in one submission.
            List<int[]> units;
            {
                Dictionary<int, List<int>> groups = [];
                units = [];
                for (int i = 0; i < count; i++)
                {
                    if (!groups.TryGetValue(recWidths[i], out List<int>? members))
                        groups[recWidths[i]] = members = [];
                    members.Add(i);
                    if (members.Count == maxBatch)
                    {
                        units.Add([.. members]);
                        members.Clear();
                    }
                }
                foreach (List<int> members in groups.Values)
                    if (members.Count > 0)
                        units.Add([.. members]);
            }

            int unitWorkers = Math.Max(1, Math.Min(workerCount, units.Count));
            // Idle workers' cores go to the units actually running: a single
            // same-width group may shard across the whole machine.
            int unitIntraOp = MathCompat.Clamp(_recIntraOpBase * _lineWorkers / unitWorkers,
                _recIntraOpBase, _recIntraOpMax);
            // GpuRecAlive flips false once a session reports its rec graph fell
            // back to CPU; then the per-unit CPU-shaped path below takes over.
            long recStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
            if (_recGpu && _recognizer.GpuRecAlive
                && _recognizer.TryRecognizeUnitsBatched(cropBuffer, offsets, bytes, widths, heights,
                    units, recWidths, recResults, _gpuCtcIntraOpThreads, _cropWorkers, returnCtcAlignment))
            {
                if (s_profileEnabled) AddProfile(4, recStart);
            }
            else if (workerCount <= 1 || units.Count <= 1)
            {
                foreach (int[] unit in units)
                    RecognizeUnit(unit, unitIntraOp, cropBuffer, offsets, bytes, widths, heights, recWidths,
                        recResults, returnCtcAlignment);
            }
            else
            {
                Parallel.For(0, unitWorkers, worker =>
                {
                    for (int u = worker; u < units.Count; u += workerCount)
                        RecognizeUnit(units[u], unitIntraOp, cropBuffer, offsets, bytes, widths, heights,
                            recWidths, recResults, returnCtcAlignment);
                });
            }

            for (int i = 0; i < count; i++)
            {
                lines[i] = MakeLine(boxes[i], recResults[i], clsScores[i], labels[i], rotations[i]);
            }
        }
        finally
        {
            PooledArrays.Return(labels);
            PooledArrays.Return(clsScores);
            PooledArrays.Return(rotations);
            PooledArrays.Return(recWidths);
        }
    }

    private void RecognizeUnit(int[] unit, int intraOpThreads, byte[] cropBuffer, int[] offsets, int[] bytes,
        int[] widths, int[] heights, int[] recWidths, PaddleOcrRecognitionResult[] recResults,
        bool returnCtcAlignment)
    {
        long recStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
        _recognizer.RecognizeBatch(cropBuffer, offsets, bytes, widths, heights, unit,
            recWidths[unit[0]], recResults, intraOpThreads, returnCtcAlignment);
        if (s_profileEnabled) AddProfile(4, recStart);
    }

    private void ClassifyRange(int first, int count, int stride, byte[] cropBuffer, int[] offsets,
        int[] bytes, int[] widths, int[] heights, uint[] labels, float[] clsScores,
        int[] rotations, int[] recWidths)
    {
        PaddleOcrClassifier? classifier = _classifier;
        for (int i = first; i < count; i += stride)
        {
            ReadOnlySpan<byte> crop = cropBuffer.AsSpan(offsets[i], bytes[i]);
            uint label = 0;
            float clsScore = 0;
            int rotation = 0;
            if (classifier is not null)
            {
                long clsStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
                PaddleOcrClassificationResult result = classifier.Classify(crop, widths[i], heights[i]);
                if (s_profileEnabled) AddProfile(3, clsStart);
                label = result.Label;
                clsScore = result.Score;
                if ((label & 1u) != 0 && clsScore > _options.ClassifierThreshold)
                {
                    PPOCRCrop.Rotate180(cropBuffer.AsSpan(offsets[i], bytes[i]), widths[i], heights[i]);
                    rotation = 180;
                }
            }
            labels[i] = label;
            clsScores[i] = clsScore;
            rotations[i] = rotation;
            recWidths[i] = _recognizer.SelectWidthForCrop(widths[i], heights[i]);
        }
    }

    // Post-batch tail of ClassifyRange: rotates 180°-flagged crops in place and
    // fills per-line REC target widths. Runs after a batched classifier call
    // that already produced labels/clsScores for every line.
    private void PostClassifyRange(int first, int count, int stride, byte[] cropBuffer,
        int[] offsets, int[] bytes, int[] widths, int[] heights, uint[] labels,
        float[] clsScores, int[] rotations, int[] recWidths)
    {
        for (int i = first; i < count; i += stride)
        {
            int rotation = 0;
            if ((labels[i] & 1u) != 0 && clsScores[i] > _options.ClassifierThreshold)
            {
                PPOCRCrop.Rotate180(cropBuffer.AsSpan(offsets[i], bytes[i]), widths[i], heights[i]);
                rotation = 180;
            }
            rotations[i] = rotation;   // rented array — must write every slot
            recWidths[i] = _recognizer.SelectWidthForCrop(widths[i], heights[i]);
        }
    }

    // Rows per crop band. Large enough that per-band setup (perspective solve,
    // source validation) stays negligible against the row work it covers.
    private const int CropRowsPerBand = 16;

    private static unsafe void CropBands(int first, int workers, nint sourceAddress, int sourceLength,
        int sourceWidth, int sourceHeight, int sourceStride, ImagePixelFormat format,
        PaddleOcrDetectionBox[] boxes,
        byte[] cropBuffer, int[] offsets, int[] bytes, int[] widths, int[] heights,
        List<(int Line, int Begin, int End)> jobs)
    {
        ReadOnlySpan<byte> source = new((void*)sourceAddress, sourceLength);
        for (int job = first; job < jobs.Count; job += workers)
        {
            (int line, int begin, int end) = jobs[job];
            PPOCRCrop.ExtractRangeInto(source, sourceWidth, sourceHeight, sourceStride,
                boxes[line], cropBuffer.AsSpan(offsets[line], bytes[line]), begin, end, format);
            if (begin == 0)
            {
                (int width, int height, _) = PPOCRCrop.GetSize(boxes[line]);
                widths[line] = width;
                heights[line] = height;
            }
        }
    }

    // Intra-op budget shared by every REC run of one image: lines split the
    // pool evenly, so the wave never oversubscribes (sessions in flight ×
    // budget ≤ base × workers). Images with fewer lines than workers still
    // hand the parked workers' cores to the sessions that do run; a static
    // budget is used rather than tracking the scheduling tail because the
    // per-line measurement showed no gain from boosting mid-drain.
    // Per-element results are identical at any thread count.
    private int LineIntraOpBudget(int lineCount)
    {
        int active = Math.Max(1, Math.Min(_lineWorkers, lineCount));
        return MathCompat.Clamp(_recIntraOpBase * _lineWorkers / active, _recIntraOpBase, _recIntraOpMax);
    }

    private void ProcessRange(int first, int count, int stride, int worker, int recIntraOp, byte[] cropBuffer,
        int[] offsets, int[] bytes, int[] widths, int[] heights, PaddleOcrDetectionBox[] boxes,
        PaddleOcrLine[] lines, bool returnCtcAlignment)
    {
        PaddleOcrClassifier? classifier = _classifier;
        PaddleOcrRecognizer recognizer = _recognizer;
        for (int i = first; i < count; i += stride)
            ProcessOne(i, classifier, recognizer, recIntraOp, cropBuffer, offsets, bytes, widths, heights,
                boxes, lines, returnCtcAlignment);
    }

    private void ProcessOne(int i, PaddleOcrClassifier? classifier, PaddleOcrRecognizer recognizer,
        int recIntraOp, byte[] cropBuffer,
        int[] offsets, int[] bytes, int[] widths, int[] heights, PaddleOcrDetectionBox[] boxes,
        PaddleOcrLine[] lines, bool returnCtcAlignment)
    {
        ReadOnlySpan<byte> crop = cropBuffer.AsSpan(offsets[i], bytes[i]);
        uint label = 0;
        float clsScore = 0;
        int rotation = 0;
        if (classifier is not null)
        {
            long clsStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
            PaddleOcrClassificationResult result = classifier.Classify(crop, widths[i], heights[i]);
            if (s_profileEnabled) AddProfile(3, clsStart);
            label = result.Label;
            clsScore = result.Score;
            if ((label & 1u) != 0 && clsScore > _options.ClassifierThreshold)
            {
                PPOCRCrop.Rotate180(cropBuffer.AsSpan(offsets[i], bytes[i]), widths[i], heights[i]);
                rotation = 180;
            }
        }
        long recStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
        PaddleOcrRecognitionResult recognition = recognizer.Recognize(crop, widths[i], heights[i],
            0, ImagePixelFormat.Bgr24, recIntraOp, returnCtcAlignment);
        if (s_profileEnabled) AddProfile(4, recStart);
        lines[i] = MakeLine(boxes[i], recognition, clsScore, label, rotation);
    }

    private static PaddleOcrLine MakeLine(in PaddleOcrDetectionBox box, in PaddleOcrRecognitionResult recognition,
        float classificationScore, uint classificationLabel, int rotation) => new PaddleOcrLine
    {
        Box = box,
        Text = recognition.Text,
        RecognitionScore = recognition.Score,
        ClassificationScore = classificationScore,
        ClassificationLabel = classificationLabel,
        AppliedRotationDegrees = rotation,
        EmittedCount = (uint)recognition.EmittedCount,
        CtcSpans = recognition.CtcSpans,
        RecognitionContentWidth = recognition.ResizedWidth,
        RecognitionTensorWidth = recognition.TensorWidth,
        RecognitionTimeSteps = recognition.TimeSteps
    };

    // DET runs in an exclusive window before crop and line workers start, so
    // it always receives the same auto budget (up to 8 threads) regardless of
    // LineWorkerCount. Dividing DET by line workers used to starve DET on
    // smaller CPUs and did not protect the later line stage. auto>8 on
    // 20-thread Zen 5 inflated rec_pool and did not improve e2e; keep 8 unless
    // DetIntraOpThreads is set explicitly.
    private static int ResolveDetectorIntraThreads(PaddleOcrOptions options)
    {
        if (options.DetIntraOpThreads > 0) return Math.Min(options.DetIntraOpThreads, 16);
        if (int.TryParse(Environment.GetEnvironmentVariable("PPOCR_DET_THREADS"), out int env) && env > 0)
            return Math.Min(env, 16);
        // The channels-last kernels are FMA-bound, so SMT siblings add
        // throughput (det 16 vs 8 threads measured ~8-25% faster on Zen 3).
        if (OnnxSharp.LayoutPlanner.IsEnabled) return Math.Min(16, Environment.ProcessorCount);
        return Math.Min(8, Environment.ProcessorCount);
    }

    // Grow-only crop workspace, one in-flight buffer per concurrent Run.
    // PooledArrays discards ≥64KiB unique sizes, so renting cropTotal every
    // image left one LOH array per request until GC — the bulk of tiny 1w
    // ΔPrivate before GC above live workspace. Best-fit prefers a buffer
    // that already covers this request so Ensure-style growth is rare.
    private byte[] RentCropBuffer(int bytes)
    {
        if (bytes <= 0) return [];
        lock (_cropLock)
        {
            int bestFit = -1, grow = -1;
            for (int i = 0; i < _cropBuffers.Count; i++)
            {
                int length = _cropBuffers[i].Length;
                if (length >= bytes)
                {
                    if (bestFit < 0 || length < _cropBuffers[bestFit].Length)
                        bestFit = i;
                }
                else if (grow < 0 || length > _cropBuffers[grow].Length)
                    grow = i;
            }
            int take = bestFit >= 0 ? bestFit : grow;
            if (take >= 0)
            {
                byte[] buffer = _cropBuffers[take];
                int last = _cropBuffers.Count - 1;
                if (take != last) _cropBuffers[take] = _cropBuffers[last];
                _cropBuffers.RemoveAt(last);
                return buffer.Length >= bytes ? buffer : new byte[bytes];
            }
        }
        return new byte[bytes];
    }

    private void ReturnCropBuffer(byte[] buffer)
    {
        if (buffer.Length == 0 || _disposed) return;
        lock (_cropLock)
        {
            int cap = Math.Max(_lineWorkers, _options.Detector.MaxPooledSessions);
            if (_cropBuffers.Count >= cap) return;
            _cropBuffers.Add(buffer);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        lock (_cropLock)
            _cropBuffers.Clear();
        _detector.Dispose();
        _classifier?.Dispose();
        _recognizer.Dispose();
        _documentOrientation?.Dispose();
        _ownedDocumentOrientationModel?.Dispose();
        _ownedModels?.Dispose();
    }
}
