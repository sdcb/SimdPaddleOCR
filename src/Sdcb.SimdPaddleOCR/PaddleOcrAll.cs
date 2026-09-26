using System.Buffers;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.ModelProvider;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Complete pure managed PP-OCR pipeline. It accepts packed BGR8 memory and
/// deliberately has no image-decoding or file-system dependency.
/// <para>
/// Thread-safe: the detector, classifier and recognizer each hold one shared
/// compiled model and pool per-call inference requests, while every run rents
/// its own crop workspace. Concurrent <see cref="Run"/> calls are safe and
/// execute without a global engine lock.
/// </para>
/// </summary>
public sealed class PaddleOcrAll : IDisposable
{
    private static bool s_profileEnabled;
    private static readonly long[] s_profileTicks = new long[6];
    private readonly PaddleOcrDetector _detector;
    private readonly PaddleOcrClassifier? _classifier;
    private readonly PaddleOcrRecognizer _recognizer;
    private readonly PaddleOcrOptions _options;
    private readonly int _lineWorkers;
    private readonly int _cropWorkers;
    private readonly int _recIntraOpBase;
    private readonly int _recIntraOpMax;
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

    public PaddleOcrAll(Model detectorModel, Model? classifierModel, Model recognizerModel,
        ReadOnlySpan<byte> dictionaryUtf8, PaddleOcrOptions? options = null)
    {
        _options = options ?? new PaddleOcrOptions();
        if (_options.LineWorkerCount is < 0 or > Parallelism.MaxLineWorkers)
            throw new ArgumentOutOfRangeException(nameof(options));
        if (_options.RecBatchLines < 1) throw new ArgumentOutOfRangeException(nameof(options));
        if (_options.ClassifierThreshold is < 0 or > 1 || !MathCompat.IsFinite(_options.ClassifierThreshold))
            throw new ArgumentOutOfRangeException(nameof(options));
        if (!_options.UseDirectionClassification) classifierModel = null;
        if (_options.UseDirectionClassification && classifierModel is null)
            throw new ArgumentNullException(nameof(classifierModel));
        _lineWorkers = Parallelism.ResolveLineWorkers(_options.LineWorkerCount);
        _cropWorkers = Parallelism.ResolveCropWorkers(_options.LineWorkerCount);
        _detector = new PaddleOcrDetector(detectorModel ?? throw new ArgumentNullException(nameof(detectorModel)), _options.Detector,
            ResolveDetectorIntraThreads(_options));
        _classifier = classifierModel is null ? null : new PaddleOcrClassifier(classifierModel, _options.Classifier);
        _recIntraOpBase = Parallelism.ResolveRecognizerIntraOp(_lineWorkers);
        // Tail boost ceiling: CompiledModel clamps to 16 anyway. The NHWC
        // kernels are FMA-bound and the detector measured faster at 16 than
        // at 8 on Zen 3, so idle line-worker cores are worth feeding even
        // past the physical-core count.
        _recIntraOpMax = Math.Min(Environment.ProcessorCount, 16);
        _recognizer = new PaddleOcrRecognizer(recognizerModel ?? throw new ArgumentNullException(nameof(recognizerModel)),
            dictionaryUtf8, _options.Recognizer, ownsModel: false,
            _recIntraOpBase);
        _recGpu = OnnxSharp.OcrSessionFactory.IsGpuBackend(_options.Recognizer.Backend);
        _recBatchEffective = _options.HasExplicitRecBatchLines ? _options.RecBatchLines
            : _recGpu ? 16 : _options.RecBatchLines;
    }

    private static byte[] ReadDictionary(Stream source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        using MemoryStream buffer = new();
        source.CopyTo(buffer);
        if (buffer.Length == 0) throw new InvalidDataException("Dictionary stream is empty.");
        return buffer.ToArray();
    }

    /// <summary>Runs DET, perspective crop, optional CLS and CTC REC.</summary>
    public PaddleOcrResult Run(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
        => Run(source, sourceWidth, sourceHeight, returnCtcAlignment: false, sourceStride, format);

    /// <summary>
    /// Same as <see cref="Run(ReadOnlySpan{byte}, int, int, int, ImagePixelFormat)"/>.
    /// <paramref name="returnCtcAlignment"/> asks this call's greedy decode to
    /// keep one <see cref="PaddleOcrCtcSpan"/> per emitted token. It does not
    /// change the graph, the session, or the tensor shape. False leaves
    /// <see cref="PaddleOcrLine.CtcSpans"/> null and allocates no span array.
    /// </summary>
    public PaddleOcrResult Run(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        bool returnCtcAlignment, int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
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
            // machine and left the phase latency-bound.
            int lineWorkerCount = Math.Min(_lineWorkers, count);
            int cropWorkers = Math.Min(Math.Max(lineWorkerCount, _cropWorkers), count);
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
                    count, labels, clsScores);
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

            List<int[]> units;
            if (_recGpu)
            {
                // GPU: a whole-graph dispatch has fixed submit/plan overhead, so
                // tiny exact-width groups lose. Sort by width and chunk by
                // maxBatch; each unit runs at its max member width — right-side
                // zero padding only adds blank columns to the CTC output.
                int[] order = Enumerable.Range(0, count).ToArray();
                Array.Sort(order, (a, b) => recWidths[a].CompareTo(recWidths[b]));
                units = [];
                for (int s = 0; s < count; s += maxBatch)
                {
                    int len = Math.Min(s + maxBatch, count) - s;
                    int[] unit = new int[len];
                    Array.Copy(order, s, unit, 0, len);
                    int wMax = 0;
                    foreach (int li in unit) wMax = Math.Max(wMax, recWidths[li]);
                    foreach (int li in unit) recWidths[li] = wMax;
                    units.Add(unit);
                }
            }
            else
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
            if (workerCount <= 1 || units.Count <= 1)
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
        _ownedModels?.Dispose();
    }
}
