using System.Buffers;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

using Sdcb.PaddleOCR.OnnxSharp;
using Sdcb.PaddleOCR.ModelProvider;

namespace Sdcb.PaddleOCR;

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
    private readonly Detector _detector;
    private readonly Classifier? _classifier;
    private readonly Recognizer _recognizer;
    private readonly PaddleOcrOptions _options;
    private readonly int _lineWorkers;
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
        _detector = new Detector(detectorModel ?? throw new ArgumentNullException(nameof(detectorModel)), _options.Detector,
            ResolveDetectorIntraThreads(_options));
        _classifier = classifierModel is null ? null : new Classifier(classifierModel, _options.Classifier);
        _recognizer = new Recognizer(recognizerModel ?? throw new ArgumentNullException(nameof(recognizerModel)),
            dictionaryUtf8, _options.Recognizer, ownsModel: false,
            Parallelism.ResolveRecognizerIntraOp(_lineWorkers));
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
        int sourceStride = 0)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrAll));
        if (sourceStride == 0) sourceStride = checked(sourceWidth * 3);
        long stageStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
        PaddleOcrDetectionResult detection = _detector.Detect(source, sourceWidth, sourceHeight, sourceStride);
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
            // Crops write disjoint buffer ranges, so they parallelize freely
            // across the same budget as the line stage below.
            int lineWorkerCount = Math.Min(_lineWorkers, count);
            if (lineWorkerCount <= 1)
            {
                for (int i = 0; i < count; i++)
                {
                    PPOCRCrop.ExtractInto(source, sourceWidth, sourceHeight, sourceStride,
                        detection.Boxes[i], cropBuffer.AsSpan(cropOffsets[i], cropBytes[i]),
                        out cropWidths[i], out cropHeights[i]);
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
                        PaddleOcrDetectionBox[] boxes = detection.Boxes;
                        Parallel.For(0, lineWorkerCount, worker =>
                            CropRange(worker, count, lineWorkerCount, sourceAddress, sourceLength,
                                sourceWidth, sourceHeight, sourceStride, boxes, cropTarget,
                                offsets, bytes, widths, heights));
                    }
                }
            }
            if (pipelineProfile) PipelineProfiler.Add(PipelineProfiler.Crop, pipelineStarted);
            if (s_profileEnabled) AddProfile(1, stageStart);

            PaddleOcrLine[] lines = new PaddleOcrLine[count];
            stageStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
            pipelineStarted = pipelineProfile ? PipelineProfiler.Now() : 0;
            int workerCount = lineWorkerCount;
            // Same-width batched REC is opt-in: it is numerically exact, but
            // per-op whole-batch execution inflates the activation working set
            // and measured slower than per-line REC on desktop CPUs.
            int maxBatch = Math.Max(1, _options.RecBatchLines);
            if (maxBatch > 1)
            {
                ProcessLinesBatched(count, workerCount, maxBatch, cropBuffer, cropOffsets,
                    cropBytes, cropWidths, cropHeights, detection.Boxes, lines);
            }
            else if (workerCount <= 1)
            {
                ProcessRange(0, count, 1, 0, cropBuffer, cropOffsets, cropBytes,
                    cropWidths, cropHeights, detection.Boxes, lines);
            }
            else
            {
                // Stride partition: crop sizes vary widely, so interleaving line
                // indices across workers balances load better than contiguous
                // ranges. Each worker touches a disjoint set of lines, so the
                // shared (read-only here) crop workspace and the result array
                // need no synchronization.
                Parallel.For(0, workerCount, worker =>
                    ProcessRange(worker, count, workerCount, worker, cropBuffer, cropOffsets,
                        cropBytes, cropWidths, cropHeights, detection.Boxes, lines));
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
        PaddleOcrLine[] lines)
    {
        uint[] labels = PooledArrays.Rent<uint>(count);
        float[] clsScores = PooledArrays.Rent<float>(count);
        int[] rotations = PooledArrays.Rent<int>(count);
        int[] recWidths = PooledArrays.Rent<int>(count);
        PaddleOcrRecognitionResult[] recResults = new PaddleOcrRecognitionResult[count];
        try
        {
            if (workerCount <= 1)
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

            Dictionary<int, List<int>> groups = [];
            List<int[]> units = [];
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

            if (workerCount <= 1 || units.Count <= 1)
            {
                foreach (int[] unit in units)
                    RecognizeUnit(unit, cropBuffer, offsets, bytes, widths, heights, recWidths, recResults);
            }
            else
            {
                Parallel.For(0, Math.Min(workerCount, units.Count), worker =>
                {
                    for (int u = worker; u < units.Count; u += workerCount)
                        RecognizeUnit(units[u], cropBuffer, offsets, bytes, widths, heights,
                            recWidths, recResults);
                });
            }

            for (int i = 0; i < count; i++)
            {
                lines[i] = new PaddleOcrLine
                {
                    Box = boxes[i],
                    Text = recResults[i].Text,
                    RecognitionScore = recResults[i].Score,
                    ClassificationScore = clsScores[i],
                    ClassificationLabel = labels[i],
                    AppliedRotationDegrees = rotations[i],
                    EmittedCount = (uint)recResults[i].EmittedCount
                };
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

    private void RecognizeUnit(int[] unit, byte[] cropBuffer, int[] offsets, int[] bytes,
        int[] widths, int[] heights, int[] recWidths, PaddleOcrRecognitionResult[] recResults)
    {
        long recStart = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
        _recognizer.RecognizeBatch(cropBuffer, offsets, bytes, widths, heights, unit,
            recWidths[unit[0]], recResults);
        if (s_profileEnabled) AddProfile(4, recStart);
    }

    private void ClassifyRange(int first, int count, int stride, byte[] cropBuffer, int[] offsets,
        int[] bytes, int[] widths, int[] heights, uint[] labels, float[] clsScores,
        int[] rotations, int[] recWidths)
    {
        Classifier? classifier = _classifier;
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

    private static unsafe void CropRange(int first, int count, int stride, nint sourceAddress,
        int sourceLength, int sourceWidth, int sourceHeight, int sourceStride,
        PaddleOcrDetectionBox[] boxes, byte[] cropBuffer, int[] offsets, int[] bytes, int[] widths,
        int[] heights)
    {
        ReadOnlySpan<byte> source = new((void*)sourceAddress, sourceLength);
        for (int i = first; i < count; i += stride)
            PPOCRCrop.ExtractInto(source, sourceWidth, sourceHeight, sourceStride,
                boxes[i], cropBuffer.AsSpan(offsets[i], bytes[i]), out widths[i], out heights[i]);
    }

    private void ProcessRange(int first, int count, int stride, int worker, byte[] cropBuffer,
        int[] offsets, int[] bytes, int[] widths, int[] heights, PaddleOcrDetectionBox[] boxes,
        PaddleOcrLine[] lines)
    {
        Classifier? classifier = _classifier;
        Recognizer recognizer = _recognizer;
        for (int i = first; i < count; i += stride)
            ProcessOne(i, classifier, recognizer, cropBuffer, offsets, bytes, widths, heights,
                boxes, lines);
    }

    private void ProcessOne(int i, Classifier? classifier, Recognizer recognizer, byte[] cropBuffer,
        int[] offsets, int[] bytes, int[] widths, int[] heights, PaddleOcrDetectionBox[] boxes,
        PaddleOcrLine[] lines)
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
        PaddleOcrRecognitionResult recognition = recognizer.Recognize(crop, widths[i], heights[i]);
        if (s_profileEnabled) AddProfile(4, recStart);
        lines[i] = new PaddleOcrLine
        {
            Box = boxes[i],
            Text = recognition.Text,
            RecognitionScore = recognition.Score,
            ClassificationScore = clsScore,
            ClassificationLabel = label,
            AppliedRotationDegrees = rotation,
            EmittedCount = (uint)recognition.EmittedCount
        };
    }

    // DET runs in an exclusive window before crop and line workers start, so
    // it always receives the same auto budget (up to 8 threads) regardless of
    // LineWorkerCount. Dividing DET by line workers used to starve DET on
    // smaller CPUs and did not protect the later line stage. auto>8 on
    // 20-thread Zen 5 inflated rec_pool and did not improve e2e; keep 8 unless
    // DetIntraOpThreads is set explicitly.
    private static int ResolveDetectorIntraThreads(PaddleOcrOptions options)
    {
        if (options.DetIntraOpThreads > 0) return Math.Min(options.DetIntraOpThreads, 16);
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
