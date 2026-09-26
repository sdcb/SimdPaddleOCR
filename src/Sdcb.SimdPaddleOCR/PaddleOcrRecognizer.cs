using System.Text;
using System.Buffers;
using System.Diagnostics;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR;

/// <summary>Pure managed PP-OCR CTC recognizer accepting BGR crop memory.</summary>
public sealed class PaddleOcrRecognizer : IDisposable
{
    private readonly Model _model;
    private readonly string[] _labels;
    private readonly int _maxLabelChars;
    private readonly PaddleOcrRecognizerOptions _options;
    private readonly CompiledModel _compiled;
    private readonly bool _ownsModel;
    private readonly List<IOcrSession> _sessions = [];
    private readonly object _poolLock = new();
    // Cleared the first time a GPU session reports it has fallen back to CPU
    // (emission/compile failure for this model). PaddleOcrAll reads it to stop
    // picking GPU-shaped (fewer, wider) line batches.
    private volatile bool _gpuRecAlive = true;
    internal bool GpuRecAlive => _gpuRecAlive;
    private int _pooledCount;
    private bool _disposed;

    public PaddleOcrRecognizer(Model model, ReadOnlySpan<byte> dictionaryUtf8, PaddleOcrRecognizerOptions? options = null)
        : this(model, dictionaryUtf8, options, ownsModel: false)
    {
    }

    public PaddleOcrRecognizer(Model model, Stream dictionaryUtf8, PaddleOcrRecognizerOptions? options = null)
        : this(model, ReadDictionary(dictionaryUtf8), options, ownsModel: false)
    {
    }

    /// <summary>Loads a recognizer model and dictionary from streams.</summary>
    public PaddleOcrRecognizer(Stream model, Stream dictionaryUtf8, PaddleOcrRecognizerOptions? options = null)
        : this(LoadOwned(model, dictionaryUtf8), options)
    {
    }

    /// <summary>Loads a recognizer model from a stream while accepting an in-memory dictionary.</summary>
    public PaddleOcrRecognizer(Stream model, ReadOnlySpan<byte> dictionaryUtf8,
        PaddleOcrRecognizerOptions? options = null)
        : this(Model.Load(model ?? throw new ArgumentNullException(nameof(model))),
            dictionaryUtf8, options, ownsModel: true)
    {
    }

    private PaddleOcrRecognizer((Model Model, byte[] Dictionary) loaded,
        PaddleOcrRecognizerOptions? options)
        : this(loaded.Model, loaded.Dictionary, options, ownsModel: true)
    {
    }

    private PaddleOcrRecognizer(Model model, ReadOnlySpan<byte> dictionaryUtf8,
        PaddleOcrRecognizerOptions? options, bool ownsModel)
        : this(model, dictionaryUtf8, options, ownsModel, intraOpThreads: 0)
    {
    }

    internal PaddleOcrRecognizer(Model model, ReadOnlySpan<byte> dictionaryUtf8,
        PaddleOcrRecognizerOptions? options, bool ownsModel, int intraOpThreads)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _ownsModel = ownsModel;
        try
        {
            _options = options ?? new PaddleOcrRecognizerOptions();
            if (_options.MaxPooledSessions < 0) throw new ArgumentOutOfRangeException(nameof(options));
            // Not a public knob: leftover cores after LineWorkerCount, cap 4.
            // Standalone Recognizer (intraOpThreads 0) assumes one in-flight
            // session. CLS stays intra-op 1; the graph is too small to shard.
            int intraOp = intraOpThreads > 0
                ? MathCompat.Clamp(intraOpThreads, 1, 16)
                : Parallelism.ResolveRecognizerIntraOp(1);
            _compiled = new CompiledModel(_model, intraOpThreads: intraOp);
            if (_options.TargetWidth <= 0 || _options.TargetWidth > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(options));
            string text;
            try { text = EncodingCompat.GetString(new UTF8Encoding(false, true), dictionaryUtf8); }
            catch (DecoderFallbackException ex) { throw new InvalidDataException("Dictionary is not valid UTF-8.", ex); }
            if (text.Length > 0 && text[0] == '\uFEFF') text = text.Substring(1);
            string[] raw = text.Split('\n');
            // C's dictionary reader does not treat the final line terminator as
            // an additional empty label. Keep the managed class numbering aligned
            // when the dictionary file ends with LF (or CRLF).
            if (raw.Length > 0 && raw[^1].Length == 0)
                raw = raw.Take(raw.Length - 1).ToArray();
            _labels = [.. raw.Select(x => x.EndsWith("\r", StringComparison.Ordinal) ? x.Substring(0, x.Length - 1) : x)];
            if (_labels.Length == 0 || !_labels.Any(x => x.Length != 0))
                throw new InvalidDataException("Dictionary is empty.");
            _maxLabelChars = _labels.Max(static x => x.Length);
        }
        catch
        {
            if (ownsModel) _model.Dispose();
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

    private static (Model Model, byte[] Dictionary) LoadOwned(Stream model, Stream dictionary)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        Model loaded = Model.Load(model);
        try
        {
            return (loaded, ReadDictionary(dictionary));
        }
        catch
        {
            loaded.Dispose();
            throw;
        }
    }

    public int ClassCount => checked(_labels.Length + 2);
    public int TargetWidth => _options.TargetWidth;

    public PaddleOcrRecognitionResult Recognize(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
        => Recognize(source, sourceWidth, sourceHeight, sourceStride, format, _compiled.IntraOpThreads, false);

    /// <summary>
    /// Same as <see cref="Recognize(ReadOnlySpan{byte}, int, int, int, ImagePixelFormat)"/>,
    /// and keeps one <see cref="PaddleOcrCtcSpan"/> per emitted token when
    /// <paramref name="returnCtcAlignment"/> is true. A span is the half-open
    /// run of that class: repeated frames are included, blank frames are excluded.
    /// The flag does not change the graph or the session. False allocates nothing
    /// beyond the text, and <see cref="PaddleOcrRecognitionResult.CtcSpans"/> stays null.
    /// </summary>
    public PaddleOcrRecognitionResult Recognize(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        bool returnCtcAlignment, int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
        => Recognize(source, sourceWidth, sourceHeight, sourceStride, format, _compiled.IntraOpThreads,
            returnCtcAlignment);

    /// <summary>
    /// Per-call intra-op override for the line stage: callers that own idle
    /// sibling workers (images whose line count is below the worker count)
    /// pass a larger budget so the session shards its convolutions across
    /// cores that would otherwise sit parked. Threading does not change any
    /// arithmetic, so results are identical at any budget.
    /// </summary>
    internal PaddleOcrRecognitionResult Recognize(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride, ImagePixelFormat format, int intraOpThreads, bool returnCtcAlignment)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrRecognizer));
        sourceStride = ImagePixels.ResolveStride(sourceWidth, sourceStride, format);
        if ((long)sourceWidth * sourceHeight > _options.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
        bool profile = PipelineProfiler.Enabled;
        long t = profile ? PipelineProfiler.Now() : 0;
        int targetWidth = SelectTargetWidth(sourceWidth, sourceHeight);
        if (profile) PipelineProfiler.Add(PipelineProfiler.RecCacheGet, t);
        t = profile ? PipelineProfiler.Now() : 0;
        IOcrSession session = RentSession(1, targetWidth);
        // Rented sessions may carry a boosted budget from a previous pooled
        // call; always rebind so the compiled default is the fallback.
        session.IntraOpThreads = intraOpThreads;
        if (profile) PipelineProfiler.Add(PipelineProfiler.RecRent, t);
        t = profile ? PipelineProfiler.Now() : 0;
        session.Reshape([1, 3, 48, targetWidth]);
        if (profile) PipelineProfiler.Add(PipelineProfiler.RecReshape, t);
        t = profile ? PipelineProfiler.Now() : 0;
        Span<float> inputSpan = session.InputData;
        if (profile) PipelineProfiler.Add(PipelineProfiler.RecPool, t);
        try
        {
            // Reuse the request's resize scratch; safe because this request is
            // exclusively held for the duration of the call.
            long started = profile ? PipelineProfiler.Now() : 0;
            int resizedWidth = PPOCRPreprocess.Rec(source, sourceWidth, sourceHeight, sourceStride,
                targetWidth, inputSpan, session.ResizeWorkspace, session.InputIsNhwc, format);
            if (profile) PipelineProfiler.Add(PipelineProfiler.RecPreprocess, started);
            started = profile ? PipelineProfiler.Now() : 0;
            CtcDecodeInput decoded = RunCtcGraph(session, inputSpan);
            if (profile) PipelineProfiler.Add(PipelineProfiler.RecGraph, started);
            try
            {
                int[] shape = session.OutputShape.Dimensions;
                if (shape.Length != 3 || shape[0] != 1 || shape[2] != ClassCount)
                    throw new InvalidDataException("Recognizer output shape is incompatible with dictionary.");
                int timeSteps = shape[1];
                if (decoded.IsCompact
                    ? decoded.Indices.Length != timeSteps || decoded.Scores.Length != timeSteps
                    : decoded.Dense.Length != checked(timeSteps * ClassCount))
                    throw new InvalidDataException("Recognizer output size mismatch.");
                started = profile ? PipelineProfiler.Now() : 0;
                PaddleOcrRecognitionResult result = Decode(decoded.Dense, decoded.Indices,
                    decoded.Scores, timeSteps, resizedWidth, targetWidth, decoded.DenseIsLogits,
                    returnCtcAlignment);
                if (profile) PipelineProfiler.Add(PipelineProfiler.RecDecode, started);
                return result;
            }
            finally { decoded.Dispose(); }
        }
        finally
        {
            long started = profile ? PipelineProfiler.Now() : 0;
            if (!session.GpuAlive) _gpuRecAlive = false;
            ReturnSession(session);
            if (profile) PipelineProfiler.Add(PipelineProfiler.RecRelease, started);
        }
    }

    private IOcrSession RentSession(int batch, int width)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrRecognizer));
        int volume = checked(batch * 3 * 48 * width);
        IOcrSession? session = TryTakeBestFit(volume);
        if (session is null)
        {
            session = OnnxSharp.OcrSessionFactory.Create(_compiled, _options.Backend);
            // RunCtcGraph stops before the vocab MatMul; do not plan the
            // [T×vocab] logits/softmax planes that path never writes.
            session.PlanForCtcProjection = true;
        }
        return session;
    }

    // ConcurrentBag handed whatever session was on top, so each of the ~8
    // line workers independently ratcheted to the widest line it had ever
    // seen. Same photo, next request: a different worker ate that max and
    // grew again. Best-fit keeps the fat buffer on the sessions that
    // already paid for it.
    private IOcrSession? TryTakeBestFit(int volume)
    {
        lock (_poolLock)
        {
            int count = _sessions.Count;
            if (count == 0) return null;
            int bestFit = -1, largest = 0;
            int bestFitWater = int.MaxValue, largestWater = -1;
            for (int i = 0; i < count; i++)
            {
                int water = _sessions[i].HighWaterInputVolume;
                if (water > largestWater)
                {
                    largestWater = water;
                    largest = i;
                }
                if (water >= volume && water < bestFitWater)
                {
                    bestFitWater = water;
                    bestFit = i;
                }
            }
            int take = bestFit >= 0 ? bestFit : largest;
            IOcrSession session = _sessions[take];
            int last = count - 1;
            if (take != last) _sessions[take] = _sessions[last];
            _sessions.RemoveAt(last);
            Interlocked.Decrement(ref _pooledCount);
            return session;
        }
    }

    internal int SelectWidthForCrop(int sourceWidth, int sourceHeight) =>
        SelectTargetWidth(sourceWidth, sourceHeight);

    /// <summary>
    /// Recognizes several same-target-width crops in one batched graph run.
    /// Every kernel iterates the batch dimension with per-sample math, so each
    /// line's result is bit-identical to a standalone <see cref="Recognize"/>.
    /// </summary>
    internal void RecognizeBatch(byte[] cropBuffer, int[] offsets, int[] cropBytes,
        int[] widths, int[] heights, ReadOnlySpan<int> lineIndices, int targetWidth,
        PaddleOcrRecognitionResult[] results, int intraOpThreads, bool returnCtcAlignment)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrRecognizer));
        int n = lineIndices.Length;
        bool profile = PipelineProfiler.Enabled;
        long t = profile ? PipelineProfiler.Now() : 0;
        IOcrSession session = RentSession(n, targetWidth);
        session.IntraOpThreads = intraOpThreads;
        if (profile) PipelineProfiler.Add(PipelineProfiler.RecRent, t);
        t = profile ? PipelineProfiler.Now() : 0;
        session.Reshape([n, 3, 48, targetWidth]);
        if (profile) PipelineProfiler.Add(PipelineProfiler.RecReshape, t);
        t = profile ? PipelineProfiler.Now() : 0;
        int sampleLength = checked(3 * 48 * targetWidth);
        Span<float> input = session.InputData;
        int[] resizedWidths = PooledArrays.Rent<int>(n);
        if (profile) PipelineProfiler.Add(PipelineProfiler.RecPool, t);
        try
        {
            long started = profile ? PipelineProfiler.Now() : 0;
            for (int k = 0; k < n; k++)
            {
                int line = lineIndices[k];
                int sourceWidth = widths[line], sourceHeight = heights[line];
                if ((long)sourceWidth * sourceHeight > _options.MaxImagePixels)
                    throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
                resizedWidths[k] = PPOCRPreprocess.Rec(
                    cropBuffer.AsSpan(offsets[line], cropBytes[line]), sourceWidth, sourceHeight,
                    checked(sourceWidth * 3), targetWidth,
                    input.Slice(k * sampleLength, sampleLength), session.ResizeWorkspace,
                    session.InputIsNhwc);
            }
            if (profile) PipelineProfiler.Add(PipelineProfiler.RecPreprocess, started);
            started = profile ? PipelineProfiler.Now() : 0;
            CtcDecodeInput decoded = RunCtcGraph(session, input);
            if (profile) PipelineProfiler.Add(PipelineProfiler.RecGraph, started);
            try
            {
                int[] shape = session.OutputShape.Dimensions;
                if (shape.Length != 3 || shape[0] != n || shape[2] != ClassCount)
                    throw new InvalidDataException("Recognizer output shape is incompatible with dictionary.");
                int timeSteps = shape[1];
                int sampleOutput = checked(timeSteps * ClassCount);
                if (decoded.IsCompact
                    ? decoded.Indices.Length != checked(n * timeSteps) ||
                      decoded.Scores.Length != checked(n * timeSteps)
                    : decoded.Dense.Length != checked(n * sampleOutput))
                    throw new InvalidDataException("Recognizer output size mismatch.");
                started = profile ? PipelineProfiler.Now() : 0;
                for (int k = 0; k < n; k++)
                    results[lineIndices[k]] = Decode(
                        decoded.IsCompact ? [] : decoded.Dense.Slice(k * sampleOutput, sampleOutput),
                        decoded.IsCompact ? decoded.Indices.Slice(k * timeSteps, timeSteps) : [],
                        decoded.IsCompact ? decoded.Scores.Slice(k * timeSteps, timeSteps) : [],
                        timeSteps, resizedWidths[k], targetWidth, decoded.DenseIsLogits, returnCtcAlignment);
                if (profile) PipelineProfiler.Add(PipelineProfiler.RecDecode, started);
            }
            finally { decoded.Dispose(); }
        }
        finally
        {
            long started = profile ? PipelineProfiler.Now() : 0;
            PooledArrays.Return(resizedWidths);
            if (!session.GpuAlive) _gpuRecAlive = false;
            ReturnSession(session);
            if (profile) PipelineProfiler.Add(PipelineProfiler.RecRelease, started);
        }
    }

    /// <summary>
    /// CTC graph + projection. Session stops before the vocab MatMul; this
    /// method owns compact ArgMax scratch via ArrayPool (Recognizer is
    /// concurrent across rented sessions, so instance fields are unsafe).
    /// </summary>
    private CtcDecodeInput RunCtcGraph(IOcrSession session, ReadOnlySpan<float> input)
    {
        if (session.TryRunUntilCtcProjection(input, out CtcProjectionOperands ops))
        {
            int rowCount = ops.RowCount;
            int[] indices = PooledArrays.Rent<int>(rowCount);
            float[] scores = PooledArrays.Rent<float>(rowCount);
            long matMulStarted = session.IsProfilingEnabled ? Stopwatch.GetTimestamp() : 0;
            if (MatMul.TryArgMax(ops.Activations, ops.Weights, ops.Bias,
                indices.AsSpan(0, rowCount), scores.AsSpan(0, rowCount),
                ops.Batch, ops.Rows, ops.Inner, ops.Columns, ops.PackedWeights,
                session.IntraOpThreads))
            {
                if (matMulStarted != 0)
                    session.NoteProfile(OperatorId.MatMul, matMulStarted, ops.MatMulNodeIndex);
                return CtcDecodeInput.Compact(indices, scores, rowCount);
            }
            PooledArrays.Return(indices);
            PooledArrays.Return(scores);
        }

        ReadOnlySpan<float> dense = session.RunInternalSkipFinalSoftmax(input, out bool logits);
        return CtcDecodeInput.FromDense(dense, logits);
    }

    private void ReturnSession(IOcrSession session)
    {
        lock (_poolLock)
        {
            if (_disposed || _sessions.Count >= _options.MaxPooledSessions)
            {
                session.Dispose();
                return;
            }
            _sessions.Add(session);
            _pooledCount = _sessions.Count;
        }
    }

    private PaddleOcrRecognitionResult Decode(ReadOnlySpan<float> dense,
        ReadOnlySpan<int> compactIndices, ReadOnlySpan<float> compactScores,
        int timeSteps, int resizedWidth, int tensorWidth, bool denseIsLogits, bool returnCtcAlignment)
        => DecodeCtc(_labels, _maxLabelChars, returnCtcAlignment,
            dense, compactIndices, compactScores, timeSteps, resizedWidth, tensorWidth, denseIsLogits);

    /// <summary>
    /// Greedy CTC decode. Each emitted token's span is the half-open run of
    /// that class: repeated frames are included, and blank frames are excluded.
    /// A run that reaches the last step keeps <paramref name="timeSteps"/> as
    /// its exclusive end.
    /// </summary>
    internal static PaddleOcrRecognitionResult DecodeCtc(string[] labels, int maxLabelChars, bool capture,
        ReadOnlySpan<float> dense, ReadOnlySpan<int> compactIndices, ReadOnlySpan<float> compactScores,
        int timeSteps, int resizedWidth, int tensorWidth, bool denseIsLogits)
    {
        bool compact = !compactIndices.IsEmpty;
        int classCount = checked(labels.Length + 2);
        int previous = 0, emitted = 0, textLength = 0, open = -1;
        double scoreSum = 0;
        // Rent scratch for this call so the recognizer stays thread-safe; only
        // the returned immutable string is allocated for the decoded text.
        // Alignment scratch is rented only when requested and holds no
        // references, so returning it does not keep dictionary strings alive.
        int requiredChars = checked(timeSteps * Math.Max(1, maxLabelChars));
        char[] rented = PooledArrays.Rent<char>(requiredChars);
        CtcRun[]? runs = capture && timeSteps > 0 ? PooledArrays.Rent<CtcRun>(timeSteps) : null;
        try
        {
            Span<char> textScratch = rented;
            for (int step = 0; step < timeSteps; step++)
            {
                ReadOnlySpan<float> rowValues = compact
                    ? []
                    : dense.Slice(step * classCount, classCount);
                int best;
                float bestValue;
                if (compact)
                {
                    best = compactIndices[step];
                    bestValue = compactScores[step];
                }
                else best = ArgMax.Find(rowValues, out bestValue);
                if (best != 0 && (step == 0 || best != previous))
                {
                    if (best == labels.Length + 1)
                        textScratch[textLength++] = ' ';
                    else
                    {
                        string label = labels[best - 1];
                        label.AsSpan().CopyTo(textScratch[textLength..]);
                        textLength += label.Length;
                    }
                    // Softmax is monotonic, so class selection used logits.
                    // Compute only the probability that contributes to the
                    // result score; blank and CTC-repeat rows need no exp at
                    // all.
                    float probability = compact
                        ? bestValue
                        : denseIsLogits
                        ? SimdKernels.SoftmaxMaximumProbability(rowValues, bestValue)
                        : bestValue;
                    scoreSum += probability;
                    if (runs is not null)
                    {
                        if (open >= 0) runs[open].End = step;
                        runs[emitted] = new CtcRun
                        {
                            Start = step,
                            End = timeSteps,
                            Label = best == labels.Length + 1 ? -1 : best - 1,
                            Score = probability
                        };
                        open = emitted;
                    }
                    emitted++;
                }
                else if (runs is not null && open >= 0 && best != previous)
                {
                    // A blank ends the run. The blank frame is excluded;
                    // repeated frames of this class stayed inside until here.
                    runs[open].End = step;
                    open = -1;
                }
                previous = best;
            }
            PaddleOcrCtcSpan[]? spans = null;
            if (capture) spans = emitted == 0 ? [] : MaterializeRuns(labels, runs!, emitted);
            return new PaddleOcrRecognitionResult(new string(rented, 0, textLength),
                emitted == 0 ? 0 : (float)(scoreSum / emitted),
                emitted, resizedWidth, timeSteps)
            {
                TensorWidth = tensorWidth,
                CtcSpans = spans
            };
        }
        finally
        {
            PooledArrays.Return(rented);
            PooledArrays.Return(runs);
        }
    }

    private static PaddleOcrCtcSpan[] MaterializeRuns(string[] labels, CtcRun[] runs, int emitted)
    {
        PaddleOcrCtcSpan[] spans = new PaddleOcrCtcSpan[emitted];
        for (int i = 0; i < emitted; i++)
        {
            CtcRun run = runs[i];
            string text = run.Label < 0 ? " " : labels[run.Label];
            spans[i] = new PaddleOcrCtcSpan(text, run.Score, run.Start, run.End);
        }
        return spans;
    }

    /// <summary>Pooled decode scratch. Label -1 is the CTC space token.</summary>
    private struct CtcRun
    {
        public int Start;
        public int End;
        public int Label;
        public float Score;
    }

    private int SelectTargetWidth(int sourceWidth, int sourceHeight)
    {
        if (!_options.AdaptiveWidth) return _options.TargetWidth;
        long scaled = ((long)48 * sourceWidth + sourceHeight - 1) / sourceHeight;
        // OpenVINO.NET's dynamic recognizer pads each batch to the next
        // multiple of 32. For the managed per-crop path, select the same
        // bucket. Adaptive mode deliberately has no 320-pixel upper cap:
        // long text lines retain their natural width instead of being
        // compressed into a fixed input tensor.
        long bucket = ((scaled + 31) / 32) * 32;
        bucket = Math.Max(bucket, 32L);
        return checked((int)bucket);
    }

    public void Dispose()
    {
        IOcrSession[] draining;
        lock (_poolLock)
        {
            if (_disposed) return;
            _disposed = true;
            draining = _sessions.ToArray();
            _sessions.Clear();
            _pooledCount = 0;
        }
        foreach (IOcrSession session in draining)
            session.Dispose();
        _compiled.Dispose();
        if (_ownsModel) _model.Dispose();
    }

    /// <summary>
    /// CTC decode view: either session dense logits/probs or compact ArgMax
    /// buffers rented for this call. Dispose returns rented arrays.
    /// </summary>
    private ref struct CtcDecodeInput
    {
        private int[]? _rentedIndices;
        private float[]? _rentedScores;
        private readonly int _compactLength;

        private CtcDecodeInput(ReadOnlySpan<float> dense, bool denseIsLogits)
        {
            Dense = dense;
            DenseIsLogits = denseIsLogits;
            _rentedIndices = null;
            _rentedScores = null;
            _compactLength = 0;
        }

        private CtcDecodeInput(int[] indices, float[] scores, int length)
        {
            Dense = [];
            DenseIsLogits = false;
            _rentedIndices = indices;
            _rentedScores = scores;
            _compactLength = length;
        }

        internal static CtcDecodeInput FromDense(ReadOnlySpan<float> dense, bool denseIsLogits)
            => new(dense, denseIsLogits);

        internal static CtcDecodeInput Compact(int[] indices, float[] scores, int length)
            => new(indices, scores, length);

        internal ReadOnlySpan<float> Dense { get; }
        internal ReadOnlySpan<int> Indices =>
            _rentedIndices is null ? [] : _rentedIndices.AsSpan(0, _compactLength);
        internal ReadOnlySpan<float> Scores =>
            _rentedScores is null ? [] : _rentedScores.AsSpan(0, _compactLength);
        internal bool DenseIsLogits { get; }
        internal bool IsCompact => _rentedIndices is not null;

        internal void Dispose()
        {
            if (_rentedIndices is not null)
            {
                PooledArrays.Return(_rentedIndices);
                _rentedIndices = null;
            }
            if (_rentedScores is not null)
            {
                PooledArrays.Return(_rentedScores);
                _rentedScores = null;
            }
        }
    }
}
