using System.Collections.Concurrent;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Pure managed two-class orientation classifier. Thread-safe: holds one shared
/// <see cref="CompiledModel"/> and pools per-call inference sessions, keeping no
/// mutable per-worker state. The input shape is fixed, so pooled sessions need
/// no reshaping.
/// </summary>
public sealed class PaddleOcrClassifier : IDisposable
{
    private const int InputWidth = 160, InputHeight = 80;
    private readonly CompiledModel _compiled;
    private readonly ConcurrentBag<IOcrSession> _sessions = [];
    private readonly bool _ownsModel;
    private readonly bool _gpuCapable;
    private int _pooledCount;
    private readonly PaddleOcrClassifierOptions _options;
    private bool _disposed;

    public PaddleOcrClassifier(Model model, PaddleOcrClassifierOptions? options = null)
        : this(model, options, ownsModel: false)
    {
    }

    /// <summary>Loads a classifier model from a stream without retaining the serialized payload.</summary>
    public PaddleOcrClassifier(Stream model, PaddleOcrClassifierOptions? options = null)
        : this(Model.Load(model ?? throw new ArgumentNullException(nameof(model))), options, ownsModel: true)
    {
    }

    private PaddleOcrClassifier(Model model, PaddleOcrClassifierOptions? options, bool ownsModel)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        try
        {
            _compiled = new CompiledModel(model, [1, 3, InputHeight, InputWidth], 1);
        }
        catch
        {
            if (ownsModel) model.Dispose();
            throw;
        }
        _ownsModel = ownsModel;
        _options = options ?? new PaddleOcrClassifierOptions();
        _gpuCapable = OcrSessionFactory.IsGpuBackend(_options.Backend);
        if (_options.MaxPooledSessions < 0)
        {
            _compiled.Dispose();
            if (ownsModel) model.Dispose();
            throw new ArgumentOutOfRangeException(nameof(options));
        }
    }

    public PaddleOcrClassificationResult Classify(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrClassifier));
        sourceStride = ImagePixels.ResolveStride(sourceWidth, sourceStride, format);
        if ((long)sourceWidth * sourceHeight > _options.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
        bool profile = PipelineProfiler.Enabled;
        long t = profile ? PipelineProfiler.Now() : 0;
        IOcrSession session = RentSession();
        if (profile) PipelineProfiler.Add(PipelineProfiler.ClsAcquire, t);
        try
        {
            // Pooled sessions may carry a padded batch shape from ClassifyBatch;
            // reshape back to the single-crop shape (no-op when already bound).
            session.Reshape([1, 3, InputHeight, InputWidth]);
            Span<float> input = session.InputData;
            long started = profile ? PipelineProfiler.Now() : 0;
            int resizedWidth = PPOCRPreprocess.Cls(source, sourceWidth, sourceHeight, sourceStride,
                input, session.ResizeWorkspace, session.InputIsNhwc, format);
            if (profile) PipelineProfiler.Add(PipelineProfiler.ClsPreprocess, started);
            started = profile ? PipelineProfiler.Now() : 0;
            if (session.TryRunUntilCtcProjection(input, out CtcProjectionOperands ops))
            {
                Span<uint> lab = stackalloc uint[1];
                Span<float> sc = stackalloc float[1];
                if (HeadFromOps(ops, 1, lab, sc))
                {
                    if (profile) PipelineProfiler.Add(PipelineProfiler.ClsGraph, started);
                    return new PaddleOcrClassificationResult(lab[0], sc[0], resizedWidth);
                }
            }
            ReadOnlySpan<float> output = session.RunInternal(input);
            if (profile) PipelineProfiler.Add(PipelineProfiler.ClsGraph, started);
            if (output.Length != 2 || !MathCompat.IsFinite(output[0]) || !MathCompat.IsFinite(output[1]))
                throw new InvalidDataException("Classifier output is invalid.");
            uint label = output[1] > output[0] ? 1u : 0u;
            return new PaddleOcrClassificationResult(label, output[(int)label], resizedWidth);
        }
        finally
        {
            long started = profile ? PipelineProfiler.Now() : 0;
            ReturnSession(session);
            if (profile) PipelineProfiler.Add(PipelineProfiler.ClsRelease, started);
        }
    }

    /// <summary>True when this classifier resolves to a GPU-backed session.</summary>
    internal bool GpuCapable => _gpuCapable;

    /// <summary>
    /// Classifies <paramref name="count"/> tightly-packed crops in one batched
    /// graph run. Input shape is fixed ([n,3,H,W]), so the whole batch shares a
    /// single submit; the tiny head matmul + softmax run on CPU via the same
    /// CTC-projection split the recognizer uses (the GPU skips the last nodes).
    /// </summary>
    internal void ClassifyBatch(byte[] cropBuffer, int[] offsets, int[] bytes,
        int[] widths, int[] heights, int count, uint[] labels, float[] scores)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrClassifier));
        const int sampleFloats = 3 * InputHeight * InputWidth;
        bool profile = PipelineProfiler.Enabled;
        long t = profile ? PipelineProfiler.Now() : 0;
        IOcrSession session = RentSession();
        if (profile) PipelineProfiler.Add(PipelineProfiler.ClsAcquire, t);
        long started = t;
        try
        {
            // Pad the batch to a multiple of 8 so per-shape execution plans stay
            // bounded across images with different line counts.
            int nPad = (count + 7) & ~7;
            session.Reshape([nPad, 3, InputHeight, InputWidth]);
            Span<float> input = session.InputData;
            for (int k = 0; k < count; k++)
            {
                PPOCRPreprocess.Cls(cropBuffer.AsSpan(offsets[k], bytes[k]), widths[k], heights[k],
                    ImagePixels.ResolveStride(widths[k], 0, ImagePixelFormat.Bgr24),
                    input.Slice(k * sampleFloats, sampleFloats), session.ResizeWorkspace,
                    session.InputIsNhwc, ImagePixelFormat.Bgr24);
            }
            input[(count * sampleFloats)..].Clear();   // pad rows → zero logits → [0.5,0.5], discarded
            if (profile) PipelineProfiler.Add(PipelineProfiler.ClsPreprocess, started);
            started = profile ? PipelineProfiler.Now() : 0;
            if (session.TryRunUntilCtcProjection(input, out CtcProjectionOperands ops) &&
                HeadFromOps(ops, count, labels, scores))
            {
                if (profile) PipelineProfiler.Add(PipelineProfiler.ClsGraph, started);
                return;
            }
            ReadOnlySpan<float> output = session.RunInternal(input);
            if (profile) PipelineProfiler.Add(PipelineProfiler.ClsGraph, started);
            for (int k = 0; k < count; k++)
            {
                float o0 = output[k * 2], o1 = output[k * 2 + 1];
                uint label = o1 > o0 ? 1u : 0u;
                labels[k] = label;
                scores[k] = label == 1 ? o1 : o0;
            }
        }
        finally
        {
            ReturnSession(session);
        }
    }

    // Per-row logits over the constant head weights, then softmax. cols==2 for
    // the shipping cls model; arbitrary small column counts work too.
    private static bool HeadFromOps(in CtcProjectionOperands ops, int count,
        Span<uint> labels, Span<float> scores)
    {
        int inner = ops.Inner, cols = ops.Columns;
        if (cols < 2 || cols > 64 || inner <= 0) return false;
        ReadOnlySpan<float> act = ops.Activations;
        ReadOnlySpan<float> w = ops.Weights;
        ReadOnlySpan<float> bias = ops.Bias;
        if (!bias.IsEmpty && bias.Length != cols) return false;
        if (w.Length < inner * cols) return false;
        int rows = Math.Min(ops.RowCount, count);
        Span<float> logits = stackalloc float[cols];
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> a = act.Slice(r * inner, inner);
            for (int j = 0; j < cols; j++)
            {
                float acc = bias.IsEmpty ? 0f : bias[j];
                for (int i = 0; i < inner; i++) acc += a[i] * w[i * cols + j];
                logits[j] = acc;
            }
            float m = logits[0];
            for (int j = 1; j < cols; j++) m = MathF.Max(m, logits[j]);
            float sum = 0f;
            for (int j = 0; j < cols; j++) sum += MathF.Exp(logits[j] - m);
            uint best = 0;
            for (int j = 1; j < cols; j++) if (logits[j] > logits[(int)best]) best = (uint)j;
            labels[r] = best;
            scores[r] = MathF.Exp(logits[(int)best] - m) / sum;
        }
        return true;
    }

    private IOcrSession RentSession()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrClassifier));
        if (_sessions.TryTake(out IOcrSession? session))
        {
            Interlocked.Decrement(ref _pooledCount);
            return session;
        }
        IOcrSession s = OcrSessionFactory.Create(_compiled, _options.Backend);
        s.PlanForCtcProjection = true;
        return s;
    }

    private void ReturnSession(IOcrSession session)
    {
        if (_disposed)
        {
            session.Dispose();
            return;
        }
        // Claim a pool slot without an O(n) ConcurrentBag.Count on the hot path;
        // when the pool is at the cap, back off and dispose instead of growing.
        int count = Interlocked.Increment(ref _pooledCount);
        if (count > _options.MaxPooledSessions)
        {
            Interlocked.Decrement(ref _pooledCount);
            session.Dispose();
            return;
        }
        _sessions.Add(session);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        while (_sessions.TryTake(out IOcrSession? session))
            session.Dispose();
        _compiled.Dispose();
        if (_ownsModel) _compiled.Model.Dispose();
    }
}
