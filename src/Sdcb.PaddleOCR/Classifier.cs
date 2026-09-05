using System.Collections.Concurrent;
using Sdcb.PaddleOCR.OnnxSharp;

namespace Sdcb.PaddleOCR;

/// <summary>
/// Pure managed two-class orientation classifier. Thread-safe: holds one shared
/// <see cref="CompiledModel"/> and pools per-call inference sessions, keeping no
/// mutable per-worker state. The input shape is fixed, so pooled sessions need
/// no reshaping.
/// </summary>
public sealed class Classifier : IDisposable
{
    private const int InputWidth = 160, InputHeight = 80;
    private readonly CompiledModel _compiled;
    private readonly ConcurrentBag<InferenceSession> _sessions = [];
    private readonly bool _ownsModel;
    private int _pooledCount;
    private readonly PaddleOcrClassifierOptions _options;
    private bool _disposed;

    public Classifier(Model model, PaddleOcrClassifierOptions? options = null)
        : this(model, options, ownsModel: false)
    {
    }

    /// <summary>Loads a classifier model from a stream without retaining the serialized payload.</summary>
    public Classifier(Stream model, PaddleOcrClassifierOptions? options = null)
        : this(Model.Load(model ?? throw new ArgumentNullException(nameof(model))), options, ownsModel: true)
    {
    }

    private Classifier(Model model, PaddleOcrClassifierOptions? options, bool ownsModel)
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
        if (_options.MaxPooledSessions < 0)
        {
            _compiled.Dispose();
            if (ownsModel) model.Dispose();
            throw new ArgumentOutOfRangeException(nameof(options));
        }
    }

    public PaddleOcrClassificationResult Classify(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride = 0)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(Classifier));
        if (sourceStride == 0) sourceStride = checked(sourceWidth * 3);
        if ((long)sourceWidth * sourceHeight > _options.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
        bool profile = PipelineProfiler.Enabled;
        long t = profile ? PipelineProfiler.Now() : 0;
        InferenceSession session = RentSession();
        if (profile) PipelineProfiler.Add(PipelineProfiler.ClsAcquire, t);
        try
        {
            Span<float> input = session.InputData;
            long started = profile ? PipelineProfiler.Now() : 0;
            int resizedWidth = PPOCRPreprocess.ClsBgrToNchw(source, sourceWidth, sourceHeight, sourceStride,
                input, session.ResizeWorkspace);
            if (profile) PipelineProfiler.Add(PipelineProfiler.ClsPreprocess, started);
            started = profile ? PipelineProfiler.Now() : 0;
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

    private InferenceSession RentSession()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(Classifier));
        if (_sessions.TryTake(out InferenceSession? session))
        {
            Interlocked.Decrement(ref _pooledCount);
            return session;
        }
        return _compiled.CreateRequest();
    }

    private void ReturnSession(InferenceSession session)
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
        while (_sessions.TryTake(out InferenceSession? session))
            session.Dispose();
        _compiled.Dispose();
        if (_ownsModel) _compiled.Model.Dispose();
    }
}
