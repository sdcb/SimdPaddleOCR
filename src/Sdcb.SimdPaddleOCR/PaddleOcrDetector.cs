using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR;

/// <summary>Pure managed DB detector. The caller supplies packed BGR bytes.</summary>
public sealed class PaddleOcrDetector : IDisposable
{
    private static bool s_profileEnabled;
    private static readonly long[] s_profileTicks = new long[3];
    private readonly Model _model;
    private readonly PaddleOcrDetectorOptions _options;
    private readonly int _intraOpThreads;
    private readonly CompiledModel _compiled;
    private readonly ConcurrentBag<InferenceSession> _sessions = [];
    private readonly ConcurrentBag<DbPostprocess.Workspace> _postprocess = [];
    private readonly int _postprocessPixels;
    private readonly bool _ownsModel;
    private int _pooledCount;
    private bool _disposed;

    public PaddleOcrDetector(Model model, PaddleOcrDetectorOptions? options = null, int intraOpThreads = 1)
        : this(model, options, intraOpThreads, ownsModel: false)
    {
    }

    /// <summary>Loads a detector model from a stream without retaining the serialized payload.</summary>
    public PaddleOcrDetector(Stream model, PaddleOcrDetectorOptions? options = null, int intraOpThreads = 1)
        : this(Model.Load(model ?? throw new ArgumentNullException(nameof(model))), options, intraOpThreads, ownsModel: true)
    {
    }

    private PaddleOcrDetector(Model model, PaddleOcrDetectorOptions? options, int intraOpThreads, bool ownsModel)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _ownsModel = ownsModel;
        try
        {
            _options = ResolveOptions(model, options);
            if (_options.MaxSessionCacheEntries < 0) throw new ArgumentOutOfRangeException(nameof(options));
            if (_options.MaxPooledSessions < 0) throw new ArgumentOutOfRangeException(nameof(options));
            _intraOpThreads = MathCompat.Clamp(intraOpThreads, 1, 16);
            // One shape-agnostic compiled model; pooled sessions are reshaped per image.
            _compiled = new CompiledModel(_model, _intraOpThreads);
            if (_options.LimitSideLength is < 32 or > 4096) throw new ArgumentOutOfRangeException(nameof(options));
            _postprocessPixels = checked(_options.LimitSideLength * _options.LimitSideLength);
            if (_options.MaxCandidates is <= 0 or > 10_000) throw new ArgumentOutOfRangeException(nameof(options));
            if (_options.BitmapThreshold is < 0 or > 1 || _options.BoxThreshold is < 0 or > 1 ||
                !MathCompat.IsFinite(_options.BitmapThreshold) || !MathCompat.IsFinite(_options.BoxThreshold) ||
                _options.UnclipRatio <= 0 || _options.UnclipRatio > 10 || !MathCompat.IsFinite(_options.UnclipRatio))
                throw new ArgumentOutOfRangeException(nameof(options));
        }
        catch
        {
            if (ownsModel) _model.Dispose();
            throw;
        }
    }

    private static PaddleOcrDetectorOptions ResolveOptions(Model model, PaddleOcrDetectorOptions? supplied)
    {
        PaddleOcrDetectorOptions source = supplied ?? new PaddleOcrDetectorOptions();
        // PaddleOCR's PP-OCRv6 configs use 0.4 for tiny and 0.45 for
        // small/medium.  Model.Load intentionally does not retain a source
        // filename, so use the first convolution's output width as a stable
        // architecture hint (tiny=16, small=24, medium=64).  An explicitly
        // supplied BoxThreshold always wins, including for non-v6 models.
        float boxThreshold = source.HasExplicitBoxThreshold
            ? source.BoxThreshold
            : IsTinyDetector(model) ? 0.4f : source.BoxThreshold;
        if (source.HasExplicitBoxThreshold && ReferenceEquals(source, supplied))
            return source;
        return new PaddleOcrDetectorOptions
        {
            MaxSessionCacheEntries = source.MaxSessionCacheEntries,
            MaxPooledSessions = source.MaxPooledSessions,
            LimitSideLength = source.LimitSideLength,
            MaxCandidates = source.MaxCandidates,
            UseDilation = source.UseDilation,
            BitmapThreshold = source.BitmapThreshold,
            BoxThreshold = boxThreshold,
            UnclipRatio = source.UnclipRatio,
            MaxImagePixels = source.MaxImagePixels
        };
    }

    private static bool IsTinyDetector(Model model)
    {
        foreach (NodeRecord node in model.Nodes)
        {
            if (node.Operator != OperatorId.Conv || node.Outputs.Length == 0) continue;
            IReadOnlyList<int> shape = model.GetTensorShape(node.Outputs[0]);
            return shape.Count >= 2 && shape[1] <= 16;
        }
        return false;
    }

    public PaddleOcrDetectionResult Detect(ReadOnlySpan<byte> source, int sourceWidth, int sourceHeight,
        int sourceStride = 0)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrDetector));
        if (sourceStride == 0) sourceStride = checked(sourceWidth * 3);
        int originalWidth = sourceWidth, originalHeight = sourceHeight;
        if ((long)sourceWidth * sourceHeight > _options.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
        bool pipelineProfile = PipelineProfiler.Enabled;
        byte[]? paddedSource = null;
        if (sourceWidth + (long)sourceHeight < 64)
        {
            int paddedWidth = Math.Max(32, sourceWidth);
            int paddedHeight = Math.Max(32, sourceHeight);
            paddedSource = PooledArrays.Rent<byte>(checked(paddedWidth * paddedHeight * 3));
            Span<byte> padded = paddedSource.AsSpan(0, checked(paddedWidth * paddedHeight * 3));
            padded.Clear();
            for (int y = 0; y < sourceHeight; y++)
            {
                source.Slice(y * sourceStride, checked(sourceWidth * 3))
                    .CopyTo(padded.Slice(y * paddedWidth * 3, checked(sourceWidth * 3)));
            }
            source = padded;
            sourceWidth = paddedWidth;
            sourceHeight = paddedHeight;
            sourceStride = checked(paddedWidth * 3);
        }

        try
        {
            (int Width, int Height, float WidthRatio, float HeightRatio) size = PPOCRPreprocess.ComputeDetSize(sourceWidth, sourceHeight, _options.LimitSideLength);
            InferenceSession session = RentSession();
            DbPostprocess.Workspace postprocess = RentPostprocess();
            session.Reshape([1, 3, size.Height, size.Width]);
            Span<float> inputSpan = session.InputData;
            try
            {
                long started = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
                long pipelineStarted = pipelineProfile ? PipelineProfiler.Now() : 0;
                PPOCRPreprocess.DetBgrToNchw(source, sourceWidth, sourceHeight, sourceStride,
                    size.Width, size.Height, inputSpan, session.ResizeWorkspace);
                if (s_profileEnabled) AddProfile(0, started);
                if (pipelineProfile) PipelineProfiler.Add(PipelineProfiler.DetPreprocess, pipelineStarted);
                started = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
                pipelineStarted = pipelineProfile ? PipelineProfiler.Now() : 0;
                ReadOnlySpan<float> probabilities = session.RunInternal(inputSpan);
                if (s_profileEnabled) AddProfile(1, started);
                if (pipelineProfile) PipelineProfiler.Add(PipelineProfiler.DetGraph, pipelineStarted);
                started = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
                pipelineStarted = pipelineProfile ? PipelineProfiler.Now() : 0;
                PaddleOcrDetectionBox[] boxes = DbPostprocess.Run(probabilities, size.Width, size.Height, _options,
                    originalWidth, originalHeight, size.WidthRatio, size.HeightRatio, postprocess);
                if (s_profileEnabled) AddProfile(2, started);
                if (pipelineProfile) PipelineProfiler.Add(PipelineProfiler.DetPostprocess, pipelineStarted);
                return new PaddleOcrDetectionResult
                {
                    Boxes = boxes,
                    ResizedWidth = size.Width,
                    ResizedHeight = size.Height,
                    WidthRatio = size.WidthRatio,
                    HeightRatio = size.HeightRatio
                };
            }
            finally
            {
                ReturnSession(session);
                ReturnPostprocess(postprocess);
            }
        }
        finally
        {
            if (paddedSource is not null) PooledArrays.Return(paddedSource);
        }
    }

    private InferenceSession RentSession()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrDetector));
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

    private DbPostprocess.Workspace RentPostprocess()
    {
        if (!_postprocess.TryTake(out DbPostprocess.Workspace? scratch))
            scratch = new DbPostprocess.Workspace();
        scratch.Ensure(_postprocessPixels);
        return scratch;
    }

    private void ReturnPostprocess(DbPostprocess.Workspace scratch)
    {
        if (_disposed) return;
        _postprocess.Add(scratch);
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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        while (_sessions.TryTake(out InferenceSession? session))
            session.Dispose();
        _compiled.Dispose();
        if (_ownsModel) _model.Dispose();
    }
}
