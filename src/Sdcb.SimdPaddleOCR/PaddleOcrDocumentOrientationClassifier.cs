using System.Collections.Concurrent;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR;

/// <summary>
/// Four-way whole-page orientation classifier (0 / 90 / 180 / 270 degrees clockwise) for
/// PP-LCNet_x1_0_doc_ori, run before OCR. It is a standalone stage, not part of
/// <see cref="PaddleOcrAll"/>:
/// <code>
/// var o = classifier.Classify(pixels, width, height);
/// (int w, int h) = PaddleOcrDocumentRotation.RotatedSize(width, height, o.OrientationDegrees);
/// byte[] upright = new byte[w * h * 3];
/// PaddleOcrDocumentRotation.Rotate(pixels, width, height, 0, ImagePixelFormat.Bgr24, o.OrientationDegrees, upright);
/// PaddleOcrResult r = ocr.Run(upright, w, h);
/// // optional: lines back in source-image coordinates
/// var lines = r.Lines.Select(l => PaddleOcrDocumentRotation.ToSource(l, width, height, o.OrientationDegrees));
/// </code>
/// Thread-safe: one shared <see cref="CompiledModel"/>, pooled per-call sessions.
/// </summary>
public sealed class PaddleOcrDocumentOrientationClassifier : IDisposable
{
    private const int Size = PPOCRPreprocess.DocumentOrientationSize;
    private readonly CompiledModel _compiled;
    private readonly ConcurrentBag<IOcrSession> _sessions = [];
    private readonly PaddleOcrDocumentOrientationOptions _options;
    private readonly bool _ownsModel;
    private int _pooledCount;
    private bool _disposed;

    /// <summary>Uses a loaded four-class model; the caller owns <paramref name="model"/>.</summary>
    public PaddleOcrDocumentOrientationClassifier(Model model, PaddleOcrDocumentOrientationOptions? options = null)
        : this(model, options, ownsModel: false)
    {
    }

    /// <summary>
    /// Loads the model from the current position of <paramref name="model"/> (read forward once, e.g.
    /// <c>DocumentOrientationModel.OpenRead()</c>). The caller still disposes the stream.
    /// </summary>
    public PaddleOcrDocumentOrientationClassifier(Stream model, PaddleOcrDocumentOrientationOptions? options = null)
        : this(Model.Load(model ?? throw new ArgumentNullException(nameof(model))), options, ownsModel: true)
    {
    }

    private PaddleOcrDocumentOrientationClassifier(Model model, PaddleOcrDocumentOrientationOptions? options, bool ownsModel)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        _options = options ?? new PaddleOcrDocumentOrientationOptions();
        _ownsModel = ownsModel;
        try
        {
            if (_options.IntraOpThreads is < 0 or > 16 || _options.MaxPooledSessions < 0 || _options.MaxImagePixels < 1)
                throw new ArgumentOutOfRangeException(nameof(options));
            int threads = _options.IntraOpThreads == 0 ? Math.Min(Environment.ProcessorCount, 4) : _options.IntraOpThreads;
            _compiled = new CompiledModel(model, [1, 3, Size, Size], threads);
            // the two-class text-line model (0 / 180) cannot correct a page; reject it up front
            using IOcrSession probe = OcrSessionFactory.Create(_compiled, OcrBackend.Cpu);
            probe.Reshape([1, 3, Size, Size]);
            if (probe.OutputShape.ElementCount != 4)
                throw new ArgumentException("Document orientation needs a model with four output classes (0, 90, 180, 270).", nameof(model));
        }
        catch
        {
            _compiled?.Dispose();
            if (ownsModel) model.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Classifies the page orientation. <paramref name="stride"/> 0 means tightly packed rows.
    /// Equal scores keep the lower class (the usual argmax).
    /// </summary>
    public PaddleOcrDocumentOrientationResult Classify(ReadOnlySpan<byte> source, int width, int height,
        int stride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrDocumentOrientationClassifier));
        stride = ImagePixels.ResolveStride(width, stride, format);
        if ((long)width * height > _options.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
        IOcrSession session = RentSession();
        try
        {
            session.Reshape([1, 3, Size, Size]);
            bool profile = PipelineProfiler.Enabled;
            long started = profile ? PipelineProfiler.Now() : 0;
            PPOCRPreprocess.DocumentOrientation(source, width, height, stride, session.InputData, session.InputIsNhwc, format);
            if (profile) PipelineProfiler.Add(PipelineProfiler.DocumentPreprocess, started);
            started = profile ? PipelineProfiler.Now() : 0;
            ReadOnlySpan<float> scores = session.RunInternal(session.InputData);
            if (profile) PipelineProfiler.Add(PipelineProfiler.DocumentGraph, started);
            if (scores.Length != 4) throw new InvalidDataException("Expected four document orientation scores.");
            int label = 0;
            for (int i = 0; i < 4; i++)
            {
                if (!MathCompat.IsFinite(scores[i])) throw new InvalidDataException("Document orientation score is not finite.");
                if (scores[i] > scores[label]) label = i;
            }
            return new PaddleOcrDocumentOrientationResult((uint)label, scores[label]);
        }
        finally
        {
            ReturnSession(session);
        }
    }

    private IOcrSession RentSession()
    {
        if (_sessions.TryTake(out IOcrSession? session))
        {
            Interlocked.Decrement(ref _pooledCount);
            return session;
        }
        return OcrSessionFactory.Create(_compiled, _options.Backend);
    }

    private void ReturnSession(IOcrSession session)
    {
        if (_disposed)
        {
            session.Dispose();
            return;
        }
        if (Interlocked.Increment(ref _pooledCount) > _options.MaxPooledSessions)
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
