using System.Runtime.InteropServices;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// CPU <see cref="IOcrSession"/> over a Sdcb.SimdOnnx <see cref="CpuSession"/>: one grow-only arena
/// per session, and the full, head-truncated and logits variants of the current shape all read the
/// same caller-written input. Not thread-safe.
/// </summary>
internal sealed class CpuOcrSession : IOcrSession
{
    private enum Kind { Full, Ctc, Logits }

    private readonly CompiledModel _compiled;
    private readonly CpuSession _session;
    private readonly int[]?[] _outShapes = new int[]?[3];   // output shape of each variant bound for the current shape
    private int[] _shape = [];
    private int _highWater, _threads;
    private bool _disposed;

    internal CpuOcrSession(CompiledModel compiled)
    {
        _compiled = compiled;
        _session = new CpuSession(compiled.Full);
        _threads = compiled.IntraOpThreads;
        if (compiled.DefaultInputShape is { } shape) Reshape(shape);
    }

    public TensorShape InputShape => new(_shape);

    public TensorShape OutputShape
    {
        get
        {
            // from a plan this shape already compiled: building the full plan only to read its shape
            // would cost a compile (and arena) the CTC path never uses
            if (_outShapes[(int)Kind.Full] is { } full) return new(full);
            if (_outShapes[(int)Kind.Ctc] is { } a && _compiled.ResolveHead(_shape) is { } head)
            {
                return new([.. a.AsSpan()[..^1], head.Columns]);
            }
            return new(Bind(Kind.Full));
        }
    }

    public Span<float> InputData => _session.Input;
    public bool InputIsNhwc => _compiled.InputIsNhwc;
    public int IntraOpThreads { get => _threads; set => _threads = MathCompat.Clamp(value, 1, 16); }
    public int HighWaterInputVolume => _highWater;
    public bool PlanForCtcProjection { get; set; }
    public bool GpuAlive => true;
    public bool IsProfilingEnabled => EngineProfiler.Enabled;

    public void Reshape(ReadOnlySpan<int> inputShape)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CpuOcrSession));
        if (inputShape.SequenceEqual(_shape) && _session.Input.Length != 0) return;
        _shape = inputShape.ToArray();
        Kind kind = PlanForCtcProjection && _compiled.ResolveHead(_shape) is not null ? Kind.Ctc : Kind.Full;
        _session.Reshape(_shape, Variant(kind));   // compiles that plan and sizes the arena once
        _highWater = Math.Max(_highWater, _session.Input.Length);
        Array.Clear(_outShapes, 0, _outShapes.Length);
        Bind(kind);
    }

    public ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input) => Run(Kind.Full, input);

    public bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands)
    {
        operands = default;
        if (_compiled.ResolveHead(_shape) is not { } head) return false;
        ReadOnlySpan<float> output = Run(Kind.Ctc, input);
        int[] a = _outShapes[(int)Kind.Ctc]!;
        int rows = a[^2], inner = a[^1];
        if (inner != head.Inner) return false;
        int batch = checked((int)(output.Length / ((long)rows * inner)));
        operands = new CtcProjectionOperands(output, head.Projection, batch, rows, -1);
        return true;
    }

    public ReadOnlySpan<float> RunInternalSkipFinalSoftmax(ReadOnlySpan<float> input, out bool outputIsLogits)
    {
        _compiled.ResolveHead(_shape);
        outputIsLogits = _compiled.Logits is not null;
        return Run(outputIsLogits ? Kind.Logits : Kind.Full, input);
    }

    public void NoteProfile(string operation, long started) =>
        EngineProfiler.Record(operation, System.Diagnostics.Stopwatch.GetTimestamp() - started);

    private ReadOnlySpan<float> Run(Kind kind, ReadOnlySpan<float> input)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CpuOcrSession));
        if (input.Length != _session.Input.Length) throw new ArgumentException("Input buffer size mismatch.", nameof(input));
        // decided before binding: binding may grow the arena (moving the input, which it preserves),
        // after which a caller span over the old InputData points at freed memory
        bool aliased = System.Runtime.CompilerServices.Unsafe.AreSame(
            ref MemoryMarshal.GetReference(input), ref MemoryMarshal.GetReference(_session.Input));
        if (!aliased) input.CopyTo(_session.Input);
        Bind(kind);
        _session.Threads = _threads;
        return _session.Run(Variant(kind));
    }

    private int[] Bind(Kind kind) =>
        _outShapes[(int)kind] ??= _session.OutputShape(Variant(kind));

    private CpuModel Variant(Kind kind) => kind switch
    {
        Kind.Ctc => _compiled.Ctc,
        Kind.Logits => _compiled.Logits,
        _ => _compiled.Full,
    } ?? throw new InvalidOperationException($"{kind} plan is unavailable for this model.");

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _session.Dispose();
    }
}