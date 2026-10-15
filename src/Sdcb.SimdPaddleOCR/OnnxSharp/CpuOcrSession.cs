using System.Runtime.InteropServices;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// CPU <see cref="IOcrSession"/> over Sdcb.SimdOnnx plans. One grow-only native arena per session;
/// the graph input sits at offset 0 of every plan, so the full, head-truncated and logits plans of
/// the current shape all read the same caller-written input. Not thread-safe.
/// </summary>
internal sealed unsafe class CpuOcrSession : IOcrSession
{
    private enum Kind { Full, Ctc, Logits }

    private readonly CompiledModel _compiled;
    private readonly ResizeWorkspace _resizeWorkspace = new();
    private readonly InferenceSession?[] _sessions = new InferenceSession?[3];
    private int[] _shape = [];
    private int _volume, _highWater, _threads;
    private nint _arena;
    private long _arenaBytes;
    private bool _disposed;

    internal CpuOcrSession(CompiledModel compiled)
    {
        _compiled = compiled;
        _threads = compiled.IntraOpThreads;
        if (compiled.DefaultInputShape is { } shape) Reshape(shape);
    }

    public TensorShape InputShape => new(_shape);

    public TensorShape OutputShape
    {
        get
        {
            if (_sessions[(int)Kind.Full] is { } full) return new(full.OutputShape);
            if (_sessions[(int)Kind.Ctc] is { } ctc && _compiled.ResolveHead(_shape) is { } head)
            {
                int[] a = ctc.OutputShape;
                return new([.. a.AsSpan()[..^1], head.Columns]);
            }
            return new(Session(Kind.Full).OutputShape);
        }
    }

    public Span<float> InputData => _arena == 0 ? [] : new Span<float>((float*)_arena, _volume);
    public ResizeWorkspace ResizeWorkspace => _resizeWorkspace;
    public bool InputIsNhwc => _compiled.InputIsNhwc;
    public int IntraOpThreads { get => _threads; set => _threads = MathCompat.Clamp(value, 1, 16); }
    public int HighWaterInputVolume => _highWater;
    public bool PlanForCtcProjection { get; set; }
    public bool GpuAlive => true;
    public bool IsProfilingEnabled => EngineProfile.Enabled;

    public void Reshape(ReadOnlySpan<int> inputShape)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CpuOcrSession));
        if (inputShape.SequenceEqual(_shape) && _arena != 0) return;
        _shape = inputShape.ToArray();
        long volume = 1;
        foreach (int d in _shape) volume = checked(volume * d);
        _volume = checked((int)volume);
        _highWater = Math.Max(_highWater, _volume);
        Array.Clear(_sessions, 0, _sessions.Length);
        bool ctc = PlanForCtcProjection && _compiled.ResolveHead(_shape) is not null;
        Session(ctc ? Kind.Ctc : Kind.Full);
    }

    public ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input) => Run(Kind.Full, input).Output;

    public bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands)
    {
        operands = default;
        if (_compiled.ResolveHead(_shape) is not { } head) return false;
        InferenceSession s = Run(Kind.Ctc, input);
        int[] a = s.OutputShape;
        int rows = a[^2], inner = a[^1];
        if (inner != head.Inner) return false;
        int batch = checked((int)(s.Output.Length / ((long)rows * inner)));
        operands = new CtcProjectionOperands(s.Output, head.Projection, batch, rows, -1);
        return true;
    }

    public ReadOnlySpan<float> RunInternalSkipFinalSoftmax(ReadOnlySpan<float> input, out bool outputIsLogits)
    {
        _compiled.ResolveHead(_shape);
        outputIsLogits = _compiled.Logits is not null;
        return Run(outputIsLogits ? Kind.Logits : Kind.Full, input).Output;
    }

    public void NoteProfile(string operation, long started) =>
        EngineProfile.AddOp(operation, System.Diagnostics.Stopwatch.GetTimestamp() - started);

    private InferenceSession Run(Kind kind, ReadOnlySpan<float> input)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CpuOcrSession));
        if (input.Length != _volume) throw new ArgumentException("Input buffer size mismatch.", nameof(input));
        bool aliased = _arena != 0 && System.Runtime.CompilerServices.Unsafe.AreSame(
            ref MemoryMarshal.GetReference(input), ref *(float*)_arena);
        InferenceSession s = Session(kind);
        if (!aliased) input.CopyTo(InputData);
        s.Threads = _threads;
        s.Run();
        return s;
    }

    private InferenceSession Session(Kind kind)
    {
        if (_sessions[(int)kind] is { } existing) return existing;
        CpuModel model = kind switch
        {
            Kind.Ctc => _compiled.Ctc,
            Kind.Logits => _compiled.Logits,
            _ => _compiled.Full,
        } ?? throw new InvalidOperationException($"{kind} plan is unavailable for this model.");
        ExecPlan plan = model.Plan(_shape);
        if (plan.BufOffset[plan.Values[plan.InputValue].Buf] != 0)
            throw new InvalidOperationException("Plan input is not at arena offset 0.");
        long need = Math.Max(Math.Max((long)plan.ArenaFloats, _volume) * sizeof(float), 64);
        if (need > _arenaBytes) Grow(need);
        return _sessions[(int)kind] = new InferenceSession(plan, _arena);
    }

    // keeps the (possibly already written) input; sessions bound to the old arena are dropped
    private void Grow(long bytes)
    {
        nint next = (nint)NativeMemory.AllocZeroed((nuint)bytes);
        if (_arena != 0)
        {
            Buffer.MemoryCopy((void*)_arena, (void*)next, bytes, Math.Min(_arenaBytes, (long)_volume * sizeof(float)));
            NativeMemory.Free((void*)_arena);
            GC.RemoveMemoryPressure(_arenaBytes);
        }
        _arena = next;
        _arenaBytes = bytes;
        GC.AddMemoryPressure(bytes);
        Array.Clear(_sessions, 0, _sessions.Length);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Array.Clear(_sessions, 0, _sessions.Length);
        if (_arena != 0)
        {
            NativeMemory.Free((void*)_arena);
            GC.RemoveMemoryPressure(_arenaBytes);
            _arena = 0;
        }
    }
}
