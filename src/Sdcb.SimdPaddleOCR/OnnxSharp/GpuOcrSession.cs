#if NET10_0_OR_GREATER
using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// GPU <see cref="IOcrSession"/> over Sdcb.SimdOnnx's Vulkan / Metal graphs: per-shape compiled
/// graphs (shared process-wide by the backend, weights in one per-model const store) and per-session
/// device buffers. The CTC / logits variants stop the graph at the projection input (the recognizer
/// and classifier run the head on CPU). Any GPU failure falls back to <see cref="CpuOcrSession"/>
/// for the rest of the session's life (<see cref="GpuAlive"/> turns false). Not thread-safe.
/// </summary>
internal sealed class GpuOcrSession : IOcrSession, IBatchedCtcSession
{
    private enum Kind { Full, Ctc, Logits }

    /// <summary>One compiled shape + output variant on the device.</summary>
    private interface IExec : IDisposable
    {
        Span<float> Input { get; }
        ReadOnlySpan<float> Output { get; }
        int[] OutputShape { get; }
        /// <summary>False when the session's borrowed buffers were regrown (recreate it).</summary>
        bool Current { get; }
        void Run();
        void Submit();
        void Wait();
    }

    private sealed class VkExec(VkInferenceSession s) : IExec
    {
        public Span<float> Input => s.Input;
        public ReadOnlySpan<float> Output => s.Output;
        public int[] OutputShape => s.OutputShape;
        public bool Current => s.BuffersCurrent;
        public void Run() => s.Run();
        public void Submit() => s.Submit();
        public void Wait() => s.Wait();
        public void Dispose() => s.Dispose();
    }

    private sealed class MtlExec(MtlInferenceSession s) : IExec
    {
        public Span<float> Input => s.Input;
        public ReadOnlySpan<float> Output => s.Output;
        public int[] OutputShape => s.OutputShape;
        public bool Current => true;
        public void Run() => s.Run();
        public void Submit() => s.Run();
        public void Wait() { }
        public void Dispose() => s.Dispose();
    }

    // recent shape × variant sessions (REC widths cycle within an image); on Vulkan they share one
    // grow-only arena / staging set, so each costs only a command buffer
    private const int MaxExecs = 32;
    private VkSessionBuffers? _vkBuffers;

    private readonly CompiledModel _compiled;
    private readonly bool _metal;
    private readonly ResizeWorkspace _resizeWorkspace = new();
    private readonly List<(string key, IExec exec)> _execs = [];
    private float[] _input = [];
    private int[] _shape = [];
    private int _volume, _highWater, _threads;
    private CpuOcrSession? _cpu;
    private bool _disposed;

    internal GpuOcrSession(CompiledModel compiled, bool metal)
    {
        _compiled = compiled;
        _metal = metal;
        _threads = compiled.IntraOpThreads;
        if (compiled.DefaultInputShape is { } shape) Reshape(shape);
    }

    public TensorShape InputShape => new(_shape);

    public TensorShape OutputShape
    {
        get
        {
            if (_cpu is not null) return _cpu.OutputShape;
            try { return new(Exec(Kind.Full).OutputShape); }
            catch (Exception ex) { return Fallback(ex).OutputShape; }
        }
    }

    public Span<float> InputData => _cpu is not null ? _cpu.InputData : _input.AsSpan(0, _volume);
    public ResizeWorkspace ResizeWorkspace => _resizeWorkspace;
    public bool InputIsNhwc => _compiled.InputIsNhwc;
    public int IntraOpThreads
    {
        get => _threads;
        set
        {
            _threads = MathCompat.Clamp(value, 1, 16);
            if (_cpu is not null) _cpu.IntraOpThreads = _threads;
        }
    }
    public int HighWaterInputVolume => _highWater;
    public bool PlanForCtcProjection { get; set; }
    public bool GpuAlive => _cpu is null;
    public bool IsProfilingEnabled => EngineProfile.Enabled;

    public void Reshape(ReadOnlySpan<int> inputShape)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuOcrSession));
        if (_cpu is not null)
        {
            _cpu.Reshape(inputShape);
            _shape = inputShape.ToArray();
            return;
        }
        if (inputShape.SequenceEqual(_shape) && _input.Length != 0) return;
        _shape = inputShape.ToArray();
        long volume = 1;
        foreach (int d in _shape) volume = checked(volume * d);
        _volume = checked((int)volume);
        _highWater = Math.Max(_highWater, _volume);
        if (_input.Length < _volume) _input = new float[_volume];
    }

    public ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input)
    {
        if (_cpu is not null) return _cpu.RunInternal(input);
        try { return Run(Kind.Full, input).Output; }
        catch (Exception ex) { return Fallback(ex).RunInternal(input); }
    }

    public bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands)
    {
        if (_cpu is not null) return _cpu.TryRunUntilCtcProjection(input, out operands);
        operands = default;
        if (_compiled.ResolveHead(_shape) is not { } head || _compiled.Ctc is null) return false;
        IExec e;
        try { e = Run(Kind.Ctc, input); }
        catch (Exception ex) { return Fallback(ex).TryRunUntilCtcProjection(input, out operands); }
        int[] a = e.OutputShape;
        int rows = a[^2], inner = a[^1];
        if (inner != head.Inner) return false;
        int batch = checked((int)(e.Output.Length / ((long)rows * inner)));
        operands = new CtcProjectionOperands(e.Output, head.Weights, head.Bias ?? [], head.Packed,
            batch, rows, inner, head.Columns, -1);
        return true;
    }

    public ReadOnlySpan<float> RunInternalSkipFinalSoftmax(ReadOnlySpan<float> input, out bool outputIsLogits)
    {
        if (_cpu is not null) return _cpu.RunInternalSkipFinalSoftmax(input, out outputIsLogits);
        _compiled.ResolveHead(_shape);
        outputIsLogits = _compiled.Logits is not null;
        try { return Run(outputIsLogits ? Kind.Logits : Kind.Full, input).Output; }
        catch (Exception ex) { return Fallback(ex).RunInternalSkipFinalSoftmax(input, out outputIsLogits); }
    }

    // ---- IBatchedCtcSession: every line group (unit) of an image through the CTC graph ----
    private int[][] _units = [];
    private float[] _unitInput = [];
    private CtcHead? _ctcHead;

    public bool CanRunMany => _cpu is null && _compiled.Ctc is not null;

    public Span<float> ReshapeMany(IReadOnlyList<int[]> shapes)
    {
        long total = 0;
        foreach (int[] sh in shapes)
        {
            long v = 1;
            foreach (int d in sh) v *= d;
            total += v;
        }
        _units = [.. shapes.Select(x => x.ToArray())];
        if (_unitInput.Length < total) _unitInput = new float[checked((int)total)];
        return _unitInput.AsSpan(0, (int)total);
    }

    public bool TryResolveManyHead(out int[] rows, out CtcHead head)
    {
        rows = [];
        head = null!;
        if (_units.Length == 0 || _compiled.ResolveHead(_units[0]) is not { } h) return false;
        head = _ctcHead ??= new CtcHead
        {
            Weights = System.Runtime.InteropServices.MemoryMarshal.AsBytes(h.Weights.AsSpan()).ToArray(),
            Bias = h.Bias is null ? null : System.Runtime.InteropServices.MemoryMarshal.AsBytes(h.Bias.AsSpan()).ToArray(),
            Packed = h.Packed,
            Inner = h.Inner,
            Columns = h.Columns,
            MatMulIndex = -1,
        };
        rows = new int[_units.Length];
        try
        {
            for (int u = 0; u < _units.Length; u++)
            {
                Reshape(_units[u]);
                int[] o = Exec(Kind.Ctc).OutputShape;
                if (o[^1] != h.Inner) return false;
                rows[u] = o[^2];
            }
        }
        catch (Exception ex)
        {
            Fallback(ex);
            return false;
        }
        return true;
    }

    public bool TryRunManyUntilCtcProjection(CtcUnitsReady onReady)
    {
        if (_cpu is not null || _ctcHead is null) return false;
        int n = _units.Length, inner = _ctcHead.Inner;
        int[] offsets = new int[n];
        long[] inOff = new long[n + 1];
        int totalActs = 0;
        var outs = new List<int>(n);
        for (int u = 0; u < n; u++)
        {
            long v = 1;
            foreach (int d in _units[u]) v *= d;
            inOff[u + 1] = inOff[u] + v;
        }
        float[] acts = [];
        try
        {
            // unit u+1 runs on the device while the caller's CPU head (MatMul + ArgMax) consumes unit u
            IExec Submit(int u)
            {
                Reshape(_units[u]);
                IExec e = Exec(Kind.Ctc);
                _unitInput.AsSpan((int)inOff[u], (int)(inOff[u + 1] - inOff[u])).CopyTo(e.Input);
                e.Submit();
                return e;
            }
            IExec cur = Submit(0);
            for (int u = 0; u < n; u++)
            {
                cur.Wait();
                ReadOnlySpan<float> o = cur.Output;
                if (acts.Length < totalActs + o.Length) Array.Resize(ref acts, Math.Max(totalActs + o.Length, acts.Length * 2));
                offsets[u] = totalActs;
                o.CopyTo(acts.AsSpan(totalActs));
                totalActs += o.Length;
                if (o.Length % inner != 0) return false;
                if (u + 1 < n) cur = Submit(u + 1);
                if (!onReady(acts, offsets, u, 1))
                {
                    if (u + 1 < n) cur.Wait();
                    return false;
                }
            }
        }
        catch (Exception ex)
        {
            Fallback(ex);
            return false;
        }
        return true;
    }

    public void NoteProfile(string operation, long started) =>
        EngineProfile.AddOp(operation, System.Diagnostics.Stopwatch.GetTimestamp() - started);

    private IExec Run(Kind kind, ReadOnlySpan<float> input)
    {
        if (input.Length != _volume) throw new ArgumentException("Input buffer size mismatch.", nameof(input));
        IExec e = Exec(kind);
        input.CopyTo(e.Input);
        long t0 = EngineProfile.Enabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        e.Run();
        if (EngineProfile.Enabled) EngineProfile.AddOp(_metal ? "MetalRun" : "VulkanRun", System.Diagnostics.Stopwatch.GetTimestamp() - t0);
        return e;
    }

    private IExec Exec(Kind kind)
    {
        string key = $"{kind}:{string.Join("x", _shape)}";
        for (int i = 0; i < _execs.Count; i++)
            if (_execs[i].key == key)
            {
                var hit = _execs[i];
                _execs.RemoveAt(i);
                if (!hit.exec.Current)
                {
                    hit.exec.Dispose();
                    break;
                }
                _execs.Add(hit);
                return hit.exec;
            }
        string? output = kind switch
        {
            Kind.Ctc => _compiled.ResolveHead(_shape)?.ActivationName,
            Kind.Logits => LogitsName(),
            _ => null,
        };
        CompileOptions opt = CompiledModel.Options with { OutputName = output };
        IExec exec = _metal
            ? new MtlExec(MetalBackend.Default.NewSession(_compiled.Model.Onnx, _shape, opt))
            : new VkExec(VkBackend.Default.NewSession(_compiled.Model.Onnx, _shape, opt,
                _vkBuffers ??= new VkSessionBuffers(VkBackend.Default.Dev)));
        _execs.Add((key, exec));
        if (_execs.Count > MaxExecs)
        {
            _execs[0].exec.Dispose();
            _execs.RemoveAt(0);
        }
        return exec;
    }

    private string? LogitsName() => _compiled.Logits?.Options.OutputName;

    /// <summary>Permanent CPU fallback; the caller's input (already in InputData) carries over.</summary>
    private CpuOcrSession Fallback(Exception ex)
    {
        if (_cpu is null)
        {
            System.Diagnostics.Debug.WriteLine($"SimdPaddleOCR: {(_metal ? "Metal" : "Vulkan")} failed, falling back to CPU: {ex}");
            var cpu = new CpuOcrSession(_compiled) { PlanForCtcProjection = PlanForCtcProjection, IntraOpThreads = _threads };
            if (_shape.Length != 0)
            {
                cpu.Reshape(_shape);
                _input.AsSpan(0, _volume).CopyTo(cpu.InputData);
            }
            _cpu = cpu;
            foreach (var (_, e) in _execs)
                try { e.Dispose(); } catch { /* device may be lost */ }
            _execs.Clear();
        }
        return _cpu;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var (_, e) in _execs)
            try { e.Dispose(); } catch { /* device may be lost */ }
        _execs.Clear();
        try { _vkBuffers?.Dispose(); } catch { /* device may be lost */ }
        _cpu?.Dispose();
    }
}
#endif
