using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends;

/// <summary>
/// GPU-backed <see cref="IOcrSession"/> shared by the Vulkan and Metal
/// backends: the whole ONNX graph runs through an <see cref="IOcrGraphRunner"/>.
/// InputData is a managed staging span uploaded on each run; the CTC tail
/// matches <see cref="InferenceSession"/>'s contract (stop before the vocab
/// MatMul, hand operands to the shared CPU ArgMax). One device is shared
/// process-wide; weights/pipelines/schedules are shared per model, while each
/// session owns its buffers, so sessions only serialize at the submissions.
/// </summary>
internal abstract class GpuSessionBase : IOcrSession, IBatchedCtcSession
{
    private readonly IOcrGraphRunner _runner;
    private readonly string _logTag;
    private readonly CompiledModel _compiled;
    private readonly Model _model;
    private readonly ResizeWorkspace _resizeWorkspace = new();
    private InferenceSession? _cpuFallback;
    private float[] _cpuNhwc = [];   // NCHW→NHWC staging for the CPU fallback

    private int[] _shape = [];
    private float[] _input = [];
    private int _curVolume;    // numel of the current shape; _input may be bigger
    private int _hwVolume;
    private bool _disposed;
    private bool _gpuDead;   // plan/run failure → serve everything from _cpuFallback

    // CTC projection resolution, cached per Reshape
    private bool _ctcResolved;
    private int _ctcMatMulIndex = -1;
    private int _ctcActTensor;
    private int _ctcBatch, _ctcRows, _ctcInner, _ctcColumns;
    private byte[]? _ctcWBytes, _ctcBiasBytes;
    private float[]? _ctcPacked;

    private CtcHead? _ctcHead;
    // Cache compact output/projection metadata only. Sessions are leased exclusively,
    // so the bounded FIFO cache does not need synchronization.
    private readonly Dictionary<(int Batch, int Channels, int Height, int Width), CtcShapePlan> _ctcPlans = new();
    private readonly Queue<(int Batch, int Channels, int Height, int Width)> _ctcPlanOrder = new();
    private const int MaxCtcShapePlans = 128;
    private sealed class CtcShapePlan
    {
        public required int[] OutputShape;
        public CtcHead? Head;
        public int ActivationTensor, Batch, Rows;
    }
    private protected GpuSessionBase(IOcrGraphRunner runner, CompiledModel compiled, string logTag)
    {
        _runner = runner;
        _compiled = compiled;
        _model = compiled.Model;
        _logTag = logTag;
    }

    public TensorShape InputShape => new(_shape);
    private int[]? _outputShape;
    public TensorShape OutputShape =>
        new(_outputShape ??= GetCtcShapePlan(_shape).OutputShape);
    public Span<float> InputData => _input.AsSpan(0, _curVolume);
    public ResizeWorkspace ResizeWorkspace => _resizeWorkspace;
    public bool InputIsNhwc => false;   // GPU conv kernels read NCHW fp32 directly
    public int IntraOpThreads { get; set; }
    public bool GpuAlive => !_gpuDead;
    public int HighWaterInputVolume => _hwVolume;
    public bool PlanForCtcProjection { get; set; }
    public bool IsProfilingEnabled => InferenceSession.ProfilingEnabled;

    public void Reshape(ReadOnlySpan<int> inputShape)
    {
        if (_disposed) throw new ObjectDisposedException(GetType().Name);
        if (inputShape.SequenceEqual(_shape)) return;
        _shape = inputShape.ToArray();
        long vol = 1;
        foreach (int d in _shape) vol *= Math.Max(d, 1);
        _curVolume = checked((int)vol);
        if (vol > _input.Length)
        {
            _input = new float[vol];
            _hwVolume = checked((int)Math.Min(vol, int.MaxValue));
        }
        _outputShape = null;
        _ctcResolved = false;
        _ctcMatMulIndex = -1;
    }

    public ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input)
    {
        if (_disposed) throw new ObjectDisposedException(GetType().Name);
        if (_gpuDead) return Cpu().RunInternal(CpuInput(input));
        try
        {
            return _runner.Run(_shape, input);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[{_logTag}] RunInternal fallback: {ex.GetType().Name} {ex.Message}");
            _gpuDead = true;
            return Cpu().RunInternal(CpuInput(input));
        }
    }

    public bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands)
    {
        operands = default;
        if (_disposed) throw new ObjectDisposedException(GetType().Name);
        if (_gpuDead) return Cpu().TryRunUntilCtcProjection(CpuInput(input), out operands);
        ResolveCtcProjection();
        if (_ctcMatMulIndex < 0) return false;

        ReadOnlySpan<float> act;
        try
        {
            act = _runner.Run(_shape, input, _ctcMatMulIndex, _ctcActTensor);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[{_logTag}] CtcProj fallback: {ex.GetType().Name} {ex.Message}");
            _gpuDead = true;
            return Cpu().TryRunUntilCtcProjection(CpuInput(input), out operands);
        }
        operands = new CtcProjectionOperands(act,
            MemoryMarshal.Cast<byte, float>(_ctcWBytes),
            _ctcBiasBytes is null ? [] : MemoryMarshal.Cast<byte, float>(_ctcBiasBytes),
            _ctcPacked, _ctcBatch, _ctcRows, _ctcInner, _ctcColumns, _ctcMatMulIndex);
        return true;
    }

    // ---- IBatchedCtcSession: several REC shapes, one GPU submission ----
    private int[][] _many = [];
    private int _manyVolume;

    public bool CanRunMany => !_gpuDead && !_disposed;

    public Span<float> ReshapeMany(IReadOnlyList<int[]> shapes)
    {
        if (_disposed) throw new ObjectDisposedException(GetType().Name);
        _many = [.. shapes];
        long vol = 0;
        foreach (int[] s in _many)
        {
            long v = 1;
            foreach (int d in s) v *= Math.Max(d, 1);
            vol += v;
        }
        _manyVolume = checked((int)vol);
        if (vol > _input.Length)
        {
            _input = new float[vol];
            _hwVolume = _manyVolume;
        }
        // single-shape members (CTC resolution, fallback) key off the first unit
        Reshape(_many[0]);
        return _input.AsSpan(0, _manyVolume);
    }

    public bool TryResolveManyHead(out int[] rows, out CtcHead head)
    {
        rows = []; head = null!;
        if (!CanRunMany || _many.Length == 0) return false;
        ResolveCtcProjection();
        if (_ctcMatMulIndex < 0) return false;
        rows = new int[_many.Length];
        for (int i = 0; i < _many.Length; i++)
        {
            CtcShapePlan plan = GetCtcShapePlan(_many[i]);
            if (plan.Head is null) return false;
            rows[i] = plan.Rows;
        }
        head = _ctcHead!;
        return true;
    }

    public bool TryRunManyUntilCtcProjection(CtcUnitsReady onReady)
    {
        if (!CanRunMany || _many.Length == 0) return false;
        ResolveCtcProjection();
        if (_ctcMatMulIndex < 0) return false;
        // consumer failures are the caller's, not a GPU fault: keep them out
        // of the fallback catch below and rethrow once the device is drained
        System.Runtime.ExceptionServices.ExceptionDispatchInfo? consumerError = null;
        bool Forward(float[] acts, int[] offsets, int first, int count)
        {
            try { return onReady(acts, offsets, first, count); }
            catch (Exception ex)
            {
                consumerError = System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex);
                return false;
            }
        }
        bool ok;
        try
        {
            ok = _runner.RunMany(_many, _input.AsSpan(0, _manyVolume), _ctcMatMulIndex, _ctcActTensor, Forward);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[{_logTag}] RunMany fallback: {ex.GetType().Name} {ex.Message}");
            _gpuDead = true;
            return false;
        }
        consumerError?.Throw();
        return ok;
    }

    private InferenceSession Cpu()
    {
        _cpuFallback ??= new InferenceSession(_compiled)
        { PlanForCtcProjection = PlanForCtcProjection };
        _cpuFallback.Reshape(_shape);
        if (IntraOpThreads > 0) _cpuFallback.IntraOpThreads = IntraOpThreads;
        return _cpuFallback;
    }

    /// <summary>Graphs without the CTC tail fall back to a lazy CPU session.</summary>
    public ReadOnlySpan<float> RunInternalSkipFinalSoftmax(ReadOnlySpan<float> input, out bool outputIsLogits)
    {
        if (_disposed) throw new ObjectDisposedException(GetType().Name);
        return Cpu().RunInternalSkipFinalSoftmax(CpuInput(input), out outputIsLogits);
    }

    /// <summary>
    /// Callers always write logical NCHW (this session reports
    /// <see cref="InputIsNhwc"/> = false). When the compiled model flags its
    /// graph input as NHWC-physical, the CPU fallback must be fed the
    /// transposed layout or it decodes garbage.
    /// </summary>
    private ReadOnlySpan<float> CpuInput(ReadOnlySpan<float> input)
    {
        if (!_compiled.InputIsNhwc) return input;
        if (_cpuNhwc.Length < input.Length) _cpuNhwc = new float[input.Length];
        Span<float> dest = _cpuNhwc.AsSpan(0, input.Length);
        Kernels.Nhwc.NchwToNhwc(input, dest, _shape[0], _shape[1],
            checked(_shape[2] * _shape[3]), Math.Max(IntraOpThreads, 1));
        return dest;
    }

    public void NoteProfile(OperatorId operation, long started, int nodeIndex)
        => InferenceSession.NoteProfileStatic(operation, started, nodeIndex);

    /// <summary>
    /// Mirrors <c>InferenceSession.TryResolveCtcProjection</c> but resolves
    /// shapes from <see cref="CompiledModel.ResolveShapesFor"/> instead of live
    /// tensor bindings — the GPU session has no CPU workspace.
    /// </summary>
    private void ResolveCtcProjection()
    {
        if (_ctcResolved) return;
        CtcShapePlan plan = GetCtcShapePlan(_shape);
        _ctcResolved = true;
        _outputShape = plan.OutputShape;
        _ctcHead = plan.Head;
        _ctcMatMulIndex = plan.Head?.MatMulIndex ?? -1;
        _ctcActTensor = plan.ActivationTensor;
        _ctcBatch = plan.Batch;
        _ctcRows = plan.Rows;
        _ctcInner = plan.Head?.Inner ?? 0;
        _ctcColumns = plan.Head?.Columns ?? 0;
        _ctcWBytes = plan.Head?.Weights;
        _ctcBiasBytes = plan.Head?.Bias;
        _ctcPacked = plan.Head?.Packed;
    }
    private CtcShapePlan GetCtcShapePlan(int[] inputShape)
    {
        if (inputShape.Length != 4) return BuildCtcShapePlan(inputShape);
        var key = (inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
        if (_ctcPlans.TryGetValue(key, out CtcShapePlan? cached)) return cached;
        CtcShapePlan plan = BuildCtcShapePlan(inputShape);
        if (_ctcPlans.Count >= MaxCtcShapePlans)
            _ctcPlans.Remove(_ctcPlanOrder.Dequeue());
        _ctcPlans.Add(key, plan);
        _ctcPlanOrder.Enqueue(key);
        return plan;
    }
    private CtcShapePlan BuildCtcShapePlan(int[] inputShape)
    {
        int[][] shapes = _compiled.ResolveShapesFor(inputShape);
        int outIdx = checked((int)_model.GraphOutputs[0]);
        var result = new CtcShapePlan { OutputShape = shapes[outIdx] };
        NodeRecord[] nodes = _model.Nodes;
        if (nodes.Length < 2) return result;
        NodeRecord softmax = nodes[^1];
        if (softmax.Operator != OperatorId.Softmax ||
            softmax.Inputs.Length != 1 || softmax.Outputs.Length != 1 ||
            softmax.Outputs[0] != (uint)outIdx) return result;
        int axis = I32(_model.GetParameters(softmax), 4);
        int[] smInShape = shapes[softmax.Inputs[0]];
        if (axis < 0) axis += smInShape.Length;
        if (axis != smInShape.Length - 1) return result;
        int terminalIndex = nodes.Length - 2;
        NodeRecord terminal = nodes[terminalIndex];
        int matMulIndex;
        byte[]? biasBytes = null;
        if (terminal.Operator == OperatorId.MatMul &&
            terminal.Outputs[0] == softmax.Inputs[0])
        {
            matMulIndex = terminalIndex;
        }
        else if (terminal.Operator == OperatorId.Add && terminal.Inputs.Length == 2 &&
            terminalIndex > 0 && terminal.Outputs[0] == softmax.Inputs[0])
        {
            matMulIndex = terminalIndex - 1;
            NodeRecord mm = nodes[matMulIndex];
            if (mm.Operator != OperatorId.MatMul || mm.Outputs.Length != 1) return result;
            uint biasIndex;
            if (terminal.Inputs[0] == mm.Outputs[0]) biasIndex = terminal.Inputs[1];
            else if (terminal.Inputs[1] == mm.Outputs[0]) biasIndex = terminal.Inputs[0];
            else return result;
            var biasMeta = _compiled.GetTensor(checked((int)biasIndex));
            int[] projShape = shapes[mm.Outputs[0]];
            if (!biasMeta.IsConstant || biasMeta.Shape.Length != 1 ||
                projShape.Length < 2 || biasMeta.Shape[0] != projShape[^1]) return result;
            biasBytes = biasMeta.Constant;
        }
        else return result;
        NodeRecord matMul = nodes[matMulIndex];
        if (matMul.Inputs.Length != 2 || matMul.Outputs.Length != 1) return result;
        var aMeta = _compiled.GetTensor(checked((int)matMul.Inputs[0]));
        var bMeta = _compiled.GetTensor(checked((int)matMul.Inputs[1]));
        int[] aShape = shapes[matMul.Inputs[0]];
        int[] bShape = shapes[matMul.Inputs[1]];
        int[] projOut = shapes[matMul.Outputs[0]];
        if (!bMeta.IsConstant || aShape.Length < 2 || bShape.Length != 2 ||
            projOut.Length < 2) return result;
        int rows = aShape[^2], inner = aShape[^1], columns = bShape[1];
        long aCount = 1;
        foreach (int d in aShape) aCount *= Math.Max(d, 1);
        if (rows <= 0 || inner <= 0 || aCount % (rows * (long)inner) != 0) return result;
        int batch = checked((int)(aCount / (rows * (long)inner)));
        if (bShape[0] != inner) return result;
        long projCount = 1;
        foreach (int d in projOut) projCount *= Math.Max(d, 1);
        if (projOut[^1] != columns || projCount != (long)batch * rows * columns) return result;
        _compiled.TryGetPackedMatMul(matMul.Inputs[1], out float[]? packed);
        // Column count does not gate here (unlike the recognizer's ArgMax
        // fusion): small-column heads like the cls [inner,2] tail resolve too.
        if (columns <= 0) return result;
        result.ActivationTensor = checked((int)matMul.Inputs[0]);
        result.Batch = batch;
        result.Rows = rows;
        result.Head = new CtcHead
        {
            Weights = bMeta.Constant, Bias = biasBytes, Packed = packed,
            Inner = inner, Columns = columns, MatMulIndex = matMulIndex,
        };
        return result;
    }

    private static int I32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadInt32LittleEndian(p[o..]);

    public void Dispose()
    {
        _disposed = true;
        _runner.Dispose();
        _cpuFallback?.Dispose();
    }
}
