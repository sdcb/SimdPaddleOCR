using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

/// <summary>
/// GPU-backed <see cref="IOcrSession"/>: the whole ONNX graph runs on Vulkan
/// via <see cref="GpuDetGraph"/>. InputData is a managed staging span uploaded
/// on each run; the CTC tail matches <see cref="InferenceSession"/>'s contract
/// (stop before the vocab MatMul, hand operands to the shared CPU ArgMax).
/// One VkDevice is shared process-wide; per-session GPU work is serialized by
/// a device-wide lock (the queue serializes anyway).
/// </summary>
internal sealed class GpuSession : IOcrSession
{
    private readonly VkDevice _dev;
    private readonly GpuDetGraph _graph;
    private readonly CompiledModel _compiled;
    private readonly Model _model;
    private readonly ResizeWorkspace _resizeWorkspace = new();
    private InferenceSession? _cpuFallback;

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

    internal GpuSession(VkDevice dev, CompiledModel compiled)
    {
        _dev = dev;
        _compiled = compiled;
        _model = compiled.Model;
        _graph = new GpuDetGraph(dev, compiled);
    }

    public TensorShape InputShape => new(_shape);
    public TensorShape OutputShape =>
        new(_compiled.ResolveShapesFor(_shape)[checked((int)_model.GraphOutputs[0])]);
    public Span<float> InputData => _input.AsSpan(0, _curVolume);
    public ResizeWorkspace ResizeWorkspace => _resizeWorkspace;
    public bool InputIsNhwc => false;   // convk_f32n reads NCHW fp32 directly
    public int IntraOpThreads { get; set; }
    public int HighWaterInputVolume => _hwVolume;
    public bool PlanForCtcProjection { get; set; }
    public bool IsProfilingEnabled => InferenceSession.ProfilingEnabled;

    public void Reshape(ReadOnlySpan<int> inputShape)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuSession));
        _shape = inputShape.ToArray();
        long vol = 1;
        foreach (int d in _shape) vol *= Math.Max(d, 1);
        _curVolume = checked((int)vol);
        if (vol > _input.Length)
        {
            _input = new float[vol];
            _hwVolume = checked((int)Math.Min(vol, int.MaxValue));
        }
        _ctcResolved = false;
        _ctcMatMulIndex = -1;
    }

    public ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuSession));
        if (_gpuDead) return Cpu().RunInternal(input);
        try
        {
            lock (_dev.Sync)
                return _graph.Run(_shape, input);
        }
        catch
        {
            _gpuDead = true;
            return Cpu().RunInternal(input);
        }
    }

    public bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands)
    {
        operands = default;
        if (_disposed) throw new ObjectDisposedException(nameof(GpuSession));
        if (_gpuDead) return Cpu().TryRunUntilCtcProjection(input, out operands);
        ResolveCtcProjection();
        if (_ctcMatMulIndex < 0) return false;

        float[] act;
        try
        {
            lock (_dev.Sync)
                act = _graph.Run(_shape, input, _ctcMatMulIndex, _ctcActTensor);
        }
        catch
        {
            _gpuDead = true;
            return Cpu().TryRunUntilCtcProjection(input, out operands);
        }
        operands = new CtcProjectionOperands(act,
            MemoryMarshal.Cast<byte, float>(_ctcWBytes),
            _ctcBiasBytes is null ? [] : MemoryMarshal.Cast<byte, float>(_ctcBiasBytes),
            _ctcPacked, _ctcBatch, _ctcRows, _ctcInner, _ctcColumns, _ctcMatMulIndex);
        return true;
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
        if (_disposed) throw new ObjectDisposedException(nameof(GpuSession));
        return Cpu().RunInternalSkipFinalSoftmax(input, out outputIsLogits);
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
        _ctcResolved = true;
        _ctcMatMulIndex = -1;

        NodeRecord[] nodes = _model.Nodes;
        if (nodes.Length < 2) return;
        int[][] shapes = _compiled.ResolveShapesFor(_shape);
        int outIdx = checked((int)_model.GraphOutputs[0]);

        NodeRecord softmax = nodes[^1];
        if (softmax.Operator != OperatorId.Softmax ||
            softmax.Inputs.Length != 1 || softmax.Outputs.Length != 1 ||
            softmax.Outputs[0] != (uint)outIdx) return;
        int axis = I32(_model.GetParameters(softmax), 4);
        int[] smInShape = shapes[softmax.Inputs[0]];
        if (axis < 0) axis += smInShape.Length;
        if (axis != smInShape.Length - 1) return;

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
            if (mm.Operator != OperatorId.MatMul || mm.Outputs.Length != 1) return;
            uint biasIndex;
            if (terminal.Inputs[0] == mm.Outputs[0]) biasIndex = terminal.Inputs[1];
            else if (terminal.Inputs[1] == mm.Outputs[0]) biasIndex = terminal.Inputs[0];
            else return;
            var biasMeta = _compiled.GetTensor(checked((int)biasIndex));
            int[] projShape = shapes[mm.Outputs[0]];
            if (!biasMeta.IsConstant || biasMeta.Shape.Length != 1 ||
                projShape.Length < 2 || biasMeta.Shape[0] != projShape[^1]) return;
            biasBytes = biasMeta.Constant;
        }
        else return;

        NodeRecord matMul = nodes[matMulIndex];
        if (matMul.Inputs.Length != 2 || matMul.Outputs.Length != 1) return;
        var aMeta = _compiled.GetTensor(checked((int)matMul.Inputs[0]));
        var bMeta = _compiled.GetTensor(checked((int)matMul.Inputs[1]));
        int[] aShape = shapes[matMul.Inputs[0]];
        int[] bShape = shapes[matMul.Inputs[1]];
        int[] projOut = shapes[matMul.Outputs[0]];
        if (!bMeta.IsConstant || aShape.Length < 2 || bShape.Length != 2 ||
            projOut.Length < 2) return;
        int rows = aShape[^2], inner = aShape[^1], columns = bShape[1];
        long aCount = 1;
        foreach (int d in aShape) aCount *= Math.Max(d, 1);
        if (rows <= 0 || inner <= 0 || aCount % (rows * (long)inner) != 0) return;
        int batch = checked((int)(aCount / (rows * (long)inner)));
        if (bShape[0] != inner) return;
        long projCount = 1;
        foreach (int d in projOut) projCount *= Math.Max(d, 1);
        if (projOut[^1] != columns || projCount != (long)batch * rows * columns) return;
        _compiled.TryGetPackedMatMul(matMul.Inputs[1], out float[]? packed);
        if (!Kernels.MatMul.CanFuseArgMax(rows, inner, columns, packed)) return;

        _ctcMatMulIndex = matMulIndex;
        _ctcActTensor = checked((int)matMul.Inputs[0]);
        _ctcBatch = batch; _ctcRows = rows; _ctcInner = inner; _ctcColumns = columns;
        _ctcWBytes = bMeta.Constant;
        _ctcBiasBytes = biasBytes;
        _ctcPacked = packed;
    }

    private static int I32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadInt32LittleEndian(p[o..]);

    public void Dispose()
    {
        _disposed = true;
        _graph.Dispose();
        _cpuFallback?.Dispose();
    }
}
