using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// One inference request bound to a <see cref="CompiledModel"/>. Holds the
/// per-request resolved shapes and activation workspace; not thread-safe.
/// Create one per in-flight inference (or pool/reuse across sequential calls),
/// but never run two inferences through the same instance concurrently. Call
/// <see cref="Reshape"/> to retarget the session to a new input shape without
/// recompiling the model.
/// </summary>
public sealed class InferenceSession : IDisposable
{
    private static bool s_profileEnabled;
    private static readonly long[] s_profileTicks = new long[26];
    private static readonly long[] s_profileCalls = new long[26];
    private static readonly long[] s_profileConvClassTicks = new long[5];
    private static readonly long[] s_profileConvClassCalls = new long[5];
    private static readonly long[] s_profileNodeTicks = new long[512];
    private static readonly long[] s_profileNodeCalls = new long[512];
    private readonly CompiledModel _compiled;
    private readonly Model _model;
    private readonly TensorValue[] _tensors;
    private readonly int _inputIndex, _outputIndex;
    private readonly int _intraOpThreads;
    private readonly ResizeWorkspace _resizeWorkspace = new();
    private float[] _workspace = [];
    private int _highWaterInputVolume;
    private bool _hasShape;
    private bool _disposed;

    public Model Model => _model;
    public CompiledModel CompiledModel => _compiled;
    public TensorShape InputShape => new(_tensors[_inputIndex].Shape);
    public TensorShape OutputShape => new(_tensors[_outputIndex].Shape);
    public int Packed1x1Count => _compiled.Packed1x1Count;

    /// <summary>
    /// Per-request resize scratch, reused across runs on this request. Safe
    /// because a request is exclusively held by one inference at a time.
    /// </summary>
    internal ResizeWorkspace ResizeWorkspace => _resizeWorkspace;

    /// <summary>
    /// Largest graph-input volume (<c>n×c×h×w</c>) this session has planned.
    /// Grow-only workspace can rerun any smaller-or-equal volume without a
    /// new allocation; the recognizer pool uses this to hand fat batches to
    /// sessions that already paid for them.
    /// </summary>
    internal int HighWaterInputVolume => _highWaterInputVolume;

    /// <summary>
    /// Graph-input window inside the planned activation workspace (after
    /// <see cref="Reshape"/>). Callers may write NCHW here and Run without a
    /// second input allocation; it is not a separate buffer.
    /// </summary>
    internal Span<float> InputData
    {
        get
        {
            EnsureShape();
            return _tensors[_inputIndex].Data;
        }
    }

    /// <summary>Creates a standalone session: compiles the model, then creates one request.</summary>
    public InferenceSession(Model model, ReadOnlySpan<int> inputShape, int intraOpThreads = 1)
        : this(new CompiledModel(model, inputShape, intraOpThreads)) { }

    /// <summary>Creates a standalone session: compiles the model, then creates one request.</summary>
    public InferenceSession(Model model, params int[] inputShape)
        : this(new CompiledModel(model, inputShape.AsSpan(), 1)) { }

    /// <summary>Creates a request over an already-compiled model.</summary>
    public InferenceSession(CompiledModel compiled)
    {
        _compiled = compiled ?? throw new ArgumentNullException(nameof(compiled));
        _model = compiled.Model;
        _intraOpThreads = compiled.IntraOpThreads;
        _inputIndex = compiled.InputIndex;
        _outputIndex = compiled.OutputIndex;
        _tensors = new TensorValue[compiled.TensorCount];
        for (int i = 0; i < _tensors.Length; i++)
        {
            CompiledModel.TensorMeta meta = compiled.GetTensor(i);
            _tensors[i] = new TensorValue(meta);
        }
        if (compiled.DefaultInputShape is int[] defaultShape)
            Reshape(defaultShape);
    }

    /// <summary>
    /// Rebinds this session to a new concrete input shape: re-resolves every
    /// tensor shape and replans the activation workspace, growing the backing
    /// buffer only when the new plan needs more storage. Constant tensors keep
    /// pointing at the shared <see cref="CompiledModel"/> weights.
    /// </summary>
    public void Reshape(ReadOnlySpan<int> inputShape)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(InferenceSession));
        // Same shape → skip ResolveShapes + PlanWorkspace (hot under adaptive REC widths).
        if (_hasShape)
        {
            int[] bound = _tensors[_inputIndex].Shape;
            if (bound.Length == inputShape.Length)
            {
                bool same = true;
                for (int i = 0; i < bound.Length; i++)
                {
                    if (bound[i] != inputShape[i]) { same = false; break; }
                }
                if (same) return;
            }
        }
        int[][] resolved = _compiled.ResolveShapesFor(inputShape);
        for (int i = 0; i < _tensors.Length; i++)
            _tensors[i].SetShape(resolved[i]);
        PlanWorkspace();
        _hasShape = true;
    }

    // Single-block activation workspace planned from the last-use table: a
    // tensor's region is released after its last consumer and reused
    // first-fit with adjacent coalescing, aligned to 16 floats (64 bytes),
    // matching lw_plan_workspace. Only allocation locations change.
    private void PlanWorkspace()
    {
        const int Align = 16;
        int tensorCount = _tensors.Length, nodeCount = _model.Nodes.Length;
        int[] offsets = new int[tensorCount];
        int[] lengths = new int[tensorCount];
        for (int i = 0; i < tensorCount; i++)
            lengths[i] = _tensors[i].IsConstant ? 0 : checked((int)_tensors[i].ElementCount);

        // Offset-sorted free list + adjacent merge, first-fit from the
        // lowest address, same as C. `assigned` is liveness; `planned`
        // remembers that a tensor already has an offset. Released
        // intermediates stay planned so we do not stack them again —
        // the old leftover pass treated every dead activation as a new
        // allocation and grew tiny DET from ~73 MB to ~687 MB.
        List<(int Offset, int Size)> free = [];
        int bump = 0;
        bool[] assigned = new bool[tensorCount];
        bool[] planned = new bool[tensorCount];

        void AddFree(int offset, int size)
        {
            if (size <= 0) return;
            int position = 0;
            while (position < free.Count && free[position].Offset < offset)
                position++;
            free.Insert(position, (offset, size));
            if (position > 0 &&
                free[position - 1].Offset + free[position - 1].Size == offset)
            {
                (int prevOffset, int prevSize) = free[position - 1];
                free[position - 1] = (prevOffset, prevSize + size);
                free.RemoveAt(position);
                position--;
                offset = free[position].Offset;
                size = free[position].Size;
            }
            if (position + 1 < free.Count &&
                offset + size == free[position + 1].Offset)
            {
                free[position] = (offset, size + free[position + 1].Size);
                free.RemoveAt(position + 1);
            }
        }

        int Allocate(int length, int preferOffset = -1)
        {
            int size = (length + Align - 1) / Align * Align;
            if (preferOffset >= 0 && TryCarve(preferOffset, size))
                return preferOffset;
            for (int i = 0; i < free.Count; i++)
            {
                if (free[i].Size < size) continue;
                (int offset, int blockSize) = free[i];
                free.RemoveAt(i);
                if (blockSize > size)
                    AddFree(offset + size, blockSize - size);
                return offset;
            }
            int bumpOffset = bump;
            bump += size;
            return bumpOffset;
        }

        bool TryCarve(int offset, int size)
        {
            for (int i = 0; i < free.Count; i++)
            {
                (int blockOffset, int blockSize) = free[i];
                if (offset < blockOffset || offset + size > blockOffset + blockSize)
                    continue;
                free.RemoveAt(i);
                if (offset > blockOffset)
                    AddFree(blockOffset, offset - blockOffset);
                int end = offset + size, blockEnd = blockOffset + blockSize;
                if (end < blockEnd)
                    AddFree(end, blockEnd - end);
                return true;
            }
            return false;
        }

        void Release(int index)
            => AddFree(offsets[index], (lengths[index] + Align - 1) / Align * Align);

        foreach (uint input in _model.GraphInputs)
        {
            int index = checked((int)input);
            if (_tensors[index].IsConstant || lengths[index] == 0) continue;
            offsets[index] = Allocate(lengths[index]);
            assigned[index] = true;
            planned[index] = true;
        }
        for (int ni = 0; ni < nodeCount; ni++)
        {
            for (int i = 0; i < tensorCount; i++)
            {
                if (assigned[i] && lengths[i] != 0 &&
                    !_compiled.HasConsumerAfter((uint)i, ni - 1) &&
                    !_compiled.IsGraphOutput(i))
                {
                    assigned[i] = false;
                    Release(i);
                }
            }
            foreach (uint output in _model.Nodes[ni].Outputs)
            {
                int index = checked((int)output);
                if (_tensors[index].IsConstant) continue;
                assigned[index] = true;
                planned[index] = true;
                if (lengths[index] != 0)
                {
                    int prefer = -1;
                    int src = _compiled.ElementwiseInPlaceSource(index);
                    if (src >= 0 && planned[src] && !assigned[src] && lengths[src] == lengths[index])
                        prefer = offsets[src];
                    offsets[index] = Allocate(lengths[index], prefer);
                }
            }
        }

        // Grow-only: shrinking here reallocates and zeros a large float[]
        // on every smaller REC (n, width), which showed up as ~70–250 ms
        // rec_reshape and regressed tiny 1w / small 4w. The recognizer pool
        // routes wide lines onto sessions that already hold this high-water
        // so sibling workers do not each copy the max buffer. Unique LOH
        // sizes are discarded by PooledArrays.
        if (_workspace.Length < bump)
            _workspace = new float[bump];
        int[] inputDims = _tensors[_inputIndex].Shape;
        int inputVolume = 1;
        for (int d = 0; d < inputDims.Length; d++)
            inputVolume = checked(inputVolume * inputDims[d]);
        if (inputVolume > _highWaterInputVolume)
            _highWaterInputVolume = inputVolume;
        for (int i = 0; i < tensorCount; i++)
        {
            TensorValue tensor = _tensors[i];
            if (tensor.IsConstant) continue;
            tensor.Bind(_workspace, offsets[i], planned[i] ? lengths[i] : 0);
        }
    }

    private void EnsureShape()
    {
        if (!_hasShape)
            throw new InvalidOperationException("The session has no input shape; call Reshape first.");
    }

    public void Run(ReadOnlySpan<float> input, Span<float> output)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(InferenceSession));
        EnsureShape();
        TensorValue inputTensor = _tensors[_inputIndex]; TensorValue outputTensor = _tensors[_outputIndex];
        if (input.Length != inputTensor.Length || output.Length < outputTensor.Length) throw new ArgumentException("Input/output buffer size mismatch.");
        Execute(input);
        outputTensor.Data.CopyTo(output);
    }

    /// <summary>Executes into the session-owned output without an extra copy.</summary>
    internal ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(InferenceSession));
        EnsureShape();
        TensorValue inputTensor = _tensors[_inputIndex];
        if (input.Length != inputTensor.Length) throw new ArgumentException("Input buffer size mismatch.");
        Execute(input);
        return _tensors[_outputIndex].Data;
    }

    /// <summary>
    /// Runs the graph through the node before the final CTC projection
    /// (MatMul, optional Add bias, Softmax). Does not write vocab logits —
    /// the caller (Recognizer) owns ArgMax scratch and calls
    /// <see cref="Sdcb.SimdPaddleOCR.Kernels.MatMul.TryArgMax"/>. Returns false when the
    /// tail pattern is unsupported; use <see cref="RunInternalSkipFinalSoftmax"/> then.
    /// </summary>
    internal bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands)
    {
        operands = default;
        if (_disposed) throw new ObjectDisposedException(nameof(InferenceSession));
        EnsureShape();
        TensorValue inputTensor = _tensors[_inputIndex];
        if (input.Length != inputTensor.Length) throw new ArgumentException("Input buffer size mismatch.");
        if (!TryResolveCtcProjection(out NodeRecord matMul, out int matMulIndex,
            out ReadOnlySpan<float> bias, out float[]? packed, out int batch, out int rows,
            out int inner, out int columns))
            return false;

        CopyInputUnlessAliased(input, inputTensor);
        for (int ni = 0; ni < matMulIndex; ni++)
            ni += ExecuteStep(ni);

        operands = new CtcProjectionOperands(
            _tensors[matMul.Inputs[0]].Data, _tensors[matMul.Inputs[1]].Data, bias, packed,
            batch, rows, inner, columns, matMulIndex);
        return true;
    }

    /// <summary>
    /// Full graph run that leaves a final last-axis Softmax as logits when the
    /// model ends that way (ArgMax-invariant). Used when fused CTC projection
    /// is unavailable.
    /// </summary>
    internal ReadOnlySpan<float> RunInternalSkipFinalSoftmax(ReadOnlySpan<float> input, out bool outputIsLogits)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(InferenceSession));
        EnsureShape();
        TensorValue inputTensor = _tensors[_inputIndex];
        if (input.Length != inputTensor.Length) throw new ArgumentException("Input buffer size mismatch.");
        outputIsLogits = HasSkippableOutputSoftmax(out NodeRecord softmax);
        Execute(input, skipFinalSoftmax: outputIsLogits);
        return outputIsLogits ? _tensors[softmax.Inputs[0]].Data : _tensors[_outputIndex].Data;
    }

    internal bool IsProfilingEnabled => s_profileEnabled;

    internal void NoteProfile(OperatorId operation, long started, int nodeIndex)
        => AddProfile((int)operation, started, nodeIndex);

    private bool TryResolveCtcProjection(out NodeRecord matMul, out int matMulIndex,
        out ReadOnlySpan<float> bias, out float[]? packed, out int batch, out int rows,
        out int inner, out int columns)
    {
        matMul = default;
        matMulIndex = -1;
        bias = [];
        packed = null;
        batch = rows = inner = columns = 0;
        if (!HasSkippableOutputSoftmax(out NodeRecord softmax))
            return false;

        int terminalIndex = _model.Nodes.Length - 2;
        NodeRecord terminal = _model.Nodes[terminalIndex];
        if (terminal.Operator == OperatorId.MatMul &&
            terminal.Outputs[0] == softmax.Inputs[0])
        {
            matMul = terminal;
            matMulIndex = terminalIndex;
        }
        else if (terminal.Operator == OperatorId.Add && terminal.Inputs.Length == 2 &&
            terminalIndex > 0 &&
            terminal.Outputs[0] == softmax.Inputs[0])
        {
            matMulIndex = terminalIndex - 1;
            matMul = _model.Nodes[matMulIndex];
            if (matMul.Operator != OperatorId.MatMul || matMul.Outputs.Length != 1)
                return false;
            uint biasIndex;
            if (terminal.Inputs[0] == matMul.Outputs[0]) biasIndex = terminal.Inputs[1];
            else if (terminal.Inputs[1] == matMul.Outputs[0]) biasIndex = terminal.Inputs[0];
            else return false;
            TensorValue biasTensor = _tensors[checked((int)biasIndex)];
            TensorValue projected = _tensors[matMul.Outputs[0]];
            if (!biasTensor.IsConstant || biasTensor.Shape.Length != 1 ||
                projected.Shape.Length < 2 || biasTensor.Length != projected.Shape[^1])
                return false;
            bias = biasTensor.Data;
        }
        else return false;

        if (matMul.Inputs.Length != 2 || matMul.Outputs.Length != 1)
            return false;
        TensorValue a = _tensors[matMul.Inputs[0]];
        TensorValue b = _tensors[matMul.Inputs[1]];
        TensorValue projectedOutput = _tensors[matMul.Outputs[0]];
        if (!b.IsConstant || a.Shape.Length < 2 || b.Shape.Length != 2 ||
            projectedOutput.Shape.Length < 2)
            return false;
        rows = a.Shape[^2];
        inner = a.Shape[^1];
        columns = b.Shape[1];
        batch = checked(a.Length / (rows * inner));
        if (b.Shape[0] != inner || projectedOutput.Shape[^1] != columns ||
            projectedOutput.Length != checked(batch * rows * columns))
            return false;
        _compiled.TryGetPackedMatMul(matMul.Inputs[1], out packed);
        return global::Sdcb.SimdPaddleOCR.Kernels.MatMul.CanFuseArgMax(rows, inner, columns, packed);
    }

    private bool HasSkippableOutputSoftmax(out NodeRecord softmax)
    {
        softmax = _model.Nodes[^1];
        if (softmax.Operator != OperatorId.Softmax ||
            softmax.Inputs.Length != 1 || softmax.Outputs.Length != 1 ||
            softmax.Outputs[0] != (uint)_outputIndex)
            return false;
        TensorValue input = _tensors[softmax.Inputs[0]];
        int axis = I32(_model.GetParameters(softmax), 4);
        if (axis < 0) axis += input.Shape.Length;
        return axis == input.Shape.Length - 1;
    }

    /// <summary>Executes using a caller-owned array directly, avoiding an input copy.</summary>
    internal ReadOnlySpan<float> RunInternal(float[] input)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(InferenceSession));
        EnsureShape();
        TensorValue inputTensor = _tensors[_inputIndex];
        if (input.Length != inputTensor.Length) throw new ArgumentException("Input buffer size mismatch.");
        if (inputTensor.IsBoundTo(input))
        {
            Execute(input, true);
            return _tensors[_outputIndex].Data;
        }
        float[]? previousBuffer = inputTensor.Buffer;
        int previousOffset = inputTensor.Offset, previousLength = inputTensor.Length;
        inputTensor.Bind(input, 0, input.Length);
        try
        {
            Execute(input, true);
            return _tensors[_outputIndex].Data;
        }
        finally
        {
            if (previousBuffer is not null)
                inputTensor.Bind(previousBuffer, previousOffset, previousLength);
        }
    }

    public float[] Run(ReadOnlySpan<float> input)
    {
        float[] output = new float[checked((int)OutputShape.ElementCount)];
        RunInternal(input).CopyTo(output);
        return output;
    }

    internal static void EnableProfiling(bool enabled)
    {
        s_profileEnabled = enabled;
        if (enabled)
        {
            Array.Clear(s_profileTicks, 0, s_profileTicks.Length); Array.Clear(s_profileCalls, 0, s_profileCalls.Length);
            Array.Clear(s_profileNodeTicks, 0, s_profileNodeTicks.Length); Array.Clear(s_profileNodeCalls, 0, s_profileNodeCalls.Length);
            Array.Clear(s_profileConvClassTicks, 0, s_profileConvClassTicks.Length); Array.Clear(s_profileConvClassCalls, 0, s_profileConvClassCalls.Length);
        }
    }

    internal static (long Ticks, long Calls)[] ProfileSnapshot()
    {
        (long, long)[] result = new (long, long)[s_profileTicks.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = (Interlocked.Read(ref s_profileTicks[i]), Interlocked.Read(ref s_profileCalls[i]));
        return result;
    }

    internal static (long Ticks, long Calls)[] NodeProfileSnapshot()
    {
        (long, long)[] result = new (long, long)[s_profileNodeTicks.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = (Interlocked.Read(ref s_profileNodeTicks[i]), Interlocked.Read(ref s_profileNodeCalls[i]));
        return result;
    }

    internal static (long Ticks, long Calls)[] ConvClassProfileSnapshot()
    {
        (long, long)[] result = new (long, long)[s_profileConvClassTicks.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = (Interlocked.Read(ref s_profileConvClassTicks[i]),
                Interlocked.Read(ref s_profileConvClassCalls[i]));
        return result;
    }

    private static void AddProfile(int operation, long started, int nodeIndex)
    {
        long elapsed = Stopwatch.GetTimestamp() - started;
        if ((uint)operation < (uint)s_profileTicks.Length)
        {
            Interlocked.Add(ref s_profileTicks[operation], elapsed);
            Interlocked.Increment(ref s_profileCalls[operation]);
        }
        if ((uint)nodeIndex < (uint)s_profileNodeTicks.Length)
        {
            Interlocked.Add(ref s_profileNodeTicks[nodeIndex], elapsed);
            Interlocked.Increment(ref s_profileNodeCalls[nodeIndex]);
        }
    }

    private void AddConvClassProfile(NodeRecord node, long started)
    {
        int classId = ProfileConvClass(_tensors[node.Inputs[0]], _tensors[node.Inputs[1]],
            _model.GetParameters(node));
        long elapsed = Stopwatch.GetTimestamp() - started;
        if ((uint)classId < (uint)s_profileConvClassTicks.Length)
        {
            Interlocked.Add(ref s_profileConvClassTicks[classId], elapsed);
            Interlocked.Increment(ref s_profileConvClassCalls[classId]);
        }
        if (s_dumpConv) AccumulateConvShape(node, elapsed);
    }

    private void AccumulateConvShape(NodeRecord node, long elapsed)
    {
        ReadOnlySpan<byte> p = _model.GetParameters(node);
        int[] id = _tensors[node.Inputs[0]].Shape, od = _tensors[node.Outputs[0]].Shape;
        string key = $"k={I32(p, 8)}x{I32(p, 12)} s={I32(p, 16)}x{I32(p, 20)} p={I32(p, 32)},{I32(p, 36)} g={U32(p, 4)} in={id[1]}x{id[2]}x{id[3]} out={od[1]}x{od[2]}x{od[3]}";
        lock (s_convShapeTicks)
            s_convShapeTicks[key] = s_convShapeTicks.TryGetValue(key, out long t) ? t + elapsed : elapsed;
    }

    private static readonly Dictionary<string, long> s_convShapeTicks = [];
    static InferenceSession()
    {
        if (s_dumpConv)
            AppDomain.CurrentDomain.ProcessExit += static (_, _) =>
            {
                lock (s_convShapeTicks)
                    foreach (KeyValuePair<string, long> e in s_convShapeTicks.OrderByDescending(static e => e.Value))
                        Console.Error.WriteLine($"conv-shape {e.Value * 1000.0 / Stopwatch.Frequency,10:F1}ms  {e.Key}");
            };
    }

    private static int ProfileConvClass(TensorValue input, TensorValue weights, ReadOnlySpan<byte> p)
    {
        int kh = I32(p, 8), kw = I32(p, 12), sh = I32(p, 16), sw = I32(p, 20);
        int groups = checked((int)U32(p, 4));
        if (kh == 1 && kw == 1) return 0; // Conv1x1
        if (kh == 3 && kw == 3 && input.Shape.Length == 4 && weights.Shape.Length == 4 &&
            input.Shape[1] > 0 && groups == input.Shape[1] && weights.Shape[0] == input.Shape[1])
            return 2; // Depthwise3x3
        if (kh == 3 && kw == 3 && sh == 2 && sw == 2) return 3; // Stride2Conv3x3
        if (kh == 3 && kw == 3) return 1; // Conv3x3
        return 4; // OtherConv
    }

    internal NodeTrace[] Trace(ReadOnlySpan<float> input)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(InferenceSession));
        EnsureShape();
        CopyInputUnlessAliased(input, _tensors[_inputIndex]);
        NodeTrace[] traces = new NodeTrace[_model.Nodes.Length];
        for (int ni = 0; ni < _model.Nodes.Length; ni++)
        {
            ExecuteNode(_model.Nodes[ni]);
            TensorValue o = _tensors[_model.Nodes[ni].Outputs[0]];
            Span<float> data = o.Data;
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            double sum = 0;
            for (int i = 0; i < data.Length; i++) { float v = data[i]; min = MathF.Min(min, v); max = MathF.Max(max, v); sum += v; }
            traces[ni] = new NodeTrace(ni, _model.Nodes[ni].Operator, [.. o.Shape], min, max, sum / data.Length);
        }
        return traces;
    }

    internal (int Index, string Operator, int[] Shape, int DataLength)[] ResolvedNodeShapes()
    {
        (int, string, int[], int)[] result = new (int, string, int[], int)[_model.Nodes.Length];
        for (int i = 0; i < result.Length; i++)
        {
            NodeRecord node = _model.Nodes[i];
            TensorValue output = _tensors[node.Outputs[0]];
            result[i] = (i, node.OpType, [.. output.Shape], checked((int)output.ElementCount));
        }
        return result;
    }

    private void CopyInputUnlessAliased(ReadOnlySpan<float> input, TensorValue inputTensor)
    {
        Span<float> dest = inputTensor.Data;
        if (input.Length != dest.Length)
            throw new ArgumentException("Input buffer size mismatch.");
        // Preprocess may write straight into the planned graph-input window.
        if (!input.Overlaps(dest))
            input.CopyTo(dest);
    }

    private void Execute(ReadOnlySpan<float> input, bool inputAlreadyBound = false,
        bool skipFinalSoftmax = false)
    {
        TensorValue inputTensor = _tensors[_inputIndex];
        if (!inputAlreadyBound)
            CopyInputUnlessAliased(input, inputTensor);
        int nodeCount = skipFinalSoftmax ? _model.Nodes.Length - 1 : _model.Nodes.Length;
        for (int ni = 0; ni < nodeCount; ni++)
            ni += ExecuteStep(ni);
    }

    private int ExecuteStep(int ni)
    {
        long started = s_profileEnabled ? Stopwatch.GetTimestamp() : 0;
        int skip = _compiled.FusedSkip(ni);
        if (skip == 4)
        {
            if (!TryExecuteGelu(ni))
                throw new InvalidOperationException($"GELU fusion failed at node {ni}.");
            if (s_profileEnabled) AddProfile((int)OperatorId.Erf, started, ni);
            return skip;
        }
        if (skip == 2)
        {
            if (TryExecuteConvHardSwish(ni))
            {
                if (s_profileEnabled) AddProfile((int)OperatorId.Conv, started, ni);
                return skip;
            }
            if (TryExecuteConvTransposeBiasRelu(ni))
            {
                if (s_profileEnabled) AddProfile((int)OperatorId.ConvTranspose, started, ni);
                return skip;
            }
            if (TryExecuteConvTransposeBiasSigmoid(ni))
            {
                if (s_profileEnabled) AddProfile((int)OperatorId.ConvTranspose, started, ni);
                return skip;
            }
            if (TryExecuteConvBiasResidualAdd(ni))
            {
                if (s_profileEnabled)
                {
                    AddProfile((int)OperatorId.Conv, started, ni);
                    AddConvClassProfile(_model.Nodes[ni], started);
                }
                return skip;
            }
            throw new InvalidOperationException($"Compiled 3-node fusion failed at node {ni}.");
        }
        if (skip == 1)
        {
            if (TryExecuteConvRelu(ni))
            {
                if (s_profileEnabled) AddProfile((int)OperatorId.Conv, started, ni);
                return skip;
            }
            if (TryExecuteHardSwish(ni))
            {
                if (s_profileEnabled) AddProfile((int)OperatorId.Mul, started, ni);
                return skip;
            }
            if (TryExecuteConvBiasAdd(ni))
            {
                if (s_profileEnabled)
                {
                    AddProfile((int)OperatorId.Conv, started, ni);
                    AddConvClassProfile(_model.Nodes[ni], started);
                }
                return skip;
            }
            throw new InvalidOperationException($"Compiled 2-node fusion failed at node {ni}.");
        }
        NodeRecord node = _model.Nodes[ni];
        if (TryExecuteInPlace(node, ni))
        {
            if (s_profileEnabled) AddProfile((int)node.Operator, started, ni);
            return 0;
        }
        ExecuteNode(node);
        if (s_profileEnabled)
        {
            AddProfile((int)node.Operator, started, ni);
            if (node.Operator == OperatorId.Conv) AddConvClassProfile(node, started);
        }
        return 0;
    }

    private void ExecuteNode(NodeRecord node)
    {
        ReadOnlySpan<byte> p = _model.GetParameters(node); TensorValue o = _tensors[node.Outputs[0]]; TensorValue x = _tensors[node.Inputs[0]];
        switch (node.Operator)
        {
            case OperatorId.Add: Binary<AddOp>(x, _tensors[node.Inputs[1]], o, _intraOpThreads); break;
            case OperatorId.Sub: Binary<SubOp>(x, _tensors[node.Inputs[1]], o, _intraOpThreads); break;
            case OperatorId.Mul: Binary<MulOp>(x, _tensors[node.Inputs[1]], o, _intraOpThreads); break;
            case OperatorId.Div: Binary<DivOp>(x, _tensors[node.Inputs[1]], o, _intraOpThreads); break;
            case OperatorId.Pow: BinaryPow(x, _tensors[node.Inputs[1]], o); break;
            case OperatorId.Relu: SimdKernels.ReluParallel(x.Data, o.Data, _intraOpThreads); break;
            case OperatorId.Sigmoid: SimdKernels.SigmoidParallel(x.Data, o.Data, _intraOpThreads); break;
            case OperatorId.Erf: SimdKernels.ErfParallel(x.Data, o.Data, _intraOpThreads); break;
            case OperatorId.Sqrt: SimdKernels.SqrtParallel(x.Data, o.Data, _intraOpThreads); break;
            case OperatorId.HardSigmoid: { float alpha = F32(p, 4), beta = F32(p, 8); SimdKernels.HardSigmoid(x.Data, o.Data, alpha, beta); break; }
            case OperatorId.BatchNormalization: BatchNorm(x, node, o); break;
            case OperatorId.Conv:
                ExecuteConv(node, o, node.Inputs.Length > 2 ? _tensors[node.Inputs[2]] : null);
                break;
            case OperatorId.ConvTranspose: ConvTranspose(x, _tensors[node.Inputs[1]], node.Inputs.Length > 2 ? _tensors[node.Inputs[2]] : null, p, o, _intraOpThreads); break;
            case OperatorId.ReduceMean: ReduceMean(x, p, o); break;
            case OperatorId.AveragePool: Pool(x, p, o, false); break;
            case OperatorId.MaxPool: Pool(x, p, o, true); break;
            case OperatorId.Transpose: Transpose(x, p, o); break;
            case OperatorId.Squeeze:
            case OperatorId.Unsqueeze:
            case OperatorId.Reshape:
                if (x.Length != o.Length)
                    throw new InvalidDataException($"{node.Operator} data size mismatch at node '{node.Name}': input=[{string.Join(",", x.Shape)}] ({x.Length}), output=[{string.Join(",", o.Shape)}] ({o.Length}).");
                x.Data.CopyTo(o.Data); break;
            case OperatorId.Concat: Concat(node, p, o); break;
            case OperatorId.MatMul:
                _compiled.TryGetPackedMatMul(node.Inputs[1], out float[]? packedMatMul);
                MatMul(x, _tensors[node.Inputs[1]], o, packedMatMul); break;
            case OperatorId.Softmax: Softmax(x, p, o); break;
            case OperatorId.Resize: Resize(x, p, o); break;
            case OperatorId.Slice: Slice(x, node, o); break;
            default: throw new NotSupportedException($"Unsupported operator {node.Operator}.");
        }
    }
    private bool TryExecuteGelu(int index)
    {
        if (index + 4 >= _model.Nodes.Length) return false;
        NodeRecord n0 = _model.Nodes[index]; NodeRecord n1 = _model.Nodes[index + 1];
        NodeRecord n2 = _model.Nodes[index + 2]; NodeRecord n3 = _model.Nodes[index + 3];
        NodeRecord n4 = _model.Nodes[index + 4];
        if (n0.Operator != OperatorId.Div || n1.Operator != OperatorId.Erf ||
            n2.Operator != OperatorId.Add || n3.Operator != OperatorId.Mul ||
            n4.Operator != OperatorId.Mul || n0.Inputs.Length != 2 ||
            n1.Inputs.Length != 1 || n2.Inputs.Length != 2 || n3.Inputs.Length != 2 ||
            n4.Inputs.Length != 2) return false;
        if (n1.Inputs[0] != n0.Outputs[0] ||
            (n2.Inputs[0] != n1.Outputs[0] && n2.Inputs[1] != n1.Outputs[0]) ||
            (n3.Inputs[0] != n0.Inputs[0] && n3.Inputs[1] != n0.Inputs[0]) ||
            (n3.Inputs[0] != n2.Outputs[0] && n3.Inputs[1] != n2.Outputs[0]) ||
            (n4.Inputs[0] != n3.Outputs[0] && n4.Inputs[1] != n3.Outputs[0])) return false;
        int n2Constant = n2.Inputs[0] == n1.Outputs[0] ? 1 : n2.Inputs[1] == n1.Outputs[0] ? 0 : -1;
        int n4Constant = n4.Inputs[0] == n3.Outputs[0] ? 1 : n4.Inputs[1] == n3.Outputs[0] ? 0 : -1;
        if (n2Constant < 0 || n4Constant < 0 ||
            !IsConstantScalar(n0.Inputs[1], 1.4142135381698608f) ||
            !IsConstantScalar(n2.Inputs[n2Constant], 1f) ||
            !IsConstantScalar(n4.Inputs[n4Constant], 0.5f) ||
            HasConsumerAfter(n0.Outputs[0], index + 3) ||
            HasConsumerAfter(n1.Outputs[0], index + 2) ||
            HasConsumerAfter(n2.Outputs[0], index + 3) ||
            HasConsumerAfter(n3.Outputs[0], index + 4)) return false;
        TensorValue src = _tensors[n0.Inputs[0]]; TensorValue dst = _tensors[n4.Outputs[0]];
        if (dst.IsConstant || src.Length == 0 || src.Length != dst.Length) return false;
        // Elementwise GELU is in-place-safe. C fuses even when the planner
        // aliases source onto the sink; rejecting that overlap dropped the
        // fusion after workspace reuse and re-ran five kernels.
        if (PartialOverlap(src, dst)) return false;
        if (_tensors[n0.Outputs[0]].Length != src.Length ||
            _tensors[n1.Outputs[0]].Length != src.Length ||
            _tensors[n2.Outputs[0]].Length != src.Length ||
            _tensors[n3.Outputs[0]].Length != src.Length) return false;
        SimdKernels.GeluParallel(src.Data, dst.Data, _intraOpThreads);
        return true;
    }

    private bool IsConstantScalar(uint tensorIndex, float expected)
    {
        TensorValue tensor = _tensors[checked((int)tensorIndex)];
        return tensor.IsConstant && tensor.Length == 1 && tensor.Data[0] == expected;
    }

    private void ExecuteConv(NodeRecord conv, TensorValue output, TensorValue? bias,
        ReadOnlySpan<float> residual = default)
    {
        ReadOnlySpan<byte> p = _model.GetParameters(conv);
        TensorValue x = _tensors[conv.Inputs[0]];
        float[]? packed = _model.GetPackedWeights(conv, Model.PackConv1x1);
        float[]? packedOc16 = _model.GetPackedWeights(conv, Model.PackConv1x1Oc16);
        float[]? packedOc8 = _model.GetPackedWeights(conv, Model.PackConv1x1Oc8);
        float[]? packed3x3 = _model.GetPackedWeights(conv, Model.PackConv3x3);
        // Disabled after the full small-model gate: quantization changed
        // texts and was slower than the AVX-512 float kernel.
        const bool EnableInt8VnniConv1x1 = false;
        PackedConv1x1Int8? packedInt8 = EnableInt8VnniConv1x1
            ? _model.GetPackedConv1x1Int8(conv)
            : null;
        Conv(x, _tensors[conv.Inputs[1]], bias, p, output, _intraOpThreads,
            packed, packed3x3, packedOc16, packedInt8, packedOc8, residual);
    }

    private bool TryExecuteConvBiasAdd(int index)
    {
        if (index + 1 >= _model.Nodes.Length) return false;
        NodeRecord conv = _model.Nodes[index], add = _model.Nodes[index + 1];
        if (conv.Operator != OperatorId.Conv || add.Operator != OperatorId.Add ||
            conv.Inputs.Length != 2 || conv.Outputs.Length != 1 ||
            add.Inputs.Length != 2 || add.Outputs.Length != 1 ||
            (add.Inputs[0] != conv.Outputs[0] && add.Inputs[1] != conv.Outputs[0]) ||
            HasConsumerAfter(conv.Outputs[0], index + 1))
            return false;
        uint biasIndex = add.Inputs[0] == conv.Outputs[0] ? add.Inputs[1] : add.Inputs[0];
        TensorValue convOutput = _tensors[conv.Outputs[0]];
        TensorValue addOutput = _tensors[add.Outputs[0]];
        TensorValue bias = _tensors[biasIndex];
        if (!bias.IsConstant || convOutput.IsConstant || addOutput.IsConstant ||
            convOutput.Length == 0 || convOutput.Length != addOutput.Length ||
            addOutput.Shape.Length != 4 || bias.Length != addOutput.Shape[1] ||
            _tensors[conv.Inputs[0]].Overlaps(addOutput))
            return false;
        convOutput.ShareStorageWith(addOutput);
        ExecuteConv(conv, addOutput, bias);
        return true;
    }

    private bool TryExecuteConvBiasResidualAdd(int index)
    {
        if (index + 2 >= _model.Nodes.Length) return false;
        NodeRecord conv = _model.Nodes[index], biasAdd = _model.Nodes[index + 1],
            residualAdd = _model.Nodes[index + 2];
        if (conv.Operator != OperatorId.Conv || biasAdd.Operator != OperatorId.Add ||
            residualAdd.Operator != OperatorId.Add ||
            conv.Inputs.Length != 2 || conv.Outputs.Length != 1 ||
            biasAdd.Inputs.Length != 2 || biasAdd.Outputs.Length != 1 ||
            residualAdd.Inputs.Length != 2 || residualAdd.Outputs.Length != 1 ||
            (biasAdd.Inputs[0] != conv.Outputs[0] && biasAdd.Inputs[1] != conv.Outputs[0]) ||
            (residualAdd.Inputs[0] != biasAdd.Outputs[0] && residualAdd.Inputs[1] != biasAdd.Outputs[0]) ||
            HasConsumerAfter(conv.Outputs[0], index + 1) ||
            HasConsumerAfter(biasAdd.Outputs[0], index + 2))
            return false;
        uint biasIndex = biasAdd.Inputs[0] == conv.Outputs[0] ? biasAdd.Inputs[1] : biasAdd.Inputs[0];
        uint skipIndex = residualAdd.Inputs[0] == biasAdd.Outputs[0]
            ? residualAdd.Inputs[1] : residualAdd.Inputs[0];
        TensorValue convOutput = _tensors[conv.Outputs[0]];
        TensorValue biasAddOutput = _tensors[biasAdd.Outputs[0]];
        TensorValue destination = _tensors[residualAdd.Outputs[0]];
        TensorValue bias = _tensors[biasIndex];
        TensorValue residual = _tensors[skipIndex];
        if (!bias.IsConstant || convOutput.IsConstant || destination.IsConstant ||
            convOutput.Length == 0 || convOutput.Length != destination.Length ||
            residual.Length != destination.Length ||
            destination.Shape.Length != 4 || bias.Length != destination.Shape[1])
            return false;
        TensorValue input = _tensors[conv.Inputs[0]];
        float[]? packedOc8 = _model.GetPackedWeights(conv, Model.PackConv1x1Oc8);
        int oc = destination.Shape[1], ic = input.Shape[1];
        if (packedOc8 is not null &&
            Conv1x1.FusesResidualInPackedEight(oc, ic, packedOc8.Length) &&
            !input.Overlaps(destination) && !PartialOverlap(residual, destination))
        {
            convOutput.ShareStorageWith(destination);
            biasAddOutput.ShareStorageWith(destination);
            ExecuteConv(conv, destination, bias, residual.Data);
            return true;
        }
        convOutput.ShareStorageWith(biasAddOutput);
        ExecuteConv(conv, convOutput, bias);
        Binary<AddOp>(convOutput, residual, destination, _intraOpThreads);
        return true;
    }

    private bool TryExecuteConvRelu(int index)
    {
        if (index + 1 >= _model.Nodes.Length) return false;
        NodeRecord conv = _model.Nodes[index], relu = _model.Nodes[index + 1];
        if (conv.Operator != OperatorId.Conv || relu.Operator != OperatorId.Relu ||
            conv.Outputs.Length != 1 || relu.Inputs.Length != 1 || relu.Outputs.Length != 1 ||
            relu.Inputs[0] != conv.Outputs[0] ||
            HasConsumerAfter(conv.Outputs[0], index + 1)) return false;
        TensorValue convOutput = _tensors[conv.Outputs[0]];
        TensorValue reluOutput = _tensors[relu.Outputs[0]];
        if (convOutput.IsConstant || reluOutput.IsConstant ||
            convOutput.Length == 0 || convOutput.Length != reluOutput.Length)
            return false;
        if (_tensors[conv.Inputs[0]].Overlaps(reluOutput)) return false;


        // The convolution writes directly into the terminal ReLU's buffer.
        // The two tensors have identical shapes, and the ReLU is the sole
        // consumer of the convolution result, so no live value is overwritten.
        convOutput.ShareStorageWith(reluOutput);
        ExecuteNode(conv);
        SimdKernels.Relu(reluOutput.Data, reluOutput.Data);
        return true;
    }

    private bool TryExecuteHardSwish(int index)
    {
        if (index + 1 >= _model.Nodes.Length) return false;
        NodeRecord hardSigmoid = _model.Nodes[index], multiply = _model.Nodes[index + 1];
        if (hardSigmoid.Operator != OperatorId.HardSigmoid || multiply.Operator != OperatorId.Mul ||
            hardSigmoid.Inputs.Length != 1 || hardSigmoid.Outputs.Length != 1 ||
            multiply.Inputs.Length != 2 || multiply.Outputs.Length != 1 ||
            (multiply.Inputs[0] != hardSigmoid.Outputs[0] && multiply.Inputs[1] != hardSigmoid.Outputs[0]))
            return false;
        uint sourceIndex = hardSigmoid.Inputs[0];
        if (multiply.Inputs[0] != sourceIndex && multiply.Inputs[1] != sourceIndex) return false;
        if (HasConsumerAfter(hardSigmoid.Outputs[0], index + 1)) return false;
        TensorValue source = _tensors[sourceIndex];
        TensorValue destination = _tensors[multiply.Outputs[0]];
        if (source.IsConstant || destination.IsConstant || source.Length == 0 ||
            source.Length != destination.Length)
            return false;
        if (PartialOverlap(source, destination)) return false;
        ReadOnlySpan<byte> p = _model.GetParameters(hardSigmoid);
        if (p.Length < 16) return false;
        SimdKernels.HardSwish(source.Data, destination.Data, F32(p, 4), F32(p, 8));
        return true;
    }

    private bool TryExecuteConvHardSwish(int index)
    {
        if (index + 2 >= _model.Nodes.Length) return false;
        NodeRecord conv = _model.Nodes[index], hardSigmoid = _model.Nodes[index + 1], multiply = _model.Nodes[index + 2];
        if (conv.Operator != OperatorId.Conv || hardSigmoid.Operator != OperatorId.HardSigmoid ||
            multiply.Operator != OperatorId.Mul || conv.Outputs.Length != 1 ||
            hardSigmoid.Inputs.Length != 1 || hardSigmoid.Outputs.Length != 1 ||
            multiply.Inputs.Length != 2 || multiply.Outputs.Length != 1 ||
            hardSigmoid.Inputs[0] != conv.Outputs[0] ||
            (multiply.Inputs[0] != hardSigmoid.Outputs[0] && multiply.Inputs[1] != hardSigmoid.Outputs[0]) ||
            (multiply.Inputs[0] != conv.Outputs[0] && multiply.Inputs[1] != conv.Outputs[0]))
            return false;
        if (HasConsumerAfter(conv.Outputs[0], index + 2) || HasConsumerAfter(hardSigmoid.Outputs[0], index + 2))
            return false;
        TensorValue convOutput = _tensors[conv.Outputs[0]];
        TensorValue destination = _tensors[multiply.Outputs[0]];
        if (convOutput.IsConstant || destination.IsConstant || convOutput.Length == 0 ||
            convOutput.Length != destination.Length)
            return false;
        if (_tensors[conv.Inputs[0]].Overlaps(destination)) return false;
        ReadOnlySpan<byte> p = _model.GetParameters(hardSigmoid);
        if (p.Length < 16) return false;
        convOutput.ShareStorageWith(destination);
        ExecuteNode(conv);
        SimdKernels.HardSwish(destination.Data, destination.Data, F32(p, 4), F32(p, 8));
        return true;
    }

    private bool TryExecuteConvTransposeBiasRelu(int index)
    {
        if (index + 2 >= _model.Nodes.Length) return false;
        NodeRecord transpose = _model.Nodes[index], add = _model.Nodes[index + 1], relu = _model.Nodes[index + 2];
        if (transpose.Operator != OperatorId.ConvTranspose || add.Operator != OperatorId.Add ||
            relu.Operator != OperatorId.Relu || transpose.Outputs.Length != 1 ||
            add.Inputs.Length != 2 || add.Outputs.Length != 1 || relu.Inputs.Length != 1 ||
            relu.Outputs.Length != 1 || relu.Inputs[0] != add.Outputs[0] ||
            (add.Inputs[0] != transpose.Outputs[0] && add.Inputs[1] != transpose.Outputs[0]) ||
            HasConsumerAfter(transpose.Outputs[0], index + 1) ||
            HasConsumerAfter(add.Outputs[0], index + 2)) return false;
        uint biasIndex = add.Inputs[0] == transpose.Outputs[0] ? add.Inputs[1] : add.Inputs[0];
        TensorValue bias = _tensors[biasIndex];
        TensorValue transposeOutput = _tensors[transpose.Outputs[0]];
        TensorValue reluOutput = _tensors[relu.Outputs[0]];
        if (!bias.IsConstant || bias.Length != transposeOutput.Shape[1] ||
            bias.Shape.Length != 4 || bias.Shape[0] != 1 || bias.Shape[2] != 1 || bias.Shape[3] != 1 ||
            transposeOutput.IsConstant || reluOutput.IsConstant || transposeOutput.Length == 0 ||
            transposeOutput.Length != reluOutput.Length) return false;
        if (_tensors[transpose.Inputs[0]].Overlaps(reluOutput)) return false;

        // ConvTranspose currently has no bias input in this graph.  Execute it
        // directly into the terminal ReLU buffer, then add the per-channel
        // constant and apply ReLU.  The original graph performs the same
        // operations in this order, so floating-point accumulation is unchanged.
        transposeOutput.ShareStorageWith(reluOutput);
        ExecuteNode(transpose);
        int channels = reluOutput.Shape[1], plane = reluOutput.Length / (reluOutput.Shape[0] * channels);
        for (int batch = 0; batch < reluOutput.Shape[0]; batch++)
            for (int channel = 0; channel < channels; channel++)
            {
                Span<float> values = reluOutput.Data.Slice((batch * channels + channel) * plane, plane);
                float addend = bias.Data[channel];
                SimdKernels.Add(values, addend, values);
                SimdKernels.Relu(values, values);
            }
        return true;
    }

    private bool TryExecuteConvTransposeBiasSigmoid(int index)
    {
        if (index + 2 >= _model.Nodes.Length) return false;
        NodeRecord transpose = _model.Nodes[index], add = _model.Nodes[index + 1], sigmoid = _model.Nodes[index + 2];
        if (transpose.Operator != OperatorId.ConvTranspose || add.Operator != OperatorId.Add ||
            sigmoid.Operator != OperatorId.Sigmoid || transpose.Outputs.Length != 1 ||
            add.Inputs.Length != 2 || add.Outputs.Length != 1 || sigmoid.Inputs.Length != 1 ||
            sigmoid.Outputs.Length != 1 || sigmoid.Inputs[0] != add.Outputs[0] ||
            (add.Inputs[0] != transpose.Outputs[0] && add.Inputs[1] != transpose.Outputs[0]) ||
            HasConsumerAfter(transpose.Outputs[0], index + 1) ||
            HasConsumerAfter(add.Outputs[0], index + 2)) return false;
        uint biasIndex = add.Inputs[0] == transpose.Outputs[0] ? add.Inputs[1] : add.Inputs[0];
        TensorValue bias = _tensors[biasIndex];
        TensorValue transposeOutput = _tensors[transpose.Outputs[0]];
        TensorValue sigmoidOutput = _tensors[sigmoid.Outputs[0]];
        if (!bias.IsConstant || bias.Length != transposeOutput.Shape[1] ||
            bias.Shape.Length != 4 || bias.Shape[0] != 1 || bias.Shape[2] != 1 || bias.Shape[3] != 1 ||
            transposeOutput.IsConstant || sigmoidOutput.IsConstant || transposeOutput.Length == 0 ||
            transposeOutput.Length != sigmoidOutput.Length) return false;
        if (_tensors[transpose.Inputs[0]].Overlaps(sigmoidOutput)) return false;

        transposeOutput.ShareStorageWith(sigmoidOutput);
        ExecuteNode(transpose);
        int channels = sigmoidOutput.Shape[1], plane = sigmoidOutput.Length / (sigmoidOutput.Shape[0] * channels);
        for (int batch = 0; batch < sigmoidOutput.Shape[0]; batch++)
            for (int channel = 0; channel < channels; channel++)
            {
                Span<float> values = sigmoidOutput.Data.Slice((batch * channels + channel) * plane, plane);
                SimdKernels.Add(values, bias.Data[channel], values);
                SimdKernels.Sigmoid(values, values);
            }
        return true;
    }

    private bool HasConsumerAfter(uint tensorIndex, int nodeIndex)
        => _compiled.HasConsumerAfter(tensorIndex, nodeIndex);

    private static bool PartialOverlap(TensorValue a, TensorValue b)
        => a.Overlaps(b) && !a.StorageEquals(b);

    private bool CanReuseTensor(uint tensorIndex, int nodeIndex)
    {
        TensorValue tensor = _tensors[checked((int)tensorIndex)];
        return !tensor.IsConstant && !HasConsumerAfter(tensorIndex, nodeIndex);
    }

    private bool TryExecuteInPlace(NodeRecord node, int nodeIndex)
    {
        if (node.Inputs.Length == 0 || node.Outputs.Length == 0 ||
            !CanReuseTensor(node.Inputs[0], nodeIndex)) return false;
        TensorValue source = _tensors[node.Inputs[0]];
        TensorValue destination = _tensors[node.Outputs[0]];
        if (destination.IsConstant || source.Length != destination.Length)
            return false;

        if (node.Operator == OperatorId.Relu)
        {
            // Write the planned destination, not the input buffer. Aliasing
            // output onto input disagrees with PlanWorkspace, which may reuse
            // the input range after this node while the output is still live.
            SimdKernels.Relu(source.Data, destination.Data);
            return true;
        }
        if (node.Operator is not (OperatorId.Add or OperatorId.Mul or OperatorId.Div or OperatorId.Sub) ||
            node.Inputs.Length < 2 || source.StorageEquals(_tensors[node.Inputs[1]]))
            return false;

        ExecuteNode(node);
        return true;
    }

    private static void Binary<TOp>(TensorValue a, TensorValue b, TensorValue o, int intraOpThreads = 1)
        where TOp : struct, IBinaryOp
    {
        if (b.Length == 1 && a.Length == o.Length)
        {
            SimdKernels.ElementwiseScalar<TOp>(a.Data, b.Data[0], o.Data);
            return;
        }
        if (a.Length == o.Length && b.Length == o.Length) { SimdKernels.ElementwiseParallel<TOp>(a.Data, b.Data, o.Data, intraOpThreads); return; }
        int[] ad = a.Shape; int[] bd = b.Shape; int[] od = o.Shape; int rank = od.Length;
        if (rank == 4 && ad.Length == 4 && bd.Length == 4 && od.Length == 4 &&
            ad[0] == od[0] && ad[1] == od[1] && ad[2] == od[2] && ad[3] == od[3] &&
            bd[0] == 1 && bd[1] == ad[1] && bd[2] == 1 && bd[3] == 1)
        {
            int plane = ad[2] * ad[3];
            SimdKernels.ElementwiseChannel<TOp>(a.Data, b.Data, o.Data, ad[0], ad[1], plane);
            return;
        }
        // Fast path for the ubiquitous trailing-channel bias: [N,C] or
        // [B,N,C] plus a one-dimensional [C] tensor.  The generic index
        // decoder is several times slower for the final REC projection.
        if (bd.Length == 1 && ad.Length == od.Length && bd[0] == ad[^1] &&
            ad.SequenceEqual(od) && rank <= 3)
        {
            int channel = bd[0], outer = checked(o.Length / channel);
            for (int block = 0; block < outer; block++)
                SimdKernels.Elementwise<TOp>(a.Data.Slice(block * channel, channel), b.Data,
                    o.Data.Slice(block * channel, channel));
            return;
        }
        TOp op = default;
        ReadOnlySpan<float> aData = a.Data, bData = b.Data;
        Span<float> oData = o.Data;
        for (int index = 0; index < oData.Length; index++)
        {
            int rem = index, ao = 0, bo = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int c = rem % od[ax]; rem /= od[ax];
                int aa = ax - (rank - ad.Length), bb = ax - (rank - bd.Length);
                if (aa >= 0 && ad[aa] != 1) ao += c * Stride(ad, aa);
                if (bb >= 0 && bd[bb] != 1) bo += c * Stride(bd, bb);
            }
            oData[index] = op.Apply(aData[ao], bData[bo]);
        }
    }
    private static int Stride(int[] d, int axis) { int s = 1; for (int i = axis + 1; i < d.Length; i++) s *= d[i]; return s; }

    private static void BinaryPow(TensorValue a, TensorValue b, TensorValue o)
    {
        if (b.Length == 1 && a.Length == o.Length)
        {
            float exponent = b.Data[0];
            ReadOnlySpan<float> powIn = a.Data; Span<float> powOut = o.Data;
            for (int i = 0; i < powOut.Length; i++) powOut[i] = MathF.Pow(powIn[i], exponent);
            return;
        }
        if (a.Length == o.Length && b.Length == o.Length)
        {
            ReadOnlySpan<float> powA = a.Data, powB = b.Data; Span<float> powO = o.Data;
            for (int i = 0; i < powO.Length; i++) powO[i] = MathF.Pow(powA[i], powB[i]);
            return;
        }
        int[] ad = a.Shape, bd = b.Shape, od = o.Shape;
        int rank = od.Length;
        if (bd.Length == 1 && ad.Length == od.Length && bd[0] == ad[^1] && ad.SequenceEqual(od))
        {
            int channel = bd[0], outer = checked(o.Length / channel);
            ReadOnlySpan<float> blkA = a.Data, blkB = b.Data; Span<float> blkO = o.Data;
            for (int block = 0; block < outer; block++)
                for (int i = 0; i < channel; i++)
                    blkO[block * channel + i] = MathF.Pow(blkA[block * channel + i], blkB[i]);
            return;
        }
        ReadOnlySpan<float> aData = a.Data, bData = b.Data;
        Span<float> oData = o.Data;
        for (int index = 0; index < oData.Length; index++)
        {
            int rem = index, ao = 0, bo = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int c = rem % od[ax]; rem /= od[ax];
                int aa = ax - (rank - ad.Length), bb = ax - (rank - bd.Length);
                if (aa >= 0 && ad[aa] != 1) ao += c * Stride(ad, aa);
                if (bb >= 0 && bd[bb] != 1) bo += c * Stride(bd, bb);
            }
            oData[index] = MathF.Pow(aData[ao], bData[bo]);
        }
    }

    private static void Conv(TensorValue x, TensorValue w, TensorValue? bias, ReadOnlySpan<byte> p,
        TensorValue o, int intraOpThreads = 1, float[]? packedWeights = null, float[]? packed3x3 = null,
        float[]? packedOc16 = null, PackedConv1x1Int8? packedInt8 = null, float[]? packedOc8 = null,
        ReadOnlySpan<float> residual = default)
    {
        int[] id = x.Shape; int[] wd = w.Shape; int[] od = o.Shape; int group = checked((int)U32(p, 4)), kh = I32(p, 8), kw = I32(p, 12), sh = I32(p, 16), sw = I32(p, 20), dh = I32(p, 24), dw = I32(p, 28), pt = I32(p, 32), pl = I32(p, 36); int n = id[0], cin = id[1], h = id[2], wi = id[3], cout = od[1], oh = od[2], ow = od[3], cpg = cin / group, opg = cout / group;
        ReadOnlySpan<float> biasData = bias is null ? [] : bias.Data;
        // A packed 3x3 kernel shards by eight-channel blocks.  For the
        // detector's 16-channel projections that would leave only two tasks
        // at intra-op=4; use the four-channel sharded kernel instead so all
        // worker lanes participate.  Keep the packed path for single-thread
        // execution and wider projections.
        if (group == 1 && kh == 3 && kw == 3 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            pt == 1 && pl == 1 && I32(p, 40) == 1 && I32(p, 44) == 1 && intraOpThreads > 1 &&
            cout == 16 && intraOpThreads > cout / 8 && oh == h && ow == wi &&
            Conv3x3.Try(x.Data, w.Data, biasData, o.Data, n, cin, h, wi, cout,
                intraOpThreads)) return;
        if (group == 1 && kh == 3 && kw == 3 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            pt == 1 && pl == 1 && I32(p, 40) == 1 && I32(p, 44) == 1 && packed3x3 is not null &&
            (cout & 7) == 0 && Conv3x3Packed.Try(x.Data, packed3x3, biasData, o.Data,
                n, cin, h, wi, cout, intraOpThreads)) return;
        if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            pt == 0 && pl == 0 && I32(p, 40) == 0 && I32(p, 44) == 0 &&
            packedOc16 is not null && group == 1 &&
            Conv1x1.TryOcMajor(x.Data, packedOc16, biasData, o.Data,
                n, cin, h, wi, cout, intraOpThreads)) return;
        if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            pt == 0 && pl == 0 && I32(p, 40) == 0 && I32(p, 44) == 0 &&
            packedWeights is not null && group == 1 &&
            Conv1x1.TryPacked(x.Data, packedWeights, biasData, o.Data,
                n, cin, h, wi, cout, intraOpThreads, packedInt8, packedOc8, residual)) return;
        if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            pt == 0 && pl == 0 && I32(p, 40) == 0 && I32(p, 44) == 0 &&
            Conv1x1.Try(x.Data, w.Data, biasData, o.Data, n, cin, h, wi, cout, group, intraOpThreads)) return;
        if (group == cin && cout == cin && wd[1] == 1 && kh == 3 && kw == 3 &&
            sh == 1 && sw == 1 && dh == 1 && dw == 1 && pt == 1 && pl == 1 &&
            I32(p, 40) == 1 && I32(p, 44) == 1 && oh == h && ow == wi &&
            Depthwise.Try3x3(x.Data, w.Data, biasData, o.Data, n, cin, h, wi)) return;
        if (group == cin && cout == cin && wd[1] == 1 && kh == 5 && kw == 5 &&
            sh == 1 && sw == 1 && dh == 1 && dw == 1 && pt == 2 && pl == 2 &&
            I32(p, 40) == 2 && I32(p, 44) == 2 && oh == h && ow == wi &&
            Depthwise.Try5x5(x.Data, w.Data, biasData, o.Data, n, cin, h, wi,
                intraOpThreads)) return;
        if (group == cin && cout == cin && wd[1] == 1 && kh == 7 && kw == 7 &&
            sh == 1 && sw == 1 && dh == 1 && dw == 1 && pt == 3 && pl == 3 &&
            I32(p, 40) == 3 && I32(p, 44) == 3 && oh == h && ow == wi &&
            Depthwise.Try7x7(x.Data, w.Data, biasData, o.Data, n, cin, h, wi,
                intraOpThreads)) return;
        if (group == cin && cout == cin && wd[1] == 1 && kh == 3 && kw == 3 &&
            sh == 2 && sw == 1 && dh == 1 && dw == 1 && pt == 1 && pl == 1 &&
            I32(p, 40) == 1 && I32(p, 44) == 1 &&
            Depthwise.Try3x3StrideHeight2(x.Data, w.Data, biasData, o.Data,
                n, cin, h, wi, oh, ow)) return;
        if (group == cin && cout == cin && wd[1] == 1 && kh == 5 && kw == 5 &&
            sh == 2 && sw == 1 && dh == 1 && dw == 1 && pt == 2 && pl == 2 &&
            I32(p, 40) == 2 && I32(p, 44) == 2 &&
            Depthwise.Try5x5StrideHeight2(x.Data, w.Data, biasData, o.Data,
                n, cin, h, wi, oh, ow)) return;
        if (group == cin && cout == cin && wd[1] == 1 && kh == 1 && kw == 5 &&
            sh == 1 && sw == 1 && dh == 1 && dw == 1 && pt == 0 && pl == 2 &&
            I32(p, 40) == 0 && I32(p, 44) == 2 && oh == h && ow == wi &&
            Depthwise.Try1x5(x.Data, w.Data, biasData, o.Data, n, cin, h, wi)) return;
        if (group == cin && cout == cin && wd[1] == 1 && kh == 3 && kw == 3 &&
            sh == 2 && sw == 2 && dh == 1 && dw == 1 && pt == 1 && pl == 1 &&
            I32(p, 40) == 1 && I32(p, 44) == 1 &&
            Depthwise.Try3x3Stride2(x.Data, w.Data, biasData, o.Data,
                n, cin, h, wi, oh, ow)) return;
        if (group == 1 && kh == 3 && kw == 3 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            pt == 1 && pl == 1 && I32(p, 40) == 1 && I32(p, 44) == 1 && oh == h && ow == wi &&
            Conv3x3.Try(x.Data, w.Data, biasData, o.Data, n, cin, h, wi, cout, intraOpThreads)) return;
        if (group == 1 && kh == 2 && kw == 2 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            pt == 0 && pl == 0 && I32(p, 40) == 1 && I32(p, 44) == 1 && oh == h && ow == wi &&
            Stride2.Try(x.Data, w.Data, biasData, o.Data, n, cin, h, wi, cout)) return;
        if (group == 1 && kh == 3 && kw == 3 && sh == 2 && sw == 2 && dh == 1 && dw == 1 &&
            pt == 1 && pl == 1 && I32(p, 40) == 1 && I32(p, 44) == 1 &&
            packed3x3 is not null && Conv3x3Stride2.TryPacked(x.Data, packed3x3,
                biasData, o.Data, n, cin, h, wi, oh, ow, cout, intraOpThreads)) return;
        if (group == 1 && kh == 3 && kw == 3 && sh == 2 && sw == 2 && dh == 1 && dw == 1 &&
            pt == 1 && pl == 1 && I32(p, 40) == 1 && I32(p, 44) == 1 &&
            Conv3x3Stride2.Try(x.Data, w.Data, biasData, o.Data, n, cin, h, wi, oh, ow, cout, intraOpThreads)) return;
        // Generic stride-1 SIMD kernels: catch the large-kernel branches (7x7,
        // 9x9 depthwise, 1x7/7x1, ...) used by the small/medium detector and
        // recognizer backbones that have no shape-specific kernel above.
        if (group == cin && cout == cin && wd[1] == 1 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            DepthwiseStride1.Try(x.Data, w.Data, biasData, o.Data, n, cin, h, wi,
                oh, ow, kh, kw, pt, pl, intraOpThreads)) return;
        if (group == 1 && sh == 1 && sw == 1 && dh == 1 && dw == 1 &&
            ConvDenseStride1.Try(x.Data, w.Data, biasData, o.Data, n, cin, h, wi,
                cout, oh, ow, kh, kw, pt, pl, intraOpThreads)) return;
        ReadOnlySpan<float> xData = x.Data, wData = w.Data; Span<float> oData = o.Data;
        if (s_dumpConv) DumpConvShape(n, cin, h, wi, cout, oh, ow, group, kh, kw, sh, sw, dh, dw, pt, pl);
        for (int b = 0; b < n; b++) for (int co = 0; co < cout; co++) for (int y = 0; y < oh; y++) for (int xx = 0; xx < ow; xx++) { float sum = bias is null ? 0 : biasData[co]; int g = co / opg; for (int ci = 0; ci < cpg; ci++) for (int ky = 0; ky < kh; ky++) { int iy = y * sh - pt + ky * dh; if ((uint)iy >= (uint)h) continue; for (int kx = 0; kx < kw; kx++) { int ix = xx * sw - pl + kx * dw; if ((uint)ix >= (uint)wi) continue; sum += xData[((b * cin + g * cpg + ci) * h + iy) * wi + ix] * wData[(((co * cpg + ci) * kh + ky) * kw + kx)]; } } oData[((b * cout + co) * oh + y) * ow + xx] = sum; }
    }

    private static readonly bool s_dumpConv = Environment.GetEnvironmentVariable("PPOCR_DUMP_CONV") is not null;
    private static readonly HashSet<string> s_dumpedConvs = [];
    private static void DumpConvShape(int n, int cin, int h, int wi, int cout, int oh, int ow,
        int group, int kh, int kw, int sh, int sw, int dh, int dw, int pt, int pl)
    {
        string key = $"generic-conv k={kh}x{kw} s={sh}x{sw} d={dh}x{dw} p={pt},{pl} g={group} in={cin}x{h}x{wi} out={cout}x{oh}x{ow}";
        lock (s_dumpedConvs)
            if (s_dumpedConvs.Add(key))
                Console.Error.WriteLine(key);
    }
    private static void ConvTranspose(TensorValue x, TensorValue w, TensorValue? bias, ReadOnlySpan<byte> p, TensorValue o,
        int intraOpThreads = 1)
    {
        int[] id = x.Shape; int[] wd = w.Shape; int[] od = o.Shape; int g = checked((int)U32(p, 4)), kh = I32(p, 8), kw = I32(p, 12), sh = I32(p, 16), sw = I32(p, 20), dh = I32(p, 24), dw = I32(p, 28), pt = I32(p, 32), pl = I32(p, 36); int n = id[0], ci = id[1], h = id[2], wi = id[3], co = od[1]; ReadOnlySpan<float> biasData = bias is null ? [] : bias.Data;
        if (g == 1 && kh == 2 && kw == 2 && sh == 2 && sw == 2 && dh == 1 && dw == 1 &&
            pt == 0 && pl == 0 && I32(p, 40) == 0 && I32(p, 44) == 0 &&
            global::Sdcb.SimdPaddleOCR.Kernels.ConvTranspose.Try(x.Data, w.Data, biasData, o.Data, n, ci, h, wi, co,
                intraOpThreads)) return;
        ReadOnlySpan<float> xData = x.Data, wData = w.Data; Span<float> oData = o.Data;
        oData.Clear(); int[] _ = wd;
        for (int b = 0; b < n; b++) for (int ic = 0; ic < ci; ic++) for (int y = 0; y < h; y++) for (int xx = 0; xx < wi; xx++) { float v = xData[((b * ci + ic) * h + y) * wi + xx]; for (int oc = 0; oc < co; oc++) for (int ky = 0; ky < kh; ky++) { int oy = y * sh - pt + ky * dh; if ((uint)oy >= (uint)od[2]) continue; for (int kx = 0; kx < kw; kx++) { int ox = xx * sw - pl + kx * dw; if ((uint)ox >= (uint)od[3]) continue; oData[((b * co + oc) * od[2] + oy) * od[3] + ox] += v * wData[(((ic * co + oc) * kh + ky) * kw + kx)]; } } }
        if (bias is not null) for (int b = 0; b < n; b++) for (int c = 0; c < co; c++) for (int y = 0; y < od[2]; y++) for (int xx = 0; xx < od[3]; xx++) oData[((b * co + c) * od[2] + y) * od[3] + xx] += biasData[c];
    }
    private void BatchNorm(TensorValue x, NodeRecord n, TensorValue o)
    {
        ReadOnlySpan<float> scale = _tensors[n.Inputs[1]].Data, bias = _tensors[n.Inputs[2]].Data,
            mean = _tensors[n.Inputs[3]].Data, vari = _tensors[n.Inputs[4]].Data;
        ReadOnlySpan<byte> p = _model.GetParameters(n);
        float eps = F32(p, 4);
        int c = x.Shape[1], plane = x.Length / (x.Shape[0] * c);
        ReadOnlySpan<float> xData = x.Data; Span<float> oData = o.Data;
        for (int b = 0; b < x.Shape[0]; b++)
            for (int ch = 0; ch < c; ch++)
            {
                float k = scale[ch] / MathF.Sqrt(vari[ch] + eps);
                for (int i = 0; i < plane; i++)
                {
                    int ix = (b * c + ch) * plane + i;
                    oData[ix] = (xData[ix] - mean[ch]) * k + bias[ch];
                }
            }
    }
    private static void ReduceMean(TensorValue x, ReadOnlySpan<byte> p, TensorValue o)
    {
        int rank = x.Shape.Length, count = U16(p, 2);
        bool keep = U32(p, 4) != 0, all = count == 0 && U32(p, 8) == 0;
        Span<bool> reduced = stackalloc bool[8];
        reduced.Slice(0, rank).Clear();
        if (all) reduced.Slice(0, rank).Fill(true);
        else
        {
            for (int k = 0; k < count; k++)
            {
                int a = I32(p, 12 + k * 4);
                if (a < 0) a += rank;
                if ((uint)a >= (uint)rank) throw new InvalidDataException("ReduceMean axis is invalid.");
                if (reduced[a]) throw new InvalidDataException("ReduceMean axis is duplicated.");
                reduced[a] = true;
            }
        }
        int reduction = 1;
        for (int a = 0; a < rank; a++) if (reduced[a]) reduction = checked(reduction * x.Shape[a]);
        if (rank == 4 && o.Shape.Length == 4 && keep && !reduced[0] && !reduced[1] &&
            reduced[2] && reduced[3] && o.Shape[0] == x.Shape[0] && o.Shape[1] == x.Shape[1] &&
            o.Shape[2] == 1 && o.Shape[3] == 1)
        {
            int spatial = checked(x.Shape[2] * x.Shape[3]);
            Sdcb.SimdPaddleOCR.Kernels.ReduceMean.SpatialNchw(x.Data, o.Data, x.Shape[0], x.Shape[1], spatial);
            return;
        }
        int[] outputStrides = new int[o.Shape.Length];
        int stride = 1;
        for (int a = o.Shape.Length - 1; a >= 0; a--) { outputStrides[a] = stride; stride = checked(stride * o.Shape[a]); }
        Span<int> inputToOutput = stackalloc int[8];
        inputToOutput.Slice(0, rank).Fill(-1);
        int outputAxis = 0;
        for (int a = 0; a < rank; a++)
        {
            if (reduced[a])
            {
                if (keep) outputAxis++;
            }
            else inputToOutput[a] = outputAxis++;
        }
        if (outputAxis != o.Shape.Length) throw new InvalidDataException("ReduceMean output rank is invalid.");
        ReadOnlySpan<float> input = x.Data;
        Span<float> output = o.Data;
        output.Clear();
        for (int ii = 0; ii < input.Length; ii++)
        {
            int rem = ii, oi = 0;
            for (int a = rank - 1; a >= 0; a--)
            {
                int c = rem % x.Shape[a]; rem /= x.Shape[a];
                int mapped = inputToOutput[a];
                if (mapped >= 0) oi += c * outputStrides[mapped];
            }
            output[oi] += input[ii];
        }
        for (int i = 0; i < output.Length; i++) output[i] /= reduction;
    }
    private static void Pool(TensorValue x, ReadOnlySpan<byte> p, TensorValue o, bool max)
    {
        int[] id = x.Shape; int[] od = o.Shape;
        int kh = I32(p, 8), kw = I32(p, 12), sh = I32(p, 16), sw = I32(p, 20);
        int pt = I32(p, 24), pl = I32(p, 28), pb = I32(p, 32), pr = I32(p, 36);
        if (max && kh == 2 && kw == 2 && sh == 1 && sw == 1 && pt == 0 && pl == 0 &&
            pb == 1 && pr == 1 && od[2] == id[2] && od[3] == id[3])
        {
            SimdKernels.MaxPool2x2PadEnd(x.Data, o.Data, id[0], id[1], id[2], id[3]);
            return;
        }
        ReadOnlySpan<float> xData = x.Data; Span<float> oData = o.Data;
        for (int b = 0; b < id[0]; b++) for (int c = 0; c < id[1]; c++)
            for (int y = 0; y < od[2]; y++) for (int xx = 0; xx < od[3]; xx++)
            {
                float z = max ? float.NegativeInfinity : 0; int count = 0;
                for (int ky = 0; ky < kh; ky++)
                {
                    int iy = y * sh - pt + ky; if ((uint)iy >= (uint)id[2]) continue;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        int ix = xx * sw - pl + kx; if ((uint)ix >= (uint)id[3]) continue;
                        float v = xData[((b * id[1] + c) * id[2] + iy) * id[3] + ix];
                        if (max) z = MathF.Max(z, v); else { z += v; count++; }
                    }
                }
                oData[((b * id[1] + c) * od[2] + y) * od[3] + xx] = max ? z : z / count;
            }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] private static float id3(float a, float b, float c, float d) => MathF.Max(MathF.Max(a, b), MathF.Max(c, d));
    private static void Transpose(TensorValue x, ReadOnlySpan<byte> p, TensorValue o)
    {
        int rank = x.Shape.Length, n = U16(p, 2);
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = I32(p, 4 + i * 4);
        ReadOnlySpan<float> xData = x.Data; Span<float> oData = o.Data;
        for (int oi = 0; oi < oData.Length; oi++) { int rem = oi, ii = 0; for (int ax = rank - 1; ax >= 0; ax--) { int c = rem % o.Shape[ax]; rem /= o.Shape[ax]; ii += c * Stride(x.Shape, perm[ax]); } oData[oi] = xData[ii]; }
    }
    private void Concat(NodeRecord n, ReadOnlySpan<byte> p, TensorValue o)
    {
        int axis = I32(p, 4);
        if (axis < 0) axis += o.Shape.Length;
        int inner = 1;
        for (int i = axis + 1; i < o.Shape.Length; i++) inner *= o.Shape[i];
        int outer = o.Length / (o.Shape[axis] * inner), dst = 0;
        Span<float> oData = o.Data;
        for (int q = 0; q < outer; q++)
            foreach (uint ti in n.Inputs)
            {
                TensorValue t = _tensors[ti];
                int chunk = t.Shape[axis] * inner;
                t.Data.Slice(q * chunk, chunk).CopyTo(oData.Slice(dst, chunk));
                dst += chunk;
            }
    }
    private static void Softmax(TensorValue x, ReadOnlySpan<byte> p, TensorValue o)
    {
        int axis = I32(p, 4);
        if (axis < 0) axis += x.Shape.Length;
        int dim = x.Shape[axis], inner = 1;
        for (int i = axis + 1; i < x.Shape.Length; i++) inner *= x.Shape[i];
        int outer = x.Length / (dim * inner);
        if (inner == 1) { SimdKernels.SoftmaxContiguous(x.Data, o.Data, outer, dim); return; }
        ReadOnlySpan<float> xData = x.Data; Span<float> oData = o.Data;
        for (int q = 0; q < outer; q++) for (int i = 0; i < inner; i++) { float m = float.NegativeInfinity; for (int j = 0; j < dim; j++) m = MathF.Max(m, xData[(q * dim + j) * inner + i]); float s = 0; for (int j = 0; j < dim; j++) { float e = MathF.Exp(xData[(q * dim + j) * inner + i] - m); oData[(q * dim + j) * inner + i] = e; s += e; } for (int j = 0; j < dim; j++) oData[(q * dim + j) * inner + i] /= s; }
    }
    private static void MatMul(TensorValue a, TensorValue b, TensorValue o,
        float[]? packedWeights = null)
    {
        int[] ad = a.Shape, bd = b.Shape, od = o.Shape;
        if (ad.Length >= 2 && bd.Length == 2 && od.Length >= 2)
        {
            int m = ad[^2], k = ad[^1], n = bd[1], batch = checked(a.Length / (m * k));
            if (global::Sdcb.SimdPaddleOCR.Kernels.MatMul.Try(a.Data, b.Data, o.Data, batch, m, k, n,
                packedWeights)) return;
        }
        // Attention uses [N,heads,L,K] x [N,heads,K,L] with identical leading
        // dims. Route each batch slice through the SIMD kernel instead of the
        // per-element broadcasting fallback below.
        if (ad.Length >= 3 && ad.Length == bd.Length && od.Length == ad.Length &&
            ad[^1] == bd[^2] && ad.AsSpan(0, ad.Length - 2).SequenceEqual(bd.AsSpan(0, bd.Length - 2)))
        {
            int m = ad[^2], k = ad[^1], n = bd[^1];
            int slices = checked(a.Length / (m * k));
            ReadOnlySpan<float> aAll = a.Data, bAll = b.Data;
            Span<float> oAll = o.Data;
            bool ok = true;
            for (int s = 0; s < slices && ok; s++)
                ok = global::Sdcb.SimdPaddleOCR.Kernels.MatMul.Try(aAll.Slice(s * m * k, m * k),
                    bAll.Slice(s * k * n, k * n), oAll.Slice(s * m * n, m * n), 1, m, k, n);
            if (ok) return;
        }

        // ONNX MatMul supports batched matrices and NumPy-style broadcasting
        // of all leading dimensions. The PP-OCRv6 attention blocks use
        // [N,heads,L,K] x [N,heads,K,L], which the old rank-2-only kernel did
        // not cover.
        int aRank = ad.Length, bRank = bd.Length;
        int aAdjustedRank = Math.Max(2, aRank), bAdjustedRank = Math.Max(2, bRank);
        int batchRank = Math.Max(aAdjustedRank, bAdjustedRank) - 2;
        int[] aBatch = new int[batchRank], bBatch = new int[batchRank], outBatch = new int[batchRank];
        for (int axis = 0; axis < batchRank; axis++)
        {
            int aAxis = axis - (batchRank - (aAdjustedRank - 2));
            int bAxis = axis - (batchRank - (bAdjustedRank - 2));
            int av = aAxis >= 0 ? ad[aAxis] : 1;
            int bv = bAxis >= 0 ? bd[bAxis] : 1;
            aBatch[axis] = av; bBatch[axis] = bv; outBatch[axis] = Math.Max(av, bv);
        }
        int mDim = aRank == 1 ? 1 : ad[^2];
        int kDim = ad[^1];
        int bKDim = bRank == 1 ? bd[0] : bd[^2];
        int nDim = bRank == 1 ? 1 : bd[^1];
        if (kDim != bKDim) throw new InvalidDataException("MatMul inner dimensions do not match.");
        int[] outAdjusted = [.. outBatch, mDim, nDim];
        int[] expected = outAdjusted;
        if (aRank == 1) expected = expected.Skip(1).ToArray();
        if (bRank == 1) expected = expected.Take(expected.Length - 1).ToArray();
        if (!expected.SequenceEqual(od)) throw new InvalidDataException($"MatMul output shape mismatch: expected [{string.Join(",", expected)}], got [{string.Join(",", od)}].");

        int outputElements = o.Length;
        ReadOnlySpan<float> aData = a.Data, bData = b.Data;
        Span<float> oData = o.Data;
        Span<int> coordinates = stackalloc int[8];
        Span<int> aCoordinates = stackalloc int[8];
        Span<int> bCoordinates = stackalloc int[8];
        for (int outputIndex = 0; outputIndex < outputElements; outputIndex++)
        {
            int remainder = outputIndex;
            for (int axis = od.Length - 1; axis >= 0; axis--)
            {
                coordinates[axis] = remainder % od[axis];
                remainder /= od[axis];
            }
            int coordinateOffset = 0;
            for (int axis = 0; axis < batchRank; axis++)
            {
                int c = coordinates[coordinateOffset + axis];
                int aAxis = axis - (batchRank - (aAdjustedRank - 2));
                int bAxis = axis - (batchRank - (bAdjustedRank - 2));
                if (aAxis >= 0) aCoordinates[aAxis] = aBatch[axis] == 1 ? 0 : c;
                if (bAxis >= 0) bCoordinates[bAxis] = bBatch[axis] == 1 ? 0 : c;
            }
            int row = aRank == 1 ? 0 : coordinates[coordinateOffset + batchRank];
            int col = bRank == 1 ? 0 : coordinates[coordinateOffset + batchRank + (aRank == 1 ? 0 : 1)];
            if (aRank > 1) { aCoordinates[aRank - 2] = row; aCoordinates[aRank - 1] = 0; }
            else aCoordinates[0] = 0;
            if (bRank > 1) { bCoordinates[bRank - 2] = 0; bCoordinates[bRank - 1] = col; }
            else bCoordinates[0] = 0;

            int aBase = 0, bBase = 0;
            for (int axis = 0; axis < aRank; axis++) aBase += aCoordinates[axis] * Stride(ad, axis);
            for (int axis = 0; axis < bRank; axis++) bBase += bCoordinates[axis] * Stride(bd, axis);
            float sum = 0;
            for (int k = 0; k < kDim; k++)
            {
                int aIndex = aRank == 1 ? k : aBase + k;
                int bIndex = bRank == 1 ? k : bBase + k * Stride(bd, bRank - 2);
                sum += aData[aIndex] * bData[bIndex];
            }
            oData[outputIndex] = sum;
        }
    }
    private static void Resize(TensorValue x, ReadOnlySpan<byte> p, TensorValue o)
    {
        int[] id = x.Shape; int[] od = o.Shape;
        float heightScale = F32(p, 12), widthScale = F32(p, 16);
        int heightFactor = (int)heightScale, widthFactor = (int)widthScale;
        if (id.Length == 4 && od.Length == 4 && F32(p, 4) == 1f && F32(p, 8) == 1f &&
            heightFactor > 0 && widthFactor > 0 && heightScale == heightFactor && widthScale == widthFactor &&
            od[0] == id[0] && od[1] == id[1] && od[2] == id[2] * heightFactor &&
            od[3] == id[3] * widthFactor)
        {
            int inputPlane = id[2] * id[3], outputPlane = od[2] * od[3];
            ReadOnlySpan<float> xData = x.Data;
            Span<float> oData = o.Data;
            for (int b = 0; b < id[0]; b++)
                for (int c = 0; c < id[1]; c++)
                {
                    int inputBase = (b * id[1] + c) * inputPlane;
                    int outputBase = (b * od[1] + c) * outputPlane;
                    for (int iy = 0; iy < id[2]; iy++)
                    {
                        int sourceRow = inputBase + iy * id[3];
                        int firstOutputRow = outputBase + iy * heightFactor * od[3];
                        int ox = 0;
                        for (int ix = 0; ix < id[3]; ix++)
                        {
                            float value = xData[sourceRow + ix];
                            for (int repeat = 0; repeat < widthFactor; repeat++) oData[firstOutputRow + ox++] = value;
                        }
                        for (int repeatY = 1; repeatY < heightFactor; repeatY++)
                            oData.Slice(firstOutputRow, od[3]).CopyTo(oData.Slice(firstOutputRow + repeatY * od[3], od[3]));
                    }
                }
            return;
        }
        ReadOnlySpan<float> xFallback = x.Data; Span<float> oFallback = o.Data;
        for (int i = 0; i < oFallback.Length; i++) { int rem = i; int ox = rem % od[^1]; rem /= od[^1]; int oy = rem % od[^2]; rem /= od[^2]; int c = rem % od[^3]; int b = rem / od[^3]; int iy = Math.Min((int)(oy / heightScale), id[^2] - 1), ix = Math.Min((int)(ox / widthScale), id[^1] - 1); oFallback[i] = xFallback[((b * id[1] + c) * id[2] + iy) * id[3] + ix]; }
    }

    private void Slice(TensorValue x, NodeRecord node, TensorValue o)
    {
        (int[] starts, int[] steps) = _compiled.ResolveSliceBounds(x.Shape, node, out _);
        int rank = x.Shape.Length;
        ReadOnlySpan<float> xData = x.Data;
        Span<float> oData = o.Data;
        for (int outputIndex = 0; outputIndex < oData.Length; outputIndex++)
        {
            int remainder = outputIndex, inputIndex = 0;
            for (int axis = rank - 1; axis >= 0; axis--)
            {
                int coordinate = remainder % o.Shape[axis];
                remainder /= o.Shape[axis];
                inputIndex += (starts[axis] + coordinate * steps[axis]) * Stride(x.Shape, axis);
            }
            oData[outputIndex] = xData[inputIndex];
        }
    }

    private static ushort U16(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt16LittleEndian(p[o..]);
    private static uint U32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt32LittleEndian(p[o..]);
    private static int I32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadInt32LittleEndian(p[o..]);
    private static float F32(ReadOnlySpan<byte> p, int o) => BitConverterCompat.Int32BitsToSingle(I32(p, o));

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _workspace = [];
    }

    /// <summary>
    /// Per-request view of one tensor. Dtype/constness come from the shared
    /// <see cref="CompiledModel.TensorMeta"/>; <see cref="Shape"/> is resolved
    /// per session by <see cref="Reshape"/>; the activation storage is a window
    /// of the session's single workspace block (or, for constants, the shared
    /// weight array). Only the binding changes during fused/in-place ops.
    /// </summary>
    private sealed class TensorValue
    {
        private readonly CompiledModel.TensorMeta _meta;
        private float[]? _buffer;
        private int _offset;
        private int _length;

        public TensorValue(CompiledModel.TensorMeta meta)
        {
            _meta = meta;
            Shape = meta.Shape;
            if (meta.IsConstant)
            {
                _buffer = meta.Data;
                _length = meta.Data.Length;
            }
        }

        public DType DType => _meta.DType;
        public bool IsConstant => _meta.IsConstant;
        public int[] Shape { get; private set; }
        public long ElementCount => Shape.Aggregate(1L, static (a, b) => checked(a * b));
        public float[]? Buffer => _buffer;
        public int Offset => _offset;
        public int Length => _length;

        /// <summary>The tensor's element window; empty while unbound/zero-sized.</summary>
        public Span<float> Data => _buffer is null ? [] : _buffer.AsSpan(_offset, _length);

        public void SetShape(int[] shape) => Shape = shape;

        public void Bind(float[] buffer, int offset, int length)
        {
            _buffer = buffer;
            _offset = offset;
            _length = length;
        }

        /// <summary>Points this tensor at another's storage (fused/in-place ops).</summary>
        public void ShareStorageWith(TensorValue other)
        {
            _buffer = other._buffer;
            _offset = other._offset;
            _length = other._length;
        }

        /// <summary>True when both tensors expose the identical storage window.</summary>
        public bool StorageEquals(TensorValue other)
            => _buffer is not null && ReferenceEquals(_buffer, other._buffer)
               && _offset == other._offset && _length == other._length;

        /// <summary>True when the two windows share any element of the same buffer.</summary>
        public bool Overlaps(TensorValue other)
            => _buffer is not null && other._buffer is not null &&
               ReferenceEquals(_buffer, other._buffer) &&
               _offset < other._offset + other._length &&
               other._offset < _offset + _length;

        /// <summary>True when this tensor is zero-copy bound to <paramref name="array"/>.</summary>
        public bool IsBoundTo(float[] array)
            => _offset == 0 && _length == array.Length && ReferenceEquals(_buffer, array);
    }
}

/// <summary>
/// Operands for the REC CTC projection after
/// <see cref="InferenceSession.TryRunUntilCtcProjection"/>. Spans alias session
/// activations/weights and are valid until the next Run* on that session.
/// </summary>
internal readonly ref struct CtcProjectionOperands
{
    internal CtcProjectionOperands(ReadOnlySpan<float> activations, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, float[]? packedWeights, int batch, int rows, int inner,
        int columns, int matMulNodeIndex)
    {
        Activations = activations;
        Weights = weights;
        Bias = bias;
        PackedWeights = packedWeights;
        Batch = batch;
        Rows = rows;
        Inner = inner;
        Columns = columns;
        MatMulNodeIndex = matMulNodeIndex;
    }

    internal ReadOnlySpan<float> Activations { get; }
    internal ReadOnlySpan<float> Weights { get; }
    internal ReadOnlySpan<float> Bias { get; }
    internal float[]? PackedWeights { get; }
    internal int Batch { get; }
    internal int Rows { get; }
    internal int Inner { get; }
    internal int Columns { get; }
    internal int MatMulNodeIndex { get; }
    internal int RowCount => checked(Batch * Rows);
}

internal readonly record struct NodeTrace(int Index, OperatorId Operator, int[] Shape,
    float Minimum, float Maximum, double Mean);
