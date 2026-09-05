using System.Buffers.Binary;

namespace Sdcb.PaddleOCR.OnnxSharp;

/// <summary>
/// Compiled form of a <see cref="Model"/>, independent of any concrete input
/// shape. Holds only immutable, shareable state: tensor shape templates
/// (dynamic dimensions stay -1), constant tensor data, packed weights, the
/// buffer-reuse plan, and per-tensor last-use info. Thread-safe after
/// construction; share one instance across any number of
/// <see cref="InferenceSession"/> requests and input shapes.
/// </summary>
public sealed class CompiledModel
{
    private readonly Model _model;
    private readonly TensorMeta[] _tensors;
    private readonly int[] _lastUse;
    private readonly int _inputIndex, _outputIndex;
    private readonly int _intraOpThreads;
    private readonly byte[] _fusedSkip;
    private readonly int[] _inplaceSource;
    private readonly int[]? _defaultInputShape;
    private bool _disposed;

    internal CompiledModel(Model model, int intraOpThreads)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _intraOpThreads = Math.Clamp(intraOpThreads, 1, 16);
        if (model.GraphInputs.Count != 1 || model.GraphOutputs.Count != 1)
            throw new NotSupportedException("Only one-input/one-output graphs are supported.");
        ValidateSupportedGraph(model);
        _inputIndex = checked((int)model.GraphInputs[0]);
        _outputIndex = checked((int)model.GraphOutputs[0]);
        _tensors = new TensorMeta[model.Tensors.Length];
        _lastUse = new int[_tensors.Length];
        _fusedSkip = new byte[model.Nodes.Length];
        _inplaceSource = new int[_tensors.Length];
        Array.Fill(_lastUse, -1);
        Array.Fill(_inplaceSource, -1);
        for (int ni = 0; ni < model.Nodes.Length; ni++)
            foreach (uint input in model.Nodes[ni].Inputs)
                _lastUse[checked((int)input)] = ni;
        foreach (uint output in model.GraphOutputs)
            _lastUse[checked((int)output)] = model.Nodes.Length;
        ExtendLastUseForFusedKernels();
        for (int i = 0; i < _tensors.Length; i++)
        {
            TensorRecord t = model.Tensors[i];
            int[] dims = t.Dimensions.Take(checked((int)t.Rank)).ToArray();
            _tensors[i] = new TensorMeta((DType)t.DType, dims,
                (t.Flags & Model.TensorConstant) != 0 ? model.GetTensorBytes(i) : []);
        }
    }

    /// <summary>Creates a compiled model whose requests default to <paramref name="inputShape"/>.</summary>
    internal CompiledModel(Model model, ReadOnlySpan<int> inputShape, int intraOpThreads)
        : this(model, intraOpThreads)
    {
        ValidateInputShape(inputShape);
        _defaultInputShape = inputShape.ToArray();
    }

    private void ValidateInputShape(ReadOnlySpan<int> inputShape)
    {
        foreach (TensorRecord t in _model.Tensors)
        {
            if ((t.Flags & Model.TensorInput) == 0) continue;
            int rank = checked((int)t.Rank);
            if (inputShape.Length != rank)
                throw new ArgumentException("Input rank does not match model.", nameof(inputShape));
            for (int d = 0; d < rank; d++)
            {
                int dim = t.Dimensions[d];
                if (dim != -1 && dim != inputShape[d])
                    throw new ArgumentException("Input shape does not match model.", nameof(inputShape));
            }
        }
    }

    public Model Model => _model;
    internal int[]? DefaultInputShape => _defaultInputShape;
    public int Packed1x1Count
    {
        get
        {
            int count = 0;
            foreach (NodeRecord node in _model.Nodes)
                if (node.Operator == OperatorId.Conv && node.Inputs.Length >= 2 &&
                    _model.GetPackedWeights(node, Model.PackConv1x1) != null)
                    count++;
            return count;
        }
    }
    internal int InputIndex => _inputIndex;
    internal int OutputIndex => _outputIndex;
    internal int IntraOpThreads => _intraOpThreads;
    internal int TensorCount => _tensors.Length;
    internal int NodeCount => _model.Nodes.Length;

    internal TensorMeta GetTensor(int index) => _tensors[index];
    internal NodeRecord GetNode(int index) => _model.Nodes[index];
    internal bool HasConsumerAfter(uint tensorIndex, int nodeIndex)
        => _lastUse[checked((int)tensorIndex)] > nodeIndex;
    internal bool IsGraphOutput(int tensorIndex) => _model.GraphOutputs.Contains((uint)tensorIndex);
    internal int FusedSkip(int nodeIndex) => _fusedSkip[nodeIndex];
    internal int ElementwiseInPlaceSource(int outputTensor) => _inplaceSource[outputTensor];

    // Same patterns InferenceSession fuses at runtime, decided once. Conv
    // fusions still write a later buffer while reading the conv input, so
    // that input's last-use is stretched. Elementwise GELU/HardSwish instead
    // carve the source's exact slot for the sink so fusion is a true in-place
    // write (C does this too) without first-fit's partial-overlap skip.
    private void ExtendLastUseForFusedKernels()
    {
        NodeRecord[] nodes = _model.Nodes;
        for (int i = 0; i < nodes.Length; i++)
        {
            if (MatchGelu(nodes, i))
            {
                _inplaceSource[checked((int)nodes[i + 4].Outputs[0])] = checked((int)nodes[i].Inputs[0]);
                _fusedSkip[i] = 4;
                i += 4;
                continue;
            }
            if (MatchConvRelu(nodes, i))
            {
                StretchLastUse(nodes[i].Inputs[0], i + 1);
                _fusedSkip[i] = 1;
                i += 1;
                continue;
            }
            if (MatchHardSwish(nodes, i))
            {
                _inplaceSource[checked((int)nodes[i + 1].Outputs[0])] = checked((int)nodes[i].Inputs[0]);
                _fusedSkip[i] = 1;
                i += 1;
                continue;
            }
            if (MatchConvHardSwish(nodes, i))
            {
                StretchLastUse(nodes[i].Inputs[0], i + 2);
                _fusedSkip[i] = 2;
                i += 2;
                continue;
            }
            if (MatchConvTransposeBiasActivation(nodes, i))
            {
                StretchLastUse(nodes[i].Inputs[0], i + 2);
                _fusedSkip[i] = 2;
                i += 2;
            }
        }
    }

    private bool MatchGelu(NodeRecord[] nodes, int i)
    {
        if (i + 4 >= nodes.Length) return false;
        NodeRecord n0 = nodes[i], n1 = nodes[i + 1], n2 = nodes[i + 2], n3 = nodes[i + 3], n4 = nodes[i + 4];
        if (n0.Operator != OperatorId.Div || n1.Operator != OperatorId.Erf ||
            n2.Operator != OperatorId.Add || n3.Operator != OperatorId.Mul ||
            n4.Operator != OperatorId.Mul || n0.Inputs.Length != 2 ||
            n1.Inputs.Length != 1 || n2.Inputs.Length != 2 || n3.Inputs.Length != 2 ||
            n4.Inputs.Length != 2 || n0.Outputs.Length == 0 || n1.Outputs.Length == 0 ||
            n2.Outputs.Length == 0 || n3.Outputs.Length == 0 || n4.Outputs.Length == 0)
            return false;
        if (n1.Inputs[0] != n0.Outputs[0] ||
            !Contains(n2.Inputs, n1.Outputs[0]) ||
            !Contains(n3.Inputs, n0.Inputs[0]) ||
            !Contains(n3.Inputs, n2.Outputs[0]) ||
            !Contains(n4.Inputs, n3.Outputs[0]))
            return false;
        int n2Constant = n2.Inputs[0] == n1.Outputs[0] ? 1 : n2.Inputs[1] == n1.Outputs[0] ? 0 : -1;
        int n4Constant = n4.Inputs[0] == n3.Outputs[0] ? 1 : n4.Inputs[1] == n3.Outputs[0] ? 0 : -1;
        return n2Constant >= 0 && n4Constant >= 0 &&
            IsConstantScalar(n0.Inputs[1], 1.4142135381698608f) &&
            IsConstantScalar(n2.Inputs[n2Constant], 1f) &&
            IsConstantScalar(n4.Inputs[n4Constant], 0.5f) &&
            !HasConsumerAfter(n0.Outputs[0], i + 3) &&
            !HasConsumerAfter(n1.Outputs[0], i + 2) &&
            !HasConsumerAfter(n2.Outputs[0], i + 3) &&
            !HasConsumerAfter(n3.Outputs[0], i + 4);
    }

    private bool MatchConvRelu(NodeRecord[] nodes, int i)
    {
        if (i + 1 >= nodes.Length) return false;
        NodeRecord conv = nodes[i], relu = nodes[i + 1];
        return conv.Operator == OperatorId.Conv && relu.Operator == OperatorId.Relu &&
            conv.Inputs.Length > 0 && conv.Outputs.Length == 1 &&
            relu.Inputs.Length == 1 && relu.Outputs.Length == 1 &&
            relu.Inputs[0] == conv.Outputs[0] &&
            !HasConsumerAfter(conv.Outputs[0], i + 1);
    }

    private bool MatchHardSwish(NodeRecord[] nodes, int i)
    {
        if (i + 1 >= nodes.Length) return false;
        NodeRecord hardSigmoid = nodes[i], multiply = nodes[i + 1];
        return hardSigmoid.Operator == OperatorId.HardSigmoid && multiply.Operator == OperatorId.Mul &&
            hardSigmoid.Inputs.Length == 1 && hardSigmoid.Outputs.Length == 1 &&
            multiply.Inputs.Length == 2 && multiply.Outputs.Length == 1 &&
            Contains(multiply.Inputs, hardSigmoid.Outputs[0]) &&
            Contains(multiply.Inputs, hardSigmoid.Inputs[0]) &&
            !HasConsumerAfter(hardSigmoid.Outputs[0], i + 1);
    }

    private bool MatchConvHardSwish(NodeRecord[] nodes, int i)
    {
        if (i + 2 >= nodes.Length) return false;
        NodeRecord conv = nodes[i], hardSigmoid = nodes[i + 1], multiply = nodes[i + 2];
        return conv.Operator == OperatorId.Conv && hardSigmoid.Operator == OperatorId.HardSigmoid &&
            multiply.Operator == OperatorId.Mul && conv.Inputs.Length > 0 &&
            conv.Outputs.Length == 1 && hardSigmoid.Inputs.Length == 1 &&
            hardSigmoid.Outputs.Length == 1 && multiply.Inputs.Length == 2 &&
            multiply.Outputs.Length == 1 && hardSigmoid.Inputs[0] == conv.Outputs[0] &&
            Contains(multiply.Inputs, hardSigmoid.Outputs[0]) &&
            Contains(multiply.Inputs, conv.Outputs[0]) &&
            !HasConsumerAfter(conv.Outputs[0], i + 2) &&
            !HasConsumerAfter(hardSigmoid.Outputs[0], i + 2);
    }

    private bool MatchConvTransposeBiasActivation(NodeRecord[] nodes, int i)
    {
        if (i + 2 >= nodes.Length) return false;
        NodeRecord transpose = nodes[i], add = nodes[i + 1], act = nodes[i + 2];
        return transpose.Operator == OperatorId.ConvTranspose && add.Operator == OperatorId.Add &&
            act.Operator is OperatorId.Relu or OperatorId.Sigmoid &&
            transpose.Inputs.Length > 0 && transpose.Outputs.Length == 1 &&
            add.Inputs.Length == 2 && add.Outputs.Length == 1 &&
            act.Inputs.Length == 1 && act.Outputs.Length == 1 &&
            Contains(add.Inputs, transpose.Outputs[0]) &&
            act.Inputs[0] == add.Outputs[0] &&
            !HasConsumerAfter(transpose.Outputs[0], i + 1) &&
            !HasConsumerAfter(add.Outputs[0], i + 2);
    }

    private bool IsConstantScalar(uint tensorIndex, float expected)
    {
        int index = checked((int)tensorIndex);
        TensorRecord tensor = _model.Tensors[index];
        if ((tensor.Flags & Model.TensorConstant) == 0) return false;
        int rank = checked((int)tensor.Rank);
        long count = 1;
        for (int d = 0; d < rank; d++)
            count *= tensor.Dimensions[d];
        if (count != 1) return false;
        ReadOnlySpan<byte> bytes = _model.GetTensorBytes(index);
        return bytes.Length >= 4 && BinaryPrimitives.ReadSingleLittleEndian(bytes) == expected;
    }

    private void StretchLastUse(uint tensorIndex, int nodeIndex)
    {
        int index = checked((int)tensorIndex);
        if (_lastUse[index] < nodeIndex)
            _lastUse[index] = nodeIndex;
    }

    private static bool Contains(uint[] inputs, uint value)
    {
        foreach (uint input in inputs)
            if (input == value) return true;
        return false;
    }

    // Packed weights are shared across all input shapes via the Model-level
    // cache; only the lookup is delegated here.
    internal bool TryGetPacked1x1(uint weightIndex, out float[]? packed)
        => TryGetPacked(weightIndex, Model.PackConv1x1, out packed);
    internal bool TryGetPacked3x3(uint weightIndex, out float[]? packed)
        => TryGetPacked(weightIndex, Model.PackConv3x3, out packed);
    internal bool TryGetPackedMatMul(uint weightIndex, out float[]? packed)
        => TryGetPacked(weightIndex, Model.PackMatMul, out packed);

    private bool TryGetPacked(uint weightIndex, int kind, out float[]? packed)
    {
        packed = null;
        foreach (NodeRecord node in _model.Nodes)
        {
            if (node.Inputs.Length < 2 || node.Inputs[1] != weightIndex) continue;
            packed = _model.GetPackedWeights(node, kind);
            if (packed != null) return true;
        }
        return false;
    }

    /// <summary>Creates a new inference request with its own activation workspace.</summary>
    public InferenceSession CreateRequest()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledModel));
        return new InferenceSession(this);
    }

    private static void ValidateSupportedGraph(Model model)
    {
        for (int i = 0; i < model.Nodes.Length; i++)
        {
            NodeRecord node = model.Nodes[i];
            if (node.Operator != OperatorId.Unknown) continue;
            string op = node.OpType.Length == 0 ? node.Operator.ToString() : node.OpType;
            string domain = node.Domain.Length == 0 ? "ai.onnx" : node.Domain;
            throw new NotSupportedException($"Unsupported ONNX operator '{op}' at node {i} ('{node.Name}'), domain='{domain}', opset={node.Opset}.");
        }
        for (int i = 0; i < model.Nodes.Length; i++)
        {
            NodeRecord node = model.Nodes[i];
            int expected = node.Operator switch
            {
                OperatorId.Conv or OperatorId.ConvTranspose or OperatorId.AveragePool or OperatorId.MaxPool => 64,
                OperatorId.BatchNormalization => 24,
                OperatorId.HardSigmoid or OperatorId.Softmax or OperatorId.Concat => 16,
                OperatorId.ReduceMean => 48,
                OperatorId.Squeeze or OperatorId.Unsqueeze or OperatorId.Transpose => 40,
                OperatorId.Resize => 32,
                _ => 0
            };
            if (expected != 0 && model.GetParameters(node).Length != expected)
            {
                string op = node.OpType.Length == 0 ? node.Operator.ToString() : node.OpType;
                throw new NotSupportedException($"Unsupported ONNX attribute layout for '{op}' at node {i} ('{node.Name}'), domain='ai.onnx', opset={node.Opset}.");
            }
        }
    }

    /// <summary>
    /// Resolves every tensor's concrete shape for one input shape into a fresh
    /// array; the shared templates keep their symbolic (-1) dimensions. Pure
    /// with respect to this instance's state, so a session may call it on every
    /// reshape. Reuses the exact operator shape logic (including the PaddleX
    /// dynamic-export reshape heuristics) so per-width decisions are identical.
    /// </summary>
    internal int[][] ResolveShapesFor(ReadOnlySpan<int> inputShape)
    {
        ValidateInputShape(inputShape);
        int[][] shapes = new int[_tensors.Length][];
        for (int i = 0; i < shapes.Length; i++)
        {
            int[] dims = (int[])_tensors[i].Shape.Clone();
            if ((_model.Tensors[i].Flags & Model.TensorInput) != 0)
                for (int d = 0; d < dims.Length; d++)
                    dims[d] = inputShape[d];
            shapes[i] = dims;
        }
        for (int ni = 0; ni < _model.Nodes.Length; ni++)
        {
            NodeRecord node = _model.Nodes[ni];
            ReadOnlySpan<byte> p = _model.GetParameters(node);
            int[] a = shapes[node.Inputs[0]];
            int[] shape = node.Operator switch
            {
                OperatorId.Add or OperatorId.Mul or OperatorId.Div or OperatorId.Sub or OperatorId.Pow => BroadcastShape(a, shapes[node.Inputs[1]], ni, node.Operator),
                OperatorId.Erf or OperatorId.HardSigmoid or OperatorId.Relu or OperatorId.Sigmoid or OperatorId.Sqrt or OperatorId.Softmax or OperatorId.BatchNormalization => [.. a],
                OperatorId.Conv => ConvShape(a, shapes[node.Inputs[1]], p),
                OperatorId.ConvTranspose => ConvTransposeShape(a, shapes[node.Inputs[1]], p),
                OperatorId.ReduceMean => ReduceShape(a, p),
                OperatorId.AveragePool or OperatorId.MaxPool => PoolShape(a, p),
                OperatorId.Squeeze => SqueezeShape(a, p),
                OperatorId.Unsqueeze => UnsqueezeShape(a, p),
                OperatorId.Transpose => TransposeShape(a, p),
                OperatorId.Reshape => node.Inputs.Length > 1
                    ? ReshapeShape(a, _tensors[node.Inputs[1]], _tensors[node.Outputs[0]].Shape)
                    : ReshapeShape(a, _tensors[node.Outputs[0]].Shape),
                OperatorId.Concat => ConcatShape([.. node.Inputs.Select(i => shapes[i])], p),
                OperatorId.MatMul => MatMulShape(a, shapes[node.Inputs[1]]),
                OperatorId.Resize => ResizeShape(a, p),
                OperatorId.Slice => SliceShape(a, node, ni),
                _ => [.. a]
            };
            shapes[node.Outputs[0]] = shape;
        }
        return shapes;
    }

    private int[] SliceShape(int[] inputShape, NodeRecord node, int nodeIndex)
    {
        (int[] starts, int[] steps) = ResolveSliceBounds(inputShape, node, out long[] ends);
        int[] output = inputShape.ToArray();
        for (int axis = 0; axis < output.Length; axis++)
        {
            long end = ends[axis];
            int step = steps[axis], start = starts[axis], dimension = inputShape[axis];
            output[axis] = SliceLength(start, end, step, dimension);
        }
        return output;
    }

    internal (int[] Starts, int[] Steps) ResolveSliceBounds(int[] inputShape, NodeRecord node, out long[] endsByAxis)
    {
        if (node.Inputs.Length < 3)
            throw new InvalidDataException("ONNX Slice requires starts and ends inputs.");
        long[] startsValues = _tensors[node.Inputs[1]].GetIntegerValues();
        long[] endsValues = _tensors[node.Inputs[2]].GetIntegerValues();
        if (startsValues.Length != endsValues.Length || startsValues.Length == 0 || startsValues.Length > inputShape.Length)
            throw new InvalidDataException("ONNX Slice starts/ends are invalid.");
        long[] axesValues = node.Inputs.Length > 3
            ? _tensors[node.Inputs[3]].GetIntegerValues()
            : Enumerable.Range(0, startsValues.Length).Select(static x => (long)x).ToArray();
        long[] stepsValues = node.Inputs.Length > 4
            ? _tensors[node.Inputs[4]].GetIntegerValues()
            : Enumerable.Repeat(1L, startsValues.Length).ToArray();
        if (axesValues.Length != startsValues.Length || stepsValues.Length != startsValues.Length)
            throw new InvalidDataException("ONNX Slice axes/steps length does not match starts.");

        int[] starts = new int[inputShape.Length];
        int[] steps = new int[inputShape.Length];
        endsByAxis = new long[inputShape.Length];
        for (int axis = 0; axis < inputShape.Length; axis++)
        {
            starts[axis] = 0;
            steps[axis] = 1;
            endsByAxis[axis] = inputShape[axis];
        }
        for (int i = 0; i < startsValues.Length; i++)
        {
            long rawAxis = axesValues[i];
            if (rawAxis < 0) rawAxis += inputShape.Length;
            if ((ulong)rawAxis >= (ulong)inputShape.Length)
                throw new InvalidDataException("ONNX Slice axis is invalid.");
            int axis = checked((int)rawAxis);
            if (steps[axis] != 1 || endsByAxis[axis] != inputShape[axis])
                throw new InvalidDataException("ONNX Slice contains duplicate axes.");
            long rawStep = stepsValues[i];
            if (rawStep == 0 || rawStep < int.MinValue || rawStep > int.MaxValue)
                throw new InvalidDataException("ONNX Slice step is invalid.");
            int step = checked((int)rawStep), dimension = inputShape[axis];
            (int start, long end) = NormalizeSlice(startsValues[i], endsValues[i], step, dimension);
            starts[axis] = start;
            steps[axis] = step;
            endsByAxis[axis] = end;
        }
        return (starts, steps);
    }

    private static (int Start, long End) NormalizeSlice(long start, long end, int step, int dimension)
    {
        if (dimension < 0) throw new InvalidDataException("ONNX Slice input has a dynamic dimension.");
        if (step > 0)
        {
            long normalizedStart = start < 0 ? start + dimension : start;
            long normalizedEnd = end < 0 ? end + dimension : end;
            normalizedStart = Math.Clamp(normalizedStart, 0, dimension);
            normalizedEnd = Math.Clamp(normalizedEnd, 0, dimension);
            return (checked((int)normalizedStart), normalizedEnd);
        }
        long negativeStart = start == long.MaxValue ? dimension - 1 : start;
        long negativeEnd = end == long.MinValue ? -1 : end;
        if (negativeStart < 0) negativeStart += dimension;
        if (negativeEnd < 0 && negativeEnd != -1) negativeEnd += dimension;
        negativeStart = Math.Clamp(negativeStart, -1, dimension - 1);
        negativeEnd = Math.Clamp(negativeEnd, -1, dimension - 1);
        return (checked((int)negativeStart), negativeEnd);
    }

    private static int SliceLength(int start, long end, int step, int dimension)
    {
        if (step > 0)
            return start < end ? checked((int)((end - start + step - 1) / step)) : 0;
        long distance = (long)start - end, stride = -(long)step;
        return distance > 0 ? checked((int)((distance + stride - 1) / stride)) : 0;
    }

    private static int[] BroadcastShape(int[] a, int[] b, int node, OperatorId op) { int r = Math.Max(a.Length, b.Length); int[] o = new int[r]; for (int i = 0; i < r; i++) { int x = i >= r - a.Length ? a[i - r + a.Length] : 1, y = i >= r - b.Length ? b[i - r + b.Length] : 1; if (x != y && x != 1 && y != 1) throw new InvalidDataException($"Broadcast shape mismatch at node {node} ({op}): [{string.Join(",", a)}] + [{string.Join(",", b)}]"); o[i] = Math.Max(x, y); } return o; }
    private static int[] ConvShape(int[] i, int[] w, ReadOnlySpan<byte> p) => [i[0], w[0], (i[2] + I32(p, 32) + I32(p, 40) - ((I32(p, 8) - 1) * I32(p, 24) + 1)) / I32(p, 16) + 1, (i[3] + I32(p, 36) + I32(p, 44) - ((I32(p, 12) - 1) * I32(p, 28) + 1)) / I32(p, 20) + 1];
    private static int[] ConvTransposeShape(int[] i, int[] w, ReadOnlySpan<byte> p) => [i[0], checked((int)(w[1] * U32(p, 4))), (i[2] - 1) * I32(p, 16) - I32(p, 32) - I32(p, 40) + (I32(p, 8) - 1) * I32(p, 24) + 1, (i[3] - 1) * I32(p, 20) - I32(p, 36) - I32(p, 44) + (I32(p, 12) - 1) * I32(p, 28) + 1];
    private static int[] PoolShape(int[] i, ReadOnlySpan<byte> p) => [i[0], i[1], (i[2] + I32(p, 24) + I32(p, 32) - I32(p, 8)) / I32(p, 16) + 1, (i[3] + I32(p, 28) + I32(p, 36) - I32(p, 12)) / I32(p, 20) + 1];
    private static int[] ReduceShape(int[] i, ReadOnlySpan<byte> p) { int count = U16(p, 2); bool keep = U32(p, 4) != 0; bool all = count == 0 && U32(p, 8) == 0; bool[] red = new bool[i.Length]; if (all) Array.Fill(red, true); else for (int k = 0; k < count; k++) { int a = I32(p, 12 + k * 4); if (a < 0) a += i.Length; if ((uint)a < (uint)i.Length) red[a] = true; } List<int> o = []; for (int k = 0; k < i.Length; k++) if (red[k]) { if (keep) o.Add(1); } else o.Add(i[k]); return [.. o]; }
    private static int[] SqueezeShape(int[] i, ReadOnlySpan<byte> p) { int count = U16(p, 2); if (count == 0) return [.. i.Where(d => d != 1)]; bool[] rm = new bool[i.Length]; for (int k = 0; k < count; k++) { int a = I32(p, 4 + k * 4); if (a < 0) a += i.Length; if ((uint)a < (uint)i.Length) rm[a] = true; } return [.. i.Where((d, k) => !rm[k])]; }
    private static int[] UnsqueezeShape(int[] i, ReadOnlySpan<byte> p) { int n = U16(p, 2); List<int> a = i.ToList(); for (int k = 0; k < n; k++) a.Insert(Math.Clamp(I32(p, 4 + k * 4), 0, a.Count), 1); return [.. a]; }
    private static int[] TransposeShape(int[] i, ReadOnlySpan<byte> p) { int n = U16(p, 2); int[] a = new int[n]; for (int k = 0; k < n; k++) a[k] = i[I32(p, 4 + k * 4)]; return a; }
    private static int[] ReshapeShape(int[] input, int[] existing)
    {
        if (existing.All(static d => d > 0)) return [.. existing];
        if (input.Any(static d => d <= 0)) return [.. input];
        long inputElements = input.Aggregate(1L, static (a, b) => checked(a * b));

        // PaddleX's PP-LCNet orientation model flattens [N,C,1,1] through a
        // dynamic Shape -> Slice -> Concat expression equivalent to [N,-1].
        // After metadata-only nodes are removed, value_info exposes this as
        // a rank-one symbolic template. Keep the batch dimension instead of
        // collapsing all four axes into [C].
        if (existing.Length == 1 && existing[0] <= 0 && input.Length >= 2 &&
            input[0] > 0 && input.Skip(2).All(static d => d == 1))
        {
            long flattened = 1;
            for (int i = 1; i < input.Length; i++) flattened = checked(flattened * input[i]);
            if (flattened <= int.MaxValue) return [input[0], (int)flattened];
        }

        // PaddleX PP-OCRv6 small/medium REC exports one attention tensor via
        // a dynamic Shape/Slice/Concat expression equivalent to
        // [0, 1, width, channels].  The metadata chain is intentionally
        // removed from the execution plan, so value_info only leaves the
        // under-constrained template [-1, -1, -1, channels].  Preserve the
        // singleton axis in its original position; the generic partitioning
        // below would otherwise choose [batch, width, 1, channels], which
        // changes the subsequent [0, 3, 1, 2] transpose and broadcasts the
        // attention residual to width x width.
        if (input.Length == 3 && existing.Length == 4 && existing[^1] == input[^1] &&
            existing.Take(3).All(static d => d <= 0) &&
            inputElements == (long)input[0] * input[1] * input[2])
            return [input[0], 1, input[1], input[2]];

        // PaddleX's dynamic exports often remove the Shape/Slice/Concat
        // metadata chain while leaving symbolic dimensions in value_info. The
        // resulting shape can still be recovered without executing integer
        // tensors: partition contiguous input dimensions into output
        // dimensions, allowing singleton dimensions to be inserted. This
        // covers flatten [N,C,1,W] -> [N,C,W], attention [N,L,8,15] ->
        // [N,L,120], and the final [N,L,120] -> [N,1,L,120] reshape used by
        // PP-OCRv6 small/medium.
        int[] inferred = new int[existing.Length];
        if (TryInferReshape(input, existing, 0, 0, inferred, inputElements))
            return inferred;

        int unknown = existing.Count(static d => d <= 0);
        if (unknown == 1)
        {
            long known = 1;
            foreach (int d in existing) if (d > 0) known = checked(known * d);
            if (known > 0 && inputElements % known == 0)
            {
                int[] result = [.. existing];
                int missing = checked((int)(inputElements / known));
                for (int k = 0; k < result.Length; k++) if (result[k] <= 0) result[k] = missing;
                return result;
            }
        }
        return [.. input];
    }

    private static bool TryInferReshape(int[] input, int[] template, int outputAxis,
        int inputAxis, int[] result, long inputElements)
    {
        if (outputAxis == template.Length)
            return inputAxis == input.Length;
        int remainingOutput = template.Length - outputAxis;
        if (inputAxis == input.Length)
        {
            // Remaining output axes may only be explicit/symbolic singleton
            // dimensions; consuming no input dimensions represents insertion
            // of a singleton axis.
            for (int i = outputAxis; i < template.Length; i++)
                if (template[i] > 1) return false;
            for (int i = outputAxis; i < template.Length; i++) result[i] = 1;
            return true;
        }

        int maxConsume = input.Length - inputAxis;
        // Prefer preserving individual axes. If that cannot satisfy the
        // remaining known dimensions, the recursion naturally backtracks to a
        // flattening group (for example 8*15=120).
        for (int consume = 1; consume <= maxConsume; consume++)
        {
            long value = 1;
            for (int i = 0; i < consume; i++) value = checked(value * input[inputAxis + i]);
            int expected = template[outputAxis];
            if (expected > 0 && value != expected) continue;
            if (value > int.MaxValue) continue;
            result[outputAxis] = (int)value;
            if (TryInferReshape(input, template, outputAxis + 1, inputAxis + consume,
                result, inputElements)) return true;
        }

        // A symbolic output dimension can represent an inserted singleton.
        // Do this after consuming real input axes so [N,L,120] maps to
        // [N,1,L,120] rather than an ambiguous alternative.
        if (template[outputAxis] <= 1)
        {
            result[outputAxis] = template[outputAxis] > 0 ? template[outputAxis] : 1;
            if (TryInferReshape(input, template, outputAxis + 1, inputAxis,
                result, inputElements)) return true;
        }
        return false;
    }

    private static int[] ReshapeShape(int[] input, TensorMeta shapeTensor, int[] existing)
    {
        if (!shapeTensor.IsConstant || shapeTensor.DType is not (DType.I64 or DType.I32))
            return ReshapeShape(input, existing);
        long[] requested = shapeTensor.GetIntegerValues();
        if (requested.Length == 0 || requested.Length > 8) return ReshapeShape(input, existing);
        long inputElements = input.Aggregate(1L, static (a, b) => checked(a * b));
        int[] result = new int[requested.Length];
        int unknown = -1;
        long known = 1;
        for (int i = 0; i < requested.Length; i++)
        {
            long value = requested[i];
            if (value == 0)
            {
                if (i >= input.Length) return ReshapeShape(input, existing);
                value = input[i];
            }
            if (value == -1)
            {
                if (unknown >= 0) return ReshapeShape(input, existing);
                unknown = i;
                result[i] = -1;
                continue;
            }
            if (value <= 0 || value > int.MaxValue) return ReshapeShape(input, existing);
            result[i] = (int)value;
            known = checked(known * value);
        }
        if (unknown >= 0)
        {
            if (known <= 0 || inputElements % known != 0) return ReshapeShape(input, existing);
            long inferred = inputElements / known;
            if (inferred <= 0 || inferred > int.MaxValue) return ReshapeShape(input, existing);
            result[unknown] = (int)inferred;
        }
        else if (known != inputElements)
        {
            return ReshapeShape(input, existing);
        }
        return result;
    }
    private static int[] ConcatShape(int[][] i, ReadOnlySpan<byte> p) { int a = I32(p, 4); if (a < 0) a += i[0].Length; int[] o = i[0].ToArray(); o[a] = i.Sum(x => x[a]); return o; }
    private static int[] MatMulShape(int[] a, int[] b)
    {
        if (a.Length == 0 || b.Length == 0)
            throw new InvalidDataException($"Invalid MatMul ranks: [{string.Join(",", a)}] x [{string.Join(",", b)}]");
        int aAdjustedRank = Math.Max(2, a.Length), bAdjustedRank = Math.Max(2, b.Length);
        int batchRank = Math.Max(aAdjustedRank, bAdjustedRank) - 2;
        List<int> output = new(batchRank + 2);
        for (int axis = 0; axis < batchRank; axis++)
        {
            int aAxis = axis - (batchRank - (aAdjustedRank - 2));
            int bAxis = axis - (batchRank - (bAdjustedRank - 2));
            int av = aAxis >= 0 ? a[aAxis] : 1;
            int bv = bAxis >= 0 ? b[bAxis] : 1;
            if (av != bv && av != 1 && bv != 1)
                throw new InvalidDataException($"MatMul batch dimensions do not broadcast: [{string.Join(",", a)}] x [{string.Join(",", b)}]");
            output.Add(Math.Max(av, bv));
        }
        int k = a[^1], bK = b.Length == 1 ? b[0] : b[^2];
        if (k != bK) throw new InvalidDataException($"MatMul inner dimensions do not match: {k} != {bK}.");
        if (a.Length != 1) output.Add(a[^2]);
        if (b.Length != 1) output.Add(b[^1]);
        return [.. output];
    }
    private static int[] ResizeShape(int[] i, ReadOnlySpan<byte> p) { int[] o = i.ToArray(); for (int k = 0; k < o.Length; k++) o[k] = Math.Max(1, (int)(o[k] * F32(p, 4 + k * 4))); return o; }
    private static ushort U16(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt16LittleEndian(p[o..]);
    private static uint U32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt32LittleEndian(p[o..]);
    private static int I32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadInt32LittleEndian(p[o..]);
    private static float F32(ReadOnlySpan<byte> p, int o) => BitConverter.Int32BitsToSingle(I32(p, o));

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (TensorMeta t in _tensors) t.Dispose();
    }

    /// <summary>
    /// Immutable compile-time view of one tensor: dtype, shape template
    /// (dynamic dimensions stay -1), and (for constants) the weight data.
    /// Concrete shapes and activation storage live in the per-request
    /// <see cref="InferenceSession"/>, never here.
    /// </summary>
    internal sealed class TensorMeta
    {
        public DType DType;
        public int[] Shape;
        public float[] Data;
        public bool IsConstant;
        private byte[] _constant;
        public TensorMeta(DType d, int[] s, ReadOnlySpan<byte> c)
        {
            DType = d;
            Shape = s;
            IsConstant = !c.IsEmpty;
            _constant = c.ToArray();
            Data = IsConstant ? System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(_constant.AsSpan()).ToArray() : [];
        }
        public void Dispose()
        {
            Data = [];
            _constant = [];
        }
        public long[] GetIntegerValues()
        {
            if (!IsConstant) return [];
            if (DType == DType.I64)
            {
                if ((_constant.Length & 7) != 0) return [];
                long[] result = new long[_constant.Length / 8];
                for (int i = 0; i < result.Length; i++)
                    result[i] = BinaryPrimitives.ReadInt64LittleEndian(_constant.AsSpan(i * 8, 8));
                return result;
            }
            if (DType == DType.I32)
            {
                if ((_constant.Length & 3) != 0) return [];
                long[] result = new long[_constant.Length / 4];
                for (int i = 0; i < result.Length; i++)
                    result[i] = BinaryPrimitives.ReadInt32LittleEndian(_constant.AsSpan(i * 4, 4));
                return result;
            }
            return [];
        }
    }
}
