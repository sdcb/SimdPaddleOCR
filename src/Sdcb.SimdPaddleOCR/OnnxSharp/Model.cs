using System.Buffers.Binary;
using System.Text;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>Validated ONNX model. The serialized source is released after parsing.</summary>
public sealed class Model : IDisposable
{
    internal const uint TensorConstant = 1, TensorInput = 2, TensorOutput = 4;
    // The serialized ONNX payload is only needed while parsing. Tensor data
    // and node parameters are copied into their own arrays below, so retaining
    // the original file here would unnecessarily keep the (often very large)
    // model blob alive for the lifetime of every pipeline.
    private int _disposed;
    private readonly byte[][] _tensorData;
    private readonly byte[][] _nodeParameters;

    // Shared packed-weight cache. Packed weights depend only on the weight
    // tensor's channel/kernel layout and the owning node's attributes, never
    // on the activation H/W, so one pack can be reused by every CompiledModel
    // (i.e. every input shape) of this model. Keyed by (weightTensorIndex,
    // packKind); computed lazily and thread-safe.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<(int WeightIndex, int Kind), float[]?> _packedWeights = new();
    private readonly System.Collections.Concurrent.ConcurrentDictionary<int, PackedConv1x1Int8?> _packedConv1x1Int8 = new();

    private Model(ModelInfo info, uint[] inputs, uint[] outputs,
        TensorRecord[] tensors, NodeRecord[] nodes, byte[][] tensorData, byte[][] nodeParameters)
    {
        Info = info;
        GraphInputs = inputs;
        GraphOutputs = outputs;
        Tensors = tensors;
        Nodes = nodes;
        _tensorData = tensorData;
        _nodeParameters = nodeParameters;
    }

    public ModelInfo Info { get; }
    public IReadOnlyList<uint> GraphInputs { get; }
    public IReadOnlyList<uint> GraphOutputs { get; }
    public IReadOnlyList<int> GetTensorShape(uint index) => Tensors[checked((int)index)].Dimensions.Take(checked((int)Tensors[checked((int)index)].Rank)).ToArray();
    internal TensorRecord[] Tensors { get; }
    internal NodeRecord[] Nodes { get; }
    internal ReadOnlySpan<byte> GetTensorBytes(int index)
    {
        if ((uint)index >= (uint)_tensorData.Length) throw new ArgumentOutOfRangeException(nameof(index));
        return Volatile.Read(ref _disposed) != 0 ? throw new ObjectDisposedException(nameof(Model)) : _tensorData[index];
    }
    internal ReadOnlySpan<byte> GetParameters(in NodeRecord node)
    {
        int index = checked((int)node.ParameterIndex);
        if ((uint)index >= (uint)_nodeParameters.Length) throw new ArgumentOutOfRangeException(nameof(node));
        return Volatile.Read(ref _disposed) != 0 ? throw new ObjectDisposedException(nameof(Model)) : _nodeParameters[index];
    }

    // Pack kinds mirror the loops previously in CompiledModel, plus OC-major 1x1.
    internal const int PackMatMul = 0, PackConv3x3 = 1, PackConv1x1 = 2, PackConv1x1Oc16 = 3,
        PackConv1x1Oc8 = 4;

    /// <summary>
    /// Returns the packed weights for a node's weight input, computing and
    /// caching them on first use. The pack layout depends only on the weight
    /// tensor shape and node attributes, so the result is identical across all
    /// input shapes and safe to share. Returns null when the node/weight does
    /// not qualify for the requested pack kind.
    /// </summary>
    internal float[]? GetPackedWeights(in NodeRecord node, int packKind)
    {
        if (node.Inputs.Length < 2) return null;
        int weightIndex = checked((int)node.Inputs[1]);
        NodeRecord captured = node; // capture by value; lambda cannot take 'in'
        return _packedWeights.GetOrAdd((weightIndex, packKind), _ => ComputePacked(captured, packKind));
    }

    /// <summary>Returns symmetric per-output-channel INT8 weights in VNNI dpbusd layout.</summary>
    internal PackedConv1x1Int8? GetPackedConv1x1Int8(in NodeRecord node)
    {
        if (node.Inputs.Length < 2) return null;
        int weightIndex = checked((int)node.Inputs[1]);
        NodeRecord captured = node;
        return _packedConv1x1Int8.GetOrAdd(weightIndex, _ => ComputePackedConv1x1Int8(captured));
    }

    private PackedConv1x1Int8? ComputePackedConv1x1Int8(in NodeRecord node)
    {
        if (node.Operator != OperatorId.Conv) return null;
        int weightIndex = checked((int)node.Inputs[1]);
        TensorRecord wr = Tensors[weightIndex];
        if ((wr.Flags & TensorConstant) == 0) return null;
        int rank = checked((int)wr.Rank);
        int[] dims = wr.Dimensions.Take(rank).ToArray();
        ReadOnlySpan<byte> p = GetParameters(node);
        if (p.Length < 48 ||
            U32(p, 4) != 1 || I32(p, 8) != 1 || I32(p, 12) != 1 ||
            I32(p, 16) != 1 || I32(p, 20) != 1 || I32(p, 24) != 1 || I32(p, 28) != 1 ||
            I32(p, 32) != 0 || I32(p, 36) != 0 || I32(p, 40) != 0 || I32(p, 44) != 0 ||
            dims.Length != 4 || dims[2] != 1 || dims[3] != 1)
            return null;

        int outputChannels = dims[0], inputChannels = dims[1];
        if (inputChannels < 192 || (inputChannels & 3) != 0 ||
            outputChannels < 8 || (outputChannels & 7) != 0)
            return null;

        ReadOnlySpan<float> weights = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(
            GetTensorBytes(weightIndex));
        byte[] packed = new byte[checked(outputChannels * inputChannels)];
        float[] scales = new float[outputChannels];
        int[] sums = new int[outputChannels];
        int groups = inputChannels / 4;
        for (int co = 0; co < outputChannels; co++)
        {
            ReadOnlySpan<float> row = weights.Slice(co * inputChannels, inputChannels);
            float absMax = 0f;
            for (int ci = 0; ci < inputChannels; ci++)
            {
                float value = row[ci];
                if (!MathCompat.IsFinite(value)) return null;
                absMax = MathF.Max(absMax, MathF.Abs(value));
            }
            if (!(absMax > 0f) || !MathCompat.IsFinite(absMax)) return null;
            float scale = absMax / 127f;
            scales[co] = scale;
            int block = co / 8, lane = co & 7;
            for (int ci = 0; ci < inputChannels; ci++)
            {
                int quantized = MathCompat.Clamp((int)MathF.Round(row[ci] / scale), -127, 127);
                packed[((block * groups + ci / 4) * 8 + lane) * 4 + (ci & 3)] =
                    unchecked((byte)(sbyte)quantized);
                sums[co] += quantized;
            }
        }
        return new PackedConv1x1Int8(packed, scales, sums);
    }

    private float[]? ComputePacked(in NodeRecord node, int packKind)
    {
        int weightIndex = checked((int)node.Inputs[1]);
        TensorRecord wr = Tensors[weightIndex];
        if ((wr.Flags & TensorConstant) == 0) return null;
        int rank = checked((int)wr.Rank);
        int[] dims = wr.Dimensions.Take(rank).ToArray();
        ReadOnlySpan<byte> raw = GetTensorBytes(weightIndex);
        ReadOnlySpan<float> w = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(raw);

        if (packKind == PackMatMul)
        {
            if (node.Operator != OperatorId.MatMul) return null;
            if (dims.Length != 2 || dims[0] < 64 || dims[1] < 1024) return null;
            int inner = dims[0], columns = dims[1];
            int fullTiles = columns / 16;
            float[] packed = new float[checked(fullTiles * inner * 16)];
            for (int tile = 0; tile < fullTiles; tile++)
                for (int k = 0; k < inner; k++)
                    w.Slice(k * columns + tile * 16, 16).CopyTo(packed.AsSpan((tile * inner + k) * 16, 16));
            return packed;
        }

        if (node.Operator != OperatorId.Conv) return null;
        ReadOnlySpan<byte> p = GetParameters(node);
        if (p.Length < 48) return null;

        if (packKind == PackConv3x3)
        {
            if (U32(p, 4) != 1 || I32(p, 8) != 3 || I32(p, 12) != 3 ||
                I32(p, 24) != 1 || I32(p, 28) != 1) return null;
            if (dims.Length != 4 || dims[0] < 8 || (dims[0] & 7) != 0) return null;
            int outputChannels = dims[0], inputChannels = dims[1];
            float[] packed3x3 = new float[checked(outputChannels * inputChannels * 9)];
            int blocks = outputChannels / 8;
            for (int block = 0; block < blocks; block++)
                for (int ci = 0; ci < inputChannels; ci++)
                    for (int k = 0; k < 9; k++)
                        for (int lane = 0; lane < 8; lane++)
                            packed3x3[((block * inputChannels + ci) * 9 + k) * 8 + lane] =
                                w[((block * 8 + lane) * inputChannels + ci) * 9 + k];
            return packed3x3;
        }

        // PackConv1x1 / PackConv1x1Oc16 share 1x1 attribute checks.
        if (U32(p, 4) != 1 || I32(p, 8) != 1 || I32(p, 12) != 1 ||
            I32(p, 16) != 1 || I32(p, 20) != 1 || I32(p, 24) != 1 || I32(p, 28) != 1 ||
            I32(p, 32) != 0 || I32(p, 36) != 0 || I32(p, 40) != 0 || I32(p, 44) != 0)
            return null;
        if (dims.Length != 4 || dims[2] != 1 || dims[3] != 1)
            return null;

        if (packKind == PackConv1x1Oc8)
        {
            // Layout [block8][ic][8]: one cache line of 8-OC weights per IC so
            // AVX-512 PackedEight broadcasts do not straddle two 4-OC blocks.
            if (dims[0] < 8 || (dims[0] & 7) != 0) return null;
            int outputChannels = dims[0], inputChannels = dims[1];
            float[] packedOc8 = new float[checked(outputChannels * inputChannels)];
            int blocks8 = outputChannels / 8;
            for (int block = 0; block < blocks8; block++)
                for (int ci = 0; ci < inputChannels; ci++)
                    for (int lane = 0; lane < 8; lane++)
                        packedOc8[(block * inputChannels + ci) * 8 + lane] =
                            w[(block * 8 + lane) * inputChannels + ci];
            return packedOc8;
        }

        if (packKind == PackConv1x1Oc16)
        {
            // Layout [ic][oc_padded_to_16]: contiguous 16-OC weight rows for
            // OC-major Avx512 (broadcast scalar input × vector weights).
            if (dims[0] < 16) return null;
            int outputChannels = dims[0], inputChannels = dims[1];
            int coutPadded = (outputChannels + 15) & ~15;
            float[] packedOc = new float[checked(inputChannels * coutPadded)];
            for (int ci = 0; ci < inputChannels; ci++)
            {
                int row = ci * coutPadded;
                for (int co = 0; co < outputChannels; co++)
                    packedOc[row + co] = w[co * inputChannels + ci];
            }
            return packedOc;
        }

        if (packKind != PackConv1x1) return null;
        if (dims[0] < 4 || (dims[0] & 3) != 0) return null;
        {
            int outputChannels = dims[0], inputChannels = dims[1];
            float[] packed = new float[checked(outputChannels * inputChannels)];
            int blocks = outputChannels / 4;
            for (int block = 0; block < blocks; block++)
                for (int ci = 0; ci < inputChannels; ci++)
                    for (int lane = 0; lane < 4; lane++)
                        packed[(block * inputChannels + ci) * 4 + lane] =
                            w[(block * 4 + lane) * inputChannels + ci];
            return packed;
        }
    }

    private static uint U32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt32LittleEndian(p[o..]);
    private static int I32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadInt32LittleEndian(p[o..]);

    public static Model Load(ReadOnlyMemory<byte> source)
    {
        if (source.Length == 0) throw new InvalidDataException("ONNX model is empty.");
        if (System.Runtime.InteropServices.MemoryMarshal.TryGetArray(source,
            out ArraySegment<byte> segment) && segment.Array is byte[] array)
        {
            using var stream = new MemoryStream(array, segment.Offset, segment.Count,
                writable: false, publiclyVisible: true);
            return Load(stream);
        }
        return LoadOnnx(source.Span);
    }

    /// <summary>Loads an ONNX model from the current stream position. The caller owns the stream.</summary>
    public static Model Load(Stream source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        return BuildModel(OnnxProtoReader.Parse(source));
    }

    /// <summary>Loads an ONNX model from the current stream position without staging the serialized payload.</summary>
    public static Task<Model> LoadAsync(Stream source, CancellationToken cancellationToken = default)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        cancellationToken.ThrowIfCancellationRequested();
        // The parser is synchronous, but it consumes the caller's stream
        // directly. Run it off-thread so the async API does not block the
        // caller while still avoiding a complete model-sized staging buffer.
        return Task.Run(() => Load(source), cancellationToken);
    }

    /// <summary>Loads a standard ONNX ModelProto without protobuf or native dependencies.</summary>
    public static Model LoadOnnx(Stream source) => Load(source);

    public static Model LoadOnnx(ReadOnlyMemory<byte> source) => Load(source);

    public static Model LoadOnnx(ReadOnlySpan<byte> source)
    {
        if (source.Length == 0) throw new InvalidDataException("ONNX model is empty.");
        // Span compatibility necessarily starts with an in-memory payload.
        // Route it through the same parser; the Stream API above is the path
        // that avoids staging a second model-sized buffer.
        using var stream = new MemoryStream(source.ToArray(), writable: false);
        return Load(stream);
    }

    private static Model BuildModel(OnnxModelData parsed)
    {
        if (parsed.SourceLength == 0) throw new InvalidDataException("ONNX model is empty.");
        OnnxGraphData graph = parsed.Graph!;
        int opset = checked((int)(parsed.Opsets.FirstOrDefault(static x => x.Domain.Length == 0 || x.Domain == "ai.onnx")?.Version ?? 0));
        BuildOnnxRecords(graph, opset, out TensorRecord[] tensors, out NodeRecord[] nodes,
            out uint[] inputs, out uint[] outputs, out byte[][] tensorData, out byte[][] nodeParameters);
        ulong weightSize = checked((ulong)tensorData.Sum(static x => x.LongLength));
        ModelInfo info = new(checked((ushort)MathCompat.Clamp(parsed.IrVersion, 0, ushort.MaxValue)), 0,
            checked((uint)tensors.Length), checked((uint)nodes.Length),
            checked((uint)inputs.Length), checked((uint)outputs.Length), parsed.SourceLength, weightSize, parsed.ContentChecksum);
        return new Model(info, inputs, outputs, tensors, nodes, tensorData, nodeParameters);
    }

    public static Model LoadOnnx(byte[] source) => LoadOnnx((ReadOnlyMemory<byte>)source);

    private static void BuildOnnxRecords(OnnxGraphData graph, int opset,
        out TensorRecord[] tensors, out NodeRecord[] nodes, out uint[] graphInputs, out uint[] graphOutputs,
        out byte[][] tensorData, out byte[][] nodeParameters)
    {
        Dictionary<string, OnnxTensorData> initializers = graph.Initializers.ToDictionary(static x => x.Name, StringComparer.Ordinal);

        // PaddleX's exported PP-LCNet ONNX graph stores learned parameters as
        // Constant(value=TensorProto) nodes instead of graph initializers.
        // Treat those nodes exactly like initializers before building the
        // managed execution plan. This keeps the public Model API unchanged
        // and avoids requiring a runtime Constant operator.
        foreach (OnnxNodeData constant in graph.Nodes)
        {
            if (constant.OpType != "Constant" || constant.Outputs.Count != 1 ||
                constant.Outputs[0].Length == 0)
                continue;
            OnnxAttributeData? value = constant.Attributes.FirstOrDefault(a => a.Name == "value");
            if (value?.Tensor is OnnxTensorData tensor)
            {
                if (tensor.Name.Length == 0) tensor.Name = constant.Outputs[0];
                initializers[constant.Outputs[0]] = tensor;
                continue;
            }

            // Also accept the scalar/list forms permitted by ONNX Constant.
            // They are uncommon in PaddleX weights but cost little to support
            // and make the conversion safe for metadata constants.
            OnnxAttributeData? scalar = constant.Attributes.FirstOrDefault(a => a.Name == "value_float");
            if (scalar is not null)
            {
                initializers[constant.Outputs[0]] = MakeFloatTensor(constant.Outputs[0], [], [scalar.Float]);
                continue;
            }
            scalar = constant.Attributes.FirstOrDefault(a => a.Name == "value_int");
            if (scalar is not null)
            {
                OnnxTensorData integer = new() { Name = constant.Outputs[0], DataType = 7 };
                integer.Int64Data.Add(scalar.Int);
                initializers[constant.Outputs[0]] = integer;
                continue;
            }
            OnnxAttributeData? list = constant.Attributes.FirstOrDefault(a => a.Name == "value_floats");
            if (list is not null)
            {
                initializers[constant.Outputs[0]] = MakeFloatTensor(constant.Outputs[0], [list.Floats.Count], list.Floats);
                continue;
            }
            list = constant.Attributes.FirstOrDefault(a => a.Name == "value_ints");
            if (list is not null)
            {
                OnnxTensorData integers = new() { Name = constant.Outputs[0], DataType = 7 };
                integers.Dims.Add(list.Ints.Count);
                integers.Int64Data.AddRange(list.Ints);
                initializers[constant.Outputs[0]] = integers;
            }
        }
        Dictionary<string, string> aliases = [with(StringComparer.Ordinal)];
        foreach (OnnxNodeData node in graph.Nodes)
            if (node.OpType == "Identity" && node.Inputs.Count == 1 && node.Outputs.Count == 1 &&
                node.Inputs[0].Length != 0 && node.Outputs[0].Length != 0)
                aliases[node.Outputs[0]] = node.Inputs[0];

        string Resolve(string name)
        {
            HashSet<string> visited = [with(StringComparer.Ordinal)];
            while (aliases.TryGetValue(name, out string? next))
            {
                if (!visited.Add(name)) throw new InvalidDataException($"ONNX Identity alias cycle involving '{name}'.");
                name = next;
            }
            return name;
        }

        static OnnxNodeData CloneNode(OnnxNodeData source, string? op = null)
        {
            OnnxNodeData clone = new() { Name = source.Name, OpType = op ?? source.OpType, Domain = source.Domain };
            clone.Inputs.AddRange(source.Inputs); clone.Outputs.AddRange(source.Outputs); clone.Attributes.AddRange(source.Attributes);
            return clone;
        }
        static OnnxAttributeData MakeInts(string name, params long[] values)
        {
            OnnxAttributeData attribute = new() { Name = name };
            attribute.Ints.AddRange(values);
            return attribute;
        }
        static OnnxAttributeData MakeInt(string name, long value) => new() { Name = name, Int = value };
        static OnnxAttributeData MakeFloats(string name, params float[] values)
        {
            OnnxAttributeData attribute = new() { Name = name };
            attribute.Floats.AddRange(values);
            return attribute;
        }

        List<OnnxNodeData> transformed = [with(graph.Nodes.Count)];
        foreach (OnnxNodeData original in graph.Nodes)
        {
            if (original.OpType is "Identity" or "Constant") continue;
            OnnxNodeData node = CloneNode(original);
            if (node.OpType == "GlobalAveragePool")
            {
                node.OpType = "ReduceMean";
                node.Attributes.Clear(); node.Attributes.Add(MakeInts("axes", 2, 3)); node.Attributes.Add(MakeInt("keepdims", 1));
            }
            else if (node.OpType == "Reshape" && node.Inputs.Count > 1)
            {
                // PP-OCR's older exports use a Shape/Slice/Concat metadata
                // chain whose output shape is present in value_info.  The
                // PaddleX PP-LCNet export instead feeds Reshape from a
                // Constant/Identity tensor. Keep that constant shape input so
                // the runtime can recover [N,C,1,1] (and similar) exactly;
                // discard only non-constant metadata chains.
                string shapeInput = Resolve(node.Inputs[1]);
                if (!initializers.ContainsKey(shapeInput))
                    node.Inputs.RemoveRange(1, node.Inputs.Count - 1);
            }
            else if (node.OpType == "Resize" && node.Inputs.Count >= 3 &&
                     initializers.TryGetValue(node.Inputs[2], out OnnxTensorData? scaleTensor))
            {
                float[] scales = TensorFloats(scaleTensor);
                if (scales.Length == 4)
                {
                    node.Inputs.RemoveRange(1, node.Inputs.Count - 1);
                    node.Attributes.RemoveAll(a => a.Name is "scales" or "sizes");
                    node.Attributes.Add(MakeFloats("scales", scales));
                }
            }
            NormalizeSameUpper(node);
            string[] resolvedInputs = [.. node.Inputs.Select(Resolve)];
            string[] resolvedOutputs = [.. node.Outputs.Select(Resolve)];
            node.Inputs.Clear(); node.Inputs.AddRange(resolvedInputs);
            node.Outputs.Clear(); node.Outputs.AddRange(resolvedOutputs);
            transformed.Add(node);
        }

        FoldConvBatchNormalization(transformed, initializers, graph);
        MergeParallelConvBranches(transformed, initializers, graph);

        Dictionary<string, OnnxValueInfoData> valueInfo = [with(StringComparer.Ordinal)];
        foreach (OnnxValueInfoData value in graph.Inputs.Concat(graph.ValueInfo).Concat(graph.Outputs))
            valueInfo[value.Name] = value;
        string[] inputNames = graph.Inputs
            .Select(static x => x.Name)
            .Where(name => !initializers.ContainsKey(name))
            .Select(Resolve)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        string[] outputNames = graph.Outputs.Select(static x => x.Name).Select(Resolve).Distinct(StringComparer.Ordinal).ToArray();

        // Drop nodes that became dead after Identity/metadata rewrites.  This
        // removes the CLS Shape -> Slice -> Concat chain while retaining real
        // data Concat nodes in DET.
        HashSet<string> needed = new(outputNames, StringComparer.Ordinal);
        bool[] keepNode = new bool[transformed.Count];
        for (int i = transformed.Count - 1; i >= 0; i--)
        {
            OnnxNodeData node = transformed[i];
            if (!node.Outputs.Any(needed.Contains)) continue;
            keepNode[i] = true;
            foreach (string input in node.Inputs) if (input.Length != 0) needed.Add(input);
        }
        transformed = [.. transformed.Where((_, i) => keepNode[i])];

        // Retain only constants reachable from executable nodes.  This drops
        // CLS's INT64 Shape metadata constant and DET's Resize roi/scales.
        HashSet<string> usedInitializers = [with(StringComparer.Ordinal)];
        foreach (OnnxNodeData node in transformed)
            foreach (string input in node.Inputs)
                if (initializers.ContainsKey(input)) usedInitializers.Add(input);

        List<string> orderedNames = [];
        void AddName(string name) { if (!orderedNames.Contains(name, StringComparer.Ordinal)) orderedNames.Add(name); }
        foreach (string name in inputNames) AddName(name);
        foreach (OnnxTensorData initializer in initializers.Values)
            if (usedInitializers.Contains(initializer.Name)) AddName(initializer.Name);
        foreach (OnnxNodeData node in transformed)
            foreach (string name in node.Outputs) if (name.Length != 0) AddName(name);
        foreach (string name in outputNames) AddName(name);

        Dictionary<string, uint> index = [with(StringComparer.Ordinal)];
        List<TensorRecord> tensorList = [with(orderedNames.Count)];
        List<byte[]> dataList = [with(orderedNames.Count)];
        foreach (string name in orderedNames)
        {
            if (initializers.TryGetValue(name, out OnnxTensorData? initializer))
            {
                (DType type, byte[] data) = TensorBytes(initializer);
                int[] dims = initializer.Dims.Select(ToDimension).ToArray();
                if (dims.Length > 8) throw new InvalidDataException($"ONNX tensor '{name}' rank exceeds 8.");
                long elements = 1;
                foreach (int dim in dims)
                {
                    if (dim <= 0) throw new InvalidDataException($"ONNX initializer '{name}' has a non-positive dimension.");
                    elements = checked(elements * dim);
                }
                int bytesPerElement = type switch { DType.F32 or DType.I32 => 4, DType.I64 => 8, DType.U8 => 1, _ => 0 };
                if (bytesPerElement == 0 || checked(elements * bytesPerElement) != data.LongLength)
                    throw new InvalidDataException($"ONNX initializer '{name}' data length does not match its shape.");
                index[name] = checked((uint)tensorList.Count);
                tensorList.Add(new TensorRecord(type, checked((uint)dims.Length), dims,
                    TensorConstant | (inputNames.Contains(name, StringComparer.Ordinal) ? TensorInput : 0) |
                    (outputNames.Contains(name, StringComparer.Ordinal) ? TensorOutput : 0)));
                dataList.Add(data);
            }
            else
            {
                OnnxValueInfoData? value = valueInfo.GetValueOrDefault(name);
                int[] dims = value?.Shape is { Length: > 0 } shape ? shape.ToArray() : [-1];
                if (dims.Length > 8) throw new InvalidDataException($"ONNX tensor '{name}' rank exceeds 8.");
                DType type = value is null ? DType.F32 : MapDType(value.ElementType);
                index[name] = checked((uint)tensorList.Count);
                tensorList.Add(new TensorRecord(type, checked((uint)dims.Length), dims,
                    (inputNames.Contains(name, StringComparer.Ordinal) ? TensorInput : 0) |
                    (outputNames.Contains(name, StringComparer.Ordinal) ? TensorOutput : 0)));
                dataList.Add([]);
            }
        }

        List<NodeRecord> nodeList = [with(transformed.Count)];
        List<byte[]> paramList = [with(transformed.Count)];
        foreach (OnnxNodeData node in transformed)
        {
            uint[] ins = node.Inputs.Where(static x => x.Length != 0).Select(name => index.TryGetValue(name, out uint i) ? i : uint.MaxValue).ToArray();
            uint[] outs = node.Outputs.Where(static x => x.Length != 0).Select(name => index.TryGetValue(name, out uint i) ? i : uint.MaxValue).ToArray();
            OperatorId op = node.Domain.Length != 0 && node.Domain != "ai.onnx" ? OperatorId.Unknown : MapOperator(node.OpType);
            byte[] parameters = EncodeParameters(node);
            nodeList.Add(new NodeRecord(op, ins, outs, checked((uint)nodeList.Count), node.Name, node.OpType, node.Domain, opset));
            paramList.Add(parameters);
        }
        tensors = [.. tensorList]; nodes = [.. nodeList]; tensorData = [.. dataList]; nodeParameters = [.. paramList];
        graphInputs = [.. inputNames.Select(name => index[name])]; graphOutputs = [.. outputNames.Select(name => index[name])];
    }

    private static void NormalizeSameUpper(OnnxNodeData node)
    {
        if (node.OpType is not ("Conv" or "MaxPool")) return;
        OnnxAttributeData? auto = node.Attributes.FirstOrDefault(a => a.Name == "auto_pad");
        if (auto is null || auto.String.Length == 0 || Encoding.UTF8.GetString(auto.String) == "NOTSET") return;
        if (Encoding.UTF8.GetString(auto.String) != "SAME_UPPER") return;
        long[] kernel = Ints(node, "kernel_shape", []), strides = Ints(node, "strides", [1, 1]), dilation = Ints(node, "dilations", [1, 1]);
        if (kernel.Length != 2 || strides.Length != 2 || dilation.Length != 2) return;
        if (strides[0] != 1 || strides[1] != 1) return;
        long ph = (kernel[0] - 1) * dilation[0], pw = (kernel[1] - 1) * dilation[1];
        node.Attributes.Remove(auto); node.Attributes.Add(new OnnxAttributeData { Name = "pads", Ints = { ph / 2, pw / 2, ph - ph / 2, pw - pw / 2 } });
    }

    private static void FoldConvBatchNormalization(List<OnnxNodeData> nodes,
        Dictionary<string, OnnxTensorData> initializers, OnnxGraphData graph)
    {
        Dictionary<string, int> producer = [with(StringComparer.Ordinal)];
        Dictionary<string, int> consumers = [with(StringComparer.Ordinal)];
        foreach (OnnxNodeData node in nodes)
        {
            foreach (string output in node.Outputs) if (output.Length != 0) producer[output] = nodes.IndexOf(node);
            foreach (string input in node.Inputs) if (input.Length != 0) consumers[input] = consumers.GetValueOrDefault(input) + 1;
        }
        Dictionary<int, int> folded = [];
        HashSet<string> graphOutputs = new(graph.Outputs.Select(static x => x.Name), StringComparer.Ordinal);
        for (int bnIndex = 0; bnIndex < nodes.Count; bnIndex++)
        {
            OnnxNodeData bn = nodes[bnIndex];
            if (bn.OpType != "BatchNormalization" || bn.Inputs.Count != 5 || bn.Outputs.Count != 1) continue;
            if (!producer.TryGetValue(bn.Inputs[0], out int convIndex) || consumers.GetValueOrDefault(bn.Inputs[0]) != 1 || convIndex >= bnIndex) continue;
            OnnxNodeData conv = nodes[convIndex];
            if (conv.OpType != "Conv" || conv.Inputs.Count is < 2 or > 3 || conv.Outputs.Count != 1 || graphOutputs.Contains(conv.Outputs[0])) continue;
            string[] names = [.. conv.Inputs.Skip(1).Concat(bn.Inputs.Skip(1))];
            if (names.Any(name => !initializers.ContainsKey(name))) continue;
            if (names.Any(name => initializers[name].DataType != 1)) continue;
            OnnxTensorData weightTensor = initializers[conv.Inputs[1]];
            float[] weights = TensorFloats(weightTensor);
            float[] gamma = TensorFloats(initializers[bn.Inputs[1]]), beta = TensorFloats(initializers[bn.Inputs[2]]);
            float[] mean = TensorFloats(initializers[bn.Inputs[3]]), variance = TensorFloats(initializers[bn.Inputs[4]]);
            long weightElements = weightTensor.Dims.Aggregate(1L, (a, b) => checked(a * b));
            if (weightTensor.Dims.Count != 4 || gamma.Length == 0 ||
                gamma.Length != beta.Length || gamma.Length != mean.Length ||
                gamma.Length != variance.Length || weights.Length != weightElements)
                continue;
            float epsilon = bn.Attributes.FirstOrDefault(a => a.Name == "epsilon")?.Float ?? 1e-5f;
            if (!MathCompat.IsFinite(epsilon) || epsilon < 0 || variance.Any(v => !MathCompat.IsFinite(v) || v < 0) ||
                gamma.Any(float.IsNaN) || beta.Any(float.IsNaN) || mean.Any(float.IsNaN)) continue;
            int outputChannels = checked((int)weightTensor.Dims[0]);
            if (outputChannels != gamma.Length) continue;
            float[] foldedWeights = new float[weights.Length];
            float[] foldedBias = new float[outputChannels];
            int perChannel = weights.Length / outputChannels;
            float[] originalBias = conv.Inputs.Count == 3 ? TensorFloats(initializers[conv.Inputs[2]]) : new float[outputChannels];
            if (originalBias.Length != outputChannels) continue;
            for (int channel = 0; channel < outputChannels; channel++)
            {
                float scale = gamma[channel] / MathF.Sqrt(variance[channel] + epsilon);
                foldedBias[channel] = (originalBias[channel] - mean[channel]) * scale + beta[channel];
                for (int i = 0; i < perChannel; i++) foldedWeights[channel * perChannel + i] = weights[channel * perChannel + i] * scale;
            }
            string weightName = UniqueName(conv.Inputs[1] + "__managed_bn_weight", initializers);
            string biasName = UniqueName((conv.Name.Length == 0 ? conv.Outputs[0] : conv.Name) + "__managed_bn_bias", initializers);
            initializers[weightName] = MakeFloatTensor(weightName, weightTensor.Dims, foldedWeights);
            initializers[biasName] = MakeFloatTensor(biasName, [outputChannels], foldedBias);
            conv.Inputs[1] = weightName;
            if (conv.Inputs.Count == 2) conv.Inputs.Add(biasName); else conv.Inputs[2] = biasName;
            conv.Outputs[0] = bn.Outputs[0];
            folded[bnIndex] = convIndex;
        }
        if (folded.Count == 0) return;
        List<OnnxNodeData> result = [with(nodes.Count - folded.Count)];
        for (int i = 0; i < nodes.Count; i++) if (!folded.ContainsKey(i)) result.Add(nodes[i]);
        nodes.Clear(); nodes.AddRange(result);
    }

    private static string UniqueName(string baseName, Dictionary<string, OnnxTensorData> initializers)
    {
        string candidate = baseName; int serial = 1;
        while (initializers.ContainsKey(candidate)) candidate = baseName + "_" + serial++;
        return candidate;
    }

    // RepLK-style blocks (small/medium PP-OCRv6 detectors) run several
    // parallel convolutions over the same input (e.g. 7x7 + 7x1 + 1x7) and
    // sum the results.  For stride-1/dilation-1/group-1 branches with odd,
    // centered kernels this is algebraically one convolution whose weights
    // are the centered sum of the branch weights, saving both FLOPs and
    // repeated input sweeps.  Results are mathematically identical (float
    // rounding of the merged accumulation differs, deterministically).
    private static void MergeParallelConvBranches(List<OnnxNodeData> nodes,
        Dictionary<string, OnnxTensorData> initializers, OnnxGraphData graph)
    {
        HashSet<string> graphOutputs = new(graph.Outputs.Select(static x => x.Name), StringComparer.Ordinal);
        bool merged = true;
        while (merged)
        {
            merged = false;
            Dictionary<string, int> producer = new(StringComparer.Ordinal);
            Dictionary<string, int> consumers = new(StringComparer.Ordinal);
            for (int i = 0; i < nodes.Count; i++)
            {
                foreach (string output in nodes[i].Outputs) if (output.Length != 0) producer[output] = i;
                foreach (string input in nodes[i].Inputs) if (input.Length != 0) consumers[input] = consumers.GetValueOrDefault(input) + 1;
            }
            for (int addIndex = 0; addIndex < nodes.Count && !merged; addIndex++)
            {
                OnnxNodeData add = nodes[addIndex];
                if (add.OpType != "Add" || add.Inputs.Count != 2 || add.Outputs.Count != 1) continue;
                if (!producer.TryGetValue(add.Inputs[0], out int firstIndex) ||
                    !producer.TryGetValue(add.Inputs[1], out int secondIndex) ||
                    firstIndex == secondIndex) continue;
                if (!TryGetMergeableConv(nodes[firstIndex], initializers, graphOutputs, out ConvBranch first) ||
                    !TryGetMergeableConv(nodes[secondIndex], initializers, graphOutputs, out ConvBranch second)) continue;
                if (!string.Equals(first.Input, second.Input, StringComparison.Ordinal)) continue;
                if (consumers.GetValueOrDefault(nodes[firstIndex].Outputs[0]) != 1 ||
                    consumers.GetValueOrDefault(nodes[secondIndex].Outputs[0]) != 1) continue;
                if (first.OutputChannels != second.OutputChannels ||
                    first.InputChannels != second.InputChannels) continue;

                int kernelH = Math.Max(first.KernelH, second.KernelH);
                int kernelW = Math.Max(first.KernelW, second.KernelW);
                float[] weights = new float[(long)first.OutputChannels * first.InputChannels * kernelH * kernelW];
                AccumulateCentered(weights, first, kernelH, kernelW);
                AccumulateCentered(weights, second, kernelH, kernelW);
                float[]? bias = null;
                if (first.Bias is not null || second.Bias is not null)
                {
                    bias = new float[first.OutputChannels];
                    for (int c = 0; c < bias.Length; c++)
                        bias[c] = (first.Bias?[c] ?? 0f) + (second.Bias?[c] ?? 0f);
                }

                OnnxNodeData conv = nodes[firstIndex];
                string weightName = UniqueName(conv.Inputs[1] + "__managed_branch_weight", initializers);
                initializers[weightName] = MakeFloatTensor(weightName,
                    [first.OutputChannels, first.InputChannels, kernelH, kernelW], weights);
                conv.Inputs[1] = weightName;
                if (bias is not null)
                {
                    string biasName = UniqueName(weightName + "_bias", initializers);
                    initializers[biasName] = MakeFloatTensor(biasName, [first.OutputChannels], bias);
                    if (conv.Inputs.Count == 2) conv.Inputs.Add(biasName); else conv.Inputs[2] = biasName;
                }
                SetIntsAttribute(conv, "kernel_shape", [kernelH, kernelW]);
                SetIntsAttribute(conv, "pads", [(kernelH - 1) / 2, (kernelW - 1) / 2, (kernelH - 1) / 2, (kernelW - 1) / 2]);
                conv.Outputs[0] = add.Outputs[0];
                int removeHigh = Math.Max(addIndex, secondIndex), removeLow = Math.Min(addIndex, secondIndex);
                nodes.RemoveAt(removeHigh);
                nodes.RemoveAt(removeLow);
                merged = true;
            }
        }
    }

    private readonly record struct ConvBranch(string Input, int OutputChannels, int InputChannels,
        int KernelH, int KernelW, float[] Weights, float[]? Bias);

    private static bool TryGetMergeableConv(OnnxNodeData node,
        Dictionary<string, OnnxTensorData> initializers, HashSet<string> graphOutputs,
        out ConvBranch branch)
    {
        branch = default;
        if (node.OpType != "Conv" || node.Inputs.Count is < 2 or > 3 || node.Outputs.Count != 1 ||
            graphOutputs.Contains(node.Outputs[0])) return false;
        if (Ints(node, "strides", [1, 1]) is not [1, 1] ||
            Ints(node, "dilations", [1, 1]) is not [1, 1]) return false;
        if ((node.Attributes.FirstOrDefault(a => a.Name == "group")?.Int ?? 1) != 1) return false;
        byte[]? autoPad = node.Attributes.FirstOrDefault(a => a.Name == "auto_pad")?.String;
        if (autoPad is { Length: > 0 } && !"NOTSET"u8.SequenceEqual(autoPad)) return false;
        if (!initializers.TryGetValue(node.Inputs[1], out OnnxTensorData? weightTensor) ||
            weightTensor.DataType != 1 || weightTensor.Dims.Count != 4) return false;
        int kernelH = checked((int)weightTensor.Dims[2]), kernelW = checked((int)weightTensor.Dims[3]);
        if ((kernelH & 1) == 0 || (kernelW & 1) == 0) return false;
        long[] kernel = Ints(node, "kernel_shape", [kernelH, kernelW]);
        if (kernel.Length != 2 || kernel[0] != kernelH || kernel[1] != kernelW) return false;
        long[] pads = Ints(node, "pads", [0, 0, 0, 0]);
        if (pads.Length != 4 || pads[0] != (kernelH - 1) / 2 || pads[2] != (kernelH - 1) / 2 ||
            pads[1] != (kernelW - 1) / 2 || pads[3] != (kernelW - 1) / 2) return false;
        float[] weights = TensorFloats(weightTensor);
        long expected = weightTensor.Dims.Aggregate(1L, static (a, b) => checked(a * b));
        if (weights.Length != expected) return false;
        float[]? bias = null;
        if (node.Inputs.Count == 3)
        {
            if (!initializers.TryGetValue(node.Inputs[2], out OnnxTensorData? biasTensor) ||
                biasTensor.DataType != 1) return false;
            bias = TensorFloats(biasTensor);
            if (bias.Length != (int)weightTensor.Dims[0]) return false;
        }
        branch = new ConvBranch(node.Inputs[0], checked((int)weightTensor.Dims[0]),
            checked((int)weightTensor.Dims[1]), kernelH, kernelW, weights, bias);
        return true;
    }

    private static void AccumulateCentered(float[] target, ConvBranch branch, int kernelH, int kernelW)
    {
        int offsetY = (kernelH - branch.KernelH) / 2, offsetX = (kernelW - branch.KernelW) / 2;
        int filters = branch.OutputChannels * branch.InputChannels;
        for (int f = 0; f < filters; f++)
            for (int ky = 0; ky < branch.KernelH; ky++)
                for (int kx = 0; kx < branch.KernelW; kx++)
                    target[(f * kernelH + ky + offsetY) * kernelW + kx + offsetX] +=
                        branch.Weights[(f * branch.KernelH + ky) * branch.KernelW + kx];
    }

    private static void SetIntsAttribute(OnnxNodeData node, string name, long[] values)
    {
        OnnxAttributeData? attribute = node.Attributes.FirstOrDefault(a => a.Name == name);
        if (attribute is null)
        {
            attribute = new OnnxAttributeData { Name = name };
            node.Attributes.Add(attribute);
        }
        attribute.Ints.Clear();
        attribute.Ints.AddRange(values);
    }
    private static OnnxTensorData MakeFloatTensor(string name, IReadOnlyList<long> dims, IReadOnlyList<float> values)
    {
        OnnxTensorData tensor = new() { Name = name, DataType = 1 };
        tensor.Dims.AddRange(dims); tensor.FloatData.AddRange(values); return tensor;
    }

    private static long[] Ints(OnnxNodeData node, string name, long[] fallback) =>
        node.Attributes.FirstOrDefault(a => a.Name == name)?.Ints.ToArray() ?? fallback;
    private static int ToDimension(long value) => value > 0 && value <= int.MaxValue ? checked((int)value) : -1;

    private static OperatorId MapOperator(string op) => op switch
    {
        "Conv" => OperatorId.Conv,
        "Add" => OperatorId.Add,
        "Mul" => OperatorId.Mul,
        "Div" => OperatorId.Div,
        "Erf" => OperatorId.Erf,
        "HardSigmoid" => OperatorId.HardSigmoid,
        "BatchNormalization" => OperatorId.BatchNormalization,
        "ReduceMean" => OperatorId.ReduceMean,
        "Relu" => OperatorId.Relu,
        "AveragePool" => OperatorId.AveragePool,
        "Squeeze" => OperatorId.Squeeze,
        "Transpose" => OperatorId.Transpose,
        "Unsqueeze" => OperatorId.Unsqueeze,
        "MatMul" => OperatorId.MatMul,
        "Softmax" => OperatorId.Softmax,
        "Reshape" => OperatorId.Reshape,
        "Concat" => OperatorId.Concat,
        "ConvTranspose" => OperatorId.ConvTranspose,
        "MaxPool" => OperatorId.MaxPool,
        "Resize" => OperatorId.Resize,
        "Sigmoid" => OperatorId.Sigmoid,
        "Sub" => OperatorId.Sub,
        "Pow" => OperatorId.Pow,
        "Sqrt" => OperatorId.Sqrt,
        "Slice" => OperatorId.Slice,
        _ => OperatorId.Unknown
    };

    private static byte[] EncodeParameters(OnnxNodeData node)
    {
        static long Get(OnnxNodeData n, string name, long fallback) =>
            n.Attributes.FirstOrDefault(a => a.Name == name)?.Int ?? fallback;
        static float GetF(OnnxNodeData n, string name, float fallback) =>
            n.Attributes.FirstOrDefault(a => a.Name == name)?.Float ?? fallback;
        static long[] GetInts(OnnxNodeData n, string name, long[] fallback) =>
            n.Attributes.FirstOrDefault(a => a.Name == name)?.Ints.ToArray() ?? fallback;
        static void PutI(byte[] buffer, int offset, long value) =>
            BinaryPrimitives.WriteInt32LittleEndian(buffer.AsSpan(offset, 4), checked((int)value));
        static void PutF(byte[] buffer, int offset, float value) =>
            BinaryPrimitives.WriteInt32LittleEndian(buffer.AsSpan(offset, 4), BitConverterCompat.SingleToInt32Bits(value));
        static void PutU16(byte[] buffer, int offset, int value) =>
            BinaryPrimitives.WriteUInt16LittleEndian(buffer.AsSpan(offset, 2), checked((ushort)value));
        static void PutU32(byte[] buffer, int offset, long value) =>
            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(offset, 4), checked((uint)value));

        if (node.OpType is "Conv" or "ConvTranspose")
        {
            long[] kernel = GetInts(node, "kernel_shape", []);
            long[] strides = GetInts(node, "strides", [1, 1]);
            long[] dilation = GetInts(node, "dilations", [1, 1]);
            long[] pads = GetInts(node, "pads", [0, 0, 0, 0]);
            if (kernel.Length != 2 || strides.Length != 2 || dilation.Length != 2 || pads.Length != 4)
                return [];

            byte[] buffer = new byte[64];
            PutU16(buffer, 0, 1);
            PutU16(buffer, 2, 2);
            PutU32(buffer, 4, Get(node, "group", 1));
            PutI(buffer, 8, kernel[0]);
            PutI(buffer, 12, kernel[1]);
            PutI(buffer, 16, strides[0]);
            PutI(buffer, 20, strides[1]);
            PutI(buffer, 24, dilation[0]);
            PutI(buffer, 28, dilation[1]);
            PutI(buffer, 32, pads[0]);
            PutI(buffer, 36, pads[1]);
            PutI(buffer, 40, pads[2]);
            PutI(buffer, 44, pads[3]);
            return buffer;
        }

        if (node.OpType == "BatchNormalization")
        {
            byte[] buffer = new byte[24];
            PutU16(buffer, 0, 1);
            PutF(buffer, 4, GetF(node, "epsilon", 1e-5f));
            PutF(buffer, 8, GetF(node, "momentum", 0.9f));
            PutU32(buffer, 12, Get(node, "training_mode", 0));
            return buffer;
        }

        if (node.OpType == "HardSigmoid")
        {
            byte[] buffer = new byte[16];
            PutU16(buffer, 0, 1);
            PutF(buffer, 4, GetF(node, "alpha", 0.2f));
            PutF(buffer, 8, GetF(node, "beta", 0.5f));
            return buffer;
        }

        if (node.OpType is "Squeeze" or "Unsqueeze" or "Transpose")
        {
            string attribute = node.OpType == "Transpose" ? "perm" : "axes";
            long[] values = GetInts(node, attribute, []);
            if (values.Length > 8) return [];

            byte[] buffer = new byte[40];
            PutU16(buffer, 0, 1);
            PutU16(buffer, 2, values.Length);
            for (int i = 0; i < values.Length; i++) PutI(buffer, 4 + i * 4, values[i]);
            return buffer;
        }

        if (node.OpType is "Softmax" or "Concat")
        {
            byte[] buffer = new byte[16];
            PutU16(buffer, 0, 1);
            PutI(buffer, 4, Get(node, "axis", 1));
            return buffer;
        }

        if (node.OpType == "ReduceMean")
        {
            long[] axes = GetInts(node, "axes", []);
            if (axes.Length > 8) return [];

            byte[] buffer = new byte[48];
            PutU16(buffer, 0, 1);
            PutU16(buffer, 2, axes.Length);
            PutU32(buffer, 4, Get(node, "keepdims", 1));
            PutU32(buffer, 8, Get(node, "noop_with_empty_axes", 0));
            for (int i = 0; i < axes.Length; i++) PutI(buffer, 12 + i * 4, axes[i]);
            return buffer;
        }

        if (node.OpType is "AveragePool" or "MaxPool")
        {
            long[] kernel = GetInts(node, "kernel_shape", []);
            long[] strides = GetInts(node, "strides", [1, 1]);
            long[] pads = GetInts(node, "pads", [0, 0, 0, 0]);
            if (kernel.Length != 2 || strides.Length != 2 || pads.Length != 4) return [];

            byte[] buffer = new byte[64];
            PutU16(buffer, 0, 1);
            PutU32(buffer, 4, 2);
            PutI(buffer, 8, kernel[0]);
            PutI(buffer, 12, kernel[1]);
            PutI(buffer, 16, strides[0]);
            PutI(buffer, 20, strides[1]);
            PutI(buffer, 24, pads[0]);
            PutI(buffer, 28, pads[1]);
            PutI(buffer, 32, pads[2]);
            PutI(buffer, 36, pads[3]);
            PutU32(buffer, 40, Get(node, "ceil_mode", 0));
            PutU32(buffer, 44, Get(node, "count_include_pad", 0));
            return buffer;
        }

        if (node.OpType == "Resize")
        {
            IReadOnlyList<float> scales = node.Attributes.FirstOrDefault(a => a.Name == "scales")?.Floats ?? [];
            if (scales.Count != 4) return [];

            byte[] buffer = new byte[32];
            PutU16(buffer, 0, 1);
            PutU16(buffer, 2, 4);
            for (int i = 0; i < 4; i++) PutF(buffer, 4 + i * 4, scales[i]);
            return buffer;
        }

        return [];
    }

    private static (DType Type, byte[] Data) TensorBytes(OnnxTensorData tensor)
    {
        DType type = MapDType(tensor.DataType);
        if (tensor.RawData.Length != 0) return (type, tensor.RawData);
        if (type == DType.F32)
        {
            byte[] data = new byte[checked(tensor.FloatData.Count * 4)];
            for (int i = 0; i < tensor.FloatData.Count; i++)
                BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(i * 4, 4), BitConverterCompat.SingleToInt32Bits(tensor.FloatData[i]));
            return (type, data);
        }
        if (type == DType.I32)
        {
            byte[] data = new byte[checked(tensor.Int32Data.Count * 4)];
            for (int i = 0; i < tensor.Int32Data.Count; i++)
                BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(i * 4, 4), tensor.Int32Data[i]);
            return (type, data);
        }
        if (type == DType.I64)
        {
            byte[] data = new byte[checked(tensor.Int64Data.Count * 8)];
            for (int i = 0; i < tensor.Int64Data.Count; i++)
                BinaryPrimitives.WriteInt64LittleEndian(data.AsSpan(i * 8, 8), tensor.Int64Data[i]);
            return (type, data);
        }
        if (type == DType.U8)
        {
            byte[] data = new byte[tensor.Int32Data.Count];
            for (int i = 0; i < data.Length; i++) data[i] = checked((byte)tensor.Int32Data[i]);
            return (type, data);
        }
        return (type, tensor.StringData.SelectMany(static x => x).ToArray());
    }

    private static float[] TensorFloats(OnnxTensorData tensor)
    {
        (DType type, byte[] data) = TensorBytes(tensor);
        if (type != DType.F32 || data.Length % 4 != 0) return [];

        float[] values = new float[data.Length / 4];
        for (int i = 0; i < values.Length; i++)
            values[i] = BitConverterCompat.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(i * 4, 4)));
        return values;
    }

    private static DType MapDType(int type) => type switch
    {
        1 => DType.F32,
        6 => DType.I32,
        7 => DType.I64,
        2 => DType.U8,
        _ => throw new NotSupportedException($"Unsupported ONNX tensor data type {type}.")
    };

    public static Model Load(byte[] source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (source.Length == 0) throw new InvalidDataException("ONNX model is empty.");
        using var stream = new MemoryStream(source, writable: false);
        return Load(stream);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Model data is managed, but can still occupy hundreds of megabytes.
        // Clear the references on Dispose so an owned model can release its
        // weight/parameter storage immediately even if the wrapper object is
        // kept alive by user code. Dependent CompiledModel instances are
        // invalid after their source model is disposed, just as before.
        Array.Clear(_tensorData, 0, _tensorData.Length);
        Array.Clear(_nodeParameters, 0, _nodeParameters.Length);
        _packedWeights.Clear();
        _packedConv1x1Int8.Clear();
        GC.SuppressFinalize(this);
    }
}

public readonly record struct ModelInfo(ushort FormatMajor, ushort FormatMinor, uint TensorCount,
    uint NodeCount, uint InputCount, uint OutputCount, ulong FileSize, ulong WeightSize, ulong ContentChecksum);
public enum DType : uint { F32 = 1, I32 = 2, I64 = 3, U8 = 4 }
public enum OperatorId : ushort
{
    Unknown = 0,
    Conv = 1, Add, Mul, Div, Erf, HardSigmoid, BatchNormalization, ReduceMean,
    Relu, AveragePool, Squeeze, Transpose, Unsqueeze, MatMul, Softmax, Reshape, Concat, ConvTranspose, MaxPool, Resize, Sigmoid,
    Sub, Pow, Sqrt, Slice
}
internal readonly record struct TensorRecord(DType DType, uint Rank, int[] Dimensions, uint Flags);
internal readonly record struct NodeRecord(OperatorId Operator, uint[] Inputs, uint[] Outputs, uint ParameterIndex,
    string Name = "", string OpType = "", string Domain = "", int Opset = 0);
