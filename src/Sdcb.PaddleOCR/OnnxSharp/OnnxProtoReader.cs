using System.Buffers.Binary;
using System.Text;

namespace Sdcb.PaddleOCR.OnnxSharp;

// A deliberately small protobuf decoder for the ONNX protobuf schema. It is
// kept independent of Google.Protobuf so the core remains entirely managed
// and has no generated-code or native runtime dependency.
//
// The reader consumes the input stream in place. Length-delimited protobuf
// messages are represented by bounded child readers rather than materialized
// as byte arrays. Only values that the model needs to retain (for example
// tensor raw_data and strings) are copied into managed storage.
internal static class OnnxProtoReader
{
    public static OnnxModelData Parse(Stream source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        ParseState state = new();
        ProtoReader reader = new(source, -1, state);
        OnnxModelData model = new();
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 0: model.IrVersion = checked((long)reader.ReadVarint()); break;
                case 8 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                        OnnxOpsetData opset = new();
                    ParseOpset(ref nested, opset);
                    nested.EnsureFullyConsumed();
                    model.Opsets.Add(opset);
                    break;
                }
                case 7 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    model.Graph = ParseGraph(ref nested);
                    nested.EnsureFullyConsumed();
                    break;
                }
                default: reader.Skip(wire); break;
            }
        }
        model.SourceLength = checked((ulong)state.Length);
        model.ContentChecksum = state.Checksum;
        if (model.Graph is null) throw new InvalidDataException("ONNX ModelProto has no graph.");
        return model;
    }

    private static void ParseOpset(ref ProtoReader reader, OnnxOpsetData result)
    {
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2: result.Domain = reader.ReadString(); break;
                case 2 when wire == 0: result.Version = checked((long)reader.ReadVarint()); break;
                default: reader.Skip(wire); break;
            }
        }
    }

    private static OnnxGraphData ParseGraph(ref ProtoReader reader)
    {
        OnnxGraphData result = new();
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                        OnnxNodeData node = new();
                    ParseNode(ref nested, node);
                    nested.EnsureFullyConsumed();
                    result.Nodes.Add(node);
                    break;
                }
                case 5 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                        OnnxTensorData tensor = new();
                    ParseTensor(ref nested, tensor);
                    nested.EnsureFullyConsumed();
                    result.Initializers.Add(tensor);
                    break;
                }
                case 11 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                        OnnxValueInfoData value = new();
                    ParseValueInfo(ref nested, value);
                    nested.EnsureFullyConsumed();
                    result.Inputs.Add(value);
                    break;
                }
                case 12 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                        OnnxValueInfoData value = new();
                    ParseValueInfo(ref nested, value);
                    nested.EnsureFullyConsumed();
                    result.Outputs.Add(value);
                    break;
                }
                case 13 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                        OnnxValueInfoData value = new();
                    ParseValueInfo(ref nested, value);
                    nested.EnsureFullyConsumed();
                    result.ValueInfo.Add(value);
                    break;
                }
                default: reader.Skip(wire); break;
            }
        }
        return result;
    }

    private static void ParseNode(ref ProtoReader reader, OnnxNodeData result)
    {
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2: result.Inputs.Add(reader.ReadString()); break;
                case 2 when wire == 2: result.Outputs.Add(reader.ReadString()); break;
                case 3 when wire == 2: result.Name = reader.ReadString(); break;
                case 4 when wire == 2: result.OpType = reader.ReadString(); break;
                case 5 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                        OnnxAttributeData attribute = new();
                    ParseAttribute(ref nested, attribute);
                    nested.EnsureFullyConsumed();
                    result.Attributes.Add(attribute);
                    break;
                }
                case 7 when wire == 2: result.Domain = reader.ReadString(); break;
                default: reader.Skip(wire); break;
            }
        }
        if (result.OpType.Length == 0)
            throw new InvalidDataException(
                $"ONNX NodeProto has no op_type (inputs={result.Inputs.Count}, " +
                $"outputs={result.Outputs.Count}, name='{result.Name}').");
    }

    private static void ParseAttribute(ref ProtoReader reader, OnnxAttributeData result)
    {
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2: result.Name = reader.ReadString(); break;
                case 20 when wire == 0: result.Type = checked((int)reader.ReadVarint()); break;
                case 2 when wire == 5: result.Float = BitConverterCompat.Int32BitsToSingle(unchecked((int)reader.ReadFixed32())); break;
                case 3 when wire == 0: result.Int = unchecked((long)reader.ReadVarint()); break;
                case 4 when wire == 2: result.String = reader.ReadBytes(); break;
                case 5 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                        OnnxTensorData tensor = new();
                    ParseTensor(ref nested, tensor);
                    nested.EnsureFullyConsumed();
                    result.Tensor = tensor;
                    break;
                }
                case 7 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ReadPackedFloat(ref nested, result.Floats);
                    nested.EnsureFullyConsumed();
                    break;
                }
                case 7 when wire == 5: result.Floats.Add(BitConverterCompat.Int32BitsToSingle(unchecked((int)reader.ReadFixed32()))); break;
                case 8 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ReadPackedInt64(ref nested, result.Ints);
                    nested.EnsureFullyConsumed();
                    break;
                }
                case 8 when wire == 0: result.Ints.Add(unchecked((long)reader.ReadVarint())); break;
                case 9 when wire == 2: result.Strings.Add(reader.ReadBytes()); break;
                default: reader.Skip(wire); break;
            }
        }
    }

    private static void ParseTensor(ref ProtoReader reader, OnnxTensorData result)
    {
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ReadPackedInt64(ref nested, result.Dims);
                    nested.EnsureFullyConsumed();
                    break;
                }
                case 1 when wire == 0: result.Dims.Add(unchecked((long)reader.ReadVarint())); break;
                case 2 when wire == 0: result.DataType = checked((int)reader.ReadVarint()); break;
                case 4 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ReadPackedFloat(ref nested, result.FloatData);
                    nested.EnsureFullyConsumed();
                    break;
                }
                case 4 when wire == 5: result.FloatData.Add(BitConverterCompat.Int32BitsToSingle(unchecked((int)reader.ReadFixed32()))); break;
                case 5 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ReadPackedInt32(ref nested, result.Int32Data);
                    nested.EnsureFullyConsumed();
                    break;
                }
                case 5 when wire == 0: result.Int32Data.Add(unchecked((int)reader.ReadVarint())); break;
                case 6 when wire == 2: result.StringData.Add(reader.ReadBytes()); break;
                case 7 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ReadPackedInt64(ref nested, result.Int64Data);
                    nested.EnsureFullyConsumed();
                    break;
                }
                case 7 when wire == 0: result.Int64Data.Add(unchecked((long)reader.ReadVarint())); break;
                case 8 when wire == 2: result.Name = reader.ReadString(); break;
                case 9 when wire == 2: result.RawData = reader.ReadBytes(); break;
                case 10 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ReadPackedDouble(ref nested, result.DoubleData);
                    nested.EnsureFullyConsumed();
                    break;
                }
                case 10 when wire == 1: result.DoubleData.Add(BitConverterCompat.Int64BitsToDouble(unchecked((long)reader.ReadFixed64()))); break;
                case 11 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ReadPackedUInt64(ref nested, result.UInt64Data);
                    nested.EnsureFullyConsumed();
                    break;
                }
                case 11 when wire == 1: result.UInt64Data.Add(reader.ReadFixed64()); break;
                default: reader.Skip(wire); break;
            }
        }
        if (result.Name.Length == 0) throw new InvalidDataException("ONNX TensorProto has no name.");
    }

    private static void ParseValueInfo(ref ProtoReader reader, OnnxValueInfoData result)
    {
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2: result.Name = reader.ReadString(); break;
                case 2 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    ParseType(ref nested, result);
                    nested.EnsureFullyConsumed();
                    break;
                }
                default: reader.Skip(wire); break;
            }
        }
    }

    private static void ParseType(ref ProtoReader reader, OnnxValueInfoData result)
    {
        while (reader.TryReadTag(out int field, out int wire))
        {
            if (field == 1 && wire == 2)
            {
                var nested = reader.EnterMessage();
                ParseTensorType(ref nested, result);
                nested.EnsureFullyConsumed();
            }
            else reader.Skip(wire);
        }
    }

    private static void ParseTensorType(ref ProtoReader reader, OnnxValueInfoData result)
    {
        while (reader.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 0: result.ElementType = checked((int)reader.ReadVarint()); break;
                case 2 when wire == 2:
                {
                    var nested = reader.EnterMessage();
                    result.Shape = ParseShape(ref nested);
                    nested.EnsureFullyConsumed();
                    break;
                }
                default: reader.Skip(wire); break;
            }
        }
    }

    private static int[] ParseShape(ref ProtoReader reader)
    {
        List<int> dims = [];
        while (reader.TryReadTag(out int field, out int wire))
        {
            if (field == 1 && wire == 2)
            {
                var dimension = reader.EnterMessage();
                long value = -1;
                while (dimension.TryReadTag(out int df, out int dw))
                {
                    if (df == 1 && dw == 0) value = unchecked((long)dimension.ReadVarint());
                    else dimension.Skip(dw);
                }
                dimension.EnsureFullyConsumed();
                dims.Add(value > 0 && value <= int.MaxValue ? (int)value : -1);
            }
            else reader.Skip(wire);
        }
        return dims.ToArray();
    }

    private static void ReadPackedFloat(ref ProtoReader reader, List<float> destination)
    {
        if ((reader.Remaining & 3) != 0) throw new InvalidDataException("Packed float field is truncated.");
        while (reader.Remaining > 0)
            destination.Add(BitConverterCompat.Int32BitsToSingle(unchecked((int)reader.ReadFixed32())));
    }

    private static void ReadPackedDouble(ref ProtoReader reader, List<double> destination)
    {
        if ((reader.Remaining & 7) != 0) throw new InvalidDataException("Packed double field is truncated.");
        while (reader.Remaining > 0)
            destination.Add(BitConverterCompat.Int64BitsToDouble(unchecked((long)reader.ReadFixed64())));
    }

    private static void ReadPackedInt32(ref ProtoReader reader, List<int> destination)
    {
        while (reader.Remaining > 0) destination.Add(unchecked((int)reader.ReadVarint()));
    }

    private static void ReadPackedInt64(ref ProtoReader reader, List<long> destination)
    {
        while (reader.Remaining > 0) destination.Add(unchecked((long)reader.ReadVarint()));
    }

    private static void ReadPackedUInt64(ref ProtoReader reader, List<ulong> destination)
    {
        while (reader.Remaining > 0) destination.Add(reader.ReadVarint());
    }

    private sealed class ParseState
    {
        public long Length;
        public ulong Checksum = 14695981039346656037UL;

        public void AddByte(byte value)
        {
            Length = checked(Length + 1);
            Checksum ^= value;
            Checksum *= 1099511628211UL;
        }

        public void Add(ReadOnlySpan<byte> bytes)
        {
            Length = checked(Length + bytes.Length);
            foreach (byte value in bytes)
            {
                Checksum ^= value;
                Checksum *= 1099511628211UL;
            }
        }
    }

    private ref struct ProtoReader
    {
        private readonly Stream _source;
        private readonly ParseState _state;
        // -1 denotes the unbounded root reader. Nested readers have an exact
        // number of bytes left and can never consume their parent message.
        private long _remaining;

        public ProtoReader(Stream source, long remaining, ParseState state)
        {
            _source = source;
            _remaining = remaining;
            _state = state;
        }

        public long Remaining => _remaining;

        public bool TryReadTag(out int field, out int wire)
        {
            int first = ReadByte(allowEof: true);
            if (first < 0) { field = wire = 0; return false; }
            ulong tag = ReadVarint(first);
            field = checked((int)(tag >> 3));
            wire = checked((int)(tag & 7));
            if (field <= 0) throw new InvalidDataException("Invalid protobuf field number.");
            return true;
        }

        public ulong ReadVarint()
        {
            int first = ReadByte(allowEof: false);
            return ReadVarint(first);
        }

        private ulong ReadVarint(int first)
        {
            ulong value = 0;
            for (int shift = 0; shift < 64; shift += 7)
            {
                int next = shift == 0 ? first : ReadByte(allowEof: false);
                byte b = checked((byte)next);
                if (shift == 63 && b > 1) throw new InvalidDataException("Protobuf varint overflows UInt64.");
                value |= (ulong)(b & 0x7f) << shift;
                if ((b & 0x80) == 0) return value;
            }
            throw new InvalidDataException("Protobuf varint is too long.");
        }

        public uint ReadFixed32()
        {
            Span<byte> bytes = stackalloc byte[4];
            ReadExact(bytes);
            return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        }

        public ulong ReadFixed64()
        {
            Span<byte> bytes = stackalloc byte[8];
            ReadExact(bytes);
            return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
        }

        public byte[] ReadBytes()
        {
            ulong length = ReadVarint();
            if (length > int.MaxValue) throw new InvalidDataException("Protobuf length-delimited field is too large.");
            int count = checked((int)length);
            byte[] value = new byte[count];
            ReadExact(value);
            return value;
        }

        public string ReadString()
        {
            try { return new UTF8Encoding(false, true).GetString(ReadBytes()); }
            catch (DecoderFallbackException ex) { throw new InvalidDataException("ONNX protobuf string is not valid UTF-8.", ex); }
        }

        public ProtoReader EnterMessage()
        {
            ulong length = ReadVarint();
            if (length > long.MaxValue || (_remaining >= 0 && length > (ulong)_remaining))
                throw new InvalidDataException("Protobuf length-delimited field is out of bounds.");
            if (_remaining >= 0) _remaining -= checked((long)length);
            return new ProtoReader(_source, checked((long)length), _state);
        }

        public void Skip(int wire)
        {
            switch (wire)
            {
                case 0: ReadVarint(); break;
                case 1: _ = ReadFixed64(); break;
                case 2: SkipLengthDelimited(); break;
                case 3:
                    while (TryReadTag(out _, out int nestedWire))
                    {
                        if (nestedWire == 4) return;
                        Skip(nestedWire);
                    }
                    throw new InvalidDataException("Unterminated protobuf group.");
                case 4: throw new InvalidDataException("Unexpected protobuf end-group.");
                case 5: _ = ReadFixed32(); break;
                default: throw new NotSupportedException($"Unsupported protobuf wire type {wire}.");
            }
        }

        public readonly void EnsureFullyConsumed()
        {
            if (_remaining != 0) throw new InvalidDataException("Truncated protobuf nested message.");
        }

        private void SkipLengthDelimited()
        {
            ulong length = ReadVarint();
            if (length > long.MaxValue || (_remaining >= 0 && length > (ulong)_remaining))
                throw new InvalidDataException("Protobuf length-delimited field is out of bounds.");
            long left = checked((long)length);
            Span<byte> scratch = stackalloc byte[4096];
            while (left > 0)
            {
                int take = (int)Math.Min((long)scratch.Length, left);
                ReadExact(scratch[..take]);
                left -= take;
            }
        }

        private int ReadByte(bool allowEof)
        {
            if (_remaining == 0)
            {
                if (allowEof) return -1;
                throw new InvalidDataException("Truncated protobuf message.");
            }
            int value = _source.ReadByte();
            if (value < 0)
            {
                if (allowEof && _remaining < 0) return -1;
                throw new InvalidDataException("Truncated protobuf message.");
            }
            _state.AddByte(checked((byte)value));
            if (_remaining > 0) _remaining--;
            return value;
        }

        private void ReadExact(scoped Span<byte> destination)
        {
            if (_remaining >= 0 && destination.Length > _remaining)
                throw new InvalidDataException("Truncated protobuf message.");
            int offset = 0;
            while (offset < destination.Length)
            {
                int read = _source.Read(destination[offset..]);
                if (read <= 0) throw new InvalidDataException("Truncated protobuf message.");
                _state.Add(destination.Slice(offset, read));
                offset += read;
                if (_remaining > 0) _remaining -= read;
            }
        }
    }
}

internal sealed class OnnxModelData
{
    public long IrVersion;
    public ulong SourceLength;
    public ulong ContentChecksum;
    public List<OnnxOpsetData> Opsets { get; } = [];
    public OnnxGraphData? Graph;
}
internal sealed class OnnxOpsetData { public string Domain = ""; public long Version; }
internal sealed class OnnxGraphData
{
    public List<OnnxNodeData> Nodes { get; } = [];
    public List<OnnxTensorData> Initializers { get; } = [];
    public List<OnnxValueInfoData> Inputs { get; } = [];
    public List<OnnxValueInfoData> Outputs { get; } = [];
    public List<OnnxValueInfoData> ValueInfo { get; } = [];
}
internal sealed class OnnxNodeData
{
    public List<string> Inputs { get; } = [];
    public List<string> Outputs { get; } = [];
    public List<OnnxAttributeData> Attributes { get; } = [];
    public string Name = "";
    public string OpType = "";
    public string Domain = "";
}
internal sealed class OnnxAttributeData
{
    public string Name = "";
    public int Type;
    public float Float;
    public long Int;
    public byte[] String = [];
    public OnnxTensorData? Tensor;
    public List<float> Floats { get; } = [];
    public List<long> Ints { get; } = [];
    public List<byte[]> Strings { get; } = [];
}
internal sealed class OnnxTensorData
{
    public string Name = "";
    public int DataType;
    public List<long> Dims { get; } = [];
    public byte[] RawData = [];
    public List<float> FloatData { get; } = [];
    public List<int> Int32Data { get; } = [];
    public List<long> Int64Data { get; } = [];
    public List<byte[]> StringData { get; } = [];
    public List<double> DoubleData { get; } = [];
    public List<ulong> UInt64Data { get; } = [];
}
internal sealed class OnnxValueInfoData
{
    public string Name = "";
    public int ElementType;
    public int[] Shape = [];
}
