using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>A parsed ONNX model (Sdcb.SimdOnnx). The serialized payload is not retained.</summary>
public sealed class Model : IDisposable
{
    private Model(OnnxModel onnx) => Onnx = onnx;

    internal OnnxModel Onnx { get; }

    public static Model Load(ReadOnlyMemory<byte> source)
    {
        if (source.Length == 0) throw new InvalidDataException("ONNX model is empty.");
        return new Model(OnnxModel.Load(source.Span));
    }

    public static Model Load(byte[] source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        return Load((ReadOnlyMemory<byte>)source);
    }

    /// <summary>Loads an ONNX model from the current stream position. The caller owns the stream.</summary>
    public static Model Load(Stream source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        return new Model(OnnxModel.Load(source));
    }

    /// <summary>Loads an ONNX model from the current stream position on a worker thread.</summary>
    public static Task<Model> LoadAsync(Stream source, CancellationToken cancellationToken = default)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        cancellationToken.ThrowIfCancellationRequested();
        return Task.Run(() => Load(source), cancellationToken);
    }

    public static Model LoadOnnx(Stream source) => Load(source);
    public static Model LoadOnnx(ReadOnlyMemory<byte> source) => Load(source);
    public static Model LoadOnnx(byte[] source) => Load(source);

    public static Model LoadOnnx(ReadOnlySpan<byte> source)
    {
        if (source.Length == 0) throw new InvalidDataException("ONNX model is empty.");
        return new Model(OnnxModel.Load(source));
    }

    /// <summary>Weights are managed; kept for API compatibility.</summary>
    public void Dispose() { }
}
