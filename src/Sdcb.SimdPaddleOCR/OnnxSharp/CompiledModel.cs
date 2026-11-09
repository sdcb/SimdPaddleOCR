using Sdcb.SimdOnnx.Decoding;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// A <see cref="Model"/> prepared for one OCR stage: per-shape CPU plans (compiled on demand, shared
/// packed weights) plus, when the graph ends in a constant projection + last-axis Softmax, a
/// head-truncated variant whose output is the projection's input (the recognizer and classifier
/// run the head themselves: fused MatMul+ArgMax / a tiny softmax). Thread-safe; share freely.
/// </summary>
internal sealed class CompiledModel : IDisposable
{
    internal static readonly CompileOptions Options = CompileOptions.DefaultNhwc
        ? new CompileOptions(Nhwc: true, PhysicalIo: true, LogicalOutput: true)
        : new CompileOptions(Nhwc: false);

    private readonly object _gate = new();
    private bool _headResolved;
    private OcrHead? _head;
    private CpuModel? _ctc, _logits;

    /// <param name="releaseInput">the caller writes a fresh input before every run and reads only the
    /// output (the detector): the CPU arena may reuse the input buffer after its last read.</param>
    internal CompiledModel(Model model, int intraOpThreads, bool releaseInput = false)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        IntraOpThreads = MathCompat.Clamp(intraOpThreads, 1, 16);
        Full = new CpuModel(model.Onnx, releaseInput ? Options with { ReleaseInputs = true } : Options);
    }

    /// <summary>Compiles <paramref name="inputShape"/> up front (sessions start bound to it).</summary>
    internal CompiledModel(Model model, ReadOnlySpan<int> inputShape, int intraOpThreads)
        : this(model, intraOpThreads)
    {
        DefaultInputShape = inputShape.ToArray();
        ResolveHead(inputShape);
    }

    internal Model Model { get; }
    internal int IntraOpThreads { get; }
    internal int[]? DefaultInputShape { get; }
    internal CpuModel Full { get; }
    /// <summary>Graph input is written channels-last ([n,h,w,c]) by the preprocessors.</summary>
    internal bool InputIsNhwc => Options.Nhwc && Options.PhysicalIo;

    /// <summary>Plans ending at the projection input (null when the graph has no such head).</summary>
    internal CpuModel? Ctc { get { lock (_gate) return _ctc; } }
    /// <summary>Plans ending at the final Softmax's input (null when the graph does not end in one).</summary>
    internal CpuModel? Logits { get { lock (_gate) return _logits; } }
    /// <summary>Name of the final Softmax's input (null when the graph does not end in one).</summary>
    internal string? LogitsName { get { lock (_gate) return _logits?.Options.OutputName; } }

    /// <summary>The projection head; resolved once from a plan of <paramref name="probeShape"/>.</summary>
    internal OcrHead? ResolveHead(ReadOnlySpan<int> probeShape)
    {
        lock (_gate)
        {
            if (_headResolved) return _head;
            if (ModelHeads.FindSoftmaxHead(Model.Onnx, probeShape, Options) is { } tail)
            {
                if (tail.LogitsName.Length != 0) _logits = Full.WithOutput(tail.LogitsName);
                if (tail.Projection is { } p)
                {
                    _head = new OcrHead(p);
                    _ctc = Full.WithOutput(p.ActivationName);
                }
            }
            _headResolved = true;
            return _head;
        }
    }

    public void Dispose() { }
}

/// <summary>Constant projection + bias in front of the final Softmax: logits = A·W + b, W [inner, columns].</summary>
internal sealed class OcrHead
{
    internal OcrHead(LinearProjection p)
    {
        ActivationName = p.ActivationName;
        Projection = new CtcProjection(p.Weights, p.Bias, p.Inner, p.Columns);
    }

    internal string ActivationName { get; }
    /// <summary>The projection fused with the per-row argmax (packed once, shared by every session).</summary>
    internal CtcProjection Projection { get; }
    internal float[] Weights => Projection.Weights;
    internal float[]? Bias => Projection.Bias;
    internal int Inner => Projection.Inner;
    internal int Columns => Projection.Columns;
}