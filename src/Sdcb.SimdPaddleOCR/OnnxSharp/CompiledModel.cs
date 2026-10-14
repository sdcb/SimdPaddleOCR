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

    internal CompiledModel(Model model, int intraOpThreads)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        IntraOpThreads = MathCompat.Clamp(intraOpThreads, 1, 16);
        Full = new CpuModel(model.Onnx, Options);
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

    /// <summary>The projection head; resolved once from a plan of <paramref name="probeShape"/>.</summary>
    internal OcrHead? ResolveHead(ReadOnlySpan<int> probeShape)
    {
        lock (_gate)
        {
            if (_headResolved) return _head;
            ExecPlan plan = GraphCompiler.Compile(Model.Onnx, probeShape, Options);
            ExecNode? softmax = Producer(plan, plan.OutputValue);
            if (softmax is { Op: ExecOp.Softmax })
            {
                string logitsName = plan.Values[softmax.In[0]].Name;
                if (logitsName.Length != 0)
                    _logits = new CpuModel(Model.Onnx, Options with { OutputName = logitsName }, packs: Full.Packs);
                _head = OcrHead.TryCreate(plan, softmax);
                if (_head is not null)
                    _ctc = new CpuModel(Model.Onnx, Options with { OutputName = _head.ActivationName }, packs: Full.Packs);
            }
            _headResolved = true;
            return _head;
        }
    }

    internal static ExecNode? Producer(ExecPlan plan, int value)
    {
        // zero-copy views (reshape/squeeze/fused tails) have no producer node of their own
        for (int hops = 0; value >= 0 && hops < 64; hops++, value = plan.Values[value].AliasOf)
            foreach (ExecNode n in plan.Nodes)
                if (n.Out.Length == 1 && n.Out[0] == value) return n;
        return null;
    }

    public void Dispose() { }
}

/// <summary>Constant projection + bias in front of the final Softmax: logits = A·W + b, W [inner, columns].</summary>
internal sealed class OcrHead
{
    private OcrHead(string activationName, float[] weights, float[]? bias, int inner, int columns)
    {
        ActivationName = activationName;
        Weights = weights;
        Bias = bias;
        Inner = inner;
        Columns = columns;
        Packed = PackForArgMax(weights, inner, columns);
    }

    internal string ActivationName { get; }
    internal float[] Weights { get; }
    internal float[]? Bias { get; }
    /// <summary>[columns/16][inner][16] tiles for the wide REC projection (null when not worth it).</summary>
    internal float[]? Packed { get; }
    internal int Inner { get; }
    internal int Columns { get; }

    internal static OcrHead? TryCreate(ExecPlan plan, ExecNode softmax)
    {
        ExecNode? p = CompiledModel.Producer(plan, softmax.In[0]);
        float[]? bias = null;
        if (p is { Op: ExecOp.Elementwise, P: EwP { Op: EwOp.Add, Act: ActivationKind.None } } && p.In.Length == 2)
        {
            int c = plan.Values[p.In[0]].Floats is not null ? 0 : plan.Values[p.In[1]].Floats is not null ? 1 : -1;
            if (c < 0) return null;
            bias = plan.Values[p.In[c]].Floats;
            p = CompiledModel.Producer(plan, p.In[1 - c]);
            if (p is { P: GemmP { BiasValue: >= 0 } }) return null;
        }
        if (p is not { Op: ExecOp.Gemm, P: GemmP g } || g.Act != ActivationKind.None) return null;
        if (g.BiasValue >= 0)
        {
            if (bias is not null) return null;
            bias = plan.Values[g.BiasValue].Floats;
            if (bias is null) return null;
        }
        CValue a = plan.Values[p.In[0]], b = plan.Values[p.In[1]];
        if (b.Floats is not { } w || b.Shape.Length != 2 || a.Name.Length == 0 || a.Shape.Length < 2) return null;
        int inner = g.TransB ? b.Shape[1] : b.Shape[0], columns = g.TransB ? b.Shape[0] : b.Shape[1];
        if (a.Shape[^1] != inner || (bias is not null && bias.Length != columns)) return null;
        if (g.TransB)
        {
            float[] t = new float[w.Length];
            for (int n = 0; n < columns; n++)
                for (int k = 0; k < inner; k++) t[k * columns + n] = w[n * inner + k];
            w = t;
        }
        return new OcrHead(a.Name, w, bias, inner, columns);
    }

    private static float[]? PackForArgMax(float[] w, int inner, int columns)
    {
        if (inner < 64 || columns < 1024) return null;
        int tiles = columns / 16;
        float[] packed = new float[checked(tiles * inner * 16)];
        for (int tile = 0; tile < tiles; tile++)
            for (int k = 0; k < inner; k++)
                Array.Copy(w, k * columns + tile * 16, packed, (tile * inner + k) * 16, 16);
        return packed;
    }
}
