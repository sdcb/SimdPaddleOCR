namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Operands for the CTC / classifier projection after <see cref="IOcrSession.TryRunUntilCtcProjection"/>.
/// Spans alias session activations / model weights and are valid until the next Run* on that session.
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
