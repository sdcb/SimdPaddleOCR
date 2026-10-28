using Sdcb.SimdOnnx.Decoding;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Operands for the CTC / classifier projection after <see cref="IOcrSession.TryRunUntilCtcProjection"/>.
/// Spans alias session activations / model weights and are valid until the next Run* on that session.
/// </summary>
internal readonly ref struct CtcProjectionOperands
{
    internal CtcProjectionOperands(ReadOnlySpan<float> activations, CtcProjection projection, int batch, int rows,
        int matMulNodeIndex)
    {
        Activations = activations;
        Projection = projection;
        Batch = batch;
        Rows = rows;
        MatMulNodeIndex = matMulNodeIndex;
    }

    internal ReadOnlySpan<float> Activations { get; }
    internal CtcProjection Projection { get; }
    internal int Batch { get; }
    internal int Rows { get; }
    internal int Inner => Projection.Inner;
    internal int Columns => Projection.Columns;
    internal int MatMulNodeIndex { get; }
    internal int RowCount => checked(Batch * Rows);
}
