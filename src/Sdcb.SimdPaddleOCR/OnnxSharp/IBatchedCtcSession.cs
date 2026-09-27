namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Optional capability of GPU-backed REC sessions: several input shapes
/// (e.g. every exact-width line group of one image) run up to the CTC
/// projection in a single device submission. Exact widths mean no extra
/// padding, so results equal per-group runs; the saving is the per-submit
/// round trip. The CPU interpreter does not implement this.
/// </summary>
internal interface IBatchedCtcSession
{
    /// <summary>False once the session has fallen back to CPU.</summary>
    bool CanRunMany { get; }

    /// <summary>Sizes the staging buffer for <paramref name="shapes"/> laid out
    /// back to back (unit i occupies numel(shapes[i]) floats) and returns it.</summary>
    Span<float> ReshapeMany(IReadOnlyList<int[]> shapes);

    /// <summary>
    /// Runs every unit set by <see cref="ReshapeMany"/>. Activations of unit i
    /// start at <paramref name="actOffsets"/>[i] and hold batch_i × rows[i] ×
    /// head.Inner floats. Returns false (nothing half-done) when the graph has
    /// no CTC tail or the GPU path failed; callers then use the per-unit path.
    /// </summary>
    bool TryRunManyUntilCtcProjection(out float[] activations, out int[] actOffsets,
        out int[] rows, out CtcHead head);
}

/// <summary>Shape-independent CTC projection operands (vocab MatMul + bias).</summary>
internal sealed class CtcHead
{
    public byte[] Weights = [];
    public byte[]? Bias;
    public float[]? Packed;
    public int Inner, Columns, MatMulIndex;
}
