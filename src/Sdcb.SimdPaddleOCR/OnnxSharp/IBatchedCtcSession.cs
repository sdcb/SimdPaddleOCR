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

    /// <summary>CTC operands for the units set by <see cref="ReshapeMany"/>:
    /// unit i yields batch_i × rows[i] activation rows of head.Inner floats.
    /// False when the graph has no CTC tail.</summary>
    bool TryResolveManyHead(out int[] rows, out CtcHead head);

    /// <summary>
    /// Runs every unit set by <see cref="ReshapeMany"/> and calls
    /// <paramref name="onReady"/> in unit order as batches of units complete
    /// (unit i's activations start at offsets[i]); later units may still be
    /// running on the device meanwhile. Returns false when the GPU path failed
    /// (callers then use the per-unit path) or <paramref name="onReady"/>
    /// returned false. Exceptions from <paramref name="onReady"/> propagate.
    /// </summary>
    bool TryRunManyUntilCtcProjection(CtcUnitsReady onReady);
}

/// <summary>Units [first, first+count) of <paramref name="activations"/> are ready.</summary>
internal delegate bool CtcUnitsReady(float[] activations, int[] offsets, int first, int count);

/// <summary>Shape-independent CTC projection operands (vocab MatMul + bias).</summary>
internal sealed class CtcHead
{
    public byte[] Weights = [];
    public byte[]? Bias;
    public float[]? Packed;
    public int Inner, Columns, MatMulIndex;
}
