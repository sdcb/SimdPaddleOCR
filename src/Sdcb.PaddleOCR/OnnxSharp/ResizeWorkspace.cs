namespace Sdcb.PaddleOCR.OnnxSharp;

/// <summary>
/// Per-request scratch storage for resize implementations. A workspace is
/// intentionally not thread-safe; the owning inference request must be used
/// by one inference at a time.
/// </summary>
internal sealed class ResizeWorkspace
{
    public int[] XOffsets = [];
    public short[] XCoefficients = [];
    public int[] Row0 = [];
    public int[] Row1 = [];

    public void Ensure(int width)
    {
        if (XOffsets.Length < width) Array.Resize(ref XOffsets, width);
        int coefficientCount = checked(width * 2);
        if (XCoefficients.Length < coefficientCount) Array.Resize(ref XCoefficients, coefficientCount);
        int rowCount = checked(width * 3);
        if (Row0.Length < rowCount) Array.Resize(ref Row0, rowCount);
        if (Row1.Length < rowCount) Array.Resize(ref Row1, rowCount);
    }
}
