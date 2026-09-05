namespace Sdcb.PaddleOCR.OnnxSharp;

public readonly record struct TensorShape(int[] Dimensions)
{
    public int Rank => Dimensions.Length;
    public long ElementCount
    {
        get { long n = 1; foreach (int d in Dimensions) n = checked(n * d); return n; }
    }
    public override string ToString() => $"[{string.Join(",", Dimensions)}]";
}

