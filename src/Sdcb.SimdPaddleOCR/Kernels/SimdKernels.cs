namespace Sdcb.SimdPaddleOCR.Kernels;

internal static class SimdKernels
{
    /// <summary>
    /// softmax(logits)[argmax] given the row maximum: 1 / Σ exp(x - max). Only the dense CTC
    /// fallback (graphs without a fusable projection head) uses it.
    /// </summary>
    public static float SoftmaxMaximumProbability(ReadOnlySpan<float> logits, float maximum)
    {
        float sum = 0;
        foreach (float v in logits) sum += MathF.Exp(v - maximum);
        return 1f / sum;
    }
}
