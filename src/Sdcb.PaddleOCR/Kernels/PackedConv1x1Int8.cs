namespace Sdcb.PaddleOCR.Kernels;

internal sealed record PackedConv1x1Int8(byte[] Weights, float[] Scales, int[] Sums);
