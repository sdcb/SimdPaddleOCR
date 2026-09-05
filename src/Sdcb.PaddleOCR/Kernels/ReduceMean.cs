using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.PaddleOCR.Kernels;

internal static class ReduceMean
{
    /// <summary>
    /// NCHW mean over H×W, writing one value per (N, C). Same ISA ladder and
    /// accumulation as the previous InferenceSession fast path.
    /// </summary>
    internal static unsafe void SpatialNchw(ReadOnlySpan<float> input, Span<float> output,
        int batch, int channels, int spatial)
    {
        if (Avx512F.IsSupported)
        {
            fixed (float* inputPtr = input)
            {
                for (int b = 0; b < batch; b++)
                    for (int channel = 0; channel < channels; channel++)
                    {
                        int baseOffset = (b * channels + channel) * spatial;
                        Vector512<float> sum0 = Vector512<float>.Zero;
                        Vector512<float> sum1 = Vector512<float>.Zero;
                        int i = 0;
                        for (; i <= spatial - 32; i += 32)
                        {
                            sum0 = Avx512F.Add(sum0, Avx512F.LoadVector512(inputPtr + baseOffset + i));
                            sum1 = Avx512F.Add(sum1, Avx512F.LoadVector512(inputPtr + baseOffset + i + 16));
                        }
                        Vector512<float> sum = Avx512F.Add(sum0, sum1);
                        float scalar = 0f;
                        for (int lane = 0; lane < 16; lane++) scalar += sum.GetElement(lane);
                        for (; i <= spatial - 16; i += 16)
                        {
                            Vector512<float> chunk = Avx512F.LoadVector512(inputPtr + baseOffset + i);
                            for (int lane = 0; lane < 16; lane++) scalar += chunk.GetElement(lane);
                        }
                        for (; i < spatial; i++) scalar += inputPtr[baseOffset + i];
                        output[b * channels + channel] = scalar / spatial;
                    }
            }
            return;
        }
        if (Avx.IsSupported)
        {
            fixed (float* inputPtr = input)
            {
                for (int b = 0; b < batch; b++)
                    for (int channel = 0; channel < channels; channel++)
                    {
                        int baseOffset = (b * channels + channel) * spatial;
                        Vector256<float> sum0 = Vector256<float>.Zero;
                        Vector256<float> sum1 = Vector256<float>.Zero;
                        int i = 0;
                        for (; i <= spatial - 16; i += 16)
                        {
                            sum0 = Avx.Add(sum0, Avx.LoadVector256(inputPtr + baseOffset + i));
                            sum1 = Avx.Add(sum1, Avx.LoadVector256(inputPtr + baseOffset + i + 8));
                        }
                        Vector256<float> sum = Avx.Add(sum0, sum1);
                        float scalar = sum.GetElement(0) + sum.GetElement(1) + sum.GetElement(2) + sum.GetElement(3) +
                            sum.GetElement(4) + sum.GetElement(5) + sum.GetElement(6) + sum.GetElement(7);
                        for (; i < spatial; i++) scalar += inputPtr[baseOffset + i];
                        output[b * channels + channel] = scalar / spatial;
                    }
            }
            return;
        }
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            fixed (float* inputPtr = input)
            {
                for (int b = 0; b < batch; b++)
                    for (int channel = 0; channel < channels; channel++)
                    {
                        int baseOffset = (b * channels + channel) * spatial;
                        Vector<float> sumVector = Vector<float>.Zero;
                        int i = 0;
                        for (; i <= spatial - width; i += width)
                            sumVector += Vector.LoadUnsafe(ref Unsafe.AsRef<float>(inputPtr + baseOffset + i));
                        float scalar = 0f;
                        for (int lane = 0; lane < width; lane++) scalar += sumVector[lane];
                        for (; i < spatial; i++) scalar += inputPtr[baseOffset + i];
                        output[b * channels + channel] = scalar / spatial;
                    }
            }
            return;
        }
        for (int b = 0; b < batch; b++)
            for (int channel = 0; channel < channels; channel++)
            {
                int baseOffset = (b * channels + channel) * spatial;
                float sum = 0;
                for (int i = 0; i < spatial; i++) sum += input[baseOffset + i];
                output[b * channels + channel] = sum / spatial;
            }
    }
}
