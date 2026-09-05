using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class DepthwiseStride1
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void DepthwiseStride1ChannelUnsafe(float* input, float* weights,
        float bias, float* output, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int xStart, int xEnd)
    {
        Vector256<float> vBias = Vector256.Create(bias);
        for (int y = 0; y < outputHeight; y++)
        {
            int kyMin = Math.Max(0, padTop - y);
            int kyMax = Math.Min(kernelH, height - y + padTop);
            if (kyMax <= kyMin)
            {
                for (int x = 0; x < outputWidth; x++)
                    DepthwiseEdgePixel(input, weights, bias, output, height, width,
                        outputWidth, kernelH, kernelW, padTop, padLeft, y, x);
                continue;
            }
            for (int x = 0; x < xStart; x++)
                DepthwiseEdgePixel(input, weights, bias, output, height, width,
                    outputWidth, kernelH, kernelW, padTop, padLeft, y, x);
            int kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            float* weightBase = weights + kyMin * kernelW;
            int x16 = xStart;
            for (; x16 <= xEnd - 16; x16 += 16)
            {
                Vector256<float> a0 = vBias, a1 = vBias;
                float* tapRow = rowBase + x16;
                float* wc = weightBase;
                for (int ky = 0; ky < kyCount; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                    {
                        Vector256<float> weight = Vector256.Create(wc[kx]);
                        a0 = AddMul(a0, Avx.LoadVector256(tapRow + kx), weight);
                        a1 = AddMul(a1, Avx.LoadVector256(tapRow + kx + 8), weight);
                    }
                    tapRow += width;
                    wc += kernelW;
                }
                Avx.Store(output + y * outputWidth + x16, a0);
                Avx.Store(output + y * outputWidth + x16 + 8, a1);
            }
            for (; x16 <= xEnd - 8; x16 += 8)
            {
                Vector256<float> a0 = vBias;
                float* tapRow = rowBase + x16;
                float* wc = weightBase;
                for (int ky = 0; ky < kyCount; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                        a0 = AddMul(a0, Avx.LoadVector256(tapRow + kx), Vector256.Create(wc[kx]));
                    tapRow += width;
                    wc += kernelW;
                }
                Avx.Store(output + y * outputWidth + x16, a0);
            }
            for (int x = x16; x < outputWidth; x++)
                DepthwiseEdgePixel(input, weights, bias, output, height, width,
                    outputWidth, kernelH, kernelW, padTop, padLeft, y, x);
        }
    }
}
