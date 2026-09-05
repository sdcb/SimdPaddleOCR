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
    private static unsafe void DepthwiseStride1ChannelAvx512Unsafe(float* input, float* weights,
        float bias, float* output, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int xStart, int xEnd)
    {
        Vector512<float> vBias = Vector512.Create(bias);
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
            for (; x16 <= xEnd - 32; x16 += 32)
            {
                Vector512<float> a0 = vBias, a1 = vBias;
                float* tapRow = rowBase + x16;
                float* wc = weightBase;
                for (int ky = 0; ky < kyCount; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                    {
                        Vector512<float> weight = Vector512.Create(wc[kx]);
                        a0 = AddMul512(a0, Avx512F.LoadVector512(tapRow + kx), weight);
                        a1 = AddMul512(a1, Avx512F.LoadVector512(tapRow + kx + 16), weight);
                    }
                    tapRow += width;
                    wc += kernelW;
                }
                Avx512F.Store(output + y * outputWidth + x16, a0);
                Avx512F.Store(output + y * outputWidth + x16 + 16, a1);
            }
            for (; x16 <= xEnd - 16; x16 += 16)
            {
                Vector512<float> a0 = vBias;
                float* tapRow = rowBase + x16;
                float* wc = weightBase;
                for (int ky = 0; ky < kyCount; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                        a0 = AddMul512(a0, Avx512F.LoadVector512(tapRow + kx), Vector512.Create(wc[kx]));
                    tapRow += width;
                    wc += kernelW;
                }
                Avx512F.Store(output + y * outputWidth + x16, a0);
            }
            for (int x = x16; x < outputWidth; x++)
                DepthwiseEdgePixel(input, weights, bias, output, height, width,
                    outputWidth, kernelH, kernelW, padTop, padLeft, y, x);
        }
    }
}
