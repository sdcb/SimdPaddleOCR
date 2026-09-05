using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class ConvDenseStride1
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void DenseStride1QuadUnsafe(float* input, float* weights, float* bias,
        float* output, int inputChannels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int co, int xStart, int xEnd)
    {
        int weightsPerOut = inputChannels * kernelH * kernelW;
        float* w0 = weights + (long)co * weightsPerOut;
        float* w1 = w0 + weightsPerOut, w2 = w1 + weightsPerOut, w3 = w2 + weightsPerOut;
        float* out0 = output + (long)co * outputHeight * outputWidth;
        float* out1 = out0 + outputHeight * outputWidth, out2 = out1 + outputHeight * outputWidth,
            out3 = out2 + outputHeight * outputWidth;
        Vector256<float> bias0 = Vector256.Create(bias == null ? 0f : bias[co]);
        Vector256<float> bias1 = Vector256.Create(bias == null ? 0f : bias[co + 1]);
        Vector256<float> bias2 = Vector256.Create(bias == null ? 0f : bias[co + 2]);
        Vector256<float> bias3 = Vector256.Create(bias == null ? 0f : bias[co + 3]);
        for (int y = 0; y < outputHeight; y++)
        {
            // Rows near the top/bottom simply restrict the valid tap range;
            // the restricted loop adds taps in the same (ci,ky,kx) order the
            // scalar edge path uses, so results are identical.
            int kyMin = Math.Max(0, padTop - y);
            int kyMax = Math.Min(kernelH, height - y + padTop);
            if (kyMax <= kyMin)
            {
                for (int x = 0; x < outputWidth; x++)
                    for (int q = 0; q < 4; q++)
                        DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                            outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
                continue;
            }
            for (int x = 0; x < xStart; x++)
                for (int q = 0; q < 4; q++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
            int x8 = xStart;
            int kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            int weightRowSkip = kyMin * kernelW, weightRowRemainder = (kernelH - kyMax) * kernelW;
            for (; x8 <= xEnd - 8; x8 += 8)
            {
                Vector256<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                float* inputChannel = rowBase + x8;
                float* wc0 = w0 + weightRowSkip, wc1 = w1 + weightRowSkip,
                    wc2 = w2 + weightRowSkip, wc3 = w3 + weightRowSkip;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    float* tapRow = inputChannel;
                    for (int ky = 0; ky < kyCount; ky++)
                    {
                        for (int kx = 0; kx < kernelW; kx++)
                        {
                            Vector256<float> value = Avx.LoadVector256(tapRow + kx);
                            a0 = AddMul(a0, value, Vector256.Create(wc0[kx]));
                            a1 = AddMul(a1, value, Vector256.Create(wc1[kx]));
                            a2 = AddMul(a2, value, Vector256.Create(wc2[kx]));
                            a3 = AddMul(a3, value, Vector256.Create(wc3[kx]));
                        }
                        tapRow += width;
                        wc0 += kernelW; wc1 += kernelW; wc2 += kernelW; wc3 += kernelW;
                    }
                    inputChannel += height * width;
                    wc0 += weightRowSkip + weightRowRemainder; wc1 += weightRowSkip + weightRowRemainder;
                    wc2 += weightRowSkip + weightRowRemainder; wc3 += weightRowSkip + weightRowRemainder;
                }
                int outOffset = y * outputWidth + x8;
                Avx.Store(out0 + outOffset, a0); Avx.Store(out1 + outOffset, a1);
                Avx.Store(out2 + outOffset, a2); Avx.Store(out3 + outOffset, a3);
            }
            for (int x = x8; x < outputWidth; x++)
                for (int q = 0; q < 4; q++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void DenseStride1SingleUnsafe(float* input, float* weights, float* bias,
        float* output, int inputChannels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int co, int xStart, int xEnd)
    {
        int weightsPerOut = inputChannels * kernelH * kernelW;
        float* w0 = weights + (long)co * weightsPerOut;
        float* out0 = output + (long)co * outputHeight * outputWidth;
        Vector256<float> bias0 = Vector256.Create(bias == null ? 0f : bias[co]);
        for (int y = 0; y < outputHeight; y++)
        {
            int kyMin = Math.Max(0, padTop - y);
            int kyMax = Math.Min(kernelH, height - y + padTop);
            if (kyMax <= kyMin)
            {
                for (int x = 0; x < outputWidth; x++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co, y, x);
                continue;
            }
            for (int x = 0; x < xStart; x++)
                DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                    outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co, y, x);
            int x8 = xStart;
            int kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            int weightRowSkip = kyMin * kernelW, weightRowRemainder = (kernelH - kyMax) * kernelW;
            for (; x8 <= xEnd - 8; x8 += 8)
            {
                Vector256<float> a0 = bias0;
                float* inputChannel = rowBase + x8;
                float* wc0 = w0 + weightRowSkip;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    float* tapRow = inputChannel;
                    for (int ky = 0; ky < kyCount; ky++)
                    {
                        for (int kx = 0; kx < kernelW; kx++)
                            a0 = AddMul(a0, Avx.LoadVector256(tapRow + kx), Vector256.Create(wc0[kx]));
                        tapRow += width;
                        wc0 += kernelW;
                    }
                    inputChannel += height * width;
                    wc0 += weightRowSkip + weightRowRemainder;
                }
                Avx.Store(out0 + y * outputWidth + x8, a0);
            }
            for (int x = x8; x < outputWidth; x++)
                DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                    outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co, y, x);
        }
    }
}
