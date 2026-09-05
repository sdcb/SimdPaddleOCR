using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class ConvDenseStride1
{
    internal static unsafe bool TryVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int padTop, int padLeft,
        int intraOpThreads)
    {
        if ((long)kernelH * kernelW * inputChannels > 1 << 20) return false;
        int xStart = Math.Clamp(padLeft, 0, outputWidth);
        int xEnd = Math.Clamp(width - kernelW + 1 + padLeft, xStart, outputWidth);
        int widthLanes = Vector<float>.Count;
        if (xEnd - xStart < widthLanes) return false;
        if (intraOpThreads > 1 && batch == 1 && outputChannels >= 2 &&
            (long)outputChannels * inputChannels * outputHeight * outputWidth * kernelH * kernelW >= 4_000_000)
        {
            int workers = Math.Min(intraOpThreads, outputChannels);
            fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = weights.Length,
                    biasLength = bias.Length, outputLength = output.Length;
                int weightsPerOut = inputChannels * kernelH * kernelW, plane = outputHeight * outputWidth;
                Parallel.For(0, workers, worker =>
                {
                    int begin = outputChannels * worker / workers, end = outputChannels * (worker + 1) / workers;
                    if (end <= begin) return;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> wSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(begin * weightsPerOut, (end - begin) * weightsPerOut);
                    ReadOnlySpan<float> bSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, end - begin);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * plane, (end - begin) * plane);
                    TryVector(inSpan, wSpan, bSpan, outSpan, 1, inputChannels, height,
                        width, end - begin, outputHeight, outputWidth, kernelH, kernelW,
                        padTop, padLeft, 1);
                });
            }
            return true;
        }

        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            float* biasOrNull = bias.Length == 0 ? null : biasPtr;
            for (int b = 0; b < batch; b++)
            {
                float* batchInput = inputPtr + (long)b * inputChannels * height * width;
                float* batchOutput = outputPtr + (long)b * outputChannels * outputHeight * outputWidth;
                int co = 0;
                for (; co <= outputChannels - 4; co += 4)
                    DenseStride1QuadVector(batchInput, weightsPtr, biasOrNull, batchOutput,
                        inputChannels, height, width, outputHeight, outputWidth, kernelH, kernelW,
                        padTop, padLeft, co, xStart, xEnd);
                for (; co < outputChannels; co++)
                    DenseStride1SingleVector(batchInput, weightsPtr, biasOrNull, batchOutput,
                        inputChannels, height, width, outputHeight, outputWidth, kernelH, kernelW,
                        padTop, padLeft, co, xStart, xEnd);
            }
        }
        return true;
    }

    private static unsafe void DenseStride1QuadVector(float* input, float* weights, float* bias,
        float* output, int inputChannels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int co, int xStart, int xEnd)
    {
        int weightsPerOut = inputChannels * kernelH * kernelW;
        int widthLanes = Vector<float>.Count;
        float* w0 = weights + (long)co * weightsPerOut;
        float* w1 = w0 + weightsPerOut, w2 = w1 + weightsPerOut, w3 = w2 + weightsPerOut;
        float* out0 = output + (long)co * outputHeight * outputWidth;
        float* out1 = out0 + outputHeight * outputWidth, out2 = out1 + outputHeight * outputWidth,
            out3 = out2 + outputHeight * outputWidth;
        Vector<float> bias0 = new(bias == null ? 0f : bias[co]);
        Vector<float> bias1 = new(bias == null ? 0f : bias[co + 1]);
        Vector<float> bias2 = new(bias == null ? 0f : bias[co + 2]);
        Vector<float> bias3 = new(bias == null ? 0f : bias[co + 3]);
        for (int y = 0; y < outputHeight; y++)
        {
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
            int xv = xStart;
            int kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            int weightRowSkip = kyMin * kernelW, weightRowRemainder = (kernelH - kyMax) * kernelW;
            for (; xv <= xEnd - widthLanes; xv += widthLanes)
            {
                Vector<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                float* inputChannel = rowBase + xv;
                float* wc0 = w0 + weightRowSkip, wc1 = w1 + weightRowSkip,
                    wc2 = w2 + weightRowSkip, wc3 = w3 + weightRowSkip;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    float* tapRow = inputChannel;
                    for (int ky = 0; ky < kyCount; ky++)
                    {
                        for (int kx = 0; kx < kernelW; kx++)
                        {
                            Vector<float> value = VectorLoad(tapRow + kx);
                            a0 = VectorAddMul(a0, value, wc0[kx]);
                            a1 = VectorAddMul(a1, value, wc1[kx]);
                            a2 = VectorAddMul(a2, value, wc2[kx]);
                            a3 = VectorAddMul(a3, value, wc3[kx]);
                        }
                        tapRow += width;
                        wc0 += kernelW; wc1 += kernelW; wc2 += kernelW; wc3 += kernelW;
                    }
                    inputChannel += height * width;
                    wc0 += weightRowSkip + weightRowRemainder; wc1 += weightRowSkip + weightRowRemainder;
                    wc2 += weightRowSkip + weightRowRemainder; wc3 += weightRowSkip + weightRowRemainder;
                }
                int outOffset = y * outputWidth + xv;
                VectorStore(out0 + outOffset, a0); VectorStore(out1 + outOffset, a1);
                VectorStore(out2 + outOffset, a2); VectorStore(out3 + outOffset, a3);
            }
            for (int x = xv; x < outputWidth; x++)
                for (int q = 0; q < 4; q++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
        }
    }

    private static unsafe void DenseStride1SingleVector(float* input, float* weights, float* bias,
        float* output, int inputChannels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int co, int xStart, int xEnd)
    {
        int weightsPerOut = inputChannels * kernelH * kernelW;
        int widthLanes = Vector<float>.Count;
        float* w0 = weights + (long)co * weightsPerOut;
        float* out0 = output + (long)co * outputHeight * outputWidth;
        Vector<float> bias0 = new(bias == null ? 0f : bias[co]);
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
            int xv = xStart;
            int kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            int weightRowSkip = kyMin * kernelW, weightRowRemainder = (kernelH - kyMax) * kernelW;
            for (; xv <= xEnd - widthLanes; xv += widthLanes)
            {
                Vector<float> a0 = bias0;
                float* inputChannel = rowBase + xv;
                float* wc0 = w0 + weightRowSkip;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    float* tapRow = inputChannel;
                    for (int ky = 0; ky < kyCount; ky++)
                    {
                        for (int kx = 0; kx < kernelW; kx++)
                            a0 = VectorAddMul(a0, VectorLoad(tapRow + kx), wc0[kx]);
                        tapRow += width;
                        wc0 += kernelW;
                    }
                    inputChannel += height * width;
                    wc0 += weightRowSkip + weightRowRemainder;
                }
                VectorStore(out0 + y * outputWidth + xv, a0);
            }
            for (int x = xv; x < outputWidth; x++)
                DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                    outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co, y, x);
        }
    }
}
