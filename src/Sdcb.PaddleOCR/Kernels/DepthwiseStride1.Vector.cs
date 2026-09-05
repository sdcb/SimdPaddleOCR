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
    internal static unsafe bool TryVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int intraOpThreads)
    {
        int xStart = Math.Clamp(padLeft, 0, outputWidth);
        int xEnd = Math.Clamp(width - kernelW + 1 + padLeft, xStart, outputWidth);
        int widthLanes = Vector<float>.Count;
        if (xEnd - xStart < widthLanes) return false;
        if (intraOpThreads > 1 && batch == 1 && channels >= 2 &&
            (long)channels * outputHeight * outputWidth * kernelH * kernelW >= 4_000_000)
        {
            int workers = Math.Min(intraOpThreads, channels);
            fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = weights.Length,
                    biasLength = bias.Length, outputLength = output.Length;
                int taps = kernelH * kernelW, inPlane = height * width, outPlane = outputHeight * outputWidth;
                Parallel.For(0, workers, worker =>
                {
                    int begin = channels * worker / workers, end = channels * (worker + 1) / workers;
                    if (end <= begin) return;
                    ReadOnlySpan<float> inSpan = new ReadOnlySpan<float>((void*)inputAddress, inputLength)
                        .Slice(begin * inPlane, (end - begin) * inPlane);
                    ReadOnlySpan<float> wSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(begin * taps, (end - begin) * taps);
                    ReadOnlySpan<float> bSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, end - begin);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * outPlane, (end - begin) * outPlane);
                    TryVector(inSpan, wSpan, bSpan, outSpan, 1, end - begin, height,
                        width, outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, 1);
                });
            }
            return true;
        }

        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            float* biasOrNull = bias.Length == 0 ? null : biasPtr;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                    DepthwiseStride1ChannelVector(
                        inputPtr + ((long)b * channels + c) * height * width,
                        weightsPtr + (long)c * kernelH * kernelW,
                        biasOrNull == null ? 0f : biasOrNull[c],
                        outputPtr + ((long)b * channels + c) * outputHeight * outputWidth,
                        height, width, outputHeight, outputWidth, kernelH, kernelW,
                        padTop, padLeft, xStart, xEnd);
        }
        return true;
    }

    private static unsafe void DepthwiseStride1ChannelVector(float* input, float* weights,
        float bias, float* output, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int xStart, int xEnd)
    {
        Vector<float> vBias = new(bias);
        int widthLanes = Vector<float>.Count;
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
            int xv = xStart;
            for (; xv <= xEnd - widthLanes; xv += widthLanes)
            {
                Vector<float> a0 = vBias;
                float* tapRow = rowBase + xv;
                float* wc = weightBase;
                for (int ky = 0; ky < kyCount; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                        a0 = VectorAddMul(a0, VectorLoad(tapRow + kx), wc[kx]);
                    tapRow += width;
                    wc += kernelW;
                }
                VectorStore(output + y * outputWidth + xv, a0);
            }
            for (int x = xv; x < outputWidth; x++)
                DepthwiseEdgePixel(input, weights, bias, output, height, width,
                    outputWidth, kernelH, kernelW, padTop, padLeft, y, x);
        }
    }
}
