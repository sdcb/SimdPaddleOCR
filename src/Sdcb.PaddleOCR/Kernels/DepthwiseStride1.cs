using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class DepthwiseStride1
{
    /// <summary>Any-kernel stride-1, dilation-1 depthwise convolution.</summary>
    internal static unsafe bool Try(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int intraOpThreads)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            int xStart = MathCompat.Clamp(padLeft, 0, outputWidth);
            int xEnd = MathCompat.Clamp(width - kernelW + 1 + padLeft, xStart, outputWidth);
            if (xEnd - xStart < 16) return false;
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
                        Try(inSpan, wSpan, bSpan, outSpan, 1, end - begin, height,
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
                        DepthwiseStride1ChannelAvx512Unsafe(
                            inputPtr + ((long)b * channels + c) * height * width,
                            weightsPtr + (long)c * kernelH * kernelW,
                            biasOrNull == null ? 0f : biasOrNull[c],
                            outputPtr + ((long)b * channels + c) * outputHeight * outputWidth,
                            height, width, outputHeight, outputWidth, kernelH, kernelW,
                            padTop, padLeft, xStart, xEnd);
            }
            return true;
        }
        else if (Avx.IsSupported)
        {
            int xStart = MathCompat.Clamp(padLeft, 0, outputWidth);
            int xEnd = MathCompat.Clamp(width - kernelW + 1 + padLeft, xStart, outputWidth);
            if (xEnd - xStart < 16) return false;
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
                        Try(inSpan, wSpan, bSpan, outSpan, 1, end - begin, height,
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
                        DepthwiseStride1ChannelUnsafe(
                            inputPtr + ((long)b * channels + c) * height * width,
                            weightsPtr + (long)c * kernelH * kernelW,
                            biasOrNull == null ? 0f : biasOrNull[c],
                            outputPtr + ((long)b * channels + c) * outputHeight * outputWidth,
                            height, width, outputHeight, outputWidth, kernelH, kernelW,
                            padTop, padLeft, xStart, xEnd);
            }
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return TryVector(input, weights, bias, output, batch, channels, height,
                width, outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, intraOpThreads);
        }
        return false;
    }

    private static unsafe void DepthwiseEdgePixel(float* input, float* weights, float bias,
        float* output, int height, int width, int outputWidth, int kernelH, int kernelW,
        int padTop, int padLeft, int y, int x)
    {
        float sum = bias;
        for (int ky = 0; ky < kernelH; ky++)
        {
            int iy = y - padTop + ky;
            if ((uint)iy >= (uint)height) continue;
            for (int kx = 0; kx < kernelW; kx++)
            {
                int ix = x - padLeft + kx;
                if ((uint)ix >= (uint)width) continue;
                sum += input[(long)iy * width + ix] * weights[ky * kernelW + kx];
            }
        }
        output[(long)y * outputWidth + x] = sum;
    }
}
