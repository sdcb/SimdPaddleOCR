using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class ConvDenseStride1
{
    /// <summary>
    /// Any-kernel stride-1, dilation-1, groups-1 convolution.  Interior pixels
    /// (all taps in bounds) run vectorized with the exact scalar accumulation
    /// order; boundary pixels reuse the scalar tap-skipping semantics.
    /// </summary>
    internal static unsafe bool Try(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int outputHeight, int outputWidth, int kernelH, int kernelW, int padTop, int padLeft,
        int intraOpThreads)
    {
        // DenseStride1 Avx512 disabled on Zen 5 (family bisect: OtherConv regresses).
        #if !NETSTANDARD2_0
        if (Avx.IsSupported && (long)kernelH * kernelW * inputChannels <= 1 << 20)
        {
            int xStart = MathCompat.Clamp(padLeft, 0, outputWidth);
            int xEnd = MathCompat.Clamp(width - kernelW + 1 + padLeft, xStart, outputWidth);
            if (xEnd - xStart < 16) return false;
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
                        Try(inSpan, wSpan, bSpan, outSpan, 1, inputChannels, height,
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
                        DenseStride1QuadUnsafe(batchInput, weightsPtr, biasOrNull, batchOutput,
                            inputChannels, height, width, outputHeight, outputWidth, kernelH, kernelW,
                            padTop, padLeft, co, xStart, xEnd);
                    for (; co < outputChannels; co++)
                        DenseStride1SingleUnsafe(batchInput, weightsPtr, biasOrNull, batchOutput,
                            inputChannels, height, width, outputHeight, outputWidth, kernelH, kernelW,
                            padTop, padLeft, co, xStart, xEnd);
                }
            }
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return TryVector(input, weights, bias, output, batch, inputChannels,
                height, width, outputChannels, outputHeight, outputWidth, kernelH, kernelW,
                padTop, padLeft, intraOpThreads);
        }
        return false;
    }

    // Exact replica of the interpreter's scalar fallback for one dense pixel
    // (tap-skipping bounds semantics, (ci,ky,kx) accumulation order).
    private static unsafe void DenseEdgePixel(float* input, float* weights, float* bias,
        float* output, int inputChannels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int co, int y, int x)
    {
        float sum = bias == null ? 0f : bias[co];
        long weightBase = (long)co * inputChannels * kernelH * kernelW;
        for (int ci = 0; ci < inputChannels; ci++)
            for (int ky = 0; ky < kernelH; ky++)
            {
                int iy = y - padTop + ky;
                if ((uint)iy >= (uint)height) continue;
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int ix = x - padLeft + kx;
                    if ((uint)ix >= (uint)width) continue;
                    sum += input[(long)(ci * height + iy) * width + ix] *
                        weights[weightBase + (ci * kernelH + ky) * kernelW + kx];
                }
            }
        output[(long)(co * outputHeight + y) * outputWidth + x] = sum;
    }
}
