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

internal static partial class Conv3x3
{
    // Scalar twin of Conv3x3FourOutputsVector: 4 output channels, lanes = 1,
    // including the edge tap-skipping path used for the image border.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv3x3Scalar(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int intraOpThreads)
    {
        int plane = checked(height * width);
        int weightsPerOutput = checked(inputChannels * 9);
        if (CanShardOutputs(intraOpThreads, batch, 1, outputChannels,
            (long)outputChannels * weightsPerOutput * plane))
        {
            int workers = ShardWorkers(intraOpThreads, outputChannels / OutputTile);
            fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = weights.Length, biasLength = bias.Length,
                    outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    (int begin, int end) = AlignedOutputShard(worker, workers, outputChannels);
                    if (end <= begin) return;
                    int count = end - begin;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> w = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(begin * weightsPerOutput, count * weightsPerOutput);
                    ReadOnlySpan<float> b = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * plane, count * plane);
                    Conv3x3ScalarKernel(inSpan, w, b, outSpan, 1, inputChannels, height, width, count);
                });
            }
            return;
        }
        Conv3x3ScalarKernel(input, weights, bias, output, batch, inputChannels, height, width,
            outputChannels);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv3x3ScalarKernel(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerOutput = checked(inputChannels * 9);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            int co = 0;
            for (int b = 0; b < batch; b++)
            {
                for (co = 0; co <= outputChannels - OutputTile; co += OutputTile)
                {
                    float* w0 = weightsPtr + co * weightsPerOutput, w1 = w0 + weightsPerOutput;
                    float* w2 = w1 + weightsPerOutput, w3 = w2 + weightsPerOutput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane, o1 = o0 + plane;
                    float* o2 = o1 + plane, o3 = o2 + plane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width;
                        for (int x = 0; x < width; x++)
                        {
                            float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float* src = batchInput + ci * plane;
                                int wb = ci * 9;
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    int iy = y + ky - 1;
                                    if ((uint)iy >= (uint)height) continue;
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        int ix = x + kx - 1;
                                        if ((uint)ix >= (uint)width) continue;
                                        float value = src[iy * width + ix];
                                        int wi = wb + ky * 3 + kx;
                                        s0 += value * w0[wi]; s1 += value * w1[wi];
                                        s2 += value * w2[wi]; s3 += value * w3[wi];
                                    }
                                }
                            }
                            o0[row + x] = s0; o1[row + x] = s1; o2[row + x] = s2; o3[row + x] = s3;
                        }
                    }
                }
                for (; co < outputChannels; co++)
                {
                    float* dst = outputPtr + (b * outputChannels + co) * plane;
                    float* w = weightsPtr + co * weightsPerOutput;
                    float initial = biasPtr == null ? 0f : biasPtr[co];
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                        for (int x = 0; x < width; x++)
                        {
                            float sum = initial;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float* src = batchInput + ci * plane;
                                int wb = ci * 9;
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    int iy = y + ky - 1;
                                    if ((uint)iy >= (uint)height) continue;
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        int ix = x + kx - 1;
                                        if ((uint)ix >= (uint)width) continue;
                                        sum += src[iy * width + ix] * w[wb + ky * 3 + kx];
                                    }
                                }
                            }
                            dst[y * width + x] = sum;
                        }
                }
            }
        }
    }
}
