using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Conv3x3
{
    internal static unsafe bool TryVector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int intraOpThreads)
    {
        int plane = checked(height * width);
        int weightsPerOutput = checked(inputChannels * 9);
        int widthLanes = Vector<float>.Count;
        if (intraOpThreads > 1 && batch == 1 && outputChannels >= 8 &&
            (long)outputChannels * weightsPerOutput * plane >= IntraOpMinWork)
        {
            int workers = Math.Min(intraOpThreads, outputChannels / 4);
            fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = weights.Length, biasLength = bias.Length,
                    outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    int begin = (outputChannels * worker / workers) & ~3;
                    int end = worker == workers - 1 ? outputChannels : (outputChannels * (worker + 1) / workers) & ~3;
                    if (end <= begin) return;
                    int count = end - begin;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> w = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(begin * weightsPerOutput, count * weightsPerOutput);
                    ReadOnlySpan<float> b = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * plane, count * plane);
                    TryVector(inSpan, w, b, outSpan, 1, inputChannels, height, width, count, 1);
                });
            }
            return true;
        }

        if ((outputChannels & 3) == 0)
        {
            Conv3x3FourOutputsVector(input, weights, bias, output, batch, inputChannels,
                height, width, outputChannels);
            return true;
        }

        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co++)
            {
                int outputOffset = (b * outputChannels + co) * plane;
                float initial = bias.IsEmpty ? 0f : bias[co];
                Vector<float> initialVector = new(initial);
                int i = 0;
                for (; i <= plane - widthLanes; i += widthLanes) VectorStore(output, outputOffset + i, initialVector);
                for (; i < plane; i++) output[outputOffset + i] = initial;
                int weightBase = co * weightsPerOutput;
                int inputBatch = b * inputChannels * plane;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    ReadOnlySpan<float> source = input.Slice(inputBatch + ci * plane, plane);
                    int channelWeights = weightBase + ci * 9;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int yBegin = ky == 0 ? 1 : 0;
                        int yEnd = ky == 2 ? height - 1 : height;
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int xBegin = kx == 0 ? 1 : 0;
                            int xEnd = kx == 2 ? width - 1 : width;
                            float weight = weights[channelWeights + ky * 3 + kx];
                            for (int y = yBegin; y < yEnd; y++)
                            {
                                int row = y * width;
                                int sourceRow = (y + ky - 1) * width;
                                int x = xBegin;
                                for (; x <= xEnd - widthLanes; x += widthLanes)
                                {
                                    Vector<float> current = VectorLoad(output, outputOffset + row + x);
                                    Vector<float> value = VectorLoad(source, sourceRow + x + kx - 1);
                                    VectorStore(output, outputOffset + row + x, VectorAddMul(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                    output[outputOffset + row + x] += source[sourceRow + x + kx - 1] * weight;
                            }
                        }
                    }
                }
            }
        return true;
    }

    private static unsafe void Conv3x3FourOutputsVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerOutput = checked(inputChannels * 9);
        int widthLanes = Vector<float>.Count;
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 4)
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
                        int row = y * width, x = 0;
                        while (x < width)
                        {
                            bool vector = y > 0 && y + 1 < height && x > 0 && x + widthLanes < width;
                            if (vector)
                            {
                                Vector<float> a0 = new(b0), a1 = new(b1), a2 = new(b2), a3 = new(b3);
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        float* srcRow = src + (y + ky - 1) * width;
                                        int ix = x - 1;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            Vector<float> value = VectorLoad(srcRow + ix + kx);
                                            int wi = wb + ky * 3 + kx;
                                            a0 = VectorAddMul(a0, value, w0[wi]);
                                            a1 = VectorAddMul(a1, value, w1[wi]);
                                            a2 = VectorAddMul(a2, value, w2[wi]);
                                            a3 = VectorAddMul(a3, value, w3[wi]);
                                        }
                                    }
                                }
                                VectorStore(o0 + row + x, a0); VectorStore(o1 + row + x, a1);
                                VectorStore(o2 + row + x, a2); VectorStore(o3 + row + x, a3);
                                x += widthLanes;
                            }
                            else
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
                                x++;
                            }
                        }
                    }
                }
        }
    }
}
