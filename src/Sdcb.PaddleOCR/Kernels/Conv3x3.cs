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
    internal static unsafe bool Try(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int intraOpThreads = 1)
    {
        #if !NETSTANDARD2_0
        if (Avx.IsSupported)
        {
            int plane = checked(height * width);
            int weightsPerOutput = checked(inputChannels * 9);
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
                        ReadOnlySpan<float> w = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength).Slice(begin * weightsPerOutput, count * weightsPerOutput);
                        ReadOnlySpan<float> b = biasLength == 0 ? [] : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength).Slice(begin * plane, count * plane);
                        Try(inSpan, w, b, outSpan, 1, inputChannels, height, width, count, 1);
                    });
                }
                return true;
            }
            if (Avx512F.IsSupported && (outputChannels & 3) == 0)
            {
                if (intraOpThreads == 1 && (outputChannels & 7) == 0)
                {
                    TryEightOutputsAvx512Unsafe(input, weights, bias, output, batch, inputChannels,
                        height, width, outputChannels);
                    return true;
                }
                TryFourOutputsAvx512Unsafe(input, weights, bias, output, batch, inputChannels, height,
                    width, outputChannels);
                return true;
            }
            if ((outputChannels & 3) == 0)
            {
                if (intraOpThreads == 1 && (outputChannels & 15) == 0)
                {
                    TrySixteenOutputsUnsafe(input, weights, bias, output, batch, inputChannels,
                        height, width, outputChannels);
                    return true;
                }
                if (intraOpThreads == 1 && (outputChannels & 7) == 0)
                {
                    TryEightOutputsUnsafe(input, weights, bias, output, batch, inputChannels,
                        height, width, outputChannels);
                    return true;
                }
                TryFourOutputsUnsafe(input, weights, bias, output, batch, inputChannels, height,
                    width, outputChannels);
                return true;
            }
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co++)
                {
                    int outputOffset = (b * outputChannels + co) * plane;
                    float initial = bias.IsEmpty ? 0f : bias[co];
                    Vector256<float> initialVector = Vector256.Create(initial);
                    int i = 0;
                    for (; i <= plane - 8; i += 8) Store(output, outputOffset + i, initialVector);
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
                                    for (; x <= xEnd - 8; x += 8)
                                    {
                                        Vector256<float> current = Load(output, outputOffset + row + x);
                                        Vector256<float> value = Load(source, sourceRow + x + kx - 1);
                                        Store(output, outputOffset + row + x, AddMul(current, value, weight));
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
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return TryVector(input, weights, bias, output, batch, inputChannels,
                height, width, outputChannels, intraOpThreads);
        }
        Conv3x3Scalar(input, weights, bias, output, batch, inputChannels, height, width,
            outputChannels, intraOpThreads);
        return true;
    }
}
