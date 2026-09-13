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

internal static partial class Stride2
{
    internal static unsafe bool Try(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int intraOpThreads = 1)
    {
        int plane = checked(height * width);
        // 2x2 stride-1 convolutions are used by the detector's feature
        // pyramid.  Their AVX2 kernels used to ignore the session's
        // intra-op budget and ran the whole output channel range on one
        // thread.  Shard complete output tiles; each shard has a disjoint
        // output slice and therefore preserves the scalar accumulation order.
        int tile = (outputChannels & 7) == 0 ? 8 : (outputChannels & 3) == 0 ? 4 : 1;
        long work = checked((long)outputChannels * inputChannels * plane * 4);
        if (tile > 1 && intraOpThreads > 1 && batch == 1 && work >= 4_000_000)
        {
            int blocks = outputChannels / tile;
            int workers = Math.Min(intraOpThreads, blocks);
            fixed (float* inputPtr = input, weightsPtr = weights,
                biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = weights.Length,
                    biasLength = bias.Length, outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    int beginBlock = blocks * worker / workers;
                    int endBlock = blocks * (worker + 1) / workers;
                    int begin = beginBlock * tile, count = (endBlock - beginBlock) * tile;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> wSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(begin * inputChannels * 4, count * inputChannels * 4);
                    ReadOnlySpan<float> bSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * plane, count * plane);
                    Try(inSpan, wSpan, bSpan, outSpan, 1, inputChannels, height, width, count, 1);
                });
            }
            return true;
        }
#if !NETSTANDARD2_0
        if (Avx.IsSupported)
        {
            if ((outputChannels & 7) == 0)
            {
                Conv2x2PadEndEightOutputsUnsafe(input, weights, bias, output, batch, inputChannels,
                    height, width, outputChannels);
                return true;
            }
            if ((outputChannels & 3) == 0)
            {
                Conv2x2PadEndFourOutputsUnsafe(input, weights, bias, output, batch, inputChannels,
                    height, width, outputChannels);
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
                    int weightBase = co * inputChannels * 4;
                    int inputBatch = b * inputChannels * plane;
                    for (int ci = 0; ci < inputChannels; ci++)
                    {
                        ReadOnlySpan<float> source = input.Slice(inputBatch + ci * plane, plane);
                        int channelWeights = weightBase + ci * 4;
                        for (int ky = 0; ky < 2; ky++)
                            for (int kx = 0; kx < 2; kx++)
                            {
                                int yEnd = height - ky, xEnd = width - kx;
                                float weight = weights[channelWeights + ky * 2 + kx];
                                for (int y = 0; y < yEnd; y++)
                                {
                                    int row = y * width, sourceRow = (y + ky) * width;
                                    int x = 0;
                                    for (; x <= xEnd - 8; x += 8)
                                    {
                                        Vector256<float> current = Load(output, outputOffset + row + x);
                                        Vector256<float> value = Load(source, sourceRow + x + kx);
                                        Store(output, outputOffset + row + x, AddMul(current, value, weight));
                                    }
                                    for (; x < xEnd; x++)
                                        output[outputOffset + row + x] += source[sourceRow + x + kx] * weight;
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
                height, width, outputChannels);
        }
        Conv2x2PadEndScalar(input, weights, bias, output, batch, inputChannels, height, width, outputChannels);
        return true;
    }
}
