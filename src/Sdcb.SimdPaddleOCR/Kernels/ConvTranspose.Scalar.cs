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

internal static partial class ConvTranspose
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ConvTranspose2x2Stride2Scalar(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputChannels, int intraOpThreads)
    {
        long work = (long)outputChannels * inputChannels * inputHeight * inputWidth * 4;
        if (CanShardChannels(intraOpThreads, batch, outputChannels, work))
        {
            int workers = Math.Min(intraOpThreads, outputChannels);
            fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = weights.Length,
                    biasLength = bias.Length, outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    int begin = outputChannels * worker / workers;
                    int end = outputChannels * (worker + 1) / workers;
                    if (end <= begin) return;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> weightSpan = new((void*)weightsAddress, weightsLength);
                    ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength);
                    Span<float> outSpan = new((void*)outputAddress, outputLength);
                    ConvTranspose2x2Stride2RangeScalar(inSpan, weightSpan, biasSpan, outSpan,
                        batch, inputChannels, inputHeight, inputWidth, outputChannels, begin, end);
                });
            }
            return;
        }
        ConvTranspose2x2Stride2RangeScalar(input, weights, bias, output, batch, inputChannels,
            inputHeight, inputWidth, outputChannels, 0, outputChannels);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ConvTranspose2x2Stride2RangeScalar(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputChannels,
        int channelBegin, int channelEnd)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputHeight = checked(inputHeight * 2);
        int outputWidth = checked(inputWidth * 2), outputPlane = checked(outputHeight * outputWidth);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = channelBegin; co < channelEnd; co++)
                {
                    float* dst = outputPtr + (b * outputChannels + co) * outputPlane;
                    float initial = biasPtr == null ? 0f : biasPtr[co];
                    for (int i = 0; i < outputPlane; i++) dst[i] = initial;
                    int inputBatch = b * inputChannels * inputPlane;
                    for (int ci = 0; ci < inputChannels; ci++)
                    {
                        float* src = inputPtr + inputBatch + ci * inputPlane;
                        float* w = weightsPtr + (ci * outputChannels + co) * 4;
                        for (int iy = 0; iy < inputHeight; iy++)
                        {
                            int inputRow = iy * inputWidth;
                            int outputRow0 = (iy * 2) * outputWidth;
                            int outputRow1 = outputRow0 + outputWidth;
                            for (int ix = 0; ix < inputWidth; ix++)
                            {
                                float value = src[inputRow + ix];
                                int px = ix * 2;
                                dst[outputRow0 + px] += value * w[0];
                                dst[outputRow0 + px + 1] += value * w[1];
                                dst[outputRow1 + px] += value * w[2];
                                dst[outputRow1 + px + 1] += value * w[3];
                            }
                        }
                    }
                }
        }
    }
}
