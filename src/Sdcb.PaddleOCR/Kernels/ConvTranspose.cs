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

internal static partial class ConvTranspose
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool Try(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputChannels,
        int intraOpThreads = 1)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            // Output channels are independent for this 2x2/stride-2 transform.
            // Shard large detector transforms in the same way as the C runtime;
            // each worker owns complete output planes, so accumulation order and
            // results remain deterministic.
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
                        ConvTranspose2x2Stride2RangeAvx512Unsafe(inSpan, weightSpan, biasSpan, outSpan,
                            batch, inputChannels, inputHeight, inputWidth, outputChannels, begin, end);
                    });
                }
                return true;
            }
            if (intraOpThreads == 1 && (outputChannels & 7) == 0)
            {
                ConvTranspose2x2Stride2EightOutputsAvx512Unsafe(input, weights, bias, output, batch,
                    inputChannels, inputHeight, inputWidth, outputChannels);
                return true;
            }
            ConvTranspose2x2Stride2Avx512Unsafe(input, weights, bias, output, batch, inputChannels,
                inputHeight, inputWidth, outputChannels);
            return true;
        }
        else if (Avx2.IsSupported)
        {
            // Output channels are independent for this 2x2/stride-2 transform.
            // Shard large detector transforms in the same way as the C runtime;
            // each worker owns complete output planes, so accumulation order and
            // results remain deterministic.
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
                        ConvTranspose2x2Stride2RangeUnsafe(inSpan, weightSpan, biasSpan, outSpan,
                            batch, inputChannels, inputHeight, inputWidth, outputChannels, begin, end);
                    });
                }
                return true;
            }
            if (intraOpThreads == 1 && (outputChannels & 7) == 0)
            {
                ConvTranspose2x2Stride2EightOutputsUnsafe(input, weights, bias, output, batch,
                    inputChannels, inputHeight, inputWidth, outputChannels);
                return true;
            }
            ConvTranspose2x2Stride2Unsafe(input, weights, bias, output, batch, inputChannels,
                inputHeight, inputWidth, outputChannels);
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return TryVector(input, weights, bias, output, batch,
                inputChannels, inputHeight, inputWidth, outputChannels, intraOpThreads);
        }
        ConvTranspose2x2Stride2Scalar(input, weights, bias, output, batch, inputChannels,
            inputHeight, inputWidth, outputChannels, intraOpThreads);
        return true;
    }
}
