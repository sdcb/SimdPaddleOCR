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

internal static partial class Conv3x3Packed
{
    internal static unsafe bool Try(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int intraOpThreads = 1)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported && outputChannels >= 8 && (outputChannels & 7) == 0 &&
            !(intraOpThreads == 1 && (outputChannels & 15) == 0 && outputChannels >= 64))
        {
            int plane = checked(height * width), blocks = outputChannels / 8;
            if (intraOpThreads > 1 && batch == 1 && blocks >= 2 &&
                (long)outputChannels * inputChannels * plane * 9 >= IntraOpMinWork)
            {
                int workers = Math.Min(intraOpThreads, blocks);
                fixed (float* inputPtr = input, weightsPtr = packedWeights,
                    biasPtr = bias, outputPtr = output)
                {
                    nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                        biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                    int inputLength = input.Length, weightsLength = packedWeights.Length,
                        biasLength = bias.Length, outputLength = output.Length;
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = blocks * worker / workers, end = blocks * (worker + 1) / workers;
                        if (end <= begin) return;
                        int count = (end - begin) * 8;
                        ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                        ReadOnlySpan<float> wSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                            .Slice(begin * inputChannels * 9 * 8, (end - begin) * inputChannels * 9 * 8);
                        ReadOnlySpan<float> bSpan = biasLength == 0 ? []
                            : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin * 8, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                            .Slice(begin * 8 * plane, count * plane);
                        Try(inSpan, wSpan, bSpan, outSpan, 1, inputChannels,
                            height, width, count, 1);
                    });
                }
                return true;
            }
            Conv3x3EightOutputsPackedAvx512Unsafe(input, packedWeights, bias, output, batch,
                inputChannels, height, width, outputChannels);
            return true;
        }
        else if (Avx.IsSupported && outputChannels >= 8 && (outputChannels & 7) == 0)
        {
            int plane = checked(height * width), blocks = outputChannels / 8;
            if (intraOpThreads > 1 && batch == 1 && blocks >= 2 &&
                (long)outputChannels * inputChannels * plane * 9 >= IntraOpMinWork)
            {
                int workers = Math.Min(intraOpThreads, blocks);
                fixed (float* inputPtr = input, weightsPtr = packedWeights,
                    biasPtr = bias, outputPtr = output)
                {
                    nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                        biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                    int inputLength = input.Length, weightsLength = packedWeights.Length,
                        biasLength = bias.Length, outputLength = output.Length;
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = blocks * worker / workers, end = blocks * (worker + 1) / workers;
                        if (end <= begin) return;
                        int count = (end - begin) * 8;
                        ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                        ReadOnlySpan<float> wSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                            .Slice(begin * inputChannels * 9 * 8, (end - begin) * inputChannels * 9 * 8);
                        ReadOnlySpan<float> bSpan = biasLength == 0 ? []
                            : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin * 8, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                            .Slice(begin * 8 * plane, count * plane);
                        Try(inSpan, wSpan, bSpan, outSpan, 1, inputChannels,
                            height, width, count, 1);
                    });
                }
                return true;
            }
            // Use the sixteen-output kernel only for wide single-threaded
            // projections; it reloads each input patch once for two adjacent
            // eight-channel tiles. Narrow projections keep the lower-pressure
            // eight-output path.
            if (intraOpThreads == 1 && (outputChannels & 15) == 0 && outputChannels >= 64)
                Conv3x3SixteenOutputsPackedUnsafe(input, packedWeights, bias, output, batch,
                    inputChannels, height, width, outputChannels);
            else
                Conv3x3EightOutputsPackedUnsafe(input, packedWeights, bias, output, batch,
                    inputChannels, height, width, outputChannels);
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated && outputChannels >= 8 && (outputChannels & 7) == 0)
        {
            return TryVector(input, packedWeights, bias, output, batch, inputChannels,
                height, width, outputChannels, intraOpThreads);
        }
        return false;
    }
}
