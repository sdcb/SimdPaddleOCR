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
    internal static unsafe bool TryVector(ReadOnlySpan<float> input,
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
                    ConvTranspose2x2Stride2RangeVector(inSpan, weightSpan, biasSpan, outSpan,
                        batch, inputChannels, inputHeight, inputWidth, outputChannels, begin, end);
                });
            }
            return true;
        }
        ConvTranspose2x2Stride2RangeVector(input, weights, bias, output, batch, inputChannels,
            inputHeight, inputWidth, outputChannels, 0, outputChannels);
        return true;
    }

    private static unsafe void ConvTranspose2x2Stride2RangeVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputChannels,
        int channelBegin, int channelEnd)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputHeight = checked(inputHeight * 2);
        int outputWidth = checked(inputWidth * 2), outputPlane = checked(outputHeight * outputWidth);
        int widthLanes = Vector<float>.Count;
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = channelBegin; co < channelEnd; co++)
                {
                    float* dst = outputPtr + (b * outputChannels + co) * outputPlane;
                    float initial = biasPtr == null ? 0f : biasPtr[co];
                    Vector<float> vb = new(initial);
                    int i = 0;
                    for (; i <= outputPlane - widthLanes; i += widthLanes)
                        VectorStore(dst + i, vb);
                    for (; i < outputPlane; i++) dst[i] = initial;
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
                            int ix = 0;
                            for (; ix <= inputWidth - widthLanes; ix += widthLanes)
                            {
                                Vector<float> values = VectorLoad(src + inputRow + ix);
                                ExpandTranspose2x(values, out Vector<float> evenLow, out Vector<float> evenHigh,
                                    out Vector<float> oddLow, out Vector<float> oddHigh);
                                int ox = ix * 2;
                                float* row0 = dst + outputRow0 + ox;
                                float* row1 = dst + outputRow1 + ox;
                                Vector<float> w0 = new(w[0]), w1 = new(w[1]), w2 = new(w[2]), w3 = new(w[3]);
                                AddStore(row0, evenLow * w0 + oddLow * w1);
                                AddStore(row0 + widthLanes, evenHigh * w0 + oddHigh * w1);
                                AddStore(row1, evenLow * w2 + oddLow * w3);
                                AddStore(row1 + widthLanes, evenHigh * w2 + oddHigh * w3);
                            }
                            for (; ix < inputWidth; ix++)
                            {
                                float value = src[inputRow + ix]; int ox = ix * 2;
                                dst[outputRow0 + ox] += value * w[0]; dst[outputRow0 + ox + 1] += value * w[1];
                                dst[outputRow1 + ox] += value * w[2]; dst[outputRow1 + ox + 1] += value * w[3];
                            }
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void AddStore(float* destination, Vector<float> value) =>
        VectorStore(destination, VectorLoad(destination) + value);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void ExpandTranspose2x(Vector<float> values,
        out Vector<float> evenLow, out Vector<float> evenHigh,
        out Vector<float> oddLow, out Vector<float> oddHigh)
    {
        evenLow = default;
        evenHigh = default;
        oddLow = default;
        oddHigh = default;
        ref float source = ref Unsafe.As<Vector<float>, float>(ref values);
        ref float evenLowRef = ref Unsafe.As<Vector<float>, float>(ref evenLow);
        ref float evenHighRef = ref Unsafe.As<Vector<float>, float>(ref evenHigh);
        ref float oddLowRef = ref Unsafe.As<Vector<float>, float>(ref oddLow);
        ref float oddHighRef = ref Unsafe.As<Vector<float>, float>(ref oddHigh);
        int width = Vector<float>.Count;
        if (width == 8)
        {
            evenLowRef = source;
            Unsafe.Add(ref evenLowRef, 2) = Unsafe.Add(ref source, 1);
            Unsafe.Add(ref evenLowRef, 4) = Unsafe.Add(ref source, 2);
            Unsafe.Add(ref evenLowRef, 6) = Unsafe.Add(ref source, 3);
            Unsafe.Add(ref oddLowRef, 1) = source;
            Unsafe.Add(ref oddLowRef, 3) = Unsafe.Add(ref source, 1);
            Unsafe.Add(ref oddLowRef, 5) = Unsafe.Add(ref source, 2);
            Unsafe.Add(ref oddLowRef, 7) = Unsafe.Add(ref source, 3);
            evenHighRef = Unsafe.Add(ref source, 4);
            Unsafe.Add(ref evenHighRef, 2) = Unsafe.Add(ref source, 5);
            Unsafe.Add(ref evenHighRef, 4) = Unsafe.Add(ref source, 6);
            Unsafe.Add(ref evenHighRef, 6) = Unsafe.Add(ref source, 7);
            Unsafe.Add(ref oddHighRef, 1) = Unsafe.Add(ref source, 4);
            Unsafe.Add(ref oddHighRef, 3) = Unsafe.Add(ref source, 5);
            Unsafe.Add(ref oddHighRef, 5) = Unsafe.Add(ref source, 6);
            Unsafe.Add(ref oddHighRef, 7) = Unsafe.Add(ref source, 7);
            return;
        }
        if (width == 4)
        {
            evenLowRef = source;
            Unsafe.Add(ref evenLowRef, 2) = Unsafe.Add(ref source, 1);
            Unsafe.Add(ref oddLowRef, 1) = source;
            Unsafe.Add(ref oddLowRef, 3) = Unsafe.Add(ref source, 1);
            evenHighRef = Unsafe.Add(ref source, 2);
            Unsafe.Add(ref evenHighRef, 2) = Unsafe.Add(ref source, 3);
            Unsafe.Add(ref oddHighRef, 1) = Unsafe.Add(ref source, 2);
            Unsafe.Add(ref oddHighRef, 3) = Unsafe.Add(ref source, 3);
            return;
        }
        for (int lane = 0; lane < width; lane++)
        {
            float value = Unsafe.Add(ref source, lane);
            int expanded = lane * 2;
            if (expanded < width)
            {
                Unsafe.Add(ref evenLowRef, expanded) = value;
                Unsafe.Add(ref oddLowRef, expanded + 1) = value;
            }
            else
            {
                Unsafe.Add(ref evenHighRef, expanded - width) = value;
                Unsafe.Add(ref oddHighRef, expanded - width + 1) = value;
            }
        }
    }
}
