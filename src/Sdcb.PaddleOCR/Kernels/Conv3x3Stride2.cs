using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Conv3x3Stride2
{
    internal static unsafe bool Try(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int inputHeight, int inputWidth, int outputHeight, int outputWidth, int outputChannels,
        int intraOpThreads = 1)
    {
        if (Avx.IsSupported)
        {
            int inputPlane = checked(inputHeight * inputWidth);
            int outputPlane = checked(outputHeight * outputWidth);
            int weightsPerOutput = checked(inputChannels * 9);
            if (intraOpThreads > 1 && batch == 1 && outputChannels >= 8 &&
                (long)outputChannels * weightsPerOutput * outputPlane >= IntraOpMinWork)
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
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength).Slice(begin * outputPlane, count * outputPlane);
                        Try(inSpan, w, b, outSpan, 1, inputChannels, inputHeight, inputWidth, outputHeight, outputWidth, count, 1);
                    });
                }
                return true;
            }
            if (Avx512F.IsSupported)
            {
                if ((outputChannels & 7) == 0)
                {
                    Conv3x3Stride2EightOutputsAvx512Unsafe(input, weights, bias, output, batch, inputChannels,
                        inputHeight, inputWidth, outputHeight, outputWidth, outputChannels);
                    return true;
                }
                if ((outputChannels & 3) == 0)
                {
                    Conv3x3Stride2FourOutputsAvx512(input, weights, bias, output, batch, inputChannels,
                        inputHeight, inputWidth, outputHeight, outputWidth, outputChannels);
                    return true;
                }
            }
            if ((outputChannels & 7) == 0)
            {
                Conv3x3Stride2EightOutputsUnsafe(input, weights, bias, output, batch, inputChannels,
                    inputHeight, inputWidth, outputHeight, outputWidth, outputChannels);
                return true;
            }
            if ((outputChannels & 3) == 0)
            {
                Conv3x3Stride2FourOutputs(input, weights, bias, output, batch, inputChannels,
                    inputHeight, inputWidth, outputHeight, outputWidth, outputChannels);
                return true;
            }
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co++)
                {
                    int outputOffset = (b * outputChannels + co) * outputPlane;
                    float initial = bias.IsEmpty ? 0f : bias[co];
                    Vector256<float> initialVector = Vector256.Create(initial);
                    int i = 0;
                    for (; i <= outputPlane - 8; i += 8) Store(output, outputOffset + i, initialVector);
                    for (; i < outputPlane; i++) output[outputOffset + i] = initial;
                    int weightBase = co * weightsPerOutput;
                    int inputBatch = b * inputChannels * inputPlane;
                    for (int ci = 0; ci < inputChannels; ci++)
                    {
                        ReadOnlySpan<float> source = input.Slice(inputBatch + ci * inputPlane, inputPlane);
                        int channelWeights = weightBase + ci * 9;
                        for (int ky = 0; ky < 3; ky++)
                        {
                            for (int kx = 0; kx < 3; kx++)
                            {
                                float weight = weights[channelWeights + ky * 3 + kx];
                                for (int oy = 0; oy < outputHeight; oy++)
                                {
                                    int sourceY = oy * 2 - 1 + ky;
                                    if ((uint)sourceY >= (uint)inputHeight) continue;
                                    int row = oy * outputWidth;
                                    int x = 0;
                                    // The first/last few columns can be outside the padded
                                    // image.  The interior uses AVX2 gathers at stride two.
                                    for (; x < outputWidth && (2 * x - 1 + kx < 0); x++) { }
                                    int vectorEnd = outputWidth;
                                    while (vectorEnd > x && 2 * (vectorEnd - 1) - 1 + kx + 15 >= inputWidth) vectorEnd--;
                                    for (; x <= vectorEnd - 8; x += 8)
                                    {
                                        int sourceX = 2 * x - 1 + kx;
                                        Vector256<float> value = LoadStride2(source, sourceY * inputWidth + sourceX);
                                        Vector256<float> current = Load(output, outputOffset + row + x);
                                        Store(output, outputOffset + row + x, AddMul(current, value, weight));
                                    }
                                    for (; x < vectorEnd; x++)
                                    {
                                        int sourceX = 2 * x - 1 + kx;
                                        output[outputOffset + row + x] += source[sourceY * inputWidth + sourceX] * weight;
                                    }
                                    for (; x < outputWidth; x++)
                                    {
                                        int sourceX = 2 * x - 1 + kx;
                                        if ((uint)sourceX < (uint)inputWidth)
                                            output[outputOffset + row + x] += source[sourceY * inputWidth + sourceX] * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            return true;
        }
        else if (Vector.IsHardwareAccelerated)
        {
            return TryVector(input, weights, bias, output, batch, inputChannels,
                inputHeight, inputWidth, outputHeight, outputWidth, outputChannels, intraOpThreads);
        }
        Conv3x3Stride2Scalar(input, weights, bias, output, batch, inputChannels,
            inputHeight, inputWidth, outputHeight, outputWidth, outputChannels, intraOpThreads);
        return true;
    }

    internal static unsafe bool TryPacked(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth,
        int outputChannels, int intraOpThreads = 1)
    {
        if (!Avx.IsSupported || outputChannels < 8 || (outputChannels & 7) != 0)
            return false;
        int outputPlane = checked(outputHeight * outputWidth);
        int blocks = outputChannels / 8;
        const int weightsPerInput = 9 * 8;
        if (intraOpThreads > 1 && batch == 1 && blocks >= 2)
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
                        .Slice(begin * inputChannels * weightsPerInput, (end - begin) * inputChannels * weightsPerInput);
                    ReadOnlySpan<float> bSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin * 8, count);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * 8 * outputPlane, count * outputPlane);
                    TryPacked(inSpan, wSpan, bSpan, outSpan, 1, inputChannels,
                        inputHeight, inputWidth, outputHeight, outputWidth, count, 1);
                });
            }
            return true;
        }
        // 8-OC Avx512; skip 16-OC×ZMM (Zen 5 spills). Workers already intraOp==1.
        if (Avx512F.IsSupported)
        {
            Conv3x3Stride2EightOutputsPackedAvx512Unsafe(input, packedWeights, bias, output, batch,
                inputChannels, inputHeight, inputWidth, outputHeight, outputWidth, outputChannels);
            return true;
        }
        if ((outputChannels & 15) == 0)
            Conv3x3Stride2SixteenOutputsPackedUnsafe(input, packedWeights, bias, output,
                batch, inputChannels, inputHeight, inputWidth, outputHeight, outputWidth,
                outputChannels);
        else
            Conv3x3Stride2EightOutputsPackedUnsafe(input, packedWeights, bias, output, batch,
                inputChannels, inputHeight, inputWidth, outputHeight, outputWidth, outputChannels);
        return true;
    }
}
