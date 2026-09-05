using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Conv1x1
{
    internal static unsafe bool Try(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int groups, int intraOpThreads = 1)
    {
        // Zen 5: prefer 4-OC / 8-OC × Vector512 spatial-16. Avoid 16-OC ZMM
        // tiles (register spills). Prefer FourOutputs; use Eight when aligned.
        if (Avx512F.IsSupported)
        {
            int inputPerGroup = inputChannels / groups;
            int outputPerGroup = outputChannels / groups;
            int plane = checked(height * width);
            if (CanShardOutputs(intraOpThreads, batch, groups, outputChannels,
                (long)outputChannels * inputPerGroup * plane))
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
                        ReadOnlySpan<float> w = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength).Slice(begin * inputPerGroup, count * inputPerGroup);
                        ReadOnlySpan<float> b = biasLength == 0 ? [] : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength).Slice(begin * plane, count * plane);
                        Try(inSpan, w, b, outSpan, 1, inputChannels, height, width, count, 1);
                    });
                }
                return true;
            }
            if (outputPerGroup >= 4 && (outputPerGroup & 3) == 0)
            {
                // Prefer 8-OC spatial-16; else 4-OC dual-spatial. Avoid 16-OC ZMM.
                if ((outputPerGroup & 7) == 0)
                {
                    Conv1x1EightOutputsAvx512(input, weights, bias, output, batch, inputChannels, height, width,
                        outputChannels, groups, inputPerGroup, outputPerGroup, plane);
                    return true;
                }
                Conv1x1FourOutputsAvx512(input, weights, bias, output, batch, inputChannels, height, width,
                    outputChannels, groups, inputPerGroup, outputPerGroup, plane);
                return true;
            }
            for (int b = 0; b < batch; b++)
            {
                int inputBatch = b * inputChannels * plane;
                int outputBatch = b * outputChannels * plane;
                for (int g = 0; g < groups; g++)
                {
                    int inputGroup = inputBatch + g * inputPerGroup * plane;
                    int outputGroup = outputBatch + g * outputPerGroup * plane;
                    for (int co = 0; co < outputPerGroup; co++)
                    {
                        int globalCo = g * outputPerGroup + co;
                        int outputOffset = outputGroup + co * plane;
                        float initial = bias.IsEmpty ? 0f : bias[globalCo];
                        Vector512<float> initialVector = Vector512.Create(initial);
                        int spatial = 0;
                        for (; spatial <= plane - 16; spatial += 16)
                            Store512(output, outputOffset + spatial, initialVector);
                        for (; spatial < plane; spatial++) output[outputOffset + spatial] = initial;
                        int weightBase = globalCo * inputPerGroup;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            ReadOnlySpan<float> inputChannel = input.Slice(inputGroup + ci * plane, plane);
                            float weight = weights[weightBase + ci];
                            spatial = 0;
                            for (; spatial <= plane - 16; spatial += 16)
                            {
                                Vector512<float> value = Load512(inputChannel, spatial);
                                Vector512<float> current = Load512(output, outputOffset + spatial);
                                Store512(output, outputOffset + spatial, AddMul512(current, value, weight));
                            }
                            for (; spatial < plane; spatial++)
                                output[outputOffset + spatial] += inputChannel[spatial] * weight;
                        }
                    }
                }
            }
            return true;
        }
        else if (Avx.IsSupported)
        {
            int inputPerGroup = inputChannels / groups;
            int outputPerGroup = outputChannels / groups;
            int plane = checked(height * width);
            if (CanShardOutputs(intraOpThreads, batch, groups, outputChannels,
                (long)outputChannels * inputPerGroup * plane))
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
                        ReadOnlySpan<float> w = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength).Slice(begin * inputPerGroup, count * inputPerGroup);
                        ReadOnlySpan<float> b = biasLength == 0 ? [] : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength).Slice(begin * plane, count * plane);
                        Try(inSpan, w, b, outSpan, 1, inputChannels, height, width, count, 1);
                    });
                }
                return true;
            }
            if ((outputPerGroup & 15) == 0 && outputPerGroup >= 16)
            {
                Conv1x1SixteenOutputs(input, weights, bias, output, batch, inputChannels, height, width,
                    outputChannels, groups, inputPerGroup, outputPerGroup, plane);
                return true;
            }
            if ((outputPerGroup & 7) == 0 && outputPerGroup >= 8)
            {
                Conv1x1EightOutputs(input, weights, bias, output, batch, inputChannels, height, width,
                    outputChannels, groups, inputPerGroup, outputPerGroup, plane);
                return true;
            }
            if (outputPerGroup >= 4)
            {
                Conv1x1FourOutputs(input, weights, bias, output, batch, inputChannels, height, width,
                    outputChannels, groups, inputPerGroup, outputPerGroup, plane);
                return true;
            }
            for (int b = 0; b < batch; b++)
            {
                int inputBatch = b * inputChannels * plane;
                int outputBatch = b * outputChannels * plane;
                for (int g = 0; g < groups; g++)
                {
                    int inputGroup = inputBatch + g * inputPerGroup * plane;
                    int outputGroup = outputBatch + g * outputPerGroup * plane;
                    for (int co = 0; co < outputPerGroup; co++)
                    {
                        int globalCo = g * outputPerGroup + co;
                        int outputOffset = outputGroup + co * plane;
                        float initial = bias.IsEmpty ? 0f : bias[globalCo];
                        Vector256<float> initialVector = Vector256.Create(initial);
                        int spatial = 0;
                        for (; spatial <= plane - 8; spatial += 8)
                            Store(output, outputOffset + spatial, initialVector);
                        for (; spatial < plane; spatial++) output[outputOffset + spatial] = initial;

                        int weightBase = globalCo * inputPerGroup;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            ReadOnlySpan<float> inputChannel = input.Slice(inputGroup + ci * plane, plane);
                            float weight = weights[weightBase + ci];
                            spatial = 0;
                            for (; spatial <= plane - 8; spatial += 8)
                            {
                                Vector256<float> value = Load(inputChannel, spatial);
                                Vector256<float> current = Load(output, outputOffset + spatial);
                                Store(output, outputOffset + spatial, AddMul(current, value, weight));
                            }
                            for (; spatial < plane; spatial++)
                                output[outputOffset + spatial] += inputChannel[spatial] * weight;
                        }
                    }
                }
            }
            return true;
        }
        else if (Vector.IsHardwareAccelerated)
        {
            return TryVector(input, weights, bias, output, batch, inputChannels,
                height, width, outputChannels, groups, intraOpThreads);
        }
        Conv1x1Scalar(input, weights, bias, output, batch, inputChannels, height, width,
            outputChannels, groups, intraOpThreads);
        return true;
    }

    /// <summary>
    /// OC-major packed 1x1: weights are [ic][oc_padded_to_16]. Used when the
    /// spatial plane is too small for spatial Avx512 tiles to amortize weight traffic.
    /// </summary>
    internal static unsafe bool TryOcMajor(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedOc16, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int intraOpThreads = 1)
    {
        if (!Avx512F.IsSupported || (outputChannels & 15) != 0 || outputChannels < 16)
            return false;
        int plane = checked(height * width);
        // Small-plane REC (weight-bandwidth bound): OC-vectorize.
        // plane >= 48 keeps spatial 8-OC packed path (best for larger H×W).
        if (plane >= 48) return false;
        int coutPadded = (outputChannels + 15) & ~15;
        if (packedOc16.Length < checked(inputChannels * coutPadded)) return false;

        if (intraOpThreads > 1 && batch == 1 && outputChannels >= 32 &&
            (long)outputChannels * inputChannels * plane >= 1_000_000)
        {
            int ocBlocks = outputChannels / 16;
            int workers = Math.Min(intraOpThreads, ocBlocks);
            fixed (float* inputPtr = input, weightsPtr = packedOc16,
                biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = packedOc16.Length,
                    biasLength = bias.Length, outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    int beginOc = (ocBlocks * worker / workers) * 16;
                    int endOc = worker == workers - 1
                        ? outputChannels
                        : (ocBlocks * (worker + 1) / workers) * 16;
                    if (endOc <= beginOc) return;
                    int shardCout = endOc - beginOc;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    // Full weight buffer; kernel indexes by absolute OC via coutPadded.
                    ReadOnlySpan<float> weightSpan = new((void*)weightsAddress, weightsLength);
                    ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(beginOc, shardCout);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(beginOc * plane, shardCout * plane);
                    Conv1x1OcMajorAvx512Unsafe(inSpan, weightSpan, biasSpan, outSpan, 1,
                        inputChannels, height, width, shardCout, coutPadded, beginOc);
                });
            }
            return true;
        }

        Conv1x1OcMajorAvx512Unsafe(input, packedOc16, bias, output, batch,
            inputChannels, height, width, outputChannels, coutPadded, 0);
        return true;
    }

    /// <summary>Runs a 1x1 convolution from C's [block4, input, lane] packed weights.</summary>
    internal static unsafe bool TryPacked(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int intraOpThreads = 1, PackedConv1x1Int8? packedInt8 = null)
    {
        if (AvxVnni.IsSupported && packedInt8 is not null &&
            inputChannels >= 192 && (inputChannels & 3) == 0 && (outputChannels & 7) == 0 &&
            packedInt8.Weights.Length == checked(inputChannels * outputChannels) &&
            packedInt8.Scales.Length == outputChannels && packedInt8.Sums.Length == outputChannels)
        {
            if (Conv1x1PackedEightOutputsInt8VnniUnsafe(input, packedInt8, bias, output,
                batch, inputChannels, height, width, outputChannels, intraOpThreads))
                return true;
        }

        // Packed [block4, ic, lane]: 16-OC×ZMM spills on Zen 5; prefer 8-OC / 4-OC.
        if (Avx512F.IsSupported && outputChannels >= 4 && (outputChannels & 3) == 0)
        {
            int plane = checked(height * width), blocks = outputChannels / 4;
            if (intraOpThreads > 1 && batch == 1 && blocks >= 2 &&
                (long)outputChannels * inputChannels * plane >= 1_000_000)
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
                        int beginBlock = blocks * worker / workers, endBlock = blocks * (worker + 1) / workers;
                        ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                        ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                            .Slice(beginBlock * inputChannels * 4, (endBlock - beginBlock) * inputChannels * 4);
                        ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                            : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(beginBlock * 4, (endBlock - beginBlock) * 4);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                            .Slice(beginBlock * 4 * plane, (endBlock - beginBlock) * 4 * plane);
                        int shardCout = (endBlock - beginBlock) * 4;
                        if ((shardCout & 15) == 0 && inputChannels <= 64)
                            Conv1x1PackedSixteenOutputsAvx512Unsafe(inSpan, weightSpan, biasSpan, outSpan, 1,
                                inputChannels, height, width, shardCout);
                        else if ((shardCout & 7) == 0)
                            Conv1x1PackedEightOutputsAvx512Unsafe(inSpan, weightSpan, biasSpan, outSpan, 1,
                                inputChannels, height, width, shardCout);
                        else
                            Conv1x1PackedAvx512Unsafe(inSpan, weightSpan, biasSpan, outSpan, 1, inputChannels,
                                height, width, shardCout);
                    });
                }
                return true;
            }
            if ((outputChannels & 15) == 0 && inputChannels <= 64)
            {
                // 16-OC helps when ic is modest (input reuse beats spill cost).
                Conv1x1PackedSixteenOutputsAvx512Unsafe(input, packedWeights, bias, output, batch,
                    inputChannels, height, width, outputChannels);
                return true;
            }
            if ((outputChannels & 7) == 0)
            {
                Conv1x1PackedEightOutputsAvx512Unsafe(input, packedWeights, bias, output, batch,
                    inputChannels, height, width, outputChannels);
                return true;
            }
            Conv1x1PackedAvx512Unsafe(input, packedWeights, bias, output, batch, inputChannels,
                height, width, outputChannels);
            return true;
        }
        else if (Avx.IsSupported && outputChannels >= 4 && (outputChannels & 3) == 0)
        {
            int plane = checked(height * width), blocks = outputChannels / 4;
            if (intraOpThreads > 1 && batch == 1 && blocks >= 2 &&
                (long)outputChannels * inputChannels * plane >= 1_000_000)
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
                        int beginBlock = blocks * worker / workers, endBlock = blocks * (worker + 1) / workers;
                        ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                        ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                            .Slice(beginBlock * inputChannels * 4, (endBlock - beginBlock) * inputChannels * 4);
                        ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                            : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(beginBlock * 4, (endBlock - beginBlock) * 4);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                            .Slice(beginBlock * 4 * plane, (endBlock - beginBlock) * 4 * plane);
                        Conv1x1PackedUnsafe(inSpan, weightSpan, biasSpan, outSpan, 1, inputChannels,
                            height, width, (endBlock - beginBlock) * 4);
                    });
                }
                return true;
            }
            if ((outputChannels & 15) == 0)
            {
                Conv1x1PackedSixteenOutputsUnsafe(input, packedWeights, bias, output, batch,
                    inputChannels, height, width, outputChannels);
                return true;
            }
            if ((outputChannels & 7) == 0)
            {
                Conv1x1PackedEightOutputsUnsafe(input, packedWeights, bias, output, batch,
                    inputChannels, height, width, outputChannels);
                return true;
            }
            Conv1x1PackedUnsafe(input, packedWeights, bias, output, batch, inputChannels,
                height, width, outputChannels);
            return true;
        }
        else if (Vector.IsHardwareAccelerated && outputChannels >= 4 && (outputChannels & 3) == 0)
        {
            return TryPackedVector(input, packedWeights, bias, output, batch, inputChannels,
                height, width, outputChannels, intraOpThreads);
        }
        if (outputChannels >= 4 && (outputChannels & 3) == 0)
        {
            Conv1x1PackedScalar(input, packedWeights, bias, output, batch, inputChannels,
                height, width, outputChannels, intraOpThreads);
            return true;
        }
        return false;
    }
}
