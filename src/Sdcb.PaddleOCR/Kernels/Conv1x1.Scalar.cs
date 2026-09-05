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

internal static partial class Conv1x1
{
    // Scalar twin of the Vector 4-output 1x1: same accumulation order, lanes = 1.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1Scalar(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int groups, int intraOpThreads)
    {
        int inputPerGroup = inputChannels / groups;
        int plane = checked(height * width);
        if (CanShardOutputs(intraOpThreads, batch, groups, outputChannels,
            (long)outputChannels * inputPerGroup * plane))
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
                        .Slice(begin * inputPerGroup, count * inputPerGroup);
                    ReadOnlySpan<float> b = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * plane, count * plane);
                    Conv1x1ScalarKernel(inSpan, w, b, outSpan, 1, inputChannels, height, width, count, 1);
                });
            }
            return;
        }
        Conv1x1ScalarKernel(input, weights, bias, output, batch, inputChannels, height, width,
            outputChannels, groups);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1ScalarKernel(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int groups)
    {
        int inputPerGroup = inputChannels / groups;
        int outputPerGroup = outputChannels / groups;
        int plane = checked(height * width);
        fixed (float* inputPtr = input, weightsPtr = weights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                int inputBatch = b * inputChannels * plane;
                int outputBatch = b * outputChannels * plane;
                for (int g = 0; g < groups; g++)
                {
                    int inputGroup = inputBatch + g * inputPerGroup * plane;
                    int outputGroup = outputBatch + g * outputPerGroup * plane;
                    int co = 0;
                    for (; co <= outputPerGroup - 8; co += 8)
                    {
                        int globalCo = g * outputPerGroup + co;
                        float* output0 = outputPtr + outputGroup + co * plane;
                        float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                        float* output4 = output3 + plane, output5 = output4 + plane, output6 = output5 + plane, output7 = output6 + plane;
                        int weight0 = globalCo * inputPerGroup, weight1 = (globalCo + 1) * inputPerGroup;
                        int weight2 = (globalCo + 2) * inputPerGroup, weight3 = (globalCo + 3) * inputPerGroup;
                        int weight4 = (globalCo + 4) * inputPerGroup, weight5 = (globalCo + 5) * inputPerGroup;
                        int weight6 = (globalCo + 6) * inputPerGroup, weight7 = (globalCo + 7) * inputPerGroup;
                        float b0 = bias.IsEmpty ? 0f : bias[globalCo], b1 = bias.IsEmpty ? 0f : bias[globalCo + 1];
                        float b2 = bias.IsEmpty ? 0f : bias[globalCo + 2], b3 = bias.IsEmpty ? 0f : bias[globalCo + 3];
                        float b4 = bias.IsEmpty ? 0f : bias[globalCo + 4], b5 = bias.IsEmpty ? 0f : bias[globalCo + 5];
                        float b6 = bias.IsEmpty ? 0f : bias[globalCo + 6], b7 = bias.IsEmpty ? 0f : bias[globalCo + 7];
                        for (int spatial = 0; spatial < plane; spatial++)
                        {
                            float a0 = b0, a1 = b1, a2 = b2, a3 = b3, a4 = b4, a5 = b5, a6 = b6, a7 = b7;
                            float* inputPixel = inputPtr + inputGroup + spatial;
                            for (int ci = 0; ci < inputPerGroup; ci++)
                            {
                                float value = *inputPixel;
                                a0 += value * weightsPtr[weight0 + ci]; a1 += value * weightsPtr[weight1 + ci];
                                a2 += value * weightsPtr[weight2 + ci]; a3 += value * weightsPtr[weight3 + ci];
                                a4 += value * weightsPtr[weight4 + ci]; a5 += value * weightsPtr[weight5 + ci];
                                a6 += value * weightsPtr[weight6 + ci]; a7 += value * weightsPtr[weight7 + ci];
                                inputPixel += plane;
                            }
                            output0[spatial] = a0; output1[spatial] = a1; output2[spatial] = a2; output3[spatial] = a3;
                            output4[spatial] = a4; output5[spatial] = a5; output6[spatial] = a6; output7[spatial] = a7;
                        }
                    }
                    for (; co <= outputPerGroup - OutputTile; co += OutputTile)
                    {
                        int globalCo = g * outputPerGroup + co;
                        float* output0 = outputPtr + outputGroup + co * plane;
                        float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                        int weight0 = globalCo * inputPerGroup, weight1 = (globalCo + 1) * inputPerGroup;
                        int weight2 = (globalCo + 2) * inputPerGroup, weight3 = (globalCo + 3) * inputPerGroup;
                        float b0 = bias.IsEmpty ? 0f : bias[globalCo], b1 = bias.IsEmpty ? 0f : bias[globalCo + 1];
                        float b2 = bias.IsEmpty ? 0f : bias[globalCo + 2], b3 = bias.IsEmpty ? 0f : bias[globalCo + 3];
                        for (int spatial = 0; spatial < plane; spatial++)
                        {
                            float a0 = b0, a1 = b1, a2 = b2, a3 = b3;
                            float* inputPixel = inputPtr + inputGroup + spatial;
                            for (int ci = 0; ci < inputPerGroup; ci++)
                            {
                                float value = *inputPixel;
                                a0 += value * weightsPtr[weight0 + ci];
                                a1 += value * weightsPtr[weight1 + ci];
                                a2 += value * weightsPtr[weight2 + ci];
                                a3 += value * weightsPtr[weight3 + ci];
                                inputPixel += plane;
                            }
                            output0[spatial] = a0; output1[spatial] = a1;
                            output2[spatial] = a2; output3[spatial] = a3;
                        }
                    }
                    for (; co < outputPerGroup; co++)
                    {
                        int globalCo = g * outputPerGroup + co;
                        float* dst = outputPtr + outputGroup + co * plane;
                        float initial = bias.IsEmpty ? 0f : bias[globalCo];
                        int weightBase = globalCo * inputPerGroup;
                        for (int spatial = 0; spatial < plane; spatial++)
                        {
                            float sum = initial;
                            float* inputPixel = inputPtr + inputGroup + spatial;
                            for (int ci = 0; ci < inputPerGroup; ci++)
                            {
                                sum += *inputPixel * weightsPtr[weightBase + ci];
                                inputPixel += plane;
                            }
                            dst[spatial] = sum;
                        }
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedScalar(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels, int intraOpThreads)
    {
        int plane = checked(height * width), blocks = outputChannels / OutputTile;
        if (intraOpThreads > 1 && batch == 1 && blocks >= 2 &&
            (long)outputChannels * inputChannels * plane >= IntraOpMinWork)
        {
            int workers = ShardWorkers(intraOpThreads, blocks);
            fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = packedWeights.Length,
                    biasLength = bias.Length, outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    (int beginBlock, int endBlock) = BlockShard(worker, workers, blocks);
                    if (endBlock <= beginBlock) return;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(beginBlock * inputChannels * 4, (endBlock - beginBlock) * inputChannels * 4);
                    ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(beginBlock * 4, (endBlock - beginBlock) * 4);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(beginBlock * 4 * plane, (endBlock - beginBlock) * 4 * plane);
                    Conv1x1PackedScalarKernel(inSpan, weightSpan, biasSpan, outSpan, 1, inputChannels,
                        height, width, (endBlock - beginBlock) * 4);
                });
            }
            return;
        }
        Conv1x1PackedScalarKernel(input, packedWeights, bias, output, batch, inputChannels,
            height, width, outputChannels);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedScalarKernel(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), blocks = outputChannels / OutputTile;
        int tileSpatial = plane;
        if (blocks > 1 && (long)inputChannels * plane * 4 > 1_048_576)
            tileSpatial = Math.Max(64, 49152 / inputChannels & ~15);
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int tileStart = 0; tileStart < plane; tileStart += tileSpatial)
                {
                    int tileEnd = Math.Min(plane, tileStart + tileSpatial);
                    for (int block = 0; block < blocks; block++)
                    {
                        int co = block * 4, inputBatch = b * inputChannels * plane;
                        float* output0 = outputPtr + (b * outputChannels + co) * plane;
                        float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                        float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                        float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                        for (int spatial = tileStart; spatial < tileEnd; spatial++)
                        {
                            float a0 = b0, a1 = b1, a2 = b2, a3 = b3;
                            float* inputChannel = inputPtr + inputBatch + spatial;
                            float* weightCursor = weightsPtr + block * inputChannels * 4;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float value = *inputChannel;
                                a0 += value * weightCursor[0]; a1 += value * weightCursor[1];
                                a2 += value * weightCursor[2]; a3 += value * weightCursor[3];
                                inputChannel += plane;
                                weightCursor += 4;
                            }
                            output0[spatial] = a0; output1[spatial] = a1;
                            output2[spatial] = a2; output3[spatial] = a3;
                        }
                    }
                }
        }
    }
}
