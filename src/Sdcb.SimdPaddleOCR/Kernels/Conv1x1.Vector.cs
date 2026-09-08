using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Conv1x1
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool TryVector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int groups, int intraOpThreads)
    {
        int inputPerGroup = inputChannels / groups;
        int outputPerGroup = outputChannels / groups;
        int plane = checked(height * width);
        int widthLanes = Vector<float>.Count;
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
                    ReadOnlySpan<float> w = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(begin * inputPerGroup, count * inputPerGroup);
                    ReadOnlySpan<float> b = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * plane, count * plane);
                    TryVector(inSpan, w, b, outSpan, 1, inputChannels, height, width, count, 1, 1);
                });
            }
        }
        else
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
                    for (; co <= outputPerGroup - 4; co += 4)
                    {
                        int globalCo = g * outputPerGroup + co;
                        int output0 = outputGroup + co * plane, output1 = output0 + plane;
                        int output2 = output1 + plane, output3 = output2 + plane;
                        int weight0 = globalCo * inputPerGroup, weight1 = (globalCo + 1) * inputPerGroup;
                        int weight2 = (globalCo + 2) * inputPerGroup, weight3 = (globalCo + 3) * inputPerGroup;
                        Vector<float> vBias0 = new(bias.IsEmpty ? 0f : bias[globalCo]);
                        Vector<float> vBias1 = new(bias.IsEmpty ? 0f : bias[globalCo + 1]);
                        Vector<float> vBias2 = new(bias.IsEmpty ? 0f : bias[globalCo + 2]);
                        Vector<float> vBias3 = new(bias.IsEmpty ? 0f : bias[globalCo + 3]);
                        int spatial = 0;
                        int dual = widthLanes * 2;
                        for (; spatial <= plane - dual; spatial += dual)
                        {
                            Vector<float> a0l = vBias0, a0h = vBias0, a1l = vBias1, a1h = vBias1;
                            Vector<float> a2l = vBias2, a2h = vBias2, a3l = vBias3, a3h = vBias3;
                            for (int ci = 0; ci < inputPerGroup; ci++)
                            {
                                int inputOffset = inputGroup + ci * plane + spatial;
                                Vector<float> valueLow = VectorLoad(input, inputOffset);
                                Vector<float> valueHigh = VectorLoad(input, inputOffset + widthLanes);
                                a0l = VectorAddMul(a0l, valueLow, weights[weight0 + ci]);
                                a0h = VectorAddMul(a0h, valueHigh, weights[weight0 + ci]);
                                a1l = VectorAddMul(a1l, valueLow, weights[weight1 + ci]);
                                a1h = VectorAddMul(a1h, valueHigh, weights[weight1 + ci]);
                                a2l = VectorAddMul(a2l, valueLow, weights[weight2 + ci]);
                                a2h = VectorAddMul(a2h, valueHigh, weights[weight2 + ci]);
                                a3l = VectorAddMul(a3l, valueLow, weights[weight3 + ci]);
                                a3h = VectorAddMul(a3h, valueHigh, weights[weight3 + ci]);
                            }
                            VectorStore(output, output0 + spatial, a0l);
                            VectorStore(output, output0 + spatial + widthLanes, a0h);
                            VectorStore(output, output1 + spatial, a1l);
                            VectorStore(output, output1 + spatial + widthLanes, a1h);
                            VectorStore(output, output2 + spatial, a2l);
                            VectorStore(output, output2 + spatial + widthLanes, a2h);
                            VectorStore(output, output3 + spatial, a3l);
                            VectorStore(output, output3 + spatial + widthLanes, a3h);
                        }
                        for (; spatial <= plane - widthLanes; spatial += widthLanes)
                        {
                            Vector<float> a0 = vBias0, a1 = vBias1, a2 = vBias2, a3 = vBias3;
                            for (int ci = 0; ci < inputPerGroup; ci++)
                            {
                                Vector<float> value = VectorLoad(input, inputGroup + ci * plane + spatial);
                                a0 = VectorAddMul(a0, value, weights[weight0 + ci]);
                                a1 = VectorAddMul(a1, value, weights[weight1 + ci]);
                                a2 = VectorAddMul(a2, value, weights[weight2 + ci]);
                                a3 = VectorAddMul(a3, value, weights[weight3 + ci]);
                            }
                            VectorStore(output, output0 + spatial, a0);
                            VectorStore(output, output1 + spatial, a1);
                            VectorStore(output, output2 + spatial, a2);
                            VectorStore(output, output3 + spatial, a3);
                        }
                        for (; spatial < plane; spatial++)
                        {
                            float a0 = bias.IsEmpty ? 0f : bias[globalCo], a1 = bias.IsEmpty ? 0f : bias[globalCo + 1];
                            float a2 = bias.IsEmpty ? 0f : bias[globalCo + 2], a3 = bias.IsEmpty ? 0f : bias[globalCo + 3];
                            for (int ci = 0; ci < inputPerGroup; ci++)
                            {
                                float value = input[inputGroup + ci * plane + spatial];
                                a0 += value * weights[weight0 + ci]; a1 += value * weights[weight1 + ci];
                                a2 += value * weights[weight2 + ci]; a3 += value * weights[weight3 + ci];
                            }
                            output[output0 + spatial] = a0; output[output1 + spatial] = a1;
                            output[output2 + spatial] = a2; output[output3 + spatial] = a3;
                        }
                    }
                    for (; co < outputPerGroup; co++)
                    {
                        int globalCo = g * outputPerGroup + co;
                        int outputOffset = outputGroup + co * plane;
                        float initial = bias.IsEmpty ? 0f : bias[globalCo];
                        Vector<float> initialVector = new(initial);
                        int spatial = 0;
                        for (; spatial <= plane - widthLanes; spatial += widthLanes)
                            VectorStore(output, outputOffset + spatial, initialVector);
                        for (; spatial < plane; spatial++) output[outputOffset + spatial] = initial;
                        int weightBase = globalCo * inputPerGroup;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            ReadOnlySpan<float> inputChannel = input.Slice(inputGroup + ci * plane, plane);
                            float weight = weights[weightBase + ci];
                            spatial = 0;
                            for (; spatial <= plane - widthLanes; spatial += widthLanes)
                            {
                                Vector<float> value = VectorLoad(inputChannel, spatial);
                                Vector<float> current = VectorLoad(output, outputOffset + spatial);
                                VectorStore(output, outputOffset + spatial, VectorAddMul(current, value, weight));
                            }
                            for (; spatial < plane; spatial++)
                                output[outputOffset + spatial] += inputChannel[spatial] * weight;
                        }
                    }
                }
            }
        }
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool TryPackedVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels, int intraOpThreads)
    {
        int plane = checked(height * width), blocks = outputChannels / 4;
        int widthLanes = Vector<float>.Count;
        if (intraOpThreads > 1 && batch == 1 && blocks >= 2 &&
            (long)outputChannels * inputChannels * plane >= IntraOpMinWork)
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
                    TryPackedVector(inSpan, weightSpan, biasSpan, outSpan, 1, inputChannels,
                        height, width, (endBlock - beginBlock) * 4, 1);
                });
            }
        }
        else if (intraOpThreads == 1 && (outputChannels & 15) == 0 && outputChannels >= 16)
        {
            int weightsPerBlock = inputChannels * 4;
            fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
            {
                for (int b = 0; b < batch; b++)
                    for (int co = 0; co < outputChannels; co += 16)
                    {
                        float* output0 = outputPtr + (b * outputChannels + co) * plane;
                        float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                        float* output4 = output3 + plane, output5 = output4 + plane, output6 = output5 + plane, output7 = output6 + plane;
                        float* output8 = output7 + plane, output9 = output8 + plane, output10 = output9 + plane, output11 = output10 + plane;
                        float* output12 = output11 + plane, output13 = output12 + plane, output14 = output13 + plane, output15 = output14 + plane;
                        Vector<float> bias0 = new(biasPtr == null ? 0f : biasPtr[co]);
                        Vector<float> bias1 = new(biasPtr == null ? 0f : biasPtr[co + 1]);
                        Vector<float> bias2 = new(biasPtr == null ? 0f : biasPtr[co + 2]);
                        Vector<float> bias3 = new(biasPtr == null ? 0f : biasPtr[co + 3]);
                        Vector<float> bias4 = new(biasPtr == null ? 0f : biasPtr[co + 4]);
                        Vector<float> bias5 = new(biasPtr == null ? 0f : biasPtr[co + 5]);
                        Vector<float> bias6 = new(biasPtr == null ? 0f : biasPtr[co + 6]);
                        Vector<float> bias7 = new(biasPtr == null ? 0f : biasPtr[co + 7]);
                        Vector<float> bias8 = new(biasPtr == null ? 0f : biasPtr[co + 8]);
                        Vector<float> bias9 = new(biasPtr == null ? 0f : biasPtr[co + 9]);
                        Vector<float> bias10 = new(biasPtr == null ? 0f : biasPtr[co + 10]);
                        Vector<float> bias11 = new(biasPtr == null ? 0f : biasPtr[co + 11]);
                        Vector<float> bias12 = new(biasPtr == null ? 0f : biasPtr[co + 12]);
                        Vector<float> bias13 = new(biasPtr == null ? 0f : biasPtr[co + 13]);
                        Vector<float> bias14 = new(biasPtr == null ? 0f : biasPtr[co + 14]);
                        Vector<float> bias15 = new(biasPtr == null ? 0f : biasPtr[co + 15]);
                        float* batchInput = inputPtr + b * inputChannels * plane;
                        float* firstWeights = weightsPtr + (co / 4) * weightsPerBlock;
                        float* secondWeights = firstWeights + weightsPerBlock;
                        float* thirdWeights = secondWeights + weightsPerBlock;
                        float* fourthWeights = thirdWeights + weightsPerBlock;
                        int spatial = 0;
                        for (; spatial <= plane - widthLanes; spatial += widthLanes)
                        {
                            Vector<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                            Vector<float> a4 = bias4, a5 = bias5, a6 = bias6, a7 = bias7;
                            Vector<float> a8 = bias8, a9 = bias9, a10 = bias10, a11 = bias11;
                            Vector<float> a12 = bias12, a13 = bias13, a14 = bias14, a15 = bias15;
                            float* inputChannel = batchInput + spatial;
                            float* w0 = firstWeights, w1 = secondWeights, w2 = thirdWeights, w3 = fourthWeights;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                Vector<float> value = VectorLoad(inputChannel);
                                VectorAddMulPacked4(ref a0, ref a1, ref a2, ref a3, value, w0);
                                VectorAddMulPacked4(ref a4, ref a5, ref a6, ref a7, value, w1);
                                VectorAddMulPacked4(ref a8, ref a9, ref a10, ref a11, value, w2);
                                VectorAddMulPacked4(ref a12, ref a13, ref a14, ref a15, value, w3);
                                inputChannel += plane;
                                w0 += 4; w1 += 4; w2 += 4; w3 += 4;
                            }
                            VectorStore(output0 + spatial, a0); VectorStore(output1 + spatial, a1);
                            VectorStore(output2 + spatial, a2); VectorStore(output3 + spatial, a3);
                            VectorStore(output4 + spatial, a4); VectorStore(output5 + spatial, a5);
                            VectorStore(output6 + spatial, a6); VectorStore(output7 + spatial, a7);
                            VectorStore(output8 + spatial, a8); VectorStore(output9 + spatial, a9);
                            VectorStore(output10 + spatial, a10); VectorStore(output11 + spatial, a11);
                            VectorStore(output12 + spatial, a12); VectorStore(output13 + spatial, a13);
                            VectorStore(output14 + spatial, a14); VectorStore(output15 + spatial, a15);
                        }
                        for (; spatial < plane; spatial++)
                        {
                            float a0 = biasPtr == null ? 0f : biasPtr[co], a1 = biasPtr == null ? 0f : biasPtr[co + 1];
                            float a2 = biasPtr == null ? 0f : biasPtr[co + 2], a3 = biasPtr == null ? 0f : biasPtr[co + 3];
                            float a4 = biasPtr == null ? 0f : biasPtr[co + 4], a5 = biasPtr == null ? 0f : biasPtr[co + 5];
                            float a6 = biasPtr == null ? 0f : biasPtr[co + 6], a7 = biasPtr == null ? 0f : biasPtr[co + 7];
                            float a8 = biasPtr == null ? 0f : biasPtr[co + 8], a9 = biasPtr == null ? 0f : biasPtr[co + 9];
                            float a10 = biasPtr == null ? 0f : biasPtr[co + 10], a11 = biasPtr == null ? 0f : biasPtr[co + 11];
                            float a12 = biasPtr == null ? 0f : biasPtr[co + 12], a13 = biasPtr == null ? 0f : biasPtr[co + 13];
                            float a14 = biasPtr == null ? 0f : biasPtr[co + 14], a15 = biasPtr == null ? 0f : biasPtr[co + 15];
                            float* inputChannel = batchInput + spatial;
                            float* w0 = firstWeights, w1 = secondWeights, w2 = thirdWeights, w3 = fourthWeights;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float value = *inputChannel;
                                a0 += value * w0[0]; a1 += value * w0[1]; a2 += value * w0[2]; a3 += value * w0[3];
                                a4 += value * w1[0]; a5 += value * w1[1]; a6 += value * w1[2]; a7 += value * w1[3];
                                a8 += value * w2[0]; a9 += value * w2[1]; a10 += value * w2[2]; a11 += value * w2[3];
                                a12 += value * w3[0]; a13 += value * w3[1]; a14 += value * w3[2]; a15 += value * w3[3];
                                inputChannel += plane; w0 += 4; w1 += 4; w2 += 4; w3 += 4;
                            }
                            output0[spatial] = a0; output1[spatial] = a1; output2[spatial] = a2; output3[spatial] = a3;
                            output4[spatial] = a4; output5[spatial] = a5; output6[spatial] = a6; output7[spatial] = a7;
                            output8[spatial] = a8; output9[spatial] = a9; output10[spatial] = a10; output11[spatial] = a11;
                            output12[spatial] = a12; output13[spatial] = a13; output14[spatial] = a14; output15[spatial] = a15;
                        }
                    }
            }
        }
        else
        {
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
                            Vector<float> bias0 = new(biasPtr == null ? 0f : biasPtr[co]);
                            Vector<float> bias1 = new(biasPtr == null ? 0f : biasPtr[co + 1]);
                            Vector<float> bias2 = new(biasPtr == null ? 0f : biasPtr[co + 2]);
                            Vector<float> bias3 = new(biasPtr == null ? 0f : biasPtr[co + 3]);
                            int spatial = tileStart;
                            int dual = widthLanes * 2;
                            for (; spatial <= tileEnd - dual; spatial += dual)
                            {
                                Vector<float> a0l = bias0, a0h = bias0, a1l = bias1, a1h = bias1;
                                Vector<float> a2l = bias2, a2h = bias2, a3l = bias3, a3h = bias3;
                                float* inputChannel = inputPtr + inputBatch + spatial;
                                float* weightCursor = weightsPtr + block * inputChannels * 4;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    Vector<float> valueLow = VectorLoad(inputChannel);
                                    Vector<float> valueHigh = VectorLoad(inputChannel + widthLanes);
                                    VectorAddMulPacked4(ref a0l, ref a1l, ref a2l, ref a3l, valueLow, weightCursor);
                                    VectorAddMulPacked4(ref a0h, ref a1h, ref a2h, ref a3h, valueHigh, weightCursor);
                                    inputChannel += plane;
                                    weightCursor += 4;
                                }
                                VectorStore(output0 + spatial, a0l);
                                VectorStore(output0 + spatial + widthLanes, a0h);
                                VectorStore(output1 + spatial, a1l);
                                VectorStore(output1 + spatial + widthLanes, a1h);
                                VectorStore(output2 + spatial, a2l);
                                VectorStore(output2 + spatial + widthLanes, a2h);
                                VectorStore(output3 + spatial, a3l);
                                VectorStore(output3 + spatial + widthLanes, a3h);
                            }
                            for (; spatial <= tileEnd - widthLanes; spatial += widthLanes)
                            {
                                Vector<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                                float* inputChannel = inputPtr + inputBatch + spatial;
                                float* weightCursor = weightsPtr + block * inputChannels * 4;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    Vector<float> value = VectorLoad(inputChannel);
                                    VectorAddMulPacked4(ref a0, ref a1, ref a2, ref a3, value, weightCursor);
                                    inputChannel += plane;
                                    weightCursor += 4;
                                }
                                VectorStore(output0 + spatial, a0);
                                VectorStore(output1 + spatial, a1);
                                VectorStore(output2 + spatial, a2);
                                VectorStore(output3 + spatial, a3);
                            }
                            for (; spatial < tileEnd; spatial++)
                            {
                                float a0 = biasPtr == null ? 0f : biasPtr[co], a1 = biasPtr == null ? 0f : biasPtr[co + 1];
                                float a2 = biasPtr == null ? 0f : biasPtr[co + 2], a3 = biasPtr == null ? 0f : biasPtr[co + 3];
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
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedSixteenOutputsVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerBlock = inputChannels * 4, widthLanes = Vector<float>.Count;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 16)
                {
                    float* output0 = outputPtr + (b * outputChannels + co) * plane;
                    float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                    float* output4 = output3 + plane, output5 = output4 + plane, output6 = output5 + plane, output7 = output6 + plane;
                    float* output8 = output7 + plane, output9 = output8 + plane, output10 = output9 + plane, output11 = output10 + plane;
                    float* output12 = output11 + plane, output13 = output12 + plane, output14 = output13 + plane, output15 = output14 + plane;
                    Vector<float> bias0 = new(biasPtr == null ? 0f : biasPtr[co]);
                    Vector<float> bias1 = new(biasPtr == null ? 0f : biasPtr[co + 1]);
                    Vector<float> bias2 = new(biasPtr == null ? 0f : biasPtr[co + 2]);
                    Vector<float> bias3 = new(biasPtr == null ? 0f : biasPtr[co + 3]);
                    Vector<float> bias4 = new(biasPtr == null ? 0f : biasPtr[co + 4]);
                    Vector<float> bias5 = new(biasPtr == null ? 0f : biasPtr[co + 5]);
                    Vector<float> bias6 = new(biasPtr == null ? 0f : biasPtr[co + 6]);
                    Vector<float> bias7 = new(biasPtr == null ? 0f : biasPtr[co + 7]);
                    Vector<float> bias8 = new(biasPtr == null ? 0f : biasPtr[co + 8]);
                    Vector<float> bias9 = new(biasPtr == null ? 0f : biasPtr[co + 9]);
                    Vector<float> bias10 = new(biasPtr == null ? 0f : biasPtr[co + 10]);
                    Vector<float> bias11 = new(biasPtr == null ? 0f : biasPtr[co + 11]);
                    Vector<float> bias12 = new(biasPtr == null ? 0f : biasPtr[co + 12]);
                    Vector<float> bias13 = new(biasPtr == null ? 0f : biasPtr[co + 13]);
                    Vector<float> bias14 = new(biasPtr == null ? 0f : biasPtr[co + 14]);
                    Vector<float> bias15 = new(biasPtr == null ? 0f : biasPtr[co + 15]);
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    float* firstWeights = weightsPtr + (co / 4) * weightsPerBlock;
                    float* secondWeights = firstWeights + weightsPerBlock;
                    float* thirdWeights = secondWeights + weightsPerBlock;
                    float* fourthWeights = thirdWeights + weightsPerBlock;
                    int spatial = 0;
                    for (; spatial <= plane - widthLanes; spatial += widthLanes)
                    {
                        Vector<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                        Vector<float> a4 = bias4, a5 = bias5, a6 = bias6, a7 = bias7;
                        Vector<float> a8 = bias8, a9 = bias9, a10 = bias10, a11 = bias11;
                        Vector<float> a12 = bias12, a13 = bias13, a14 = bias14, a15 = bias15;
                        float* inputChannel = batchInput + spatial;
                        float* w0 = firstWeights, w1 = secondWeights, w2 = thirdWeights, w3 = fourthWeights;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            Vector<float> value = VectorLoad(inputChannel);
                            a0 = VectorAddMul(a0, value, w0[0]); a1 = VectorAddMul(a1, value, w0[1]);
                            a2 = VectorAddMul(a2, value, w0[2]); a3 = VectorAddMul(a3, value, w0[3]);
                            a4 = VectorAddMul(a4, value, w1[0]); a5 = VectorAddMul(a5, value, w1[1]);
                            a6 = VectorAddMul(a6, value, w1[2]); a7 = VectorAddMul(a7, value, w1[3]);
                            a8 = VectorAddMul(a8, value, w2[0]); a9 = VectorAddMul(a9, value, w2[1]);
                            a10 = VectorAddMul(a10, value, w2[2]); a11 = VectorAddMul(a11, value, w2[3]);
                            a12 = VectorAddMul(a12, value, w3[0]); a13 = VectorAddMul(a13, value, w3[1]);
                            a14 = VectorAddMul(a14, value, w3[2]); a15 = VectorAddMul(a15, value, w3[3]);
                            inputChannel += plane;
                            w0 += 4; w1 += 4; w2 += 4; w3 += 4;
                        }
                        VectorStore(output0 + spatial, a0); VectorStore(output1 + spatial, a1);
                        VectorStore(output2 + spatial, a2); VectorStore(output3 + spatial, a3);
                        VectorStore(output4 + spatial, a4); VectorStore(output5 + spatial, a5);
                        VectorStore(output6 + spatial, a6); VectorStore(output7 + spatial, a7);
                        VectorStore(output8 + spatial, a8); VectorStore(output9 + spatial, a9);
                        VectorStore(output10 + spatial, a10); VectorStore(output11 + spatial, a11);
                        VectorStore(output12 + spatial, a12); VectorStore(output13 + spatial, a13);
                        VectorStore(output14 + spatial, a14); VectorStore(output15 + spatial, a15);
                    }
                    for (; spatial < plane; spatial++)
                    {
                        float a0 = biasPtr == null ? 0f : biasPtr[co], a1 = biasPtr == null ? 0f : biasPtr[co + 1];
                        float a2 = biasPtr == null ? 0f : biasPtr[co + 2], a3 = biasPtr == null ? 0f : biasPtr[co + 3];
                        float a4 = biasPtr == null ? 0f : biasPtr[co + 4], a5 = biasPtr == null ? 0f : biasPtr[co + 5];
                        float a6 = biasPtr == null ? 0f : biasPtr[co + 6], a7 = biasPtr == null ? 0f : biasPtr[co + 7];
                        float a8 = biasPtr == null ? 0f : biasPtr[co + 8], a9 = biasPtr == null ? 0f : biasPtr[co + 9];
                        float a10 = biasPtr == null ? 0f : biasPtr[co + 10], a11 = biasPtr == null ? 0f : biasPtr[co + 11];
                        float a12 = biasPtr == null ? 0f : biasPtr[co + 12], a13 = biasPtr == null ? 0f : biasPtr[co + 13];
                        float a14 = biasPtr == null ? 0f : biasPtr[co + 14], a15 = biasPtr == null ? 0f : biasPtr[co + 15];
                        float* inputChannel = batchInput + spatial;
                        float* w0 = firstWeights, w1 = secondWeights, w2 = thirdWeights, w3 = fourthWeights;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            float value = *inputChannel;
                            a0 += value * w0[0]; a1 += value * w0[1]; a2 += value * w0[2]; a3 += value * w0[3];
                            a4 += value * w1[0]; a5 += value * w1[1]; a6 += value * w1[2]; a7 += value * w1[3];
                            a8 += value * w2[0]; a9 += value * w2[1]; a10 += value * w2[2]; a11 += value * w2[3];
                            a12 += value * w3[0]; a13 += value * w3[1]; a14 += value * w3[2]; a15 += value * w3[3];
                            inputChannel += plane; w0 += 4; w1 += 4; w2 += 4; w3 += 4;
                        }
                        output0[spatial] = a0; output1[spatial] = a1; output2[spatial] = a2; output3[spatial] = a3;
                        output4[spatial] = a4; output5[spatial] = a5; output6[spatial] = a6; output7[spatial] = a7;
                        output8[spatial] = a8; output9[spatial] = a9; output10[spatial] = a10; output11[spatial] = a11;
                        output12[spatial] = a12; output13[spatial] = a13; output14[spatial] = a14; output15[spatial] = a15;
                    }
                }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool TryPackedEightVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedOc8, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels, int intraOpThreads)
    {
        int plane = checked(height * width), blocks8 = outputChannels / 8;
        int widthLanes = Vector<float>.Count;
        if (intraOpThreads > 1 && batch == 1 && blocks8 >= 2 &&
            (long)outputChannels * inputChannels * plane >= 1_000_000)
        {
            int workers = Math.Min(intraOpThreads, blocks8);
            fixed (float* inputPtr = input, weightsPtr = packedOc8, biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = packedOc8.Length,
                    biasLength = bias.Length, outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    int beginBlock = blocks8 * worker / workers, endBlock = blocks8 * (worker + 1) / workers;
                    if (endBlock <= beginBlock) return;
                    int shardCout = (endBlock - beginBlock) * 8;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(beginBlock * inputChannels * 8, (endBlock - beginBlock) * inputChannels * 8);
                    ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(beginBlock * 8, shardCout);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(beginBlock * 8 * plane, shardCout * plane);
                    TryPackedEightVector(inSpan, weightSpan, biasSpan, outSpan, 1, inputChannels,
                        height, width, shardCout, 1);
                });
            }
        }
        else
        {
            int dual = widthLanes * 2;
            int tileSpatial = plane;
            if (blocks8 > 1 && (long)inputChannels * plane * 8 > 1_048_576)
                tileSpatial = Math.Max(64, 49152 / inputChannels & ~15);
            fixed (float* inputPtr = input, weightsPtr = packedOc8, biasPtr = bias, outputPtr = output)
            {
                for (int b = 0; b < batch; b++)
                    for (int tileStart = 0; tileStart < plane; tileStart += tileSpatial)
                    {
                        int tileEnd = Math.Min(plane, tileStart + tileSpatial);
                        for (int block = 0; block < blocks8; block++)
                        {
                            int co = block * 8;
                            float* output0 = outputPtr + (b * outputChannels + co) * plane;
                            float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                            float* output4 = output3 + plane, output5 = output4 + plane;
                            float* output6 = output5 + plane, output7 = output6 + plane;
                            Vector<float> bias0 = new(biasPtr == null ? 0f : biasPtr[co]);
                            Vector<float> bias1 = new(biasPtr == null ? 0f : biasPtr[co + 1]);
                            Vector<float> bias2 = new(biasPtr == null ? 0f : biasPtr[co + 2]);
                            Vector<float> bias3 = new(biasPtr == null ? 0f : biasPtr[co + 3]);
                            Vector<float> bias4 = new(biasPtr == null ? 0f : biasPtr[co + 4]);
                            Vector<float> bias5 = new(biasPtr == null ? 0f : biasPtr[co + 5]);
                            Vector<float> bias6 = new(biasPtr == null ? 0f : biasPtr[co + 6]);
                            Vector<float> bias7 = new(biasPtr == null ? 0f : biasPtr[co + 7]);
                            float* blockWeights = weightsPtr + block * inputChannels * 8;
                            int inputBatch = b * inputChannels * plane;
                            int spatial = tileStart;
                            for (; spatial <= tileEnd - dual; spatial += dual)
                            {
                                Vector<float> a0l = bias0, a0h = bias0, a1l = bias1, a1h = bias1;
                                Vector<float> a2l = bias2, a2h = bias2, a3l = bias3, a3h = bias3;
                                Vector<float> a4l = bias4, a4h = bias4, a5l = bias5, a5h = bias5;
                                Vector<float> a6l = bias6, a6h = bias6, a7l = bias7, a7h = bias7;
                                float* inputChannel = inputPtr + inputBatch + spatial;
                                float* w = blockWeights;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    Vector<float> valueLow = VectorLoad(inputChannel);
                                    Vector<float> valueHigh = VectorLoad(inputChannel + widthLanes);
                                    // Packed8 is AdvSimd FMLA-by-element. On ns2 / Vector256
                                    // the 8-ref helper does not inline in this Eight loop and
                                    // was the 9c54d56 win-x64 ns2 regression; expand VectorAddMul.
#if !NETSTANDARD2_0
                                    if (AdvSimd.Arm64.IsSupported && Vector<float>.Count == 4)
                                    {
                                        VectorAddMulPacked8(ref a0l, ref a1l, ref a2l, ref a3l, ref a4l, ref a5l, ref a6l, ref a7l, valueLow, w);
                                        VectorAddMulPacked8(ref a0h, ref a1h, ref a2h, ref a3h, ref a4h, ref a5h, ref a6h, ref a7h, valueHigh, w);
                                    }
                                    else
#endif
                                    {
                                        a0l = VectorAddMul(a0l, valueLow, w[0]); a0h = VectorAddMul(a0h, valueHigh, w[0]);
                                        a1l = VectorAddMul(a1l, valueLow, w[1]); a1h = VectorAddMul(a1h, valueHigh, w[1]);
                                        a2l = VectorAddMul(a2l, valueLow, w[2]); a2h = VectorAddMul(a2h, valueHigh, w[2]);
                                        a3l = VectorAddMul(a3l, valueLow, w[3]); a3h = VectorAddMul(a3h, valueHigh, w[3]);
                                        a4l = VectorAddMul(a4l, valueLow, w[4]); a4h = VectorAddMul(a4h, valueHigh, w[4]);
                                        a5l = VectorAddMul(a5l, valueLow, w[5]); a5h = VectorAddMul(a5h, valueHigh, w[5]);
                                        a6l = VectorAddMul(a6l, valueLow, w[6]); a6h = VectorAddMul(a6h, valueHigh, w[6]);
                                        a7l = VectorAddMul(a7l, valueLow, w[7]); a7h = VectorAddMul(a7h, valueHigh, w[7]);
                                    }
                                    inputChannel += plane;
                                    w += 8;
                                }
                                VectorStore(output0 + spatial, a0l); VectorStore(output0 + spatial + widthLanes, a0h);
                                VectorStore(output1 + spatial, a1l); VectorStore(output1 + spatial + widthLanes, a1h);
                                VectorStore(output2 + spatial, a2l); VectorStore(output2 + spatial + widthLanes, a2h);
                                VectorStore(output3 + spatial, a3l); VectorStore(output3 + spatial + widthLanes, a3h);
                                VectorStore(output4 + spatial, a4l); VectorStore(output4 + spatial + widthLanes, a4h);
                                VectorStore(output5 + spatial, a5l); VectorStore(output5 + spatial + widthLanes, a5h);
                                VectorStore(output6 + spatial, a6l); VectorStore(output6 + spatial + widthLanes, a6h);
                                VectorStore(output7 + spatial, a7l); VectorStore(output7 + spatial + widthLanes, a7h);
                            }
                            for (; spatial <= tileEnd - widthLanes; spatial += widthLanes)
                            {
                                Vector<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                                Vector<float> a4 = bias4, a5 = bias5, a6 = bias6, a7 = bias7;
                                float* inputChannel = inputPtr + inputBatch + spatial;
                                float* w = blockWeights;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    Vector<float> value = VectorLoad(inputChannel);
                                    // Same Packed8 / VectorAddMul split as the dual tile above.
#if !NETSTANDARD2_0
                                    if (AdvSimd.Arm64.IsSupported && Vector<float>.Count == 4)
                                        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, w);
                                    else
#endif
                                    {
                                        a0 = VectorAddMul(a0, value, w[0]); a1 = VectorAddMul(a1, value, w[1]);
                                        a2 = VectorAddMul(a2, value, w[2]); a3 = VectorAddMul(a3, value, w[3]);
                                        a4 = VectorAddMul(a4, value, w[4]); a5 = VectorAddMul(a5, value, w[5]);
                                        a6 = VectorAddMul(a6, value, w[6]); a7 = VectorAddMul(a7, value, w[7]);
                                    }
                                    inputChannel += plane;
                                    w += 8;
                                }
                                VectorStore(output0 + spatial, a0); VectorStore(output1 + spatial, a1);
                                VectorStore(output2 + spatial, a2); VectorStore(output3 + spatial, a3);
                                VectorStore(output4 + spatial, a4); VectorStore(output5 + spatial, a5);
                                VectorStore(output6 + spatial, a6); VectorStore(output7 + spatial, a7);
                            }
                            for (; spatial < tileEnd; spatial++)
                            {
                                float a0 = biasPtr == null ? 0f : biasPtr[co], a1 = biasPtr == null ? 0f : biasPtr[co + 1];
                                float a2 = biasPtr == null ? 0f : biasPtr[co + 2], a3 = biasPtr == null ? 0f : biasPtr[co + 3];
                                float a4 = biasPtr == null ? 0f : biasPtr[co + 4], a5 = biasPtr == null ? 0f : biasPtr[co + 5];
                                float a6 = biasPtr == null ? 0f : biasPtr[co + 6], a7 = biasPtr == null ? 0f : biasPtr[co + 7];
                                float* inputChannel = inputPtr + inputBatch + spatial;
                                float* w = blockWeights;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float value = *inputChannel;
                                    a0 += value * w[0]; a1 += value * w[1]; a2 += value * w[2]; a3 += value * w[3];
                                    a4 += value * w[4]; a5 += value * w[5]; a6 += value * w[6]; a7 += value * w[7];
                                    inputChannel += plane;
                                    w += 8;
                                }
                                output0[spatial] = a0; output1[spatial] = a1; output2[spatial] = a2; output3[spatial] = a3;
                                output4[spatial] = a4; output5[spatial] = a5; output6[spatial] = a6; output7[spatial] = a7;
                            }
                        }
                    }
            }
        }
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void Conv1x1OcMajorVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedOc16, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int coutPadded, int weightOcBase)
    {
        int plane = checked(height * width);
        int ocTile = Vector<float>.Count >= 8 ? 8 : 4;
        if (plane <= 0 || plane >= 48) return;
        fixed (float* inputPtr = input, weightsPtr = packedOc16, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                float* batchInput = inputPtr + b * inputChannels * plane;
                float* batchOutput = outputPtr + b * outputChannels * plane;
                for (int co = 0; co < outputChannels; co += ocTile)
                {
                    Vector<float> vBias = LoadOcTile(biasPtr, co, ocTile);
                    float* weightBase = weightsPtr + weightOcBase + co;
                    float* out0 = batchOutput + co * plane;
                    int spatial = 0;
                    for (; spatial <= plane - 4; spatial += 4)
                    {
                        Vector<float> a0 = vBias, a1 = vBias, a2 = vBias, a3 = vBias;
                        float* in0 = batchInput + spatial;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            Vector<float> w = LoadOcTile(weightBase + ci * coutPadded, 0, ocTile);
                            a0 = VectorAddMul(a0, w, in0[0]);
                            a1 = VectorAddMul(a1, w, in0[1]);
                            a2 = VectorAddMul(a2, w, in0[2]);
                            a3 = VectorAddMul(a3, w, in0[3]);
                            in0 += plane;
                        }
                        StoreOcTile(out0, plane, spatial, a0, ocTile);
                        StoreOcTile(out0, plane, spatial + 1, a1, ocTile);
                        StoreOcTile(out0, plane, spatial + 2, a2, ocTile);
                        StoreOcTile(out0, plane, spatial + 3, a3, ocTile);
                    }
                    for (; spatial < plane; spatial++)
                    {
                        Vector<float> acc = vBias;
                        float* inChannel = batchInput + spatial;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            Vector<float> w = LoadOcTile(weightBase + ci * coutPadded, 0, ocTile);
                            acc = VectorAddMul(acc, w, *inChannel);
                            inChannel += plane;
                        }
                        StoreOcTile(out0, plane, spatial, acc, ocTile);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector<float> LoadOcTile(float* source, int offset, int ocTile)
    {
        if (source == null)
            return Vector<float>.Zero;
        if (Vector<float>.Count == ocTile)
            return VectorLoad(source + offset);
        Vector<float> value = default;
        ref float d = ref Unsafe.As<Vector<float>, float>(ref value);
        for (int lane = 0; lane < ocTile; lane++)
            Unsafe.Add(ref d, lane) = source[offset + lane];
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreOcTile(float* out0, int plane, int spatial,
        Vector<float> acc, int ocTile)
    {
        for (int lane = 0; lane < ocTile; lane++)
            out0[lane * plane + spatial] = acc.GetElement(lane);
    }
}
