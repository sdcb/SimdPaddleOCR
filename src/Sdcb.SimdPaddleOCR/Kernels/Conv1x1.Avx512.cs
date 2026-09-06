using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Conv1x1
{
    /// <summary>
    /// OC-major 1x1: weights [ic][oc_padded]. Accumulate into a contiguous
    /// [plane][16] tile (cheap Store512), then transpose to NCHW planar output.
    /// Multi-spatial FMA tiles reuse each weight vector on small REC planes.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1OcMajorAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int coutPadded, int weightOcBase)
    {
        int plane = checked(height * width);
        // Scratch is plane*16 floats; keep bounded for stackalloc.
        // Scratch is plane*16 floats; keep bounded for stackalloc.
        if (plane <= 0 || plane >= 48) return;
        float* tile = stackalloc float[plane * 16];
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                float* batchInput = inputPtr + b * inputChannels * plane;
                float* batchOutput = outputPtr + b * outputChannels * plane;
                for (int co = 0; co < outputChannels; co += 16)
                {
                    Vector512<float> vBias = biasPtr == null
                        ? Vector512<float>.Zero
                        : Avx512F.LoadVector512(biasPtr + co);
                    float* weightBase = weightsPtr + weightOcBase + co;

                    int spatial = 0;
                    // 8 spatial × OC-16: one weight load feeds eight FMAs (≤8 ZMM acc).
                    for (; spatial <= plane - 8; spatial += 8)
                    {
                        Vector512<float> a0 = vBias, a1 = vBias, a2 = vBias, a3 = vBias;
                        Vector512<float> a4 = vBias, a5 = vBias, a6 = vBias, a7 = vBias;
                        float* in0 = batchInput + spatial;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            Vector512<float> w = Avx512F.LoadVector512(weightBase + ci * coutPadded);
                            a0 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[0]), a0);
                            a1 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[1]), a1);
                            a2 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[2]), a2);
                            a3 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[3]), a3);
                            a4 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[4]), a4);
                            a5 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[5]), a5);
                            a6 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[6]), a6);
                            a7 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[7]), a7);
                            in0 += plane;
                        }
                        Avx512F.Store(tile + spatial * 16, a0);
                        Avx512F.Store(tile + (spatial + 1) * 16, a1);
                        Avx512F.Store(tile + (spatial + 2) * 16, a2);
                        Avx512F.Store(tile + (spatial + 3) * 16, a3);
                        Avx512F.Store(tile + (spatial + 4) * 16, a4);
                        Avx512F.Store(tile + (spatial + 5) * 16, a5);
                        Avx512F.Store(tile + (spatial + 6) * 16, a6);
                        Avx512F.Store(tile + (spatial + 7) * 16, a7);
                    }
                    for (; spatial <= plane - 4; spatial += 4)
                    {
                        Vector512<float> a0 = vBias, a1 = vBias, a2 = vBias, a3 = vBias;
                        float* in0 = batchInput + spatial;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            Vector512<float> w = Avx512F.LoadVector512(weightBase + ci * coutPadded);
                            a0 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[0]), a0);
                            a1 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[1]), a1);
                            a2 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[2]), a2);
                            a3 = Avx512F.FusedMultiplyAdd(w, Vector512.Create(in0[3]), a3);
                            in0 += plane;
                        }
                        Avx512F.Store(tile + spatial * 16, a0);
                        Avx512F.Store(tile + (spatial + 1) * 16, a1);
                        Avx512F.Store(tile + (spatial + 2) * 16, a2);
                        Avx512F.Store(tile + (spatial + 3) * 16, a3);
                    }
                    for (; spatial < plane; spatial++)
                    {
                        Vector512<float> acc = vBias;
                        float* inChannel = batchInput + spatial;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            Vector512<float> w = Avx512F.LoadVector512(weightBase + ci * coutPadded);
                            acc = Avx512F.FusedMultiplyAdd(w, Vector512.Create(*inChannel), acc);
                            inChannel += plane;
                        }
                        Avx512F.Store(tile + spatial * 16, acc);
                    }

                    // Transpose [plane][16] → 16 planar channels at this OC block.
                    float* outBase = batchOutput + co * plane;
                    for (int lane = 0; lane < 16; lane++)
                    {
                        float* dst = outBase + lane * plane;
                        float* src = tile + lane;
                        for (int s = 0; s < plane; s++)
                            dst[s] = src[s * 16];
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedSixteenOutputsAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerBlock = inputChannels * 4;
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
                    Vector512<float> bias0 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co]);
                    Vector512<float> bias1 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                    Vector512<float> bias2 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                    Vector512<float> bias3 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                    Vector512<float> bias4 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 4]);
                    Vector512<float> bias5 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 5]);
                    Vector512<float> bias6 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 6]);
                    Vector512<float> bias7 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 7]);
                    Vector512<float> bias8 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 8]);
                    Vector512<float> bias9 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 9]);
                    Vector512<float> bias10 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 10]);
                    Vector512<float> bias11 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 11]);
                    Vector512<float> bias12 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 12]);
                    Vector512<float> bias13 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 13]);
                    Vector512<float> bias14 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 14]);
                    Vector512<float> bias15 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 15]);
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    float* firstWeights = weightsPtr + (co / 4) * weightsPerBlock;
                    float* secondWeights = firstWeights + weightsPerBlock;
                    float* thirdWeights = secondWeights + weightsPerBlock;
                    float* fourthWeights = thirdWeights + weightsPerBlock;
                    int spatial = 0;
                    for (; spatial <= plane - 16; spatial += 16)
                    {
                        Vector512<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                        Vector512<float> a4 = bias4, a5 = bias5, a6 = bias6, a7 = bias7;
                        Vector512<float> a8 = bias8, a9 = bias9, a10 = bias10, a11 = bias11;
                        Vector512<float> a12 = bias12, a13 = bias13, a14 = bias14, a15 = bias15;
                        float* inputChannel = batchInput + spatial;
                        float* w0 = firstWeights, w1 = secondWeights, w2 = thirdWeights, w3 = fourthWeights;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            // One input Vector512 reused across 16 OC; prefetch next channel.
                            Vector512<float> value = Avx512F.LoadVector512(inputChannel);
                            if (ci + 1 < inputChannels)
                                Sse.Prefetch0(inputChannel + plane);
                            AddFourPacked512(ref a0, ref a1, ref a2, ref a3, value, w0);
                            AddFourPacked512(ref a4, ref a5, ref a6, ref a7, value, w1);
                            AddFourPacked512(ref a8, ref a9, ref a10, ref a11, value, w2);
                            AddFourPacked512(ref a12, ref a13, ref a14, ref a15, value, w3);
                            inputChannel += plane;
                            w0 += 4; w1 += 4; w2 += 4; w3 += 4;
                        }
                        Avx512F.Store(output0 + spatial, a0); Avx512F.Store(output1 + spatial, a1);
                        Avx512F.Store(output2 + spatial, a2); Avx512F.Store(output3 + spatial, a3);
                        Avx512F.Store(output4 + spatial, a4); Avx512F.Store(output5 + spatial, a5);
                        Avx512F.Store(output6 + spatial, a6); Avx512F.Store(output7 + spatial, a7);
                        Avx512F.Store(output8 + spatial, a8); Avx512F.Store(output9 + spatial, a9);
                        Avx512F.Store(output10 + spatial, a10); Avx512F.Store(output11 + spatial, a11);
                        Avx512F.Store(output12 + spatial, a12); Avx512F.Store(output13 + spatial, a13);
                        Avx512F.Store(output14 + spatial, a14); Avx512F.Store(output15 + spatial, a15);
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
    private static void Conv1x1EightOutputsAvx512(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels, int height,
        int width, int outputChannels, int groups, int inputPerGroup, int outputPerGroup, int plane)
    {
        for (int b = 0; b < batch; b++)
        {
            int inputBatch = b * inputChannels * plane, outputBatch = b * outputChannels * plane;
            for (int g = 0; g < groups; g++)
            {
                int inputGroup = inputBatch + g * inputPerGroup * plane;
                int outputGroup = outputBatch + g * outputPerGroup * plane;
                for (int co = 0; co < outputPerGroup; co += 8)
                {
                    int globalCo = g * outputPerGroup + co;
                    int output0 = outputGroup + co * plane, output1 = output0 + plane;
                    int output2 = output1 + plane, output3 = output2 + plane;
                    int output4 = output3 + plane, output5 = output4 + plane;
                    int output6 = output5 + plane, output7 = output6 + plane;
                    int weight0 = globalCo * inputPerGroup, weight1 = (globalCo + 1) * inputPerGroup;
                    int weight2 = (globalCo + 2) * inputPerGroup, weight3 = (globalCo + 3) * inputPerGroup;
                    int weight4 = (globalCo + 4) * inputPerGroup, weight5 = (globalCo + 5) * inputPerGroup;
                    int weight6 = (globalCo + 6) * inputPerGroup, weight7 = (globalCo + 7) * inputPerGroup;
                    Vector512<float> vBias0 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo]);
                    Vector512<float> vBias1 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 1]);
                    Vector512<float> vBias2 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 2]);
                    Vector512<float> vBias3 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 3]);
                    Vector512<float> vBias4 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 4]);
                    Vector512<float> vBias5 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 5]);
                    Vector512<float> vBias6 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 6]);
                    Vector512<float> vBias7 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 7]);
                    // Spatial-16 only (8 ZMM acc). Dual-spatial would be 16 acc.
                    int spatial = 0;
                    for (; spatial <= plane - 16; spatial += 16)
                    {
                        Vector512<float> a0 = vBias0, a1 = vBias1, a2 = vBias2, a3 = vBias3;
                        Vector512<float> a4 = vBias4, a5 = vBias5, a6 = vBias6, a7 = vBias7;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            Vector512<float> value = Load512(input, inputGroup + ci * plane + spatial);
                            a0 = AddMul512(a0, value, weights[weight0 + ci]);
                            a1 = AddMul512(a1, value, weights[weight1 + ci]);
                            a2 = AddMul512(a2, value, weights[weight2 + ci]);
                            a3 = AddMul512(a3, value, weights[weight3 + ci]);
                            a4 = AddMul512(a4, value, weights[weight4 + ci]);
                            a5 = AddMul512(a5, value, weights[weight5 + ci]);
                            a6 = AddMul512(a6, value, weights[weight6 + ci]);
                            a7 = AddMul512(a7, value, weights[weight7 + ci]);
                        }
                        Store512(output, output0 + spatial, a0); Store512(output, output1 + spatial, a1);
                        Store512(output, output2 + spatial, a2); Store512(output, output3 + spatial, a3);
                        Store512(output, output4 + spatial, a4); Store512(output, output5 + spatial, a5);
                        Store512(output, output6 + spatial, a6); Store512(output, output7 + spatial, a7);
                    }
                    for (; spatial < plane; spatial++)
                    {
                        float a0 = bias.IsEmpty ? 0f : bias[globalCo], a1 = bias.IsEmpty ? 0f : bias[globalCo + 1];
                        float a2 = bias.IsEmpty ? 0f : bias[globalCo + 2], a3 = bias.IsEmpty ? 0f : bias[globalCo + 3];
                        float a4 = bias.IsEmpty ? 0f : bias[globalCo + 4], a5 = bias.IsEmpty ? 0f : bias[globalCo + 5];
                        float a6 = bias.IsEmpty ? 0f : bias[globalCo + 6], a7 = bias.IsEmpty ? 0f : bias[globalCo + 7];
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            float value = input[inputGroup + ci * plane + spatial];
                            a0 += value * weights[weight0 + ci]; a1 += value * weights[weight1 + ci];
                            a2 += value * weights[weight2 + ci]; a3 += value * weights[weight3 + ci];
                            a4 += value * weights[weight4 + ci]; a5 += value * weights[weight5 + ci];
                            a6 += value * weights[weight6 + ci]; a7 += value * weights[weight7 + ci];
                        }
                        output[output0 + spatial] = a0; output[output1 + spatial] = a1;
                        output[output2 + spatial] = a2; output[output3 + spatial] = a3;
                        output[output4 + spatial] = a4; output[output5 + spatial] = a5;
                        output[output6 + spatial] = a6; output[output7 + spatial] = a7;
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void Conv1x1SixteenOutputsAvx512(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels, int height,
        int width, int outputChannels, int groups, int inputPerGroup, int outputPerGroup, int plane)
    {
        for (int b = 0; b < batch; b++)
        {
            int inputBatch = b * inputChannels * plane, outputBatch = b * outputChannels * plane;
            for (int g = 0; g < groups; g++)
            {
                int inputGroup = inputBatch + g * inputPerGroup * plane;
                int outputGroup = outputBatch + g * outputPerGroup * plane;
                for (int co = 0; co < outputPerGroup; co += 16)
                {
                    int globalCo = g * outputPerGroup + co;
                    int output0 = outputGroup + co * plane, output1 = output0 + plane;
                    int output2 = output1 + plane, output3 = output2 + plane;
                    int output4 = output3 + plane, output5 = output4 + plane;
                    int output6 = output5 + plane, output7 = output6 + plane;
                    int output8 = output7 + plane, output9 = output8 + plane;
                    int output10 = output9 + plane, output11 = output10 + plane;
                    int output12 = output11 + plane, output13 = output12 + plane;
                    int output14 = output13 + plane, output15 = output14 + plane;
                    int weight0 = globalCo * inputPerGroup, weight1 = (globalCo + 1) * inputPerGroup;
                    int weight2 = (globalCo + 2) * inputPerGroup, weight3 = (globalCo + 3) * inputPerGroup;
                    int weight4 = (globalCo + 4) * inputPerGroup, weight5 = (globalCo + 5) * inputPerGroup;
                    int weight6 = (globalCo + 6) * inputPerGroup, weight7 = (globalCo + 7) * inputPerGroup;
                    int weight8 = (globalCo + 8) * inputPerGroup, weight9 = (globalCo + 9) * inputPerGroup;
                    int weight10 = (globalCo + 10) * inputPerGroup, weight11 = (globalCo + 11) * inputPerGroup;
                    int weight12 = (globalCo + 12) * inputPerGroup, weight13 = (globalCo + 13) * inputPerGroup;
                    int weight14 = (globalCo + 14) * inputPerGroup, weight15 = (globalCo + 15) * inputPerGroup;
                    Vector512<float> vBias0 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo]);
                    Vector512<float> vBias1 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 1]);
                    Vector512<float> vBias2 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 2]);
                    Vector512<float> vBias3 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 3]);
                    Vector512<float> vBias4 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 4]);
                    Vector512<float> vBias5 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 5]);
                    Vector512<float> vBias6 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 6]);
                    Vector512<float> vBias7 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 7]);
                    Vector512<float> vBias8 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 8]);
                    Vector512<float> vBias9 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 9]);
                    Vector512<float> vBias10 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 10]);
                    Vector512<float> vBias11 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 11]);
                    Vector512<float> vBias12 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 12]);
                    Vector512<float> vBias13 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 13]);
                    Vector512<float> vBias14 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 14]);
                    Vector512<float> vBias15 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 15]);
                    int spatial = 0;
                    for (; spatial <= plane - 16; spatial += 16)
                    {
                        Vector512<float> a0 = vBias0, a1 = vBias1, a2 = vBias2, a3 = vBias3;
                        Vector512<float> a4 = vBias4, a5 = vBias5, a6 = vBias6, a7 = vBias7;
                        Vector512<float> a8 = vBias8, a9 = vBias9, a10 = vBias10, a11 = vBias11;
                        Vector512<float> a12 = vBias12, a13 = vBias13, a14 = vBias14, a15 = vBias15;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            Vector512<float> value = Load512(input, inputGroup + ci * plane + spatial);
                            a0 = AddMul512(a0, value, weights[weight0 + ci]);
                            a1 = AddMul512(a1, value, weights[weight1 + ci]);
                            a2 = AddMul512(a2, value, weights[weight2 + ci]);
                            a3 = AddMul512(a3, value, weights[weight3 + ci]);
                            a4 = AddMul512(a4, value, weights[weight4 + ci]);
                            a5 = AddMul512(a5, value, weights[weight5 + ci]);
                            a6 = AddMul512(a6, value, weights[weight6 + ci]);
                            a7 = AddMul512(a7, value, weights[weight7 + ci]);
                            a8 = AddMul512(a8, value, weights[weight8 + ci]);
                            a9 = AddMul512(a9, value, weights[weight9 + ci]);
                            a10 = AddMul512(a10, value, weights[weight10 + ci]);
                            a11 = AddMul512(a11, value, weights[weight11 + ci]);
                            a12 = AddMul512(a12, value, weights[weight12 + ci]);
                            a13 = AddMul512(a13, value, weights[weight13 + ci]);
                            a14 = AddMul512(a14, value, weights[weight14 + ci]);
                            a15 = AddMul512(a15, value, weights[weight15 + ci]);
                        }
                        Store512(output, output0 + spatial, a0); Store512(output, output1 + spatial, a1);
                        Store512(output, output2 + spatial, a2); Store512(output, output3 + spatial, a3);
                        Store512(output, output4 + spatial, a4); Store512(output, output5 + spatial, a5);
                        Store512(output, output6 + spatial, a6); Store512(output, output7 + spatial, a7);
                        Store512(output, output8 + spatial, a8); Store512(output, output9 + spatial, a9);
                        Store512(output, output10 + spatial, a10); Store512(output, output11 + spatial, a11);
                        Store512(output, output12 + spatial, a12); Store512(output, output13 + spatial, a13);
                        Store512(output, output14 + spatial, a14); Store512(output, output15 + spatial, a15);
                    }
                    for (; spatial < plane; spatial++)
                    {
                        float a0 = bias.IsEmpty ? 0f : bias[globalCo], a1 = bias.IsEmpty ? 0f : bias[globalCo + 1];
                        float a2 = bias.IsEmpty ? 0f : bias[globalCo + 2], a3 = bias.IsEmpty ? 0f : bias[globalCo + 3];
                        float a4 = bias.IsEmpty ? 0f : bias[globalCo + 4], a5 = bias.IsEmpty ? 0f : bias[globalCo + 5];
                        float a6 = bias.IsEmpty ? 0f : bias[globalCo + 6], a7 = bias.IsEmpty ? 0f : bias[globalCo + 7];
                        float a8 = bias.IsEmpty ? 0f : bias[globalCo + 8], a9 = bias.IsEmpty ? 0f : bias[globalCo + 9];
                        float a10 = bias.IsEmpty ? 0f : bias[globalCo + 10], a11 = bias.IsEmpty ? 0f : bias[globalCo + 11];
                        float a12 = bias.IsEmpty ? 0f : bias[globalCo + 12], a13 = bias.IsEmpty ? 0f : bias[globalCo + 13];
                        float a14 = bias.IsEmpty ? 0f : bias[globalCo + 14], a15 = bias.IsEmpty ? 0f : bias[globalCo + 15];
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            float value = input[inputGroup + ci * plane + spatial];
                            a0 += value * weights[weight0 + ci]; a1 += value * weights[weight1 + ci];
                            a2 += value * weights[weight2 + ci]; a3 += value * weights[weight3 + ci];
                            a4 += value * weights[weight4 + ci]; a5 += value * weights[weight5 + ci];
                            a6 += value * weights[weight6 + ci]; a7 += value * weights[weight7 + ci];
                            a8 += value * weights[weight8 + ci]; a9 += value * weights[weight9 + ci];
                            a10 += value * weights[weight10 + ci]; a11 += value * weights[weight11 + ci];
                            a12 += value * weights[weight12 + ci]; a13 += value * weights[weight13 + ci];
                            a14 += value * weights[weight14 + ci]; a15 += value * weights[weight15 + ci];
                        }
                        output[output0 + spatial] = a0; output[output1 + spatial] = a1;
                        output[output2 + spatial] = a2; output[output3 + spatial] = a3;
                        output[output4 + spatial] = a4; output[output5 + spatial] = a5;
                        output[output6 + spatial] = a6; output[output7 + spatial] = a7;
                        output[output8 + spatial] = a8; output[output9 + spatial] = a9;
                        output[output10 + spatial] = a10; output[output11 + spatial] = a11;
                        output[output12 + spatial] = a12; output[output13 + spatial] = a13;
                        output[output14 + spatial] = a14; output[output15 + spatial] = a15;
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedEightOutputsAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        ReadOnlySpan<float> residual = default)
    {
        int plane = checked(height * width), blocks = outputChannels / 8;
        // Rec activations often fit in Zen 5 L2 (~0.5–1.5MB). Tiny spatial
        // tiles were meant to keep a panel in L1, but they re-stream packed
        // weights once per tile and lose the 8-OC reuse that the outer block
        // loop is built for. Only tile when the input plane itself exceeds L2.
        int tileSpatial = plane;
        if (blocks > 1 && (long)inputChannels * plane * 4 > 524_288)
        {
            int target = Math.Max(64, 262144 / Math.Max(1, inputChannels) / 4);
            tileSpatial = Math.Max(64, Math.Min(plane, target) & ~15);
        }
        bool hasResidual = residual.Length == output.Length;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        fixed (float* residualPtr = residual)
        {
            for (int b = 0; b < batch; b++)
                for (int tileStart = 0; tileStart < plane; tileStart += tileSpatial)
                {
                    int tileEnd = Math.Min(plane, tileStart + tileSpatial);
                    for (int block = 0; block < blocks; block++)
                    {
                        int co = block * 8;
                        float* output0 = outputPtr + (b * outputChannels + co) * plane;
                        float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                        float* output4 = output3 + plane, output5 = output4 + plane;
                        float* output6 = output5 + plane, output7 = output6 + plane;
                        Vector512<float> bias0 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co]);
                        Vector512<float> bias1 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                        Vector512<float> bias2 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                        Vector512<float> bias3 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                        Vector512<float> bias4 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 4]);
                        Vector512<float> bias5 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 5]);
                        Vector512<float> bias6 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 6]);
                        Vector512<float> bias7 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 7]);
                        float* blockWeights = weightsPtr + block * inputChannels * 8;
                        int inputBatch = b * inputChannels * plane;
                        int spatial = tileStart;
                        for (; spatial <= tileEnd - 16; spatial += 16)
                        {
                            Vector512<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                            Vector512<float> a4 = bias4, a5 = bias5, a6 = bias6, a7 = bias7;
                            float* inputChannel = inputPtr + inputBatch + spatial;
                            float* w = blockWeights;
                            int ci = 0;
                            for (; ci <= inputChannels - 4; ci += 4)
                            {
                                Vector512<float> v0 = Avx512F.LoadVector512(inputChannel);
                                Vector512<float> v1 = Avx512F.LoadVector512(inputChannel + plane);
                                Vector512<float> v2 = Avx512F.LoadVector512(inputChannel + plane * 2);
                                Vector512<float> v3 = Avx512F.LoadVector512(inputChannel + plane * 3);
                                a0 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 0), a0);
                                a1 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 1), a1);
                                a2 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 2), a2);
                                a3 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 3), a3);
                                a4 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 4), a4);
                                a5 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 5), a5);
                                a6 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 6), a6);
                                a7 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 7), a7);
                                a0 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 8), a0);
                                a1 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 9), a1);
                                a2 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 10), a2);
                                a3 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 11), a3);
                                a4 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 12), a4);
                                a5 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 13), a5);
                                a6 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 14), a6);
                                a7 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 15), a7);
                                a0 = Avx512F.FusedMultiplyAdd(v2, BroadcastWeight512(w + 16), a0);
                                a1 = Avx512F.FusedMultiplyAdd(v2, BroadcastWeight512(w + 17), a1);
                                a2 = Avx512F.FusedMultiplyAdd(v2, BroadcastWeight512(w + 18), a2);
                                a3 = Avx512F.FusedMultiplyAdd(v2, BroadcastWeight512(w + 19), a3);
                                a4 = Avx512F.FusedMultiplyAdd(v2, BroadcastWeight512(w + 20), a4);
                                a5 = Avx512F.FusedMultiplyAdd(v2, BroadcastWeight512(w + 21), a5);
                                a6 = Avx512F.FusedMultiplyAdd(v2, BroadcastWeight512(w + 22), a6);
                                a7 = Avx512F.FusedMultiplyAdd(v2, BroadcastWeight512(w + 23), a7);
                                a0 = Avx512F.FusedMultiplyAdd(v3, BroadcastWeight512(w + 24), a0);
                                a1 = Avx512F.FusedMultiplyAdd(v3, BroadcastWeight512(w + 25), a1);
                                a2 = Avx512F.FusedMultiplyAdd(v3, BroadcastWeight512(w + 26), a2);
                                a3 = Avx512F.FusedMultiplyAdd(v3, BroadcastWeight512(w + 27), a3);
                                a4 = Avx512F.FusedMultiplyAdd(v3, BroadcastWeight512(w + 28), a4);
                                a5 = Avx512F.FusedMultiplyAdd(v3, BroadcastWeight512(w + 29), a5);
                                a6 = Avx512F.FusedMultiplyAdd(v3, BroadcastWeight512(w + 30), a6);
                                a7 = Avx512F.FusedMultiplyAdd(v3, BroadcastWeight512(w + 31), a7);
                                inputChannel += plane * 4; w += 32;
                            }
                            for (; ci <= inputChannels - 2; ci += 2)
                            {
                                Vector512<float> v0 = Avx512F.LoadVector512(inputChannel);
                                Vector512<float> v1 = Avx512F.LoadVector512(inputChannel + plane);
                                a0 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 0), a0);
                                a1 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 1), a1);
                                a2 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 2), a2);
                                a3 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 3), a3);
                                a4 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 4), a4);
                                a5 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 5), a5);
                                a6 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 6), a6);
                                a7 = Avx512F.FusedMultiplyAdd(v0, BroadcastWeight512(w + 7), a7);
                                a0 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 8), a0);
                                a1 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 9), a1);
                                a2 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 10), a2);
                                a3 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 11), a3);
                                a4 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 12), a4);
                                a5 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 13), a5);
                                a6 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 14), a6);
                                a7 = Avx512F.FusedMultiplyAdd(v1, BroadcastWeight512(w + 15), a7);
                                inputChannel += plane * 2; w += 16;
                            }
                            for (; ci < inputChannels; ci++)
                            {
                                Vector512<float> value = Avx512F.LoadVector512(inputChannel);
                                a0 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(w + 0), a0);
                                a1 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(w + 1), a1);
                                a2 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(w + 2), a2);
                                a3 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(w + 3), a3);
                                a4 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(w + 4), a4);
                                a5 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(w + 5), a5);
                                a6 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(w + 6), a6);
                                a7 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(w + 7), a7);
                                inputChannel += plane; w += 8;
                            }
                            if (hasResidual)
                            {
                                float* residual0 = residualPtr + (b * outputChannels + co) * plane;
                                a0 = Avx512F.Add(a0, Avx512F.LoadVector512(residual0 + spatial));
                                a1 = Avx512F.Add(a1, Avx512F.LoadVector512(residual0 + plane + spatial));
                                a2 = Avx512F.Add(a2, Avx512F.LoadVector512(residual0 + plane * 2 + spatial));
                                a3 = Avx512F.Add(a3, Avx512F.LoadVector512(residual0 + plane * 3 + spatial));
                                a4 = Avx512F.Add(a4, Avx512F.LoadVector512(residual0 + plane * 4 + spatial));
                                a5 = Avx512F.Add(a5, Avx512F.LoadVector512(residual0 + plane * 5 + spatial));
                                a6 = Avx512F.Add(a6, Avx512F.LoadVector512(residual0 + plane * 6 + spatial));
                                a7 = Avx512F.Add(a7, Avx512F.LoadVector512(residual0 + plane * 7 + spatial));
                            }
                            Avx512F.Store(output0 + spatial, a0); Avx512F.Store(output1 + spatial, a1);
                            Avx512F.Store(output2 + spatial, a2); Avx512F.Store(output3 + spatial, a3);
                            Avx512F.Store(output4 + spatial, a4); Avx512F.Store(output5 + spatial, a5);
                            Avx512F.Store(output6 + spatial, a6); Avx512F.Store(output7 + spatial, a7);
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
                                inputChannel += plane; w += 8;
                            }
                            if (hasResidual)
                            {
                                float* residual0 = residualPtr + (b * outputChannels + co) * plane;
                                a0 += residual0[spatial];
                                a1 += residual0[plane + spatial];
                                a2 += residual0[plane * 2 + spatial];
                                a3 += residual0[plane * 3 + spatial];
                                a4 += residual0[plane * 4 + spatial];
                                a5 += residual0[plane * 5 + spatial];
                                a6 += residual0[plane * 6 + spatial];
                                a7 += residual0[plane * 7 + spatial];
                            }
                            output0[spatial] = a0; output1[spatial] = a1; output2[spatial] = a2; output3[spatial] = a3;
                            output4[spatial] = a4; output5[spatial] = a5; output6[spatial] = a6; output7[spatial] = a7;
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), blocks = outputChannels / 4;
        // Every four-output block streams the full input plane, so large
        // activations (detector) fall out of L2 once per block.  Tile the
        // spatial axis so one input tile (~192 KB) stays cache-resident
        // across all output blocks; per-element accumulation order is
        // unchanged, so results stay bit-identical.
        int tileSpatial = plane;
        if (blocks > 1 && (long)inputChannels * plane * 4 > 1_048_576)
            tileSpatial = Math.Max(64, 24576 / Math.Max(1, inputChannels) & ~15);
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
                        Vector512<float> bias0 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co]);
                        Vector512<float> bias1 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                        Vector512<float> bias2 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                        Vector512<float> bias3 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                        // 4-OC × dual spatial-16 (= 8 ZMM acc). Zen 5 stays
                        // within the register budget while amortizing weight loads.
                        int spatial = tileStart;
                        for (; spatial <= tileEnd - 32; spatial += 32)
                        {
                            Vector512<float> a0l = bias0; Vector512<float> a0h = bias0;
                            Vector512<float> a1l = bias1; Vector512<float> a1h = bias1;
                            Vector512<float> a2l = bias2; Vector512<float> a2h = bias2;
                            Vector512<float> a3l = bias3; Vector512<float> a3h = bias3;
                            float* inputChannel = inputPtr + inputBatch + spatial;
                            float* weightCursor = weightsPtr + block * inputChannels * 4;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                Vector512<float> valueLow = Avx512F.LoadVector512(inputChannel);
                                Vector512<float> valueHigh = Avx512F.LoadVector512(inputChannel + 16);
                                float* wb = weightCursor;
                                a0l = AddMul512(a0l, valueLow, wb[0]); a0h = AddMul512(a0h, valueHigh, wb[0]);
                                a1l = AddMul512(a1l, valueLow, wb[1]); a1h = AddMul512(a1h, valueHigh, wb[1]);
                                a2l = AddMul512(a2l, valueLow, wb[2]); a2h = AddMul512(a2h, valueHigh, wb[2]);
                                a3l = AddMul512(a3l, valueLow, wb[3]); a3h = AddMul512(a3h, valueHigh, wb[3]);
                                inputChannel += plane;
                                weightCursor += 4;
                            }
                            Avx512F.Store(output0 + spatial, a0l); Avx512F.Store(output0 + spatial + 16, a0h);
                            Avx512F.Store(output1 + spatial, a1l); Avx512F.Store(output1 + spatial + 16, a1h);
                            Avx512F.Store(output2 + spatial, a2l); Avx512F.Store(output2 + spatial + 16, a2h);
                            Avx512F.Store(output3 + spatial, a3l); Avx512F.Store(output3 + spatial + 16, a3h);
                        }
                        for (; spatial <= tileEnd - 16; spatial += 16)
                        {
                            Vector512<float> a0 = bias0; Vector512<float> a1 = bias1; Vector512<float> a2 = bias2; Vector512<float> a3 = bias3;
                            float* inputChannel = inputPtr + inputBatch + spatial;
                            float* weightCursor = weightsPtr + block * inputChannels * 4;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                Vector512<float> value = Avx512F.LoadVector512(inputChannel);
                                float* wb = weightCursor;
                                a0 = AddMul512(a0, value, wb[0]); a1 = AddMul512(a1, value, wb[1]);
                                a2 = AddMul512(a2, value, wb[2]); a3 = AddMul512(a3, value, wb[3]);
                                inputChannel += plane;
                                weightCursor += 4;
                            }
                            Avx512F.Store(output0 + spatial, a0); Avx512F.Store(output1 + spatial, a1);
                            Avx512F.Store(output2 + spatial, a2); Avx512F.Store(output3 + spatial, a3);
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
                                float* wb = weightCursor;
                                a0 += value * wb[0]; a1 += value * wb[1]; a2 += value * wb[2]; a3 += value * wb[3];
                                inputChannel += plane;
                                weightCursor += 4;
                            }
                            output0[spatial] = a0; output1[spatial] = a1; output2[spatial] = a2; output3[spatial] = a3;
                        }
                    }
                }
        }
    }

    private static void Conv1x1FourOutputsAvx512(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels, int height,
        int width, int outputChannels, int groups, int inputPerGroup, int outputPerGroup, int plane)
    {
        for (int b = 0; b < batch; b++)
        {
            int inputBatch = b * inputChannels * plane, outputBatch = b * outputChannels * plane;
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
                    Vector512<float> vBias0 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo]);
                    Vector512<float> vBias1 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 1]);
                    Vector512<float> vBias2 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 2]);
                    Vector512<float> vBias3 = Vector512.Create(bias.IsEmpty ? 0f : bias[globalCo + 3]);
                    // 4-OC × dual spatial-16 (= 8 ZMM acc).
                    int spatial = 0;
                    for (; spatial <= plane - 32; spatial += 32)
                    {
                        Vector512<float> a0l = vBias0; Vector512<float> a0h = vBias0;
                        Vector512<float> a1l = vBias1; Vector512<float> a1h = vBias1;
                        Vector512<float> a2l = vBias2; Vector512<float> a2h = vBias2;
                        Vector512<float> a3l = vBias3; Vector512<float> a3h = vBias3;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            int inputOffset = inputGroup + ci * plane + spatial;
                            Vector512<float> valueLow = Load512(input, inputOffset);
                            Vector512<float> valueHigh = Load512(input, inputOffset + 16);
                            a0l = AddMul512(a0l, valueLow, weights[weight0 + ci]);
                            a0h = AddMul512(a0h, valueHigh, weights[weight0 + ci]);
                            a1l = AddMul512(a1l, valueLow, weights[weight1 + ci]);
                            a1h = AddMul512(a1h, valueHigh, weights[weight1 + ci]);
                            a2l = AddMul512(a2l, valueLow, weights[weight2 + ci]);
                            a2h = AddMul512(a2h, valueHigh, weights[weight2 + ci]);
                            a3l = AddMul512(a3l, valueLow, weights[weight3 + ci]);
                            a3h = AddMul512(a3h, valueHigh, weights[weight3 + ci]);
                        }
                        Store512(output, output0 + spatial, a0l); Store512(output, output0 + spatial + 16, a0h);
                        Store512(output, output1 + spatial, a1l); Store512(output, output1 + spatial + 16, a1h);
                        Store512(output, output2 + spatial, a2l); Store512(output, output2 + spatial + 16, a2h);
                        Store512(output, output3 + spatial, a3l); Store512(output, output3 + spatial + 16, a3h);
                    }
                    for (; spatial <= plane - 16; spatial += 16)
                    {
                        Vector512<float> a0 = vBias0;
                        Vector512<float> a1 = vBias1;
                        Vector512<float> a2 = vBias2;
                        Vector512<float> a3 = vBias3;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            Vector512<float> value = Load512(input, inputGroup + ci * plane + spatial);
                            a0 = AddMul512(a0, value, weights[weight0 + ci]);
                            a1 = AddMul512(a1, value, weights[weight1 + ci]);
                            a2 = AddMul512(a2, value, weights[weight2 + ci]);
                            a3 = AddMul512(a3, value, weights[weight3 + ci]);
                        }
                        Store512(output, output0 + spatial, a0); Store512(output, output1 + spatial, a1);
                        Store512(output, output2 + spatial, a2); Store512(output, output3 + spatial, a3);
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
                    int globalCo = g * outputPerGroup + co, outputOffset = outputGroup + co * plane;
                    float initial = bias.IsEmpty ? 0f : bias[globalCo];
                    int spatial = 0;
                    for (; spatial <= plane - 16; spatial += 16) Store512(output, outputOffset + spatial, Vector512.Create(initial));
                    for (; spatial < plane; spatial++) output[outputOffset + spatial] = initial;
                    int weightBase = globalCo * inputPerGroup;
                    for (int ci = 0; ci < inputPerGroup; ci++)
                    {
                        float weight = weights[weightBase + ci];
                        ReadOnlySpan<float> source = input.Slice(inputGroup + ci * plane, plane);
                        for (spatial = 0; spatial <= plane - 16; spatial += 16)
                            Store512(output, outputOffset + spatial, AddMul512(Load512(output, outputOffset + spatial), Load512(source, spatial), weight));
                        for (; spatial < plane; spatial++) output[outputOffset + spatial] += source[spatial] * weight;
                    }
                }
            }
        }
    }
}
