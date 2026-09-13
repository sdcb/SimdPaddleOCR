using System.Numerics;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Conv1x1
{
#if !NETSTANDARD2_0
    // oneDNN's AVX2 1x1 primitive uses a spatial-major (NHWC) working layout:
    // a vector contains adjacent output channels, while four spatial rows are
    // kept live to hide the scalar input broadcasts.  The graph itself stays
    // NCHW, so this path transposes through ArrayPool scratch for large dense
    // pointwise convolutions.  Bias is the initial accumulator, preserving the
    // same per-output accumulation order as the NCHW kernels.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe bool Conv1x1PackedNhwcAvx2(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedOc8, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int intraOpThreads)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported || Vector<float>.Count != 8 ||
            batch <= 0 || inputChannels < 32 || outputChannels < 16 ||
            (outputChannels & 15) != 0)
            return false;
        int plane = checked(height * width), rows = checked(batch * plane);
        long work = checked((long)inputChannels * outputChannels * plane);
        // Transpose overhead dominates tiny projections and small detector
        // tails.  Keep the existing packed kernel for those shapes.
        if (plane < 32 || plane > 4096 || work < 4_000_000)
            return false;
        int inputVolume = checked(rows * inputChannels);
        int outputVolume = checked(rows * outputChannels);
        float[] inputNhwc = ArrayPool<float>.Shared.Rent(inputVolume);
        float[] outputNhwc = ArrayPool<float>.Shared.Rent(outputVolume);
        try
        {
            fixed (float* inputPtr = input, inputWork = inputNhwc,
                outputWork = outputNhwc, outputPtr = output,
                weightsPtr = packedOc8, biasPtr = bias)
            {
                nint inputAddress = (nint)inputPtr, inputWorkAddress = (nint)inputWork,
                    outputWorkAddress = (nint)outputWork, outputAddress = (nint)outputPtr,
                    weightsAddress = (nint)weightsPtr, biasAddress = (nint)biasPtr;
                // Parallel.For setup is more expensive than the arithmetic for
                // the small pointwise layers in the detector head.  Keep
                // those on the caller thread and reserve nested parallelism
                // for a genuinely compute-heavy projection.
                int workers = intraOpThreads > 1 && work >= 16_000_000
                    ? Math.Min(intraOpThreads, Math.Max(1, rows / 4)) : 1;
                if (workers > 1)
                {
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = (rows * worker / workers) & ~3;
                        int end = worker == workers - 1 ? rows : (rows * (worker + 1) / workers) & ~3;
                        TransposeNchwToNhwc((float*)inputAddress, (float*)inputWorkAddress,
                            begin, end, plane, inputChannels);
                    });
                }
                else
                    TransposeNchwToNhwc(inputPtr, inputWork, 0, rows, plane, inputChannels);

                if (workers > 1)
                {
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = (rows * worker / workers) & ~3;
                        int end = worker == workers - 1 ? rows : (rows * (worker + 1) / workers) & ~3;
                        Conv1x1NhwcRows((float*)inputWorkAddress + begin * inputChannels,
                            (float*)weightsAddress, (float*)biasAddress,
                            (float*)outputWorkAddress + begin * outputChannels,
                            end - begin, inputChannels, outputChannels);
                    });
                }
                else
                    Conv1x1NhwcRows(inputWork, weightsPtr, biasPtr, outputWork,
                        rows, inputChannels, outputChannels);

                if (workers > 1)
                {
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = (rows * worker / workers) & ~3;
                        int end = worker == workers - 1 ? rows : (rows * (worker + 1) / workers) & ~3;
                        TransposeNhwcToNchw((float*)outputWorkAddress, (float*)outputAddress, begin, end,
                            plane, outputChannels);
                    });
                }
                else
                    TransposeNhwcToNchw(outputWork, outputPtr, 0, rows,
                        plane, outputChannels);
            }
            return true;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(inputNhwc);
            ArrayPool<float>.Shared.Return(outputNhwc);
        }
    }

    // Tile the NCHW↔NHWC conversion for very wide planes.  Keeping one tile
    // spatial-major lets the 16-OC body reuse each input value across all
    // output-channel blocks without allocating a full activation transpose.
    // This is primarily for REC (48 x 320+); the whole-plane path above is
    // cheaper for detector-sized planes that fit its bounded scratch budget.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe bool Conv1x1PackedNhwcTiledAvx2(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedOc8, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int intraOpThreads)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported || Vector<float>.Count != 8 ||
            batch <= 0 || inputChannels < 32 || outputChannels < 16 ||
            (outputChannels & 15) != 0)
            return false;
        int plane = checked(height * width);
        long work = checked((long)inputChannels * outputChannels * plane);
        if (plane <= 4096 || work < 4_000_000)
            return false;


        const int tileSpatial = 64;
        int tilesPerBatch = (plane + tileSpatial - 1) / tileSpatial;
        int tileCount = checked(batch * tilesPerBatch);
        int workers = intraOpThreads > 1
            ? Math.Min(intraOpThreads, tileCount) : 1;

        fixed (float* inputPtr = input, weightsPtr = packedOc8,
            biasPtr = bias, outputPtr = output)
        {
            nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
            // The tiled path can have hundreds of tiles for a single REC
            // sample.  Keep one scratch pair per worker and walk that worker's
            // tiles in a strided loop; renting two arrays and scheduling a
            // task for every tile otherwise costs a measurable fraction of
            // the convolution itself.
            Action<int> runWorker = worker =>
            {
                float[] inScratch = ArrayPool<float>.Shared.Rent(checked(tileSpatial * inputChannels));
                float[] outScratch = ArrayPool<float>.Shared.Rent(checked(tileSpatial * outputChannels));
                try
                {
                    fixed (float* inWork = inScratch, outWork = outScratch)
                    {
                        for (int tileIndex = worker; tileIndex < tileCount; tileIndex += workers)
                        {
                            int b = tileIndex / tilesPerBatch;
                            int tile = tileIndex - b * tilesPerBatch;
                            int begin = tile * tileSpatial;
                            int rows = Math.Min(tileSpatial, plane - begin);
                            float* source = (float*)inputAddress + (long)b * inputChannels * plane + begin;
                            // Walk each NCHW channel contiguously while writing
                            // the row-major scratch.  The inverse row-major loop
                            // would create one hardware-prefetch stream per
                            // channel.
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float* src = source + (long)ci * plane;
                                for (int r = 0; r < rows; r++)
                                    inWork[r * inputChannels + ci] = src[r];
                            }
                            Conv1x1NhwcRows(inWork, (float*)weightsAddress, (float*)biasAddress, outWork,
                                rows, inputChannels, outputChannels);
                            float* destination = (float*)outputAddress + (long)b * outputChannels * plane + begin;
                            for (int co = 0; co < outputChannels; co++)
                            {
                                float* dst = destination + (long)co * plane;
                                for (int r = 0; r < rows; r++)
                                    dst[r] = outWork[r * outputChannels + co];
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(inScratch);
                    ArrayPool<float>.Shared.Return(outScratch);
                }
            };
            if (workers > 1)
                Parallel.For(0, workers, runWorker);
            else
                runWorker(0);
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void TransposeNchwToNhwc(float* source, float* destination,
        int rowBegin, int rowEnd, int plane, int channels)
    {
        for (int row = rowBegin; row < rowEnd; row++)
        {
            int batchIndex = row / plane, spatial = row - batchIndex * plane;
            float* src = source + batchIndex * channels * plane + spatial;
            float* dst = destination + row * channels;
            for (int ci = 0; ci < channels; ci++, src += plane)
                dst[ci] = *src;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void TransposeNhwcToNchw(float* source, float* destination,
        int rowBegin, int rowEnd, int plane, int channels)
    {
        for (int row = rowBegin; row < rowEnd; row++)
        {
            int batchIndex = row / plane, spatial = row - batchIndex * plane;
            float* src = source + row * channels;
            float* dst = destination + batchIndex * channels * plane + spatial;
            for (int co = 0; co < channels; co++, dst += plane)
                *dst = src[co];
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1NhwcRows(float* input, float* weights,
        float* bias, float* output, int rows, int inputChannels, int outputChannels)
    {
        // The four-row body has the best register balance on Zen 3/4 AVX2;
        // wider row tiles spill accumulators on this target.
        Conv1x1NhwcRowsFour(input, weights, bias, output, rows, inputChannels, outputChannels);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1NhwcRowsFour(float* input, float* weights,
        float* bias, float* output, int rows, int inputChannels, int outputChannels)
    {
        for (int row = 0; row <= rows - 4; row += 4)
        {
            float* in0 = input + row * inputChannels;
            float* in1 = in0 + inputChannels, in2 = in1 + inputChannels, in3 = in2 + inputChannels;
            float* out0 = output + row * outputChannels;
            for (int co = 0; co < outputChannels; co += 16)
            {
                Vector256<float> biasLow = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias + co);
                Vector256<float> biasHigh = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias + co + 8);
                Vector256<float> a0 = biasLow, a1 = biasLow, a2 = biasLow, a3 = biasLow;
                Vector256<float> b0 = biasHigh, b1 = biasHigh, b2 = biasHigh, b3 = biasHigh;
                float* weight0 = weights + co * inputChannels;
                float* weight1 = weight0 + inputChannels * 8;
                for (int ci = 0; ci < inputChannels; ci++, weight0 += 8, weight1 += 8)
                {
                    Vector256<float> w0 = Avx.LoadVector256(weight0);
                    Vector256<float> w1 = Avx.LoadVector256(weight1);
                    Vector256<float> v0 = Avx.BroadcastScalarToVector256(in0 + ci);
                    Vector256<float> v1 = Avx.BroadcastScalarToVector256(in1 + ci);
                    Vector256<float> v2 = Avx.BroadcastScalarToVector256(in2 + ci);
                    Vector256<float> v3 = Avx.BroadcastScalarToVector256(in3 + ci);
                    a0 = Fma.MultiplyAdd(v0, w0, a0); b0 = Fma.MultiplyAdd(v0, w1, b0);
                    a1 = Fma.MultiplyAdd(v1, w0, a1); b1 = Fma.MultiplyAdd(v1, w1, b1);
                    a2 = Fma.MultiplyAdd(v2, w0, a2); b2 = Fma.MultiplyAdd(v2, w1, b2);
                    a3 = Fma.MultiplyAdd(v3, w0, a3); b3 = Fma.MultiplyAdd(v3, w1, b3);
                }
                Avx.Store(out0 + co, a0); Avx.Store(out0 + co + 8, b0);
                Avx.Store(out0 + co + outputChannels, a1); Avx.Store(out0 + co + outputChannels + 8, b1);
                Avx.Store(out0 + co + outputChannels * 2, a2); Avx.Store(out0 + co + outputChannels * 2 + 8, b2);
                Avx.Store(out0 + co + outputChannels * 3, a3); Avx.Store(out0 + co + outputChannels * 3 + 8, b3);
            }
        }
        for (int row = rows & ~3; row < rows; row++)
        {
            float* inPtr = input + row * inputChannels;
            float* outPtr = output + row * outputChannels;
            for (int co = 0; co < outputChannels; co++)
            {
                float sum = bias == null ? 0f : bias[co];
                float* weight = weights + (co / 8) * inputChannels * 8 + (co & 7);
                for (int ci = 0; ci < inputChannels; ci++, weight += 8)
                    sum += inPtr[ci] * *weight;
                outPtr[co] = sum;
            }
        }
    }

#endif

    // Four adjacent packed four-channel tiles share each input vector.  The
    // sixteen-output kernel is useful for the detector's wide pointwise
    // projections and is restricted to single-threaded execution so the
    // existing four-channel sharding remains available to worker pools.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedSixteenOutputsUnsafe(ReadOnlySpan<float> input,
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
                    Vector256<float> bias0 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co]);
                    Vector256<float> bias1 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                    Vector256<float> bias2 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                    Vector256<float> bias3 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                    Vector256<float> bias4 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 4]);
                    Vector256<float> bias5 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 5]);
                    Vector256<float> bias6 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 6]);
                    Vector256<float> bias7 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 7]);
                    Vector256<float> bias8 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 8]);
                    Vector256<float> bias9 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 9]);
                    Vector256<float> bias10 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 10]);
                    Vector256<float> bias11 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 11]);
                    Vector256<float> bias12 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 12]);
                    Vector256<float> bias13 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 13]);
                    Vector256<float> bias14 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 14]);
                    Vector256<float> bias15 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 15]);
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    float* firstWeights = weightsPtr + (co / 4) * weightsPerBlock;
                    float* secondWeights = firstWeights + weightsPerBlock;
                    float* thirdWeights = secondWeights + weightsPerBlock;
                    float* fourthWeights = thirdWeights + weightsPerBlock;
                    int spatial = 0;
                    for (; spatial <= plane - 8; spatial += 8)
                    {
                        Vector256<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                        Vector256<float> a4 = bias4, a5 = bias5, a6 = bias6, a7 = bias7;
                        Vector256<float> a8 = bias8, a9 = bias9, a10 = bias10, a11 = bias11;
                        Vector256<float> a12 = bias12, a13 = bias13, a14 = bias14, a15 = bias15;
                        float* inputChannel = batchInput + spatial;
                        float* w0 = firstWeights, w1 = secondWeights, w2 = thirdWeights, w3 = fourthWeights;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            Vector256<float> value = Avx.LoadVector256(inputChannel);
                            AddFourPacked(ref a0, ref a1, ref a2, ref a3, value, w0);
                            AddFourPacked(ref a4, ref a5, ref a6, ref a7, value, w1);
                            AddFourPacked(ref a8, ref a9, ref a10, ref a11, value, w2);
                            AddFourPacked(ref a12, ref a13, ref a14, ref a15, value, w3);
                            // The packed layout is [block, input-channel, lane].
                            // Advance both pairs together for the next input channel.
                            inputChannel += plane;
                            w0 += 4; w1 += 4; w2 += 4; w3 += 4;
                        }
                        Avx.Store(output0 + spatial, a0); Avx.Store(output1 + spatial, a1);
                        Avx.Store(output2 + spatial, a2); Avx.Store(output3 + spatial, a3);
                        Avx.Store(output4 + spatial, a4); Avx.Store(output5 + spatial, a5);
                        Avx.Store(output6 + spatial, a6); Avx.Store(output7 + spatial, a7);
                        Avx.Store(output8 + spatial, a8); Avx.Store(output9 + spatial, a9);
                        Avx.Store(output10 + spatial, a10); Avx.Store(output11 + spatial, a11);
                        Avx.Store(output12 + spatial, a12); Avx.Store(output13 + spatial, a13);
                        Avx.Store(output14 + spatial, a14); Avx.Store(output15 + spatial, a15);
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
    private static void Conv1x1EightOutputs(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
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
                    Vector256<float> vBias0 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo]);
                    Vector256<float> vBias1 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 1]);
                    Vector256<float> vBias2 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 2]);
                    Vector256<float> vBias3 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 3]);
                    Vector256<float> vBias4 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 4]);
                    Vector256<float> vBias5 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 5]);
                    Vector256<float> vBias6 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 6]);
                    Vector256<float> vBias7 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 7]);
                    int spatial = 0;
                    for (; spatial <= plane - 16; spatial += 16)
                    {
                        Vector256<float> a0l = vBias0, a0h = vBias0, a1l = vBias1, a1h = vBias1;
                        Vector256<float> a2l = vBias2, a2h = vBias2, a3l = vBias3, a3h = vBias3;
                        Vector256<float> a4l = vBias4, a4h = vBias4, a5l = vBias5, a5h = vBias5;
                        Vector256<float> a6l = vBias6, a6h = vBias6, a7l = vBias7, a7h = vBias7;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            int inputOffset = inputGroup + ci * plane + spatial;
                            Vector256<float> valueLow = Load(input, inputOffset);
                            Vector256<float> valueHigh = Load(input, inputOffset + 8);
                            a0l = AddMul(a0l, valueLow, weights[weight0 + ci]);
                            a0h = AddMul(a0h, valueHigh, weights[weight0 + ci]);
                            a1l = AddMul(a1l, valueLow, weights[weight1 + ci]);
                            a1h = AddMul(a1h, valueHigh, weights[weight1 + ci]);
                            a2l = AddMul(a2l, valueLow, weights[weight2 + ci]);
                            a2h = AddMul(a2h, valueHigh, weights[weight2 + ci]);
                            a3l = AddMul(a3l, valueLow, weights[weight3 + ci]);
                            a3h = AddMul(a3h, valueHigh, weights[weight3 + ci]);
                            a4l = AddMul(a4l, valueLow, weights[weight4 + ci]);
                            a4h = AddMul(a4h, valueHigh, weights[weight4 + ci]);
                            a5l = AddMul(a5l, valueLow, weights[weight5 + ci]);
                            a5h = AddMul(a5h, valueHigh, weights[weight5 + ci]);
                            a6l = AddMul(a6l, valueLow, weights[weight6 + ci]);
                            a6h = AddMul(a6h, valueHigh, weights[weight6 + ci]);
                            a7l = AddMul(a7l, valueLow, weights[weight7 + ci]);
                            a7h = AddMul(a7h, valueHigh, weights[weight7 + ci]);
                        }
                        Store(output, output0 + spatial, a0l); Store(output, output0 + spatial + 8, a0h);
                        Store(output, output1 + spatial, a1l); Store(output, output1 + spatial + 8, a1h);
                        Store(output, output2 + spatial, a2l); Store(output, output2 + spatial + 8, a2h);
                        Store(output, output3 + spatial, a3l); Store(output, output3 + spatial + 8, a3h);
                        Store(output, output4 + spatial, a4l); Store(output, output4 + spatial + 8, a4h);
                        Store(output, output5 + spatial, a5l); Store(output, output5 + spatial + 8, a5h);
                        Store(output, output6 + spatial, a6l); Store(output, output6 + spatial + 8, a6h);
                        Store(output, output7 + spatial, a7l); Store(output, output7 + spatial + 8, a7h);
                    }
                    for (; spatial <= plane - 8; spatial += 8)
                    {
                        Vector256<float> a0 = vBias0, a1 = vBias1, a2 = vBias2, a3 = vBias3;
                        Vector256<float> a4 = vBias4, a5 = vBias5, a6 = vBias6, a7 = vBias7;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            Vector256<float> value = Load(input, inputGroup + ci * plane + spatial);
                            a0 = AddMul(a0, value, weights[weight0 + ci]);
                            a1 = AddMul(a1, value, weights[weight1 + ci]);
                            a2 = AddMul(a2, value, weights[weight2 + ci]);
                            a3 = AddMul(a3, value, weights[weight3 + ci]);
                            a4 = AddMul(a4, value, weights[weight4 + ci]);
                            a5 = AddMul(a5, value, weights[weight5 + ci]);
                            a6 = AddMul(a6, value, weights[weight6 + ci]);
                            a7 = AddMul(a7, value, weights[weight7 + ci]);
                        }
                        Store(output, output0 + spatial, a0); Store(output, output1 + spatial, a1);
                        Store(output, output2 + spatial, a2); Store(output, output3 + spatial, a3);
                        Store(output, output4 + spatial, a4); Store(output, output5 + spatial, a5);
                        Store(output, output6 + spatial, a6); Store(output, output7 + spatial, a7);
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
    private static void Conv1x1SixteenOutputs(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
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
                    Vector256<float> vBias0 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo]);
                    Vector256<float> vBias1 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 1]);
                    Vector256<float> vBias2 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 2]);
                    Vector256<float> vBias3 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 3]);
                    Vector256<float> vBias4 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 4]);
                    Vector256<float> vBias5 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 5]);
                    Vector256<float> vBias6 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 6]);
                    Vector256<float> vBias7 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 7]);
                    Vector256<float> vBias8 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 8]);
                    Vector256<float> vBias9 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 9]);
                    Vector256<float> vBias10 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 10]);
                    Vector256<float> vBias11 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 11]);
                    Vector256<float> vBias12 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 12]);
                    Vector256<float> vBias13 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 13]);
                    Vector256<float> vBias14 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 14]);
                    Vector256<float> vBias15 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 15]);
                    int spatial = 0;
                    for (; spatial <= plane - 8; spatial += 8)
                    {
                        Vector256<float> a0 = vBias0, a1 = vBias1, a2 = vBias2, a3 = vBias3;
                        Vector256<float> a4 = vBias4, a5 = vBias5, a6 = vBias6, a7 = vBias7;
                        Vector256<float> a8 = vBias8, a9 = vBias9, a10 = vBias10, a11 = vBias11;
                        Vector256<float> a12 = vBias12, a13 = vBias13, a14 = vBias14, a15 = vBias15;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            Vector256<float> value = Load(input, inputGroup + ci * plane + spatial);
                            a0 = AddMul(a0, value, weights[weight0 + ci]);
                            a1 = AddMul(a1, value, weights[weight1 + ci]);
                            a2 = AddMul(a2, value, weights[weight2 + ci]);
                            a3 = AddMul(a3, value, weights[weight3 + ci]);
                            a4 = AddMul(a4, value, weights[weight4 + ci]);
                            a5 = AddMul(a5, value, weights[weight5 + ci]);
                            a6 = AddMul(a6, value, weights[weight6 + ci]);
                            a7 = AddMul(a7, value, weights[weight7 + ci]);
                            a8 = AddMul(a8, value, weights[weight8 + ci]);
                            a9 = AddMul(a9, value, weights[weight9 + ci]);
                            a10 = AddMul(a10, value, weights[weight10 + ci]);
                            a11 = AddMul(a11, value, weights[weight11 + ci]);
                            a12 = AddMul(a12, value, weights[weight12 + ci]);
                            a13 = AddMul(a13, value, weights[weight13 + ci]);
                            a14 = AddMul(a14, value, weights[weight14 + ci]);
                            a15 = AddMul(a15, value, weights[weight15 + ci]);
                        }
                        Store(output, output0 + spatial, a0); Store(output, output1 + spatial, a1);
                        Store(output, output2 + spatial, a2); Store(output, output3 + spatial, a3);
                        Store(output, output4 + spatial, a4); Store(output, output5 + spatial, a5);
                        Store(output, output6 + spatial, a6); Store(output, output7 + spatial, a7);
                        Store(output, output8 + spatial, a8); Store(output, output9 + spatial, a9);
                        Store(output, output10 + spatial, a10); Store(output, output11 + spatial, a11);
                        Store(output, output12 + spatial, a12); Store(output, output13 + spatial, a13);
                        Store(output, output14 + spatial, a14); Store(output, output15 + spatial, a15);
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
    private static unsafe void Conv1x1PackedEightOutputsUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), blocks = outputChannels / 4;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int block = 0; block < blocks; block += 2)
                {
                    int co = block * 4;
                    float* output0 = outputPtr + (b * outputChannels + co) * plane;
                    float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                    float* output4 = output3 + plane, output5 = output4 + plane;
                    float* output6 = output5 + plane, output7 = output6 + plane;
                    Vector256<float> bias0 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co]);
                    Vector256<float> bias1 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                    Vector256<float> bias2 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                    Vector256<float> bias3 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                    Vector256<float> bias4 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 4]);
                    Vector256<float> bias5 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 5]);
                    Vector256<float> bias6 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 6]);
                    Vector256<float> bias7 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 7]);
                    float* firstWeights = weightsPtr + block * inputChannels * 4;
                    float* secondWeights = firstWeights + inputChannels * 4;
                    int inputBatch = b * inputChannels * plane;
                    int spatial = 0;
                    for (; spatial <= plane - 8; spatial += 8)
                    {
                        Vector256<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                        Vector256<float> a4 = bias4, a5 = bias5, a6 = bias6, a7 = bias7;
                        float* inputChannel = inputPtr + inputBatch + spatial;
                        float* w0 = firstWeights, w4 = secondWeights;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            Vector256<float> value = Avx.LoadVector256(inputChannel);
                            a0 = AddMul(a0, value, w0[0]); a1 = AddMul(a1, value, w0[1]);
                            a2 = AddMul(a2, value, w0[2]); a3 = AddMul(a3, value, w0[3]);
                            a4 = AddMul(a4, value, w4[0]); a5 = AddMul(a5, value, w4[1]);
                            a6 = AddMul(a6, value, w4[2]); a7 = AddMul(a7, value, w4[3]);
                            inputChannel += plane; w0 += 4; w4 += 4;
                        }
                        Avx.Store(output0 + spatial, a0); Avx.Store(output1 + spatial, a1);
                        Avx.Store(output2 + spatial, a2); Avx.Store(output3 + spatial, a3);
                        Avx.Store(output4 + spatial, a4); Avx.Store(output5 + spatial, a5);
                        Avx.Store(output6 + spatial, a6); Avx.Store(output7 + spatial, a7);
                    }
                    for (; spatial < plane; spatial++)
                    {
                        float a0 = biasPtr == null ? 0f : biasPtr[co], a1 = biasPtr == null ? 0f : biasPtr[co + 1];
                        float a2 = biasPtr == null ? 0f : biasPtr[co + 2], a3 = biasPtr == null ? 0f : biasPtr[co + 3];
                        float a4 = biasPtr == null ? 0f : biasPtr[co + 4], a5 = biasPtr == null ? 0f : biasPtr[co + 5];
                        float a6 = biasPtr == null ? 0f : biasPtr[co + 6], a7 = biasPtr == null ? 0f : biasPtr[co + 7];
                        float* inputChannel = inputPtr + inputBatch + spatial;
                        float* w0 = firstWeights, w4 = secondWeights;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            float value = *inputChannel;
                            a0 += value * w0[0]; a1 += value * w0[1]; a2 += value * w0[2]; a3 += value * w0[3];
                            a4 += value * w4[0]; a5 += value * w4[1]; a6 += value * w4[2]; a7 += value * w4[3];
                            inputChannel += plane; w0 += 4; w4 += 4;
                        }
                        output0[spatial] = a0; output1[spatial] = a1; output2[spatial] = a2; output3[spatial] = a3;
                        output4[spatial] = a4; output5[spatial] = a5; output6[spatial] = a6; output7[spatial] = a7;
                    }
                }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int spatialBegin = 0, int spatialEnd = -1)
    {
        int plane = checked(height * width), blocks = outputChannels / 4;
        // Every four-output block streams the full input plane, so large
        // activations (detector) fall out of L2 once per block.  Tile the
        // spatial axis so one input tile (~192 KB) stays cache-resident
        // across all output blocks; per-element accumulation order is
        // unchanged, so results stay bit-identical.
        int tileSpatial = plane;
        if (blocks > 1 && (long)inputChannels * plane * 4 > 1_048_576)
            tileSpatial = Math.Max(64, 49152 / inputChannels & ~15);
        if (spatialEnd < 0) spatialEnd = plane;
        if ((uint)spatialBegin > (uint)spatialEnd || spatialEnd > plane)
            throw new ArgumentOutOfRangeException(nameof(spatialBegin));
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int tileStart = spatialBegin; tileStart < spatialEnd; tileStart += tileSpatial)
                {
                    int tileEnd = Math.Min(spatialEnd, tileStart + tileSpatial);
                    for (int block = 0; block < blocks; block++)
                    {
                        int co = block * 4, inputBatch = b * inputChannels * plane;
                        float* output0 = outputPtr + (b * outputChannels + co) * plane;
                        float* output1 = output0 + plane, output2 = output1 + plane, output3 = output2 + plane;
                        Vector256<float> bias0 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co]);
                        Vector256<float> bias1 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                        Vector256<float> bias2 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                        Vector256<float> bias3 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                        int spatial = tileStart;
                        for (; spatial <= tileEnd - 16; spatial += 16)
                        {
                            Vector256<float> a0l = bias0; Vector256<float> a0h = bias0; Vector256<float> a1l = bias1; Vector256<float> a1h = bias1;
                            Vector256<float> a2l = bias2; Vector256<float> a2h = bias2; Vector256<float> a3l = bias3; Vector256<float> a3h = bias3;
                            float* inputChannel = inputPtr + inputBatch + spatial;
                            float* weightCursor = weightsPtr + block * inputChannels * 4;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                Vector256<float> valueLow = Avx.LoadVector256(inputChannel);
                                Vector256<float> valueHigh = Avx.LoadVector256(inputChannel + 8);
                                float* wb = weightCursor;
                                a0l = AddMul(a0l, valueLow, wb[0]); a0h = AddMul(a0h, valueHigh, wb[0]);
                                a1l = AddMul(a1l, valueLow, wb[1]); a1h = AddMul(a1h, valueHigh, wb[1]);
                                a2l = AddMul(a2l, valueLow, wb[2]); a2h = AddMul(a2h, valueHigh, wb[2]);
                                a3l = AddMul(a3l, valueLow, wb[3]); a3h = AddMul(a3h, valueHigh, wb[3]);
                                inputChannel += plane;
                                weightCursor += 4;
                            }
                            Avx.Store(output0 + spatial, a0l); Avx.Store(output0 + spatial + 8, a0h);
                            Avx.Store(output1 + spatial, a1l); Avx.Store(output1 + spatial + 8, a1h);
                            Avx.Store(output2 + spatial, a2l); Avx.Store(output2 + spatial + 8, a2h);
                            Avx.Store(output3 + spatial, a3l); Avx.Store(output3 + spatial + 8, a3h);
                        }
                        for (; spatial <= tileEnd - 8; spatial += 8)
                        {
                            Vector256<float> a0 = bias0; Vector256<float> a1 = bias1; Vector256<float> a2 = bias2; Vector256<float> a3 = bias3;
                            float* inputChannel = inputPtr + inputBatch + spatial;
                            float* weightCursor = weightsPtr + block * inputChannels * 4;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                Vector256<float> value = Avx.LoadVector256(inputChannel);
                                float* wb = weightCursor;
                                a0 = AddMul(a0, value, wb[0]); a1 = AddMul(a1, value, wb[1]);
                                a2 = AddMul(a2, value, wb[2]); a3 = AddMul(a3, value, wb[3]);
                                inputChannel += plane;
                                weightCursor += 4;
                            }
                            Avx.Store(output0 + spatial, a0); Avx.Store(output1 + spatial, a1);
                            Avx.Store(output2 + spatial, a2); Avx.Store(output3 + spatial, a3);
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

    private static void Conv1x1FourOutputs(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
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
                    Vector256<float> vBias0 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo]);
                    Vector256<float> vBias1 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 1]);
                    Vector256<float> vBias2 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 2]);
                    Vector256<float> vBias3 = Vector256.Create(bias.IsEmpty ? 0f : bias[globalCo + 3]);
                    int spatial = 0;
                    // Keep two spatial vectors live per output plane on x64,
                    // matching the C AVX2 packed kernel and halving weight
                    // broadcast/loop overhead for the common 1x1 case.
                    for (; spatial <= plane - 16; spatial += 16)
                    {
                        Vector256<float> a0l = vBias0; Vector256<float> a0h = vBias0;
                        Vector256<float> a1l = vBias1; Vector256<float> a1h = vBias1;
                        Vector256<float> a2l = vBias2; Vector256<float> a2h = vBias2;
                        Vector256<float> a3l = vBias3; Vector256<float> a3h = vBias3;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            int inputOffset = inputGroup + ci * plane + spatial;
                            Vector256<float> valueLow = Load(input, inputOffset);
                            Vector256<float> valueHigh = Load(input, inputOffset + 8);
                            a0l = AddMul(a0l, valueLow, weights[weight0 + ci]);
                            a0h = AddMul(a0h, valueHigh, weights[weight0 + ci]);
                            a1l = AddMul(a1l, valueLow, weights[weight1 + ci]);
                            a1h = AddMul(a1h, valueHigh, weights[weight1 + ci]);
                            a2l = AddMul(a2l, valueLow, weights[weight2 + ci]);
                            a2h = AddMul(a2h, valueHigh, weights[weight2 + ci]);
                            a3l = AddMul(a3l, valueLow, weights[weight3 + ci]);
                            a3h = AddMul(a3h, valueHigh, weights[weight3 + ci]);
                        }
                        Store(output, output0 + spatial, a0l); Store(output, output0 + spatial + 8, a0h);
                        Store(output, output1 + spatial, a1l); Store(output, output1 + spatial + 8, a1h);
                        Store(output, output2 + spatial, a2l); Store(output, output2 + spatial + 8, a2h);
                        Store(output, output3 + spatial, a3l); Store(output, output3 + spatial + 8, a3h);
                    }
                    for (; spatial <= plane - 8; spatial += 8)
                    {
                        Vector256<float> a0 = vBias0;
                        Vector256<float> a1 = vBias1;
                        Vector256<float> a2 = vBias2;
                        Vector256<float> a3 = vBias3;
                        for (int ci = 0; ci < inputPerGroup; ci++)
                        {
                            Vector256<float> value = Load(input, inputGroup + ci * plane + spatial);
                            a0 = AddMul(a0, value, weights[weight0 + ci]);
                            a1 = AddMul(a1, value, weights[weight1 + ci]);
                            a2 = AddMul(a2, value, weights[weight2 + ci]);
                            a3 = AddMul(a3, value, weights[weight3 + ci]);
                        }
                        Store(output, output0 + spatial, a0); Store(output, output1 + spatial, a1);
                        Store(output, output2 + spatial, a2); Store(output, output3 + spatial, a3);
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
                    for (; spatial <= plane - 8; spatial += 8) Store(output, outputOffset + spatial, Vector256.Create(initial));
                    for (; spatial < plane; spatial++) output[outputOffset + spatial] = initial;
                    int weightBase = globalCo * inputPerGroup;
                    for (int ci = 0; ci < inputPerGroup; ci++)
                    {
                        float weight = weights[weightBase + ci];
                        ReadOnlySpan<float> source = input.Slice(inputGroup + ci * plane, plane);
                        for (spatial = 0; spatial <= plane - 8; spatial += 8)
                            Store(output, outputOffset + spatial, AddMul(Load(output, outputOffset + spatial), Load(source, spatial), weight));
                        for (; spatial < plane; spatial++) output[outputOffset + spatial] += source[spatial] * weight;
                    }
                }
            }
        }
    }
}
