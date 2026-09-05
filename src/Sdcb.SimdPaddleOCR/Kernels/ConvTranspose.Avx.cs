using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class ConvTranspose
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExpandTranspose2x256(Vector256<float> values,
        out Vector256<float> evenLow, out Vector256<float> evenHigh,
        out Vector256<float> oddLow, out Vector256<float> oddHigh)
    {
        Vector256<float> zero = Vector256<float>.Zero;
        Vector256<float> evenLo = Avx.UnpackLow(values, zero), evenHi = Avx.UnpackHigh(values, zero);
        Vector256<float> oddLo = Avx.UnpackLow(zero, values), oddHi = Avx.UnpackHigh(zero, values);
        evenLow = Avx2.Permute2x128(evenLo, evenHi, 0x20);
        evenHigh = Avx2.Permute2x128(evenLo, evenHi, 0x31);
        oddLow = Avx2.Permute2x128(oddLo, oddHi, 0x20);
        oddHigh = Avx2.Permute2x128(oddLo, oddHi, 0x31);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ConvTranspose2x2Stride2EightOutputsUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputHeight = checked(inputHeight * 2);
        int outputWidth = checked(inputWidth * 2), outputPlane = checked(outputHeight * outputWidth);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* d0 = outputPtr + (b * outputChannels + co) * outputPlane;
                    float* d1 = d0 + outputPlane, d2 = d1 + outputPlane, d3 = d2 + outputPlane;
                    float* d4 = d3 + outputPlane, d5 = d4 + outputPlane, d6 = d5 + outputPlane, d7 = d6 + outputPlane;
                    float v0 = biasPtr == null ? 0f : biasPtr[co], v1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float v2 = biasPtr == null ? 0f : biasPtr[co + 2], v3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float v4 = biasPtr == null ? 0f : biasPtr[co + 4], v5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float v6 = biasPtr == null ? 0f : biasPtr[co + 6], v7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    Vector256<float> b0 = Vector256.Create(v0), b1 = Vector256.Create(v1);
                    Vector256<float> b2 = Vector256.Create(v2), b3 = Vector256.Create(v3);
                    Vector256<float> b4 = Vector256.Create(v4), b5 = Vector256.Create(v5);
                    Vector256<float> b6 = Vector256.Create(v6), b7 = Vector256.Create(v7);
                    int inputBatch = b * inputChannels * inputPlane;
                    for (int i = 0; i < outputPlane; i += 8)
                    {
                        Avx.Store(d0 + i, b0); Avx.Store(d1 + i, b1); Avx.Store(d2 + i, b2); Avx.Store(d3 + i, b3);
                        Avx.Store(d4 + i, b4); Avx.Store(d5 + i, b5); Avx.Store(d6 + i, b6); Avx.Store(d7 + i, b7);
                    }
                    for (int ci = 0; ci < inputChannels; ci++)
                    {
                        float* src = inputPtr + inputBatch + ci * inputPlane;
                        int wb0 = (ci * outputChannels + co) * 4;
                        int wb1 = wb0 + 4, wb2 = wb1 + 4, wb3 = wb2 + 4;
                        int wb4 = wb3 + 4, wb5 = wb4 + 4, wb6 = wb5 + 4, wb7 = wb6 + 4;
                        for (int iy = 0; iy < inputHeight; iy++)
                        {
                            int inputRow = iy * inputWidth;
                            int outputRow0 = (iy * 2) * outputWidth, outputRow1 = outputRow0 + outputWidth;
                            int ix = 0;
                            for (; ix <= inputWidth - 8; ix += 8)
                            {
                                Vector256<float> values = Avx.LoadVector256(src + inputRow + ix);
                                Vector256<float> zero = Vector256<float>.Zero;
                                Vector256<float> evenLo = Avx.UnpackLow(values, zero), evenHi = Avx.UnpackHigh(values, zero);
                                Vector256<float> oddLo = Avx.UnpackLow(zero, values), oddHi = Avx.UnpackHigh(zero, values);
                                Vector256<float> evenLow = Avx2.Permute2x128(evenLo, evenHi, 0x20);
                                Vector256<float> evenHigh = Avx2.Permute2x128(evenLo, evenHi, 0x31);
                                Vector256<float> oddLow = Avx2.Permute2x128(oddLo, oddHi, 0x20);
                                Vector256<float> oddHigh = Avx2.Permute2x128(oddLo, oddHi, 0x31);
                                int ox = ix * 2;
                                float w00 = weights[wb0], w01 = weights[wb0 + 1], w02 = weights[wb0 + 2], w03 = weights[wb0 + 3];
                                float w10 = weights[wb1], w11 = weights[wb1 + 1], w12 = weights[wb1 + 2], w13 = weights[wb1 + 3];
                                float w20 = weights[wb2], w21 = weights[wb2 + 1], w22 = weights[wb2 + 2], w23 = weights[wb2 + 3];
                                float w30 = weights[wb3], w31 = weights[wb3 + 1], w32 = weights[wb3 + 2], w33 = weights[wb3 + 3];
                                float w40 = weights[wb4], w41 = weights[wb4 + 1], w42 = weights[wb4 + 2], w43 = weights[wb4 + 3];
                                float w50 = weights[wb5], w51 = weights[wb5 + 1], w52 = weights[wb5 + 2], w53 = weights[wb5 + 3];
                                float w60 = weights[wb6], w61 = weights[wb6 + 1], w62 = weights[wb6 + 2], w63 = weights[wb6 + 3];
                                float w70 = weights[wb7], w71 = weights[wb7 + 1], w72 = weights[wb7 + 2], w73 = weights[wb7 + 3];
                                AddStorePtr(d0 + outputRow0 + ox, evenLow, w00); AddStorePtr(d0 + outputRow0 + ox + 8, evenHigh, w00);
                                AddStorePtr(d0 + outputRow0 + ox, oddLow, w01); AddStorePtr(d0 + outputRow0 + ox + 8, oddHigh, w01);
                                AddStorePtr(d0 + outputRow1 + ox, evenLow, w02); AddStorePtr(d0 + outputRow1 + ox + 8, evenHigh, w02);
                                AddStorePtr(d0 + outputRow1 + ox, oddLow, w03); AddStorePtr(d0 + outputRow1 + ox + 8, oddHigh, w03);
                                AddStorePtr(d1 + outputRow0 + ox, evenLow, w10); AddStorePtr(d1 + outputRow0 + ox + 8, evenHigh, w10);
                                AddStorePtr(d1 + outputRow0 + ox, oddLow, w11); AddStorePtr(d1 + outputRow0 + ox + 8, oddHigh, w11);
                                AddStorePtr(d1 + outputRow1 + ox, evenLow, w12); AddStorePtr(d1 + outputRow1 + ox + 8, evenHigh, w12);
                                AddStorePtr(d1 + outputRow1 + ox, oddLow, w13); AddStorePtr(d1 + outputRow1 + ox + 8, oddHigh, w13);
                                AddStorePtr(d2 + outputRow0 + ox, evenLow, w20); AddStorePtr(d2 + outputRow0 + ox + 8, evenHigh, w20);
                                AddStorePtr(d2 + outputRow0 + ox, oddLow, w21); AddStorePtr(d2 + outputRow0 + ox + 8, oddHigh, w21);
                                AddStorePtr(d2 + outputRow1 + ox, evenLow, w22); AddStorePtr(d2 + outputRow1 + ox + 8, evenHigh, w22);
                                AddStorePtr(d2 + outputRow1 + ox, oddLow, w23); AddStorePtr(d2 + outputRow1 + ox + 8, oddHigh, w23);
                                AddStorePtr(d3 + outputRow0 + ox, evenLow, w30); AddStorePtr(d3 + outputRow0 + ox + 8, evenHigh, w30);
                                AddStorePtr(d3 + outputRow0 + ox, oddLow, w31); AddStorePtr(d3 + outputRow0 + ox + 8, oddHigh, w31);
                                AddStorePtr(d3 + outputRow1 + ox, evenLow, w32); AddStorePtr(d3 + outputRow1 + ox + 8, evenHigh, w32);
                                AddStorePtr(d3 + outputRow1 + ox, oddLow, w33); AddStorePtr(d3 + outputRow1 + ox + 8, oddHigh, w33);
                                AddStorePtr(d4 + outputRow0 + ox, evenLow, w40); AddStorePtr(d4 + outputRow0 + ox + 8, evenHigh, w40);
                                AddStorePtr(d4 + outputRow0 + ox, oddLow, w41); AddStorePtr(d4 + outputRow0 + ox + 8, oddHigh, w41);
                                AddStorePtr(d4 + outputRow1 + ox, evenLow, w42); AddStorePtr(d4 + outputRow1 + ox + 8, evenHigh, w42);
                                AddStorePtr(d4 + outputRow1 + ox, oddLow, w43); AddStorePtr(d4 + outputRow1 + ox + 8, oddHigh, w43);
                                AddStorePtr(d5 + outputRow0 + ox, evenLow, w50); AddStorePtr(d5 + outputRow0 + ox + 8, evenHigh, w50);
                                AddStorePtr(d5 + outputRow0 + ox, oddLow, w51); AddStorePtr(d5 + outputRow0 + ox + 8, oddHigh, w51);
                                AddStorePtr(d5 + outputRow1 + ox, evenLow, w52); AddStorePtr(d5 + outputRow1 + ox + 8, evenHigh, w52);
                                AddStorePtr(d5 + outputRow1 + ox, oddLow, w53); AddStorePtr(d5 + outputRow1 + ox + 8, oddHigh, w53);
                                AddStorePtr(d6 + outputRow0 + ox, evenLow, w60); AddStorePtr(d6 + outputRow0 + ox + 8, evenHigh, w60);
                                AddStorePtr(d6 + outputRow0 + ox, oddLow, w61); AddStorePtr(d6 + outputRow0 + ox + 8, oddHigh, w61);
                                AddStorePtr(d6 + outputRow1 + ox, evenLow, w62); AddStorePtr(d6 + outputRow1 + ox + 8, evenHigh, w62);
                                AddStorePtr(d6 + outputRow1 + ox, oddLow, w63); AddStorePtr(d6 + outputRow1 + ox + 8, oddHigh, w63);
                                AddStorePtr(d7 + outputRow0 + ox, evenLow, w70); AddStorePtr(d7 + outputRow0 + ox + 8, evenHigh, w70);
                                AddStorePtr(d7 + outputRow0 + ox, oddLow, w71); AddStorePtr(d7 + outputRow0 + ox + 8, oddHigh, w71);
                                AddStorePtr(d7 + outputRow1 + ox, evenLow, w72); AddStorePtr(d7 + outputRow1 + ox + 8, evenHigh, w72);
                                AddStorePtr(d7 + outputRow1 + ox, oddLow, w73); AddStorePtr(d7 + outputRow1 + ox + 8, oddHigh, w73);
                            }
                            for (; ix < inputWidth; ix++)
                            {
                                float value = src[inputRow + ix]; int ox = ix * 2;
                                d0[outputRow0 + ox] += value * weights[wb0]; d0[outputRow0 + ox + 1] += value * weights[wb0 + 1];
                                d0[outputRow1 + ox] += value * weights[wb0 + 2]; d0[outputRow1 + ox + 1] += value * weights[wb0 + 3];
                                d1[outputRow0 + ox] += value * weights[wb1]; d1[outputRow0 + ox + 1] += value * weights[wb1 + 1];
                                d1[outputRow1 + ox] += value * weights[wb1 + 2]; d1[outputRow1 + ox + 1] += value * weights[wb1 + 3];
                                d2[outputRow0 + ox] += value * weights[wb2]; d2[outputRow0 + ox + 1] += value * weights[wb2 + 1];
                                d2[outputRow1 + ox] += value * weights[wb2 + 2]; d2[outputRow1 + ox + 1] += value * weights[wb2 + 3];
                                d3[outputRow0 + ox] += value * weights[wb3]; d3[outputRow0 + ox + 1] += value * weights[wb3 + 1];
                                d3[outputRow1 + ox] += value * weights[wb3 + 2]; d3[outputRow1 + ox + 1] += value * weights[wb3 + 3];
                                d4[outputRow0 + ox] += value * weights[wb4]; d4[outputRow0 + ox + 1] += value * weights[wb4 + 1];
                                d4[outputRow1 + ox] += value * weights[wb4 + 2]; d4[outputRow1 + ox + 1] += value * weights[wb4 + 3];
                                d5[outputRow0 + ox] += value * weights[wb5]; d5[outputRow0 + ox + 1] += value * weights[wb5 + 1];
                                d5[outputRow1 + ox] += value * weights[wb5 + 2]; d5[outputRow1 + ox + 1] += value * weights[wb5 + 3];
                                d6[outputRow0 + ox] += value * weights[wb6]; d6[outputRow0 + ox + 1] += value * weights[wb6 + 1];
                                d6[outputRow1 + ox] += value * weights[wb6 + 2]; d6[outputRow1 + ox + 1] += value * weights[wb6 + 3];
                                d7[outputRow0 + ox] += value * weights[wb7]; d7[outputRow0 + ox + 1] += value * weights[wb7 + 1];
                                d7[outputRow1 + ox] += value * weights[wb7 + 2]; d7[outputRow1 + ox + 1] += value * weights[wb7 + 3];
                            }
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ConvTranspose2x2Stride2Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputHeight = checked(inputHeight * 2);
        int outputWidth = checked(inputWidth * 2), outputPlane = checked(outputHeight * outputWidth);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co++)
                {
                    float* dst = outputPtr + (b * outputChannels + co) * outputPlane;
                    float initial = biasPtr == null ? 0f : biasPtr[co];
                    Vector256<float> vb = Vector256.Create(initial);
                    int i = 0;
                    for (; i <= outputPlane - 8; i += 8) Avx.Store(dst + i, vb);
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
                            for (; ix <= inputWidth - 8; ix += 8)
                            {
                                Vector256<float> values = Avx.LoadVector256(src + inputRow + ix);
                                Vector256<float> zero = Vector256<float>.Zero;
                                Vector256<float> evenLo = Avx.UnpackLow(values, zero); Vector256<float> evenHi = Avx.UnpackHigh(values, zero);
                                Vector256<float> oddLo = Avx.UnpackLow(zero, values); Vector256<float> oddHi = Avx.UnpackHigh(zero, values);
                                Vector256<float> evenLow = Avx2.Permute2x128(evenLo, evenHi, 0x20);
                                Vector256<float> evenHigh = Avx2.Permute2x128(evenLo, evenHi, 0x31);
                                Vector256<float> oddLow = Avx2.Permute2x128(oddLo, oddHi, 0x20);
                                Vector256<float> oddHigh = Avx2.Permute2x128(oddLo, oddHi, 0x31);
                                int ox = ix * 2;
                                AddStorePtr(dst + outputRow0 + ox, evenLow, w[0]); AddStorePtr(dst + outputRow0 + ox + 8, evenHigh, w[0]);
                                AddStorePtr(dst + outputRow0 + ox, oddLow, w[1]); AddStorePtr(dst + outputRow0 + ox + 8, oddHigh, w[1]);
                                AddStorePtr(dst + outputRow1 + ox, evenLow, w[2]); AddStorePtr(dst + outputRow1 + ox + 8, evenHigh, w[2]);
                                AddStorePtr(dst + outputRow1 + ox, oddLow, w[3]); AddStorePtr(dst + outputRow1 + ox + 8, oddHigh, w[3]);
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

    // A transposed 2x2/stride-2 convolution has independent output channels.
    // Keep four channels in flight so each input vector is expanded only once
    // instead of once per output channel.  Each channel still visits inputs in
    // the original ci/y/x order, preserving deterministic accumulation.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ConvTranspose2x2Stride2FourOutputsUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputHeight = checked(inputHeight * 2);
        int outputWidth = checked(inputWidth * 2), outputPlane = checked(outputHeight * outputWidth);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 4)
                {
                    float* d0 = outputPtr + (b * outputChannels + co) * outputPlane;
                    float* d1 = d0 + outputPlane, d2 = d1 + outputPlane, d3 = d2 + outputPlane;
                    float v0 = biasPtr == null ? 0f : biasPtr[co], v1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float v2 = biasPtr == null ? 0f : biasPtr[co + 2], v3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    Vector256<float> b0 = Vector256.Create(v0), b1 = Vector256.Create(v1);
                    Vector256<float> b2 = Vector256.Create(v2), b3 = Vector256.Create(v3);
                    int inputBatch = b * inputChannels * inputPlane;
                    int outputBase = (b * outputChannels + co) * outputPlane;
                    for (int i = 0; i < outputPlane; i += 8)
                    {
                        Avx.Store(d0 + i, b0); Avx.Store(d1 + i, b1);
                        Avx.Store(d2 + i, b2); Avx.Store(d3 + i, b3);
                    }
                    for (int ci = 0; ci < inputChannels; ci++)
                    {
                        float* src = inputPtr + inputBatch + ci * inputPlane;
                        int wb0 = (ci * outputChannels + co) * 4;
                        int wb1 = wb0 + 4, wb2 = wb1 + 4, wb3 = wb2 + 4;
                        for (int iy = 0; iy < inputHeight; iy++)
                        {
                            int inputRow = iy * inputWidth;
                            int outputRow0 = (iy * 2) * outputWidth;
                            int outputRow1 = outputRow0 + outputWidth;
                            int ix = 0;
                            for (; ix <= inputWidth - 8; ix += 8)
                            {
                                Vector256<float> values = Avx.LoadVector256(src + inputRow + ix);
                                Vector256<float> zero = Vector256<float>.Zero;
                                Vector256<float> evenLo = Avx.UnpackLow(values, zero), evenHi = Avx.UnpackHigh(values, zero);
                                Vector256<float> oddLo = Avx.UnpackLow(zero, values), oddHi = Avx.UnpackHigh(zero, values);
                                Vector256<float> evenLow = Avx2.Permute2x128(evenLo, evenHi, 0x20);
                                Vector256<float> evenHigh = Avx2.Permute2x128(evenLo, evenHi, 0x31);
                                Vector256<float> oddLow = Avx2.Permute2x128(oddLo, oddHi, 0x20);
                                Vector256<float> oddHigh = Avx2.Permute2x128(oddLo, oddHi, 0x31);
                                int ox = ix * 2;
                                float w00 = weights[wb0], w01 = weights[wb0 + 1], w02 = weights[wb0 + 2], w03 = weights[wb0 + 3];
                                float w10 = weights[wb1], w11 = weights[wb1 + 1], w12 = weights[wb1 + 2], w13 = weights[wb1 + 3];
                                float w20 = weights[wb2], w21 = weights[wb2 + 1], w22 = weights[wb2 + 2], w23 = weights[wb2 + 3];
                                float w30 = weights[wb3], w31 = weights[wb3 + 1], w32 = weights[wb3 + 2], w33 = weights[wb3 + 3];
                                AddStorePtr(d0 + outputRow0 + ox, evenLow, w00); AddStorePtr(d0 + outputRow0 + ox + 8, evenHigh, w00);
                                AddStorePtr(d0 + outputRow0 + ox, oddLow, w01); AddStorePtr(d0 + outputRow0 + ox + 8, oddHigh, w01);
                                AddStorePtr(d0 + outputRow1 + ox, evenLow, w02); AddStorePtr(d0 + outputRow1 + ox + 8, evenHigh, w02);
                                AddStorePtr(d0 + outputRow1 + ox, oddLow, w03); AddStorePtr(d0 + outputRow1 + ox + 8, oddHigh, w03);
                                AddStorePtr(d1 + outputRow0 + ox, evenLow, w10); AddStorePtr(d1 + outputRow0 + ox + 8, evenHigh, w10);
                                AddStorePtr(d1 + outputRow0 + ox, oddLow, w11); AddStorePtr(d1 + outputRow0 + ox + 8, oddHigh, w11);
                                AddStorePtr(d1 + outputRow1 + ox, evenLow, w12); AddStorePtr(d1 + outputRow1 + ox + 8, evenHigh, w12);
                                AddStorePtr(d1 + outputRow1 + ox, oddLow, w13); AddStorePtr(d1 + outputRow1 + ox + 8, oddHigh, w13);
                                AddStorePtr(d2 + outputRow0 + ox, evenLow, w20); AddStorePtr(d2 + outputRow0 + ox + 8, evenHigh, w20);
                                AddStorePtr(d2 + outputRow0 + ox, oddLow, w21); AddStorePtr(d2 + outputRow0 + ox + 8, oddHigh, w21);
                                AddStorePtr(d2 + outputRow1 + ox, evenLow, w22); AddStorePtr(d2 + outputRow1 + ox + 8, evenHigh, w22);
                                AddStorePtr(d2 + outputRow1 + ox, oddLow, w23); AddStorePtr(d2 + outputRow1 + ox + 8, oddHigh, w23);
                                AddStorePtr(d3 + outputRow0 + ox, evenLow, w30); AddStorePtr(d3 + outputRow0 + ox + 8, evenHigh, w30);
                                AddStorePtr(d3 + outputRow0 + ox, oddLow, w31); AddStorePtr(d3 + outputRow0 + ox + 8, oddHigh, w31);
                                AddStorePtr(d3 + outputRow1 + ox, evenLow, w32); AddStorePtr(d3 + outputRow1 + ox + 8, evenHigh, w32);
                                AddStorePtr(d3 + outputRow1 + ox, oddLow, w33); AddStorePtr(d3 + outputRow1 + ox + 8, oddHigh, w33);
                            }
                            for (; ix < inputWidth; ix++)
                            {
                                float value = src[inputRow + ix]; int ox = ix * 2;
                                output[outputBase + outputRow0 + ox] += value * weights[wb0]; output[outputBase + outputRow0 + ox + 1] += value * weights[wb0 + 1];
                                output[outputBase + outputRow1 + ox] += value * weights[wb0 + 2]; output[outputBase + outputRow1 + ox + 1] += value * weights[wb0 + 3];
                                output[outputBase + outputRow0 + ox + outputPlane] += value * weights[wb1]; output[outputBase + outputRow0 + ox + outputPlane + 1] += value * weights[wb1 + 1];
                                output[outputBase + outputRow1 + ox + outputPlane] += value * weights[wb1 + 2]; output[outputBase + outputRow1 + ox + outputPlane + 1] += value * weights[wb1 + 3];
                                output[outputBase + outputRow0 + ox + outputPlane * 2] += value * weights[wb2]; output[outputBase + outputRow0 + ox + outputPlane * 2 + 1] += value * weights[wb2 + 1];
                                output[outputBase + outputRow1 + ox + outputPlane * 2] += value * weights[wb2 + 2]; output[outputBase + outputRow1 + ox + outputPlane * 2 + 1] += value * weights[wb2 + 3];
                                output[outputBase + outputRow0 + ox + outputPlane * 3] += value * weights[wb3]; output[outputBase + outputRow0 + ox + outputPlane * 3 + 1] += value * weights[wb3 + 1];
                                output[outputBase + outputRow1 + ox + outputPlane * 3] += value * weights[wb3 + 2]; output[outputBase + outputRow1 + ox + outputPlane * 3 + 1] += value * weights[wb3 + 3];
                            }
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void ConvTranspose2x2Stride2RangeUnsafe(ReadOnlySpan<float> input,
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
                    Vector256<float> vb = Vector256.Create(initial);
                    int i = 0;
                    for (; i <= outputPlane - 8; i += 8) Avx.Store(dst + i, vb);
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
                            for (; ix <= inputWidth - 8; ix += 8)
                            {
                                Vector256<float> values = Avx.LoadVector256(src + inputRow + ix);
                                Vector256<float> zero = Vector256<float>.Zero;
                                Vector256<float> evenLo = Avx.UnpackLow(values, zero); Vector256<float> evenHi = Avx.UnpackHigh(values, zero);
                                Vector256<float> oddLo = Avx.UnpackLow(zero, values); Vector256<float> oddHi = Avx.UnpackHigh(zero, values);
                                Vector256<float> evenLow = Avx2.Permute2x128(evenLo, evenHi, 0x20);
                                Vector256<float> evenHigh = Avx2.Permute2x128(evenLo, evenHi, 0x31);
                                Vector256<float> oddLow = Avx2.Permute2x128(oddLo, oddHi, 0x20);
                                Vector256<float> oddHigh = Avx2.Permute2x128(oddLo, oddHi, 0x31);
                                int ox = ix * 2;
                                AddStorePtr(dst + outputRow0 + ox, evenLow, w[0]); AddStorePtr(dst + outputRow0 + ox + 8, evenHigh, w[0]);
                                AddStorePtr(dst + outputRow0 + ox, oddLow, w[1]); AddStorePtr(dst + outputRow0 + ox + 8, oddHigh, w[1]);
                                AddStorePtr(dst + outputRow1 + ox, evenLow, w[2]); AddStorePtr(dst + outputRow1 + ox + 8, evenHigh, w[2]);
                                AddStorePtr(dst + outputRow1 + ox, oddLow, w[3]); AddStorePtr(dst + outputRow1 + ox + 8, oddHigh, w[3]);
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
    private static unsafe void AddStorePtr(float* destination, Vector256<float> value, float weight)
    {
        Vector256<float> current = Avx.LoadVector256(destination);
        Vector256<float> weightVector = Vector256.Create(weight);
        Avx.Store(destination, Fma.IsSupported
            ? Fma.MultiplyAdd(value, weightVector, current)
            : Avx.Add(current, Avx.Multiply(value, weightVector)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddStore(Span<float> destination, int offset, Vector256<float> value, float weight)
    {
        Vector256<float> current = Load(destination, offset);
        Store(destination, offset, Avx.Add(current, Avx.Multiply(value, Vector256.Create(weight))));
    }
}
