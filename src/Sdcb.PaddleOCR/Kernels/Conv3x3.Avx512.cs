using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Conv3x3
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void TrySixteenOutputsAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerOutput = checked(inputChannels * 9);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 16)
                {
                    float* w0 = weightsPtr + co * weightsPerOutput;
                    float* w1 = w0 + weightsPerOutput, w2 = w1 + weightsPerOutput;
                    float* w3 = w2 + weightsPerOutput, w4 = w3 + weightsPerOutput;
                    float* w5 = w4 + weightsPerOutput, w6 = w5 + weightsPerOutput;
                    float* w7 = w6 + weightsPerOutput, w8 = w7 + weightsPerOutput;
                    float* w9 = w8 + weightsPerOutput, w10 = w9 + weightsPerOutput;
                    float* w11 = w10 + weightsPerOutput, w12 = w11 + weightsPerOutput;
                    float* w13 = w12 + weightsPerOutput, w14 = w13 + weightsPerOutput;
                    float* w15 = w14 + weightsPerOutput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane;
                    float* o1 = o0 + plane, o2 = o1 + plane, o3 = o2 + plane;
                    float* o4 = o3 + plane, o5 = o4 + plane, o6 = o5 + plane, o7 = o6 + plane;
                    float* o8 = o7 + plane, o9 = o8 + plane, o10 = o9 + plane, o11 = o10 + plane;
                    float* o12 = o11 + plane, o13 = o12 + plane, o14 = o13 + plane, o15 = o14 + plane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4], b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6], b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    float b8 = biasPtr == null ? 0f : biasPtr[co + 8], b9 = biasPtr == null ? 0f : biasPtr[co + 9];
                    float b10 = biasPtr == null ? 0f : biasPtr[co + 10], b11 = biasPtr == null ? 0f : biasPtr[co + 11];
                    float b12 = biasPtr == null ? 0f : biasPtr[co + 12], b13 = biasPtr == null ? 0f : biasPtr[co + 13];
                    float b14 = biasPtr == null ? 0f : biasPtr[co + 14], b15 = biasPtr == null ? 0f : biasPtr[co + 15];
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        while (x < width)
                        {
                            bool vector = y > 0 && y + 1 < height && x > 0 && x + 16 < width;
                            if (vector)
                            {
                                Vector512<float> a0 = Vector512.Create(b0), a1 = Vector512.Create(b1);
                                Vector512<float> a2 = Vector512.Create(b2), a3 = Vector512.Create(b3);
                                Vector512<float> a4 = Vector512.Create(b4), a5 = Vector512.Create(b5);
                                Vector512<float> a6 = Vector512.Create(b6), a7 = Vector512.Create(b7);
                                Vector512<float> a8 = Vector512.Create(b8), a9 = Vector512.Create(b9);
                                Vector512<float> a10 = Vector512.Create(b10), a11 = Vector512.Create(b11);
                                Vector512<float> a12 = Vector512.Create(b12), a13 = Vector512.Create(b13);
                                Vector512<float> a14 = Vector512.Create(b14), a15 = Vector512.Create(b15);
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        float* srcRow = src + (y + ky - 1) * width;
                                        int ix = x - 1;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            Vector512<float> value = Avx512F.LoadVector512(srcRow + ix + kx);
                                            int wi = wb + ky * 3 + kx;
                                            a0 = AddMul512(a0, value, w0[wi]); a1 = AddMul512(a1, value, w1[wi]);
                                            a2 = AddMul512(a2, value, w2[wi]); a3 = AddMul512(a3, value, w3[wi]);
                                            a4 = AddMul512(a4, value, w4[wi]); a5 = AddMul512(a5, value, w5[wi]);
                                            a6 = AddMul512(a6, value, w6[wi]); a7 = AddMul512(a7, value, w7[wi]);
                                            a8 = AddMul512(a8, value, w8[wi]); a9 = AddMul512(a9, value, w9[wi]);
                                            a10 = AddMul512(a10, value, w10[wi]); a11 = AddMul512(a11, value, w11[wi]);
                                            a12 = AddMul512(a12, value, w12[wi]); a13 = AddMul512(a13, value, w13[wi]);
                                            a14 = AddMul512(a14, value, w14[wi]); a15 = AddMul512(a15, value, w15[wi]);
                                        }
                                    }
                                }
                                Avx512F.Store(o0 + row + x, a0); Avx512F.Store(o1 + row + x, a1);
                                Avx512F.Store(o2 + row + x, a2); Avx512F.Store(o3 + row + x, a3);
                                Avx512F.Store(o4 + row + x, a4); Avx512F.Store(o5 + row + x, a5);
                                Avx512F.Store(o6 + row + x, a6); Avx512F.Store(o7 + row + x, a7);
                                Avx512F.Store(o8 + row + x, a8); Avx512F.Store(o9 + row + x, a9);
                                Avx512F.Store(o10 + row + x, a10); Avx512F.Store(o11 + row + x, a11);
                                Avx512F.Store(o12 + row + x, a12); Avx512F.Store(o13 + row + x, a13);
                                Avx512F.Store(o14 + row + x, a14); Avx512F.Store(o15 + row + x, a15);
                                x += 16;
                            }
                            else
                            {
                                float s0 = b0, s1 = b1, s2 = b2, s3 = b3, s4 = b4, s5 = b5, s6 = b6, s7 = b7;
                                float s8 = b8, s9 = b9, s10 = b10, s11 = b11, s12 = b12, s13 = b13, s14 = b14, s15 = b15;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = y + ky - 1;
                                        if ((uint)iy >= (uint)height) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = x + kx - 1;
                                            if ((uint)ix >= (uint)width) continue;
                                            float value = src[iy * width + ix];
                                            int wi = wb + ky * 3 + kx;
                                            s0 += value * w0[wi]; s1 += value * w1[wi]; s2 += value * w2[wi]; s3 += value * w3[wi];
                                            s4 += value * w4[wi]; s5 += value * w5[wi]; s6 += value * w6[wi]; s7 += value * w7[wi];
                                            s8 += value * w8[wi]; s9 += value * w9[wi]; s10 += value * w10[wi]; s11 += value * w11[wi];
                                            s12 += value * w12[wi]; s13 += value * w13[wi]; s14 += value * w14[wi]; s15 += value * w15[wi];
                                        }
                                    }
                                }
                                o0[row + x] = s0; o1[row + x] = s1; o2[row + x] = s2; o3[row + x] = s3;
                                o4[row + x] = s4; o5[row + x] = s5; o6[row + x] = s6; o7[row + x] = s7;
                                o8[row + x] = s8; o9[row + x] = s9; o10[row + x] = s10; o11[row + x] = s11;
                                o12[row + x] = s12; o13[row + x] = s13; o14[row + x] = s14; o15[row + x] = s15;
                                x++;
                            }
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void TryEightOutputsAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerOutput = checked(inputChannels * 9);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* w0 = weightsPtr + co * weightsPerOutput;
                    float* w1 = w0 + weightsPerOutput, w2 = w1 + weightsPerOutput;
                    float* w3 = w2 + weightsPerOutput, w4 = w3 + weightsPerOutput;
                    float* w5 = w4 + weightsPerOutput, w6 = w5 + weightsPerOutput;
                    float* w7 = w6 + weightsPerOutput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane;
                    float* o1 = o0 + plane, o2 = o1 + plane, o3 = o2 + plane;
                    float* o4 = o3 + plane, o5 = o4 + plane, o6 = o5 + plane, o7 = o6 + plane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co];
                    float b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2];
                    float b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4];
                    float b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6];
                    float b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        while (x < width)
                        {
                            bool vector = y > 0 && y + 1 < height && x > 0 && x + 16 < width;
                            if (vector)
                            {
                                Vector512<float> a0 = Vector512.Create(b0), a1 = Vector512.Create(b1);
                                Vector512<float> a2 = Vector512.Create(b2), a3 = Vector512.Create(b3);
                                Vector512<float> a4 = Vector512.Create(b4), a5 = Vector512.Create(b5);
                                Vector512<float> a6 = Vector512.Create(b6), a7 = Vector512.Create(b7);
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        float* srcRow = src + (y + ky - 1) * width;
                                        int ix = x - 1;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            Vector512<float> value = Avx512F.LoadVector512(srcRow + ix + kx);
                                            int wi = wb + ky * 3 + kx;
                                            a0 = AddMul512(a0, value, w0[wi]); a1 = AddMul512(a1, value, w1[wi]);
                                            a2 = AddMul512(a2, value, w2[wi]); a3 = AddMul512(a3, value, w3[wi]);
                                            a4 = AddMul512(a4, value, w4[wi]); a5 = AddMul512(a5, value, w5[wi]);
                                            a6 = AddMul512(a6, value, w6[wi]); a7 = AddMul512(a7, value, w7[wi]);
                                        }
                                    }
                                }
                                Avx512F.Store(o0 + row + x, a0); Avx512F.Store(o1 + row + x, a1);
                                Avx512F.Store(o2 + row + x, a2); Avx512F.Store(o3 + row + x, a3);
                                Avx512F.Store(o4 + row + x, a4); Avx512F.Store(o5 + row + x, a5);
                                Avx512F.Store(o6 + row + x, a6); Avx512F.Store(o7 + row + x, a7);
                                x += 16;
                            }
                            else
                            {
                                float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
                                float s4 = b4, s5 = b5, s6 = b6, s7 = b7;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = y + ky - 1;
                                        if ((uint)iy >= (uint)height) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = x + kx - 1;
                                            if ((uint)ix >= (uint)width) continue;
                                            float value = src[iy * width + ix];
                                            int wi = wb + ky * 3 + kx;
                                            s0 += value * w0[wi]; s1 += value * w1[wi];
                                            s2 += value * w2[wi]; s3 += value * w3[wi];
                                            s4 += value * w4[wi]; s5 += value * w5[wi];
                                            s6 += value * w6[wi]; s7 += value * w7[wi];
                                        }
                                    }
                                }
                                o0[row + x] = s0; o1[row + x] = s1; o2[row + x] = s2; o3[row + x] = s3;
                                o4[row + x] = s4; o5[row + x] = s5; o6[row + x] = s6; o7[row + x] = s7;
                                x++;
                            }
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void TryFourOutputsAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerOutput = checked(inputChannels * 9);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 4)
                {
                    float* w0 = weightsPtr + co * weightsPerOutput, w1 = w0 + weightsPerOutput;
                    float* w2 = w1 + weightsPerOutput, w3 = w2 + weightsPerOutput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane, o1 = o0 + plane;
                    float* o2 = o1 + plane, o3 = o2 + plane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        while (x < width)
                        {
                            bool vector16 = y > 0 && y + 1 < height && x > 0 && x + 32 < width;
                            if (vector16)
                            {
                                Vector512<float> a0l = Vector512.Create(b0), a0h = a0l;
                                Vector512<float> a1l = Vector512.Create(b1), a1h = a1l;
                                Vector512<float> a2l = Vector512.Create(b2), a2h = a2l;
                                Vector512<float> a3l = Vector512.Create(b3), a3h = a3l;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        float* srcRow = src + (y + ky - 1) * width;
                                        int ix = x - 1;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            Vector512<float> valueLow = Avx512F.LoadVector512(srcRow + ix + kx);
                                            Vector512<float> valueHigh = Avx512F.LoadVector512(srcRow + ix + kx + 16);
                                            int wi = wb + ky * 3 + kx;
                                            a0l = AddMul512(a0l, valueLow, w0[wi]); a0h = AddMul512(a0h, valueHigh, w0[wi]);
                                            a1l = AddMul512(a1l, valueLow, w1[wi]); a1h = AddMul512(a1h, valueHigh, w1[wi]);
                                            a2l = AddMul512(a2l, valueLow, w2[wi]); a2h = AddMul512(a2h, valueHigh, w2[wi]);
                                            a3l = AddMul512(a3l, valueLow, w3[wi]); a3h = AddMul512(a3h, valueHigh, w3[wi]);
                                        }
                                    }
                                }
                                Avx512F.Store(o0 + row + x, a0l); Avx512F.Store(o0 + row + x + 16, a0h);
                                Avx512F.Store(o1 + row + x, a1l); Avx512F.Store(o1 + row + x + 16, a1h);
                                Avx512F.Store(o2 + row + x, a2l); Avx512F.Store(o2 + row + x + 16, a2h);
                                Avx512F.Store(o3 + row + x, a3l); Avx512F.Store(o3 + row + x + 16, a3h);
                                x += 32;
                                continue;
                            }
                            bool vector = y > 0 && y + 1 < height && x > 0 && x + 16 < width;
                            if (vector)
                            {
                                Vector512<float> a0 = Vector512.Create(b0); Vector512<float> a1 = Vector512.Create(b1);
                                Vector512<float> a2 = Vector512.Create(b2); Vector512<float> a3 = Vector512.Create(b3);
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        float* srcRow = src + (y + ky - 1) * width;
                                        int ix = x - 1;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            Vector512<float> value = Avx512F.LoadVector512(srcRow + ix + kx);
                                            int wi = wb + ky * 3 + kx;
                                            a0 = AddMul512(a0, value, w0[wi]); a1 = AddMul512(a1, value, w1[wi]);
                                            a2 = AddMul512(a2, value, w2[wi]); a3 = AddMul512(a3, value, w3[wi]);
                                        }
                                    }
                                }
                                Avx512F.Store(o0 + row + x, a0); Avx512F.Store(o1 + row + x, a1);
                                Avx512F.Store(o2 + row + x, a2); Avx512F.Store(o3 + row + x, a3);
                                x += 16;
                            }
                            else
                            {
                                float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane; int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = y + ky - 1; if ((uint)iy >= (uint)height) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = x + kx - 1; if ((uint)ix >= (uint)width) continue;
                                            float v = src[iy * width + ix]; int wi = wb + ky * 3 + kx;
                                            s0 += v * w0[wi]; s1 += v * w1[wi]; s2 += v * w2[wi]; s3 += v * w3[wi];
                                        }
                                    }
                                }
                                o0[row + x] = s0; o1[row + x] = s1; o2[row + x] = s2; o3[row + x] = s3;
                                x++;
                            }
                        }
                    }
                }
        }
    }

    private static void TryFourOutputsAvx512(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels, int height,
        int width, int outputChannels, int plane, int weightsPerOutput)
    {
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co += 4)
            {
                int inputBatch = b * inputChannels * plane;
                int outputBatch = b * outputChannels * plane;
                int weightBase0 = co * weightsPerOutput, weightBase1 = (co + 1) * weightsPerOutput;
                int weightBase2 = (co + 2) * weightsPerOutput, weightBase3 = (co + 3) * weightsPerOutput;
                float bias0 = bias.IsEmpty ? 0f : bias[co], bias1 = bias.IsEmpty ? 0f : bias[co + 1];
                float bias2 = bias.IsEmpty ? 0f : bias[co + 2], bias3 = bias.IsEmpty ? 0f : bias[co + 3];
                for (int y = 0; y < height; y++)
                {
                    int row = y * width;
                    int x = 0;
                    while (x < width)
                    {
                        bool vector = y > 0 && y + 1 < height && x > 0 && x + 16 < width;
                        if (vector)
                        {
                            Vector512<float> a0 = Vector512.Create(bias0); Vector512<float> a1 = Vector512.Create(bias1);
                            Vector512<float> a2 = Vector512.Create(bias2); Vector512<float> a3 = Vector512.Create(bias3);
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                ReadOnlySpan<float> source = input.Slice(inputBatch + ci * plane, plane);
                                int wb0 = weightBase0 + ci * 9, wb1 = weightBase1 + ci * 9;
                                int wb2 = weightBase2 + ci * 9, wb3 = weightBase3 + ci * 9;
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    int sourceRow = (y + ky - 1) * width;
                                    int sourceOffset = sourceRow + x - 1;
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        Vector512<float> value = Load512(source, sourceOffset + kx);
                                        a0 = AddMul512(a0, value, weights[wb0 + ky * 3 + kx]);
                                        a1 = AddMul512(a1, value, weights[wb1 + ky * 3 + kx]);
                                        a2 = AddMul512(a2, value, weights[wb2 + ky * 3 + kx]);
                                        a3 = AddMul512(a3, value, weights[wb3 + ky * 3 + kx]);
                                    }
                                }
                            }
                            Store512(output, outputBatch + co * plane + row + x, a0);
                            Store512(output, outputBatch + (co + 1) * plane + row + x, a1);
                            Store512(output, outputBatch + (co + 2) * plane + row + x, a2);
                            Store512(output, outputBatch + (co + 3) * plane + row + x, a3);
                            x += 16;
                            continue;
                        }
                        float s0 = bias0, s1 = bias1, s2 = bias2, s3 = bias3;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            ReadOnlySpan<float> source = input.Slice(inputBatch + ci * plane, plane);
                            int wb0 = weightBase0 + ci * 9, wb1 = weightBase1 + ci * 9;
                            int wb2 = weightBase2 + ci * 9, wb3 = weightBase3 + ci * 9;
                            for (int ky = 0; ky < 3; ky++)
                            {
                                int iy = y + ky - 1;
                                if ((uint)iy >= (uint)height) continue;
                                for (int kx = 0; kx < 3; kx++)
                                {
                                    int ix = x + kx - 1;
                                    if ((uint)ix >= (uint)width) continue;
                                    float v = source[iy * width + ix];
                                    s0 += v * weights[wb0 + ky * 3 + kx]; s1 += v * weights[wb1 + ky * 3 + kx];
                                    s2 += v * weights[wb2 + ky * 3 + kx]; s3 += v * weights[wb3 + ky * 3 + kx];
                                }
                            }
                        }
                        output[outputBatch + co * plane + row + x] = s0;
                        output[outputBatch + (co + 1) * plane + row + x] = s1;
                        output[outputBatch + (co + 2) * plane + row + x] = s2;
                        output[outputBatch + (co + 3) * plane + row + x] = s3;
                        x++;
                    }
                }
            }
    }
}
